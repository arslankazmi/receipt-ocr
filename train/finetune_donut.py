"""
Fine-tune Donut on receipt data using LoRA via PEFT + Accelerate.

Usage:
  # Single GPU (local):
  python train/finetune_donut.py [--config config.yaml] [--dry-run]

  # Multi-GPU (scale):
  accelerate launch --multi_gpu train/finetune_donut.py [--config config.yaml]

  # Checkpoint to S3:
  set config.yaml model.path to s3://bucket/checkpoints/donut-receipt-lora

--dry-run: loads model, runs 1 batch forward pass, saves nothing. For CI/verification.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
import yaml
from accelerate import Accelerator
from peft import LoraConfig, TaskType, get_peft_model
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import DonutProcessor, VisionEncoderDecoderModel
from transformers import get_linear_schedule_with_warmup

from train.donut_dataset import DonutReceiptDataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def save_checkpoint(model, processor, path: str) -> None:
    """Save model + processor; supports local paths and s3:// URIs."""
    if path.startswith("s3://"):
        import tempfile, os
        from data.storage import get_backend

        with tempfile.TemporaryDirectory() as tmp:
            model.save_pretrained(tmp)
            processor.save_pretrained(tmp)
            # Push every file to the storage backend
            backend = get_backend({"type": "s3", "uri": path})
            for local_file in Path(tmp).rglob("*"):
                if local_file.is_file():
                    rel = local_file.relative_to(tmp)
                    backend.upload(str(local_file), str(rel))
    else:
        Path(path).mkdir(parents=True, exist_ok=True)
        model.save_pretrained(path)
        processor.save_pretrained(path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune Donut on receipts")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run one batch forward pass then exit without saving",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]

    # ------------------------------------------------------------------
    # Accelerator — handles distributed / mixed-precision automatically
    # ------------------------------------------------------------------
    accelerator = Accelerator()
    device = accelerator.device

    if accelerator.is_main_process:
        print(f"[init] device={device}  detected={detect_device()}")

    # ------------------------------------------------------------------
    # Processor + model
    # ------------------------------------------------------------------
    base = model_cfg["base"]
    processor = DonutProcessor.from_pretrained(base)
    model = VisionEncoderDecoderModel.from_pretrained(base)

    # Add task tokens to the vocabulary
    new_tokens = [DonutReceiptDataset.TASK_TOKEN, DonutReceiptDataset.END_TOKEN]
    processor.tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
    model.decoder.resize_token_embeddings(len(processor.tokenizer))

    # Set decoder config tokens
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids(
        DonutReceiptDataset.TASK_TOKEN
    )

    # ------------------------------------------------------------------
    # LoRA via PEFT
    # ------------------------------------------------------------------
    lora_cfg = LoraConfig(
        r=train_cfg["lora_r"],
        lora_alpha=train_cfg["lora_alpha"],
        target_modules=["query", "value"],
        task_type=TaskType.SEQ_2_SEQ_LM,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    if accelerator.is_main_process:
        model.print_trainable_parameters()

    # ------------------------------------------------------------------
    # Datasets + DataLoaders
    # ------------------------------------------------------------------
    max_length = train_cfg["max_length"]
    batch_size = train_cfg["batch_size"]

    train_ds = DonutReceiptDataset(cfg, processor, split="train", max_length=max_length)
    val_ds = DonutReceiptDataset(cfg, processor, split="val", max_length=max_length)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        prefetch_factor=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        prefetch_factor=2,
        pin_memory=True,
    )

    # ------------------------------------------------------------------
    # Optimizer + scheduler (linear warmup over 10 % of total steps)
    # ------------------------------------------------------------------
    epochs = train_cfg["epochs"]
    optimizer = AdamW(model.parameters(), lr=train_cfg["lr"])

    total_steps = math.ceil(len(train_loader) / accelerator.gradient_accumulation_steps) * epochs
    warmup_steps = max(1, int(0.1 * total_steps))

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # ------------------------------------------------------------------
    # Wrap with Accelerate
    # ------------------------------------------------------------------
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    # ------------------------------------------------------------------
    # Dry-run: one forward pass then exit
    # ------------------------------------------------------------------
    if args.dry_run:
        model.train()
        batch = next(iter(train_loader))
        pixel_values = batch["pixel_values"]
        labels = batch["labels"]
        outputs = model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss
        accelerator.print(f"[dry-run] loss={loss.item():.4f}  EXIT")
        sys.exit(0)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    for epoch in range(1, epochs + 1):
        # --- train ---
        model.train()
        running_loss = 0.0
        for step, batch in enumerate(train_loader, 1):
            outputs = model(pixel_values=batch["pixel_values"], labels=batch["labels"])
            loss = outputs.loss
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            running_loss += loss.item()

        avg_train = running_loss / len(train_loader)

        # --- eval ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                outputs = model(pixel_values=batch["pixel_values"], labels=batch["labels"])
                val_loss += outputs.loss.item()
        avg_val = val_loss / max(len(val_loader), 1)

        accelerator.print(
            f"epoch={epoch}/{epochs}  train_loss={avg_train:.4f}  val_loss={avg_val:.4f}"
        )

    # ------------------------------------------------------------------
    # Save checkpoint
    # ------------------------------------------------------------------
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        checkpoint_path = model_cfg["path"]
        save_checkpoint(unwrapped, processor, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")


if __name__ == "__main__":
    main()
