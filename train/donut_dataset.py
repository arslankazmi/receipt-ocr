"""
DonutReceiptDataset — wraps CordDataset for Donut seq2seq training.

Donut expects:
  pixel_values: FloatTensor [C, H, W] (from DonutProcessor)
  labels: LongTensor of token ids (target JSON string)

Target format: <s_receipt>{schema_json}</s_receipt>
Truncated to max_length tokens.

Supports both local images (via CordDataset) and WebDataset shards (via ShardDataset)
— controlled by config: data.source = "cord" | "shards"
"""
from __future__ import annotations

import json
from typing import Literal

from torch.utils.data import Dataset
from transformers import DonutProcessor

from data.cord_dataset import CordDataset


class DonutReceiptDataset(Dataset):
    TASK_TOKEN = "<s_receipt>"
    END_TOKEN = "</s_receipt>"

    def __init__(
        self,
        cfg: dict,
        processor: DonutProcessor,
        split: Literal["train", "val"] = "train",
        max_length: int = 512,
    ):
        source = cfg.get("data", {}).get("source", "cord")
        if source == "shards":
            from data.shard_dataset import ShardDataset

            url = cfg["data"]["shard_url"]
            self._items = list(ShardDataset(url))
        else:
            cord_ds = CordDataset(cfg["data"], split=split)
            self._items = list(cord_ds)

        self.processor = processor
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> dict:
        img, annotation = self._items[idx]

        # Process image → pixel_values [C, H, W]
        pixel_values = self.processor(img, return_tensors="pt").pixel_values.squeeze(0)

        # Build target sequence
        target_json = json.dumps(annotation, ensure_ascii=False)
        target_seq = f"{self.TASK_TOKEN}{target_json}{self.END_TOKEN}"

        # Tokenize target
        target_enc = self.processor.tokenizer(
            target_seq,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        labels = target_enc.input_ids.squeeze(0)
        # Mask padding tokens so they are ignored in the loss
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {"pixel_values": pixel_values, "labels": labels}
