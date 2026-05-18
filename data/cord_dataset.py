"""
CordDataset — loads CORD annotations.jsonl paired with images.

Config keys used (from config.yaml data section):
  annotations: absolute path to annotations.jsonl
  images_dir:  absolute path to images directory
  val_ratio:   fraction of data for val split (default 0.1)
  seed:        random seed for split (default 42)

Usage:
  ds = CordDataset(cfg["data"], split="train")  # or "val"
  for img, annotation in ds:
      ...  # PIL.Image.Image, dict
"""
from __future__ import annotations
import json
import random
from pathlib import Path
from typing import Iterator, Literal
from PIL import Image


class CordDataset:
    def __init__(
        self,
        cfg: dict,
        split: Literal["train", "val"] = "train",
    ):
        annotations_path = Path(cfg["annotations"])
        images_dir = Path(cfg["images_dir"])
        val_ratio = cfg.get("val_ratio", 0.1)
        seed = cfg.get("seed", 42)

        records: list[dict] = []
        with annotations_path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

        # deterministic split
        rng = random.Random(seed)
        indices = list(range(len(records)))
        rng.shuffle(indices)
        n_val = max(1, int(len(records) * val_ratio))
        val_indices = set(indices[:n_val])

        self._pairs: list[tuple[Path, dict]] = []
        for i, rec in enumerate(records):
            img_path_str = rec.get("metadata", {}).get("image_path", "")
            if img_path_str:
                img_path = images_dir / Path(img_path_str).name
            else:
                # fallback: infer from index
                img_path = images_dir / f"cord_{i:05d}.jpg"

            in_val = i in val_indices
            if (split == "val") == in_val:
                self._pairs.append((img_path, rec))

    def __len__(self) -> int:
        return len(self._pairs)

    def __iter__(self) -> Iterator[tuple[Image.Image, dict]]:
        for img_path, rec in self._pairs:
            if img_path.exists():
                yield Image.open(img_path).convert("RGB"), rec

    def __getitem__(self, idx: int) -> tuple[Image.Image, dict]:
        img_path, rec = self._pairs[idx]
        return Image.open(img_path).convert("RGB"), rec
