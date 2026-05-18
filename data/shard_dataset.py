"""
ShardDataset — streams receipt data from WebDataset tar shards.

Shards are created by scripts/create_shards.py and can be stored locally or on S3.
Each shard is a .tar file containing:
  {idx:06d}.jpg  — the receipt image
  {idx:06d}.json — the schema-conforming annotation

Usage (local):
  ds = ShardDataset("data/shards/{000..009}.tar")

Usage (S3):
  ds = ShardDataset("s3://my-bucket/shards/{000..999}.tar")

This enables streaming over 550+ GB without loading the dataset into RAM.
"""
from __future__ import annotations
import json
from typing import Iterator
from PIL import Image
import io


def ShardDataset(url_pattern: str, *, shuffle_buffer: int = 1000) -> Iterator[tuple[Image.Image, dict]]:
    """
    Returns an iterator of (PIL.Image, annotation_dict) pairs.
    url_pattern: WebDataset URL pattern, e.g. "data/shards/{000..009}.tar" or "s3://..."
    """
    try:
        import webdataset as wds
    except ImportError:
        raise ImportError("Install webdataset: pip install webdataset")

    dataset = (
        wds.WebDataset(url_pattern, shardshuffle=True)
        .shuffle(shuffle_buffer)
        .decode("pil")
        .to_tuple("jpg", "json")
    )
    for img, annotation in dataset:
        if isinstance(annotation, (str, bytes)):
            annotation = json.loads(annotation)
        yield img, annotation
