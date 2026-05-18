"""
Creates WebDataset tar shards from a local annotations.jsonl + images directory.

Usage:
  python scripts/create_shards.py \\
    --annotations /path/to/annotations.jsonl \\
    --images /path/to/images/ \\
    --out data/shards/ \\
    --shard-size 1000   # images per shard

Output: data/shards/000000.tar, data/shards/000001.tar, ...

To upload shards to S3:
  aws s3 sync data/shards/ s3://my-bucket/shards/
"""
from __future__ import annotations
import argparse
import json
import tarfile
import io
from pathlib import Path


def create_shards(
    annotations_path: Path,
    images_dir: Path,
    out_dir: Path,
    shard_size: int = 1000,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    with annotations_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    shard_idx = 0
    buf: list[tuple[str, bytes, dict]] = []

    def flush_shard(items: list, idx: int) -> None:
        shard_path = out_dir / f"{idx:06d}.tar"
        with tarfile.open(shard_path, "w") as tar:
            for i, (img_name, img_bytes, annotation) in enumerate(items):
                # image
                img_info = tarfile.TarInfo(name=f"{i:06d}.jpg")
                img_info.size = len(img_bytes)
                tar.addfile(img_info, io.BytesIO(img_bytes))
                # annotation
                ann_bytes = json.dumps(annotation).encode()
                ann_info = tarfile.TarInfo(name=f"{i:06d}.json")
                ann_info.size = len(ann_bytes)
                tar.addfile(ann_info, io.BytesIO(ann_bytes))
        print(f"  wrote {shard_path} ({len(items)} items)")

    for global_idx, rec in enumerate(records):
        img_path_str = rec.get("metadata", {}).get("image_path", "")
        if img_path_str:
            img_path = images_dir / Path(img_path_str).name
        else:
            img_path = images_dir / f"cord_{global_idx:05d}.jpg"

        if not img_path.exists():
            continue

        img_bytes = img_path.read_bytes()
        buf.append((img_path.name, img_bytes, rec))

        if len(buf) >= shard_size:
            flush_shard(buf, shard_idx)
            shard_idx += 1
            buf = []

    if buf:
        flush_shard(buf, shard_idx)

    print(f"Done: {shard_idx + (1 if buf else 0)} shards in {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--shard-size", type=int, default=1000)
    args = parser.parse_args()
    create_shards(
        Path(args.annotations), Path(args.images), Path(args.out), args.shard_size
    )
