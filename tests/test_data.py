"""Tests for data layer — uses local CORD data."""
import os
import json
from pathlib import Path
import pytest

CORD_ANNOTATIONS = os.environ.get(
    "CORD_ANNOTATIONS",
    "/Users/akazmi/Documents/arslan/arslan-projects/sandbox/datasets/receipts-grocery/public/cord/annotations.jsonl",
)
CORD_IMAGES = os.environ.get(
    "CORD_IMAGES",
    "/Users/akazmi/Documents/arslan/arslan-projects/sandbox/datasets/receipts-grocery/public/cord/images",
)

CORD_CFG = {
    "annotations": CORD_ANNOTATIONS,
    "images_dir": CORD_IMAGES,
    "val_ratio": 0.1,
    "seed": 42,
}


@pytest.mark.skipif(
    not Path(CORD_ANNOTATIONS).exists(), reason="CORD data not found"
)
def test_cord_dataset_splits():
    from data.cord_dataset import CordDataset
    train = CordDataset(CORD_CFG, split="train")
    val = CordDataset(CORD_CFG, split="val")
    assert len(train) + len(val) > 0
    assert len(val) >= 1
    assert len(train) > len(val)


@pytest.mark.skipif(
    not Path(CORD_ANNOTATIONS).exists(), reason="CORD data not found"
)
def test_cord_dataset_returns_image_and_dict():
    from data.cord_dataset import CordDataset
    val = CordDataset(CORD_CFG, split="val")
    img, rec = val[0]
    from PIL import Image
    assert isinstance(img, Image.Image)
    assert "items" in rec
    assert isinstance(rec["items"], list)


def test_local_backend():
    import tempfile, os
    from data.storage import LocalBackend
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        f.write(b"hello")
        tmp = f.name
    backend = LocalBackend(base=str(Path(tmp).parent))
    assert backend.exists(Path(tmp).name)
    data = backend.open(Path(tmp).name).read()
    assert data == b"hello"
    os.unlink(tmp)
