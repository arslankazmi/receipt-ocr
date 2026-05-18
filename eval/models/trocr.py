"""
TrOCRRunner — uses microsoft/trocr-base-printed

TrOCR extracts raw text from images. It has no understanding of receipt structure.
We use it as a "raw OCR text quality" baseline — it will score 0 on structured fields
but shows how well the underlying text is being read.

Since TrOCR outputs text (not JSON), this runner returns a minimal dict with
store.name and total attempted via simple regex parsing of the text.
"""
from __future__ import annotations

import re

from PIL import Image

from eval.models.base import ModelRunner

_HF_MODEL = "microsoft/trocr-base-printed"


def _extract_total(text: str) -> float | None:
    """
    Attempt to extract a total value from raw OCR text.

    Tries patterns like:
      TOTAL 12.34
      Total: $12.34
      TOTAL AMOUNT 9.99
    Falls back to the last price-like number in the text.
    """
    # Explicit TOTAL label patterns
    total_patterns = [
        r"(?i)total\s*(?:amount)?[:\s]*\$?\s*([\d,]+\.\d{2})",
        r"(?i)amount\s+due[:\s]*\$?\s*([\d,]+\.\d{2})",
        r"(?i)grand\s+total[:\s]*\$?\s*([\d,]+\.\d{2})",
    ]
    for pattern in total_patterns:
        m = re.search(pattern, text)
        if m:
            try:
                return float(m.group(1).replace(",", ""))
            except ValueError:
                continue

    # Fallback: last price-like number (e.g. 12.34 or 1,234.56)
    prices = re.findall(r"\b([\d,]+\.\d{2})\b", text)
    if prices:
        try:
            return float(prices[-1].replace(",", ""))
        except ValueError:
            pass

    return None


def _extract_store_name(text: str) -> str | None:
    """Return the first non-empty, non-numeric line as a store name guess."""
    for line in text.splitlines():
        line = line.strip()
        if line and not re.match(r"^[\d\s\-/:.]+$", line):
            return line
    return None


class TrOCRRunner(ModelRunner):
    model_id = "trocr-base"

    def __init__(self) -> None:
        self._processor = None
        self._model = None
        self._device = None

    def _load(self) -> None:
        if self._model is not None:
            return

        import torch
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        self._device = device
        self._processor = TrOCRProcessor.from_pretrained(_HF_MODEL)
        self._model = VisionEncoderDecoderModel.from_pretrained(_HF_MODEL)
        self._model.to(device)
        self._model.eval()

    def extract(self, image: Image.Image) -> dict:
        import torch

        self._load()

        pixel_values = self._processor(
            image.convert("RGB"),
            return_tensors="pt",
        ).pixel_values.to(self._device)

        with torch.no_grad():
            generated_ids = self._model.generate(pixel_values)

        raw_text = self._processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]

        store_name = _extract_store_name(raw_text)
        total = _extract_total(raw_text)

        result: dict = {
            "store": {"name": store_name} if store_name else {},
            "date": "1970-01-01",
            "items": [],
            "metadata": {"notes": raw_text},
        }
        if total is not None:
            result["total"] = total

        return result
