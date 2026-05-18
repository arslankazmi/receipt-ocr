"""
DonutCordRunner — uses naver-clova-ix/donut-base-finetuned-cord-v2

The model outputs JSON in CORD's internal format:
  {"menu": [{"nm": "...", "cnt": "1", "unitprice": "...", "price": "..."}],
   "sub_total": {"subtotal_price": "...", "storenm": "..."},
   "total": {"total_price": "..."}}

This runner remaps that to our schema format.
"""
from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from PIL import Image

from eval.models.base import ModelRunner

if TYPE_CHECKING:
    pass

_HF_MODEL = "naver-clova-ix/donut-base-finetuned-cord-v2"
_TASK_PROMPT = "<s_cord-v2>"


def _parse_number(val) -> float | None:
    """Convert a value to float, handling KRW locale, commas, Rp prefix, etc."""
    if val is None:
        return None
    text = str(val).strip()
    # Remove currency prefixes/symbols
    text = re.sub(r"^(Rp\.?|KRW|₩|\$|£|€)\s*", "", text, flags=re.IGNORECASE)
    # Remove thousands separators (commas)
    text = text.replace(",", "")
    # Remove trailing non-numeric characters (e.g. "원")
    text = re.sub(r"[^\d.+-]", "", text)
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


class DonutCordRunner(ModelRunner):
    model_id = "donut-cord"

    def __init__(self) -> None:
        self._processor = None
        self._model = None
        self._device = None

    def _load(self) -> None:
        if self._model is not None:
            return

        import torch
        from transformers import DonutProcessor, VisionEncoderDecoderModel

        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        self._device = device
        self._processor = DonutProcessor.from_pretrained(_HF_MODEL)
        self._model = VisionEncoderDecoderModel.from_pretrained(_HF_MODEL)
        self._model.to(device)
        self._model.eval()

    def _decode_cord_json(self, sequence: str) -> dict:
        """Extract and parse the JSON payload between <s_cord-v2> and </s_cord-v2>."""
        match = re.search(r"<s_cord-v2>(.*?)</s_cord-v2>", sequence, re.DOTALL)
        if match:
            payload = match.group(1).strip()
        else:
            # Fallback: try to parse the whole sequence as JSON
            payload = sequence.strip()
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            return {}

    def _remap(self, cord: dict) -> dict:
        """Remap CORD internal format to our schema."""
        items = []
        for idx, menu_item in enumerate(cord.get("menu") or [], start=1):
            qty_raw = menu_item.get("cnt")
            try:
                qty = float(qty_raw) if qty_raw else 1.0
            except (ValueError, TypeError):
                qty = 1.0

            items.append({
                "line_number": idx,
                "name": menu_item.get("nm") or "",
                "quantity": qty,
                "unit": "each",
                "unit_price": _parse_number(menu_item.get("unitprice")),
                "total": _parse_number(menu_item.get("price")),
            })

        sub_total = cord.get("sub_total") or {}
        total_block = cord.get("total") or {}

        store_name = sub_total.get("storenm") or ""
        total_val = _parse_number(total_block.get("total_price"))

        result: dict = {
            "store": {"name": store_name},
            "date": "1970-01-01",
            "currency": "KRW",
            "items": items,
        }
        if total_val is not None:
            result["total"] = total_val

        return result

    def extract(self, image: Image.Image) -> dict:
        import torch

        self._load()

        pixel_values = self._processor(
            image.convert("RGB"),
            return_tensors="pt",
        ).pixel_values.to(self._device)

        decoder_input_ids = self._processor.tokenizer(
            _TASK_PROMPT,
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids.to(self._device)

        with torch.no_grad():
            outputs = self._model.generate(
                pixel_values,
                decoder_input_ids=decoder_input_ids,
                max_length=self._model.decoder.config.max_position_embeddings,
                early_stopping=True,
                pad_token_id=self._processor.tokenizer.pad_token_id,
                eos_token_id=self._processor.tokenizer.eos_token_id,
                use_cache=True,
                num_beams=1,
                bad_words_ids=[[self._processor.tokenizer.unk_token_id]],
                return_dict_in_generate=True,
            )

        sequence = self._processor.batch_decode(
            outputs.sequences, skip_special_tokens=False
        )[0]
        sequence = self._processor.tokenizer.convert_tokens_to_string(
            self._processor.tokenizer.convert_ids_to_tokens(
                outputs.sequences[0].tolist()
            )
        )
        # Use the raw decoded sequence for JSON extraction
        raw_sequence = self._processor.batch_decode(
            outputs.sequences, skip_special_tokens=False
        )[0]

        cord_data = self._decode_cord_json(raw_sequence)
        return self._remap(cord_data)
