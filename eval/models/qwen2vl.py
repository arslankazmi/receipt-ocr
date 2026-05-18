"""
Qwen2VLRunner — uses Qwen/Qwen2-VL-2B-Instruct

Prompts the model to extract receipt data as JSON conforming to our schema.
Requires ~4GB RAM/VRAM. On CPU it will be slow but will run.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from PIL import Image

from eval.models.base import ModelRunner

_HF_MODEL = "Qwen/Qwen2-VL-2B-Instruct"
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _load_schema() -> dict:
    schema_path = _PROJECT_ROOT / "schema.json"
    with open(schema_path) as f:
        return json.load(f)


def _strip_fences(text: str) -> str:
    """Remove markdown code fences if present."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        # Strip opening fence (e.g. ```json or ```)
        start = 1
        # Strip closing fence
        end = len(lines) - 1 if lines[-1].strip() == "```" else len(lines)
        text = "\n".join(lines[start:end])
    return text.strip()


class Qwen2VLRunner(ModelRunner):
    model_id = "qwen2vl-2b"

    def __init__(self) -> None:
        self._processor = None
        self._model = None
        self._device = None
        self._schema_json: str | None = None

    def _load(self) -> None:
        if self._model is not None:
            return

        import torch
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        self._device = device
        self._processor = AutoProcessor.from_pretrained(_HF_MODEL)
        self._model = Qwen2VLForConditionalGeneration.from_pretrained(
            _HF_MODEL,
            torch_dtype="auto",
        )
        self._model.to(device)
        self._model.eval()
        self._schema_json = json.dumps(_load_schema(), indent=2)

    def extract(self, image: Image.Image) -> dict:
        import torch

        self._load()

        prompt_text = (
            "Extract this receipt as JSON. Return only valid JSON matching this schema"
            " — no markdown fences:\n" + self._schema_json
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image.convert("RGB")},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

        # Apply chat template
        text = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Process inputs
        inputs = self._processor(
            text=[text],
            images=[image.convert("RGB")],
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=1024,
            )

        # Decode only newly generated tokens
        generated_ids = [
            out[len(inp):]
            for inp, out in zip(inputs["input_ids"], output_ids)
        ]
        response = self._processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        cleaned = _strip_fences(response)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return {}
