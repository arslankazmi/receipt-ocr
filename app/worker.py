"""
Celery worker for async receipt extraction at scale.

Start worker:
  celery -A app.worker worker --concurrency=4

Tasks:
  extract_receipt(image_bytes_b64: str) -> dict  — runs Donut inference, returns schema JSON
"""
from __future__ import annotations

import base64
import io
import json
import os

from celery import Celery

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
celery_app = Celery("receipt_worker", broker=REDIS_URL, backend=REDIS_URL)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
)


@celery_app.task(bind=True, max_retries=3, default_retry_delay=5)
def extract_receipt(self, image_bytes_b64: str) -> dict:
    """Extract structured data from a base64-encoded receipt image."""
    try:
        import yaml
        from PIL import Image
        from transformers import DonutProcessor, VisionEncoderDecoderModel

        with open("config.yaml") as f:
            cfg = yaml.safe_load(f)

        image_bytes = base64.b64decode(image_bytes_b64)
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        processor = DonutProcessor.from_pretrained(cfg["model"]["path"])
        model = VisionEncoderDecoderModel.from_pretrained(cfg["model"]["path"])
        model.eval()

        pixel_values = processor(img, return_tensors="pt").pixel_values
        task_prompt = "<s_receipt>"
        decoder_input_ids = processor.tokenizer(
            task_prompt, add_special_tokens=False, return_tensors="pt"
        ).input_ids

        eos_id = processor.tokenizer.convert_tokens_to_ids(["</s_receipt>"])[0]

        outputs = model.generate(
            pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_length=512,
            early_stopping=True,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=eos_id,
        )
        seq = processor.batch_decode(outputs, skip_special_tokens=False)[0]
        seq = seq.replace(processor.tokenizer.eos_token, "").replace(
            processor.tokenizer.pad_token, ""
        )
        # Extract JSON between task tokens
        start = seq.find("<s_receipt>") + len("<s_receipt>")
        end = seq.find("</s_receipt>")
        json_str = seq[start:end].strip() if end > start else seq
        return json.loads(json_str)
    except Exception as exc:
        raise self.retry(exc=exc)
