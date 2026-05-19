"""
Receipt OCR API

Endpoints:
  POST /extract        — sync: upload image, get GroceryReceipt JSON immediately
  POST /extract/async  — async: returns {"job_id": "..."}, poll GET /jobs/{job_id}
  GET  /jobs/{job_id}  — poll async job status: {"status": "pending|done|failed", "result": {...}}
  GET  /healthz        — {"status":"ok","model":"...","backend":"local|s3"}
  GET  /docs           — Swagger UI (FastAPI auto)
  GET  /               — redirect to /docs

Config loaded from config.yaml on startup.
CORS: allow all origins (for demo/index.html running from file://).
"""
from __future__ import annotations

import base64
import io
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global state (populated during lifespan startup)
# ---------------------------------------------------------------------------
_model = None
_processor = None
_config: dict = {}
_schema: dict = {}


def load_model(model_path: str, base_model: str):
    """Load DonutProcessor + VisionEncoderDecoderModel, applying LoRA if present.

    Tries local checkpoint first; falls back to HF cache (local_files_only=False
    allows reading from the HF cache dir without forcing a new network download
    when the model is already cached).
    """
    import torch
    from transformers import DonutProcessor, VisionEncoderDecoderModel

    checkpoint = Path(model_path)
    if checkpoint.exists():
        source = str(checkpoint)
    else:
        # model_path is a HF repo id — load from HF cache (already downloaded)
        source = model_path

    logger.info("Loading processor from %s", source)
    processor = DonutProcessor.from_pretrained(source)

    logger.info("Loading model from %s", source)
    model = VisionEncoderDecoderModel.from_pretrained(source)

    # Apply PEFT LoRA adapter if adapter_config.json exists alongside checkpoint
    adapter_cfg = checkpoint / "adapter_config.json"
    if adapter_cfg.exists():
        try:
            from peft import PeftModel
            logger.info("Applying LoRA adapter from %s", checkpoint)
            model = PeftModel.from_pretrained(model, str(checkpoint))
            model = model.merge_and_unload()
        except ImportError:
            logger.warning("peft not installed — skipping LoRA adapter loading")

    model.eval()
    return model, processor


def run_inference(model, processor, image_bytes: bytes) -> dict:
    """Run Donut-CORD inference on raw image bytes; return our schema dict."""
    import torch
    from PIL import Image

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Prepare inputs
    pixel_values = processor(img, return_tensors="pt").pixel_values

    # Detect device
    device = next(model.parameters()).device
    pixel_values = pixel_values.to(device)

    # Decode with CORD task token
    task_prompt = "<s_cord-v2>"
    decoder_input_ids = processor.tokenizer(
        task_prompt, add_special_tokens=False, return_tensors="pt"
    ).input_ids.to(device)

    with torch.no_grad():
        outputs = model.generate(
            pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_length=model.decoder.config.max_position_embeddings,
            early_stopping=True,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            use_cache=True,
            num_beams=1,
            bad_words_ids=[[processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
        ).sequences

    seq = processor.batch_decode(outputs, skip_special_tokens=False)[0]
    cord_data = processor.token2json(seq)
    if "cord-v2" in cord_data:
        cord_data = cord_data["cord-v2"]

    return _remap_cord_to_schema(cord_data)


def _remap_cord_to_schema(cord: dict) -> dict:
    """Remap CORD token2json output to our GroceryReceipt schema."""
    import re

    def _parse_num(v) -> float:
        if v is None:
            return 0.0
        s = re.sub(r"[^\d.,]", "", str(v))
        if "," in s and "." in s:
            s = s.replace(",", "") if s.index(",") < s.index(".") else s.replace(".", "").replace(",", ".")
        elif "," in s:
            s = s.replace(",", ".")
        try:
            return float(s)
        except Exception:
            return 0.0

    menu_raw = cord.get("menu", [])
    if isinstance(menu_raw, dict):
        menu_raw = [menu_raw]

    items = []
    for i, item in enumerate(menu_raw, 1):
        items.append({
            "line_number": i,
            "name": item.get("nm", "UNKNOWN"),
            "normalized_name": None,
            "quantity": _parse_num(item.get("cnt", "1")) or 1.0,
            "unit": "each",
            "unit_price": _parse_num(item.get("unitprice")) or None,
            "total": _parse_num(item.get("price", "0")),
            "sku": None,
            "category": None,
            "tax_code": None,
            "discount_applied": None,
        })

    sub = cord.get("sub_total", {})
    total_sec = cord.get("total", {})

    return {
        "store": {"name": sub.get("storenm", "") or "Unknown"},
        "date": "1970-01-01",
        "items": items or [{"line_number": 1, "name": "UNKNOWN", "quantity": 1, "unit": "each", "total": 0.0}],
        "total": _parse_num(total_sec.get("total_price", "0")),
        "currency": "KRW",
        "metadata": {"source": "user_scan", "annotator": "donut-cord"},
    }


def validate_against_schema(data: dict, schema: dict) -> list[str]:
    """Return list of validation error messages (empty if valid)."""
    if not schema:
        return []
    try:
        import jsonschema
        jsonschema.validate(instance=data, schema=schema)
        return []
    except jsonschema.ValidationError as exc:
        return [exc.message]
    except jsonschema.SchemaError as exc:
        logger.warning("Invalid schema: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _processor, _config, _schema

    # Load config
    config_path = Path(os.environ.get("CONFIG_PATH", "config.yaml"))
    if config_path.exists():
        with open(config_path) as f:
            _config = yaml.safe_load(f)
    else:
        logger.warning("config.yaml not found — using defaults")
        _config = {
            "model": {"path": "train/checkpoints/donut-receipt-lora", "base": "naver-clova-ix/donut-base"},
            "api": {"host": "0.0.0.0", "port": 8000, "cors_origins": ["*"]},
            "storage": {"backend": "local"},
        }

    # Load schema
    schema_path = Path("schema.json")
    if schema_path.exists():
        with open(schema_path) as f:
            _schema = json.load(f)

    # Load model
    try:
        _model, _processor = load_model(
            _config["model"]["path"],
            _config["model"]["base"],
        )
        logger.info("Model loaded successfully")
    except Exception as exc:
        logger.error("Failed to load model: %s", exc)
        # Allow app to start without model so /healthz still works

    yield

    # Shutdown — nothing to clean up
    _model = None
    _processor = None


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Receipt OCR API",
    description="Donut-based receipt extraction with LoRA fine-tuning",
    version="0.1.0",
    lifespan=lifespan,
)


@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    logger.exception("Unhandled error: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"error": type(exc).__name__, "detail": str(exc)},
    )


# CORS — allow all origins so the file:// demo page can call localhost:8000
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # overridden after config load if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")


@app.get("/healthz", tags=["ops"])
async def healthz():
    model_path = _config.get("model", {}).get("path", "unknown")
    backend = _config.get("storage", {}).get("backend", "local")
    return {
        "status": "ok",
        "model": model_path,
        "backend": backend,
        "model_loaded": _model is not None,
    }


@app.post("/extract", tags=["inference"])
async def extract(image: UploadFile = File(..., description="Receipt image (JPEG/PNG/WebP)")):
    """Synchronous receipt extraction — returns GroceryReceipt JSON immediately."""
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Only image/* content types accepted")

    if _model is None or _processor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    image_bytes = await image.read()

    try:
        result = run_inference(_model, _processor, image_bytes)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    errors = validate_against_schema(result, _schema)
    if errors:
        return JSONResponse(
            status_code=422,
            content={"error": "Schema validation failed", "detail": errors, "raw": result},
        )

    return result


@app.post("/extract/async", tags=["inference"])
async def extract_async(image: UploadFile = File(..., description="Receipt image (JPEG/PNG/WebP)")):
    """Async receipt extraction — returns job_id, poll GET /jobs/{job_id} for result."""
    from app.worker import extract_receipt  # imported lazily so app starts without Celery

    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Only image/* content types accepted")

    image_bytes = await image.read()
    image_b64 = base64.b64encode(image_bytes).decode()

    task = extract_receipt.delay(image_b64)
    return {"job_id": task.id}


@app.get("/jobs/{job_id}", tags=["inference"])
async def get_job(job_id: str):
    """Poll async job status."""
    from celery.result import AsyncResult
    from app.worker import celery_app

    result: AsyncResult = celery_app.AsyncResult(job_id)

    if result.state == "PENDING":
        return {"status": "pending", "result": None}
    elif result.state == "SUCCESS":
        return {"status": "done", "result": result.result}
    elif result.state == "FAILURE":
        return {"status": "failed", "result": str(result.result)}
    else:
        return {"status": result.state.lower(), "result": None}
