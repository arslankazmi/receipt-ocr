# receipt-ocr

End-to-end receipt OCR and structured extraction pipeline using Donut + LoRA fine-tuning.

## What it does

receipt-ocr takes a photo of a grocery receipt and outputs a structured JSON object conforming to `schema.json` — including store name, line items with quantities and prices, subtotal, taxes, and total. It fine-tunes a Donut (document understanding transformer) model on CORD data using LoRA for parameter-efficient training. An async FastAPI service exposes the model for real-time inference.

## Architecture

```
Image
  │
  ▼
┌──────────────────────────────────────────────────────────┐
│  Ingest                                                  │
│  scripts/fetch_public.py  ──►  annotations.jsonl         │
│  scripts/create_shards.py ──►  data/shards/*.tar         │
└──────────────────────┬───────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│  Data Layer                                              │
│  data/cord_dataset.py  (local, ≤200 images)             │
│  data/shard_dataset.py (WebDataset stream, any scale)   │
│  data/storage.py       (LocalBackend / S3Backend)       │
└──────────────────────┬───────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│  Train                                                   │
│  train/finetune_donut.py                                 │
│  Donut-base + LoRA (r=8, alpha=16)                       │
│  Accelerate multi-GPU / fp16                             │
│  Saves to train/checkpoints/donut-receipt-lora/          │
└──────────────────────┬───────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│  Eval                                                    │
│  eval/run_eval.py                                        │
│  RapidFuzz token-set F1 per field                        │
│  Output: eval/results.json                               │
└──────────────────────┬───────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│  Serve                                                   │
│  app/main.py — FastAPI + Celery/Redis async queue        │
│  POST /infer   ──► {"image": "<base64>"}                 │
│  GET  /health                                            │
└──────────────────────────────────────────────────────────┘
```

### Scale swim-lane

| Concern          | ≤200 images (dev)        | 10k+ images (scale)                    |
|------------------|--------------------------|----------------------------------------|
| Storage          | Local filesystem         | S3 (`storage.backend: s3`)             |
| Data loading     | `CordDataset`            | `ShardDataset` (WebDataset tar shards) |
| Training         | Single GPU               | `accelerate launch --multi_gpu`        |
| Inference        | Sync FastAPI handler     | Celery workers + Redis queue           |
| Checkpoints      | `train/checkpoints/`     | `s3://bucket/checkpoints/`             |

## Results

| Model                  | store.name F1 | total F1 | item_name F1 | item_total F1 | overall F1 |
|------------------------|---------------|----------|--------------|---------------|------------|
| Donut-base (zero-shot) | —             | —        | —            | —             | —          |
| Donut + LoRA (ours)    | —             | —        | —            | —             | —          |

_Populate after running `python eval/run_eval.py`._

## Quick start

```bash
# 1. Install dependencies
uv sync --extra dev

# 2. Fetch CORD data (200 samples)
python scripts/fetch_public.py --dataset cord --out public/cord/

# 3. Set paths in config.yaml
#    data.annotations: /absolute/path/to/public/cord/annotations.jsonl
#    data.images_dir:  /absolute/path/to/public/cord/images

# 4. Evaluate (requires trained checkpoint)
python eval/run_eval.py

# 5. Fine-tune
python train/finetune_donut.py

# 6. Start API server
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Scaling beyond 200 images

**S3 backend** — set `storage.backend: s3` in `config.yaml` with your `bucket` and `prefix`. `S3Backend` in `data/storage.py` handles pagination and a simple 200-object in-memory cache to reduce GET costs.

**WebDataset shards** — run `scripts/create_shards.py` to pack images + annotations into `.tar` shards (1000 images each). Upload with `aws s3 sync data/shards/ s3://bucket/shards/`. Then use `ShardDataset("s3://bucket/shards/{000..999}.tar")` — this streams the full dataset without loading it into RAM, enabling 550+ GB datasets.

**Multi-GPU training** — install Accelerate and run:
```bash
accelerate config   # one-time setup
accelerate launch train/finetune_donut.py
```
The trainer wraps the model with `Accelerator` so no code changes are needed for DDP or FSDP.

**Celery async inference** — start a Redis broker and Celery worker:
```bash
redis-server &
celery -A app.tasks worker --loglevel=info
```
The FastAPI `/infer` endpoint enqueues tasks and returns a job ID; poll `/status/{job_id}` for results.

## Dataset

CORD (Consolidated Receipt Dataset) by NAVER CLOVA — available at [naver-clova-ix/cord-v2](https://huggingface.co/datasets/naver-clova-ix/cord-v2) on Hugging Face. Licensed under **CC BY 4.0**. Please cite the original authors when publishing results based on this dataset.

## License

MIT
