"""
Distributed batch eval via Celery.

At scale (500k+ images), run:
  celery -A eval.batch_eval worker --concurrency=8
  python eval/batch_eval.py --queue --annotations /path/to/all_annotations.jsonl

Each Celery task processes one receipt image and writes result to Redis.
The --queue command reads a JSONL file of {"image_path": "...", "annotation": {...}}
and dispatches one Celery task per line.

To collect results after workers finish:
  python eval/batch_eval.py --collect --task-ids-file task_ids.json
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from celery import Celery

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
app = Celery("receipt_eval", broker=REDIS_URL, backend=REDIS_URL)

# Celery config: JSON serialization, task acks on failure prevention
app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    result_expires=86400,  # 24 h
)


@app.task(bind=True, max_retries=3, default_retry_delay=2)
def eval_receipt(self, image_path: str, annotation: dict, schema: dict) -> dict:
    """Process a single receipt. Called by the distributed workers."""
    import base64

    import anthropic

    from eval.metrics import score_receipt

    try:
        client = anthropic.Anthropic()
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()
        schema_json = json.dumps(schema)
        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2048,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": img_b64,
                            },
                        },
                        {
                            "type": "text",
                            "text": (
                                "Extract this receipt as JSON conforming to this schema. "
                                f"Return JSON only.\nSchema: {schema_json}"
                            ),
                        },
                    ],
                }
            ],
        )
        raw = resp.content[0].text.strip()
        # Strip accidental markdown fences
        if raw.startswith("```"):
            lines = raw.splitlines()
            raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        pred = json.loads(raw)
        score = score_receipt(pred, annotation)
        return {"status": "ok", "pred": pred, "score": score.__dict__}
    except Exception as exc:
        raise self.retry(exc=exc)


# ---------------------------------------------------------------------------
# CLI: queue jobs or collect results
# ---------------------------------------------------------------------------

def _queue_jobs(annotations_file: str, schema_path: str) -> None:
    """Read a JSONL annotations file and dispatch one task per line."""
    schema_file = Path(schema_path)
    if not schema_file.exists():
        raise FileNotFoundError(f"Schema not found: {schema_path}")
    with open(schema_file) as f:
        schema = json.load(f)

    task_ids = []
    with open(annotations_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            image_path = record["image_path"]
            annotation = record["annotation"]
            result = eval_receipt.delay(image_path, annotation, schema)
            task_ids.append(result.id)

    ids_file = Path("task_ids.json")
    with open(ids_file, "w") as f:
        json.dump(task_ids, f, indent=2)
    print(f"Queued {len(task_ids)} tasks. IDs written to {ids_file}")


def _collect_results(task_ids_file: str, output_path: str) -> None:
    """Collect completed task results and write aggregated output."""
    from eval.metrics import ReceiptScore, aggregate_scores

    with open(task_ids_file) as f:
        task_ids = json.load(f)

    from celery.result import AsyncResult

    results = []
    pending = 0
    failed = 0
    for task_id in task_ids:
        ar = AsyncResult(task_id, app=app)
        if ar.successful():
            results.append(ar.result)
        elif ar.failed():
            failed += 1
        else:
            pending += 1

    print(f"Collected: {len(results)} ok | {failed} failed | {pending} pending")

    if results:
        scores = []
        for r in results:
            s = r.get("score", {})
            from eval.metrics import FieldScore
            receipt_score = ReceiptScore(
                store_name=FieldScore(**s.get("store_name", {})),
                total=FieldScore(**s.get("total", {})),
                item_count=FieldScore(**s.get("item_count", {})),
                item_name=FieldScore(**s.get("item_name", {})),
                item_total=FieldScore(**s.get("item_total", {})),
                item_qty=FieldScore(**s.get("item_qty", {})),
            )
            scores.append(receipt_score)

        agg = aggregate_scores(scores)
        output = {
            "summary": {
                "overall_f1": agg.overall_f1(),
                "store_name": agg.store_name.__dict__,
                "total": agg.total.__dict__,
                "item_count": agg.item_count.__dict__,
                "item_name": agg.item_name.__dict__,
                "item_total": agg.item_total.__dict__,
                "item_qty": agg.item_qty.__dict__,
            },
            "samples": results,
        }
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Results written to {out_file}")


def main() -> None:
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Distributed batch eval via Celery")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--queue", action="store_true",
                      help="Queue all jobs from annotations JSONL")
    mode.add_argument("--collect", action="store_true",
                      help="Collect completed results from Redis")
    parser.add_argument("--annotations", default=None,
                        help="Path to JSONL file with {image_path, annotation} records")
    parser.add_argument("--schema", default="schema.json",
                        help="Path to schema.json")
    parser.add_argument("--task-ids-file", default="task_ids.json",
                        help="JSON file of task IDs (input for --collect)")
    parser.add_argument("--output", default="eval/batch_results.json",
                        help="Output file for collected results")
    args = parser.parse_args()

    if args.queue:
        if not args.annotations:
            print("ERROR: --annotations required with --queue", file=sys.stderr)
            sys.exit(1)
        _queue_jobs(args.annotations, args.schema)
    elif args.collect:
        _collect_results(args.task_ids_file, args.output)


if __name__ == "__main__":
    main()
