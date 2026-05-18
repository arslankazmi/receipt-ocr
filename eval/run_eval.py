"""
Receipt OCR eval runner.

Usage:
    python eval/run_eval.py [--config config.yaml] [--async] [--max-samples N]

Requires:
    ANTHROPIC_API_KEY environment variable.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import sys
from dataclasses import asdict
from io import BytesIO
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Path setup — allow running as script from project root
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import anthropic
from data.cord_dataset import CordDataset
from eval.metrics import ReceiptScore, aggregate_scores, score_receipt
from eval.models.claude_runner import (
    ClaudeRunner,
    _build_messages,
    _encode_image,
    _parse_response,
    _PROMPT_TEMPLATE,
    _RETRY_PROMPT,
)


# ---------------------------------------------------------------------------
# Sync runner
# ---------------------------------------------------------------------------

class SyncEvalRunner:
    def __init__(self, client: anthropic.Anthropic, schema: dict, cfg: dict) -> None:
        self.client = client
        self.schema = schema
        self.schema_json = json.dumps(schema)
        self.cfg = cfg
        self.model = "claude-sonnet-4-6"
        self.max_tokens = 2048

    def _call_model(self, messages: list[dict], extra_messages: list[dict] | None = None) -> str:
        full_messages = messages + (extra_messages or [])
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=full_messages,
        )
        return resp.content[0].text

    def _extract(self, img, gt: dict) -> dict:
        img_b64, media_type = _encode_image(img)
        messages = _build_messages(img_b64, media_type, self.schema_json)
        raw = self._call_model(messages)
        pred = _parse_response(raw)

        if pred is None:
            # Retry once
            retry_messages = [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": _RETRY_PROMPT},
            ]
            raw2 = self._call_model(messages, retry_messages)
            pred = _parse_response(raw2) or {}

        return pred

    def run(self, dataset: CordDataset, max_samples: int | None) -> list[dict]:
        results = []
        limit = max_samples if max_samples is not None else len(dataset)
        for idx in range(min(limit, len(dataset))):
            img, gt = dataset[idx]
            pred = self._extract(img, gt)
            score = score_receipt(pred, gt)
            results.append({"pred": pred, "gt": gt, "score": score})
            print(f"  [{idx + 1}/{limit}] overall_f1={score.overall_f1():.3f}", flush=True)
        return results


# ---------------------------------------------------------------------------
# Async runner
# ---------------------------------------------------------------------------

class AsyncEvalRunner:
    def __init__(
        self,
        client: anthropic.AsyncAnthropic,
        schema: dict,
        cfg: dict,
        concurrency: int = 50,
    ) -> None:
        self.client = client
        self.schema = schema
        self.schema_json = json.dumps(schema)
        self.cfg = cfg
        self.concurrency = concurrency
        self.model = "claude-sonnet-4-6"
        self.max_tokens = 2048

    async def _call_model(self, messages: list[dict], extra_messages: list[dict] | None = None) -> str:
        full_messages = messages + (extra_messages or [])
        resp = await self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=full_messages,
        )
        return resp.content[0].text

    async def _extract(self, img, gt: dict) -> dict:
        img_b64, media_type = _encode_image(img)
        messages = _build_messages(img_b64, media_type, self.schema_json)
        raw = await self._call_model(messages)
        pred = _parse_response(raw)

        if pred is None:
            retry_messages = [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": _RETRY_PROMPT},
            ]
            raw2 = await self._call_model(messages, retry_messages)
            pred = _parse_response(raw2) or {}

        return pred

    async def _process_one(
        self,
        sem: asyncio.Semaphore,
        idx: int,
        total: int,
        img,
        gt: dict,
    ) -> dict:
        async with sem:
            pred = await self._extract(img, gt)
            score = score_receipt(pred, gt)
            print(f"  [{idx + 1}/{total}] overall_f1={score.overall_f1():.3f}", flush=True)
            return {"pred": pred, "gt": gt, "score": score}

    async def run(self, dataset: CordDataset, max_samples: int | None) -> list[dict]:
        limit = max_samples if max_samples is not None else len(dataset)
        sem = asyncio.Semaphore(self.concurrency)
        tasks = [
            self._process_one(sem, idx, limit, *dataset[idx])
            for idx in range(min(limit, len(dataset)))
        ]
        return await asyncio.gather(*tasks)


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _print_summary(agg: ReceiptScore) -> None:
    fields = [
        ("store_name", agg.store_name),
        ("total", agg.total),
        ("item_count", agg.item_count),
        ("item_name", agg.item_name),
        ("item_total", agg.item_total),
        ("item_qty", agg.item_qty),
    ]
    header = f"{'Field':<15} {'Precision':>10} {'Recall':>8} {'F1':>8}"
    sep = "─" * len(header)
    print(header)
    print(sep)
    for name, fs in fields:
        print(f"{name:<15} {fs.precision:>10.2f} {fs.recall:>8.2f} {fs.f1:>8.2f}")
    print(sep)
    print(f"{'overall':<15} {'-':>10} {'-':>8} {agg.overall_f1():>8.2f}")


def _build_output(results: list[dict], agg: ReceiptScore) -> dict:
    def _field_dict(fs):
        return {
            "precision": fs.precision,
            "recall": fs.recall,
            "f1": fs.f1,
            "n_pred": fs.n_pred,
            "n_gt": fs.n_gt,
        }

    summary = {
        "store_name": _field_dict(agg.store_name),
        "total": _field_dict(agg.total),
        "item_count": _field_dict(agg.item_count),
        "item_name": _field_dict(agg.item_name),
        "item_total": _field_dict(agg.item_total),
        "item_qty": _field_dict(agg.item_qty),
        "overall_f1": agg.overall_f1(),
    }

    samples = []
    for r in results:
        score: ReceiptScore = r["score"]
        samples.append({
            "pred": r["pred"],
            "gt": r["gt"],
            "scores": asdict(score),
        })

    return {"summary": summary, "samples": samples}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Receipt OCR eval runner")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--async", dest="use_async", action="store_true",
                        help="Use async runner (overrides config)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit number of samples evaluated")
    args = parser.parse_args()

    # --- API key check ---
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    # --- Load config ---
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    eval_cfg = cfg.get("eval", {})
    data_cfg = cfg.get("data", {})

    max_samples = args.max_samples if args.max_samples is not None else eval_cfg.get("max_samples")
    concurrency = eval_cfg.get("concurrency", 1)
    use_async = args.use_async or concurrency > 1
    output_path = Path(eval_cfg.get("output", "eval/results.json"))

    # --- Load schema ---
    schema_path = _PROJECT_ROOT / "schema.json"
    if not schema_path.exists():
        print(f"ERROR: schema.json not found at {schema_path}", file=sys.stderr)
        sys.exit(1)
    with open(schema_path) as f:
        schema = json.load(f)

    # --- Build dataset ---
    dataset = CordDataset(data_cfg, split="val")
    total = min(max_samples, len(dataset)) if max_samples else len(dataset)
    print(f"Evaluating {total} samples (async={use_async}, concurrency={concurrency})")

    # --- Run ---
    if use_async:
        async_client = anthropic.AsyncAnthropic(api_key=api_key)
        runner = AsyncEvalRunner(async_client, schema, cfg, concurrency=concurrency)
        results = asyncio.run(runner.run(dataset, max_samples))
    else:
        sync_client = anthropic.Anthropic(api_key=api_key)
        runner = SyncEvalRunner(sync_client, schema, cfg)
        results = runner.run(dataset, max_samples)

    # --- Aggregate & print ---
    scores: list[ReceiptScore] = [r["score"] for r in results]
    agg = aggregate_scores(scores)
    print()
    _print_summary(agg)

    # --- Write results ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = _build_output(results, agg)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults written to {output_path}")


if __name__ == "__main__":
    main()
