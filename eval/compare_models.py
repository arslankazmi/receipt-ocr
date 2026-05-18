"""
Compare pre-trained OCR models on CORD val set.

Usage:
  python eval/compare_models.py [--config config.yaml] [--max-samples N] [--models donut-cord,qwen2vl-2b,trocr-base,claude-sonnet]

Default: runs donut-cord, qwen2vl-2b, trocr-base (Claude skipped unless ANTHROPIC_API_KEY set)

Output:
  - Prints comparison table to stdout
  - Writes eval/comparison_results.json

Example output:
  Model          store_name  total   item_name  item_total  item_qty   overall
  ─────────────────────────────────────────────────────────────────────────────
  donut-cord     0.72        0.68    0.61       0.65        0.58       0.65
  qwen2vl-2b     0.81        0.79    0.74       0.72        0.69       0.75
  trocr-base     0.41        0.38    0.00       0.00        0.00       0.16
  claude-sonnet  —           —       —          —           —          —  (API key not set)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from data.cord_dataset import CordDataset
from eval.metrics import ReceiptScore, aggregate_scores, score_receipt

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

_DEFAULT_MODELS = ["donut-cord", "qwen2vl-2b", "trocr-base"]

_MODEL_CLASSES: dict[str, str] = {
    "donut-cord": "eval.models.donut_cord.DonutCordRunner",
    "qwen2vl-2b": "eval.models.qwen2vl.Qwen2VLRunner",
    "trocr-base": "eval.models.trocr.TrOCRRunner",
    "claude-sonnet": "eval.models.claude_runner.ClaudeRunner",
}


def _load_runner(model_id: str):
    """Dynamically import and instantiate a runner by model_id."""
    dotpath = _MODEL_CLASSES[model_id]
    module_path, class_name = dotpath.rsplit(".", 1)
    import importlib
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _run_model(runner, dataset: CordDataset, max_samples: int | None) -> list[ReceiptScore] | None:
    limit = max_samples if max_samples is not None else len(dataset)
    limit = min(limit, len(dataset))
    scores: list[ReceiptScore] = []
    for idx in range(limit):
        img, gt = dataset[idx]
        pred = runner.extract(img)
        score = score_receipt(pred, gt)
        scores.append(score)
        log.info("  [%d/%d] overall_f1=%.3f", idx + 1, limit, score.overall_f1())
    return scores


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

_COL_WIDTH = 12
_COLS = ["store_name", "total", "item_name", "item_total", "item_qty", "overall"]


def _format_row(model_id: str, agg: ReceiptScore | None, note: str = "") -> str:
    if agg is None:
        values = ["—"] * len(_COLS)
        suffix = f"  ({note})" if note else ""
        return f"{model_id:<16}" + "".join(f"{v:>{_COL_WIDTH}}" for v in values) + suffix

    field_f1s = {
        "store_name": agg.store_name.f1,
        "total": agg.total.f1,
        "item_name": agg.item_name.f1,
        "item_total": agg.item_total.f1,
        "item_qty": agg.item_qty.f1,
        "overall": agg.overall_f1(),
    }
    return f"{model_id:<16}" + "".join(
        f"{field_f1s[col]:>{_COL_WIDTH}.2f}" for col in _COLS
    )


def _print_table(rows: list[tuple[str, ReceiptScore | None, str]]) -> None:
    header = f"{'Model':<16}" + "".join(f"{c:>{_COL_WIDTH}}" for c in _COLS)
    sep = "─" * len(header)
    print(header)
    print(sep)
    for model_id, agg, note in rows:
        print(_format_row(model_id, agg, note))


def _build_json_output(rows: list[tuple[str, ReceiptScore | None, str]]) -> dict:
    results = {}
    for model_id, agg, note in rows:
        if agg is None:
            results[model_id] = {"skipped": True, "reason": note}
        else:
            results[model_id] = {
                "store_name_f1": agg.store_name.f1,
                "total_f1": agg.total.f1,
                "item_name_f1": agg.item_name.f1,
                "item_total_f1": agg.item_total.f1,
                "item_qty_f1": agg.item_qty.f1,
                "overall_f1": agg.overall_f1(),
            }
    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Compare OCR models on CORD val set")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit number of val samples")
    parser.add_argument(
        "--models",
        default=None,
        help="Comma-separated model IDs to run (default: donut-cord,qwen2vl-2b,trocr-base)",
    )
    args = parser.parse_args()

    # Parse requested models
    if args.models:
        requested = [m.strip() for m in args.models.split(",")]
    else:
        requested = list(_DEFAULT_MODELS)

    # Auto-add Claude if API key is present and not explicitly listed
    if "claude-sonnet" not in requested and os.environ.get("ANTHROPIC_API_KEY"):
        requested.append("claude-sonnet")

    # Validate model IDs
    unknown = [m for m in requested if m not in _MODEL_CLASSES]
    if unknown:
        print(f"ERROR: Unknown model IDs: {unknown}. Valid: {list(_MODEL_CLASSES.keys())}", file=sys.stderr)
        sys.exit(1)

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg.get("data", {})
    eval_cfg = cfg.get("eval", {})
    max_samples = args.max_samples if args.max_samples is not None else eval_cfg.get("max_samples")

    # Load dataset
    dataset = CordDataset(data_cfg, split="val")
    total = min(max_samples, len(dataset)) if max_samples else len(dataset)
    print(f"Dataset: {total} val samples")
    print(f"Models:  {', '.join(requested)}\n")

    # Run each model
    rows: list[tuple[str, ReceiptScore | None, str]] = []

    for model_id in requested:
        print(f"--- {model_id} ---")
        try:
            runner = _load_runner(model_id)
        except EnvironmentError as exc:
            log.warning("Skipping %s: %s", model_id, exc)
            rows.append((model_id, None, str(exc)))
            continue
        except Exception as exc:
            log.error("Failed to load %s: %s", model_id, exc)
            rows.append((model_id, None, f"load error: {exc}"))
            continue

        try:
            scores = _run_model(runner, dataset, max_samples)
            agg = aggregate_scores(scores)
            rows.append((model_id, agg, ""))
            print(f"  overall_f1={agg.overall_f1():.3f}\n")
        except Exception as exc:
            log.error("Error running %s: %s", model_id, exc, exc_info=True)
            rows.append((model_id, None, f"runtime error: {exc}"))

    # Print table
    print()
    _print_table(rows)

    # Write JSON output
    output_path = _PROJECT_ROOT / "eval" / "comparison_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(_build_json_output(rows), f, indent=2)
    print(f"\nResults written to {output_path}")


if __name__ == "__main__":
    main()
