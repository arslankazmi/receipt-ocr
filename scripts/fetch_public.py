"""
Download public receipt datasets and normalise them to our schema.

Supports:
  - SROIE (ICDAR 2019) — printed receipts with text + key-info annotations
  - CORD  (Consolidated Receipt Dataset) — Korean/multi-lang with detailed line items

Both are available on HuggingFace.

Usage:
    python fetch_public.py --dataset sroie --out ../public/sroie/
    python fetch_public.py --dataset cord  --out ../public/cord/

Note: SROIE's stock schema captures {company, date, address, total}.
      CORD captures full line-item structure.
      This script normalises both to our schema (../../schema.json).

Requires:
    pip install datasets pillow
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def fetch_sroie(out_dir: Path, max_samples: int = 200) -> int:
    """Download SROIE and normalise to our schema. Returns count of receipts."""
    from datasets import load_dataset  # type: ignore

    print(f"Loading SROIE from HuggingFace (mychen76/invoices-and-receipts_ocr_v1), streaming up to {max_samples} samples...")
    # SROIE is mirrored on HF under several names; this one is the closest direct port
    ds = load_dataset("mychen76/invoices-and-receipts_ocr_v1", split="train", streaming=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = out_dir / "images"
    images_dir.mkdir(exist_ok=True)
    out_jsonl = out_dir / "annotations.jsonl"

    count = 0
    with out_jsonl.open("w") as f:
        for i, row in enumerate(ds):
            if count >= max_samples:
                break
            try:
                # Save image
                img = row.get("image")
                if img is None:
                    continue
                img_path = images_dir / f"sroie_{i:05d}.jpg"
                img.save(img_path, "JPEG", quality=90)

                # Parse the parsed_data field (varies by source)
                parsed: dict[str, Any] = row.get("parsed_data", {}) or {}
                if isinstance(parsed, str):
                    parsed = json.loads(parsed)

                # Build our schema's normalised form
                # Schema requires minItems:1 — add a sentinel item for SROIE (no line-item data)
                placeholder_items = [{"line_number": 1, "name": "UNKNOWN", "quantity": 1, "unit": "each", "total": 0.0}]
                receipt = {
                    "store": {"name": str(parsed.get("company", "UNKNOWN")).strip() or "UNKNOWN"},
                    "date": str(parsed.get("date", "1970-01-01"))[:10],
                    "items": placeholder_items,  # SROIE doesn't always have line items — placeholder
                    "subtotal": None,
                    "discounts": [],
                    "taxes": [],
                    "total": float(parsed.get("total", 0.0) or 0.0),
                    "currency": "USD",  # SROIE varies; user should override per row
                    "payment": {"method": "other"},
                    "metadata": {
                        "image_path": str(img_path.relative_to(out_dir)),
                        "source": "sroie",
                        "annotator": "fetch_public.py",
                        "notes": "Sparse — only store/date/total reliably populated. Line items must be re-annotated.",
                    },
                }
                line_out = json.dumps(receipt).replace("\n", " ").replace("\r", " ")
                f.write(line_out + "\n")
                count += 1
            except Exception as exc:
                print(f"  [skip row {i}]: {exc}")

    print(f"✓ Wrote {count} SROIE receipts to {out_jsonl}")
    return count


def _parse_number(val: Any) -> float:
    """Parse KRW / Indonesian-locale number strings like '1,346,000' or '35.000,00'."""
    if val is None:
        return 0.0
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip()
    # Remove currency prefixes like 'Rp'
    s = s.lstrip("RrpP ").strip()
    # European format: 35.000,00 → 35000.00
    if "," in s and "." in s:
        if s.rindex(",") > s.rindex("."):
            # comma is decimal separator (European)
            s = s.replace(".", "").replace(",", ".")
        else:
            # period is decimal separator, comma is thousands
            s = s.replace(",", "")
    elif "," in s:
        # Comma as thousands separator (no decimal): 1,346,000
        s = s.replace(",", "")
    # Remove trailing/leading non-numeric chars
    s = s.strip(" ,.")
    try:
        return float(s)
    except ValueError:
        return 0.0


def fetch_cord(out_dir: Path, max_samples: int = 200) -> int:
    """Download CORD and normalise to our schema. Returns count of receipts."""
    from datasets import load_dataset  # type: ignore

    print(f"Loading CORD from HuggingFace (naver-clova-ix/cord-v2), streaming up to {max_samples} samples...")
    ds = load_dataset("naver-clova-ix/cord-v2", split="train", streaming=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = out_dir / "images"
    images_dir.mkdir(exist_ok=True)
    out_jsonl = out_dir / "annotations.jsonl"

    count = 0
    with out_jsonl.open("w") as f:
        for i, row in enumerate(ds):
            if count >= max_samples:
                break
            try:
                img = row.get("image")
                if img is None:
                    continue
                img_path = images_dir / f"cord_{i:05d}.jpg"
                img.convert("RGB").save(img_path, "JPEG", quality=90)

                # CORD stores parsed data in 'ground_truth' as JSON string
                gt = row.get("ground_truth") or row.get("gt_parse_str")
                if isinstance(gt, str):
                    gt = json.loads(gt)
                gt_parse = gt.get("gt_parse", gt) if isinstance(gt, dict) else {}

                menu = gt_parse.get("menu", [])
                if not isinstance(menu, list):
                    menu = [menu]

                items = []
                for j, m in enumerate(menu, start=1):
                    if not isinstance(m, dict):
                        continue
                    cnt_raw = m.get("cnt", 1)
                    items.append({
                        "line_number": j,
                        "name": str(m.get("nm", "?")),
                        "quantity": _parse_number(cnt_raw) if cnt_raw not in (None, "") else 1.0,
                        "unit": "each",
                        "unit_price": _parse_number(m.get("unitprice")) if m.get("unitprice") else None,
                        "total": _parse_number(m.get("price", 0)),
                        "tax_code": None,
                        "discount_applied": None,
                    })

                total_obj = gt_parse.get("total", {}) or {}
                # Schema requires minItems:1 — add placeholder if CORD has no menu items
                if not items:
                    items = [{"line_number": 1, "name": "UNKNOWN", "quantity": 1, "unit": "each", "total": 0.0}]
                receipt = {
                    "store": {"name": str(gt_parse.get("sub_total", {}).get("storenm", "UNKNOWN") or "UNKNOWN")},
                    "date": "1970-01-01",
                    "items": items,
                    "subtotal": _parse_number(gt_parse.get("sub_total", {}).get("subtotal_price")) or None,
                    "discounts": [],
                    "taxes": [],
                    "total": _parse_number(total_obj.get("total_price", 0)),
                    "currency": "KRW",
                    "payment": {"method": "other"},
                    "metadata": {
                        "image_path": str(img_path.relative_to(out_dir)),
                        "source": "cord",
                        "annotator": "fetch_public.py",
                        "notes": "Line items populated. Currency is KRW unless otherwise marked.",
                    },
                }
                # Ensure no embedded newlines break the JSONL format
                line_out = json.dumps(receipt).replace("\n", " ").replace("\r", " ")
                f.write(line_out + "\n")
                count += 1
            except Exception as exc:
                print(f"  [skip row {i}]: {exc}")

    print(f"✓ Wrote {count} CORD receipts to {out_jsonl}")
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["sroie", "cord"])
    parser.add_argument("--out", type=Path, required=True, help="Output directory")
    args = parser.parse_args()

    if args.dataset == "sroie":
        fetch_sroie(args.out)
    else:
        fetch_cord(args.out)


if __name__ == "__main__":
    main()
