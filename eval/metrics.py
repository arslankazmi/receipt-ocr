"""
Per-field scoring for receipt extraction.

Scored fields (CORD-compatible):
  - store_name:   fuzzy string match (rapidfuzz token_sort_ratio >= 80 = match)
  - total:        numeric within ±2%
  - item_count:   exact match
  - item_name:    per-item fuzzy match >= 70, then precision/recall/f1 over items
  - item_total:   per-item numeric ±2%, then P/R/F1
  - item_qty:     per-item exact match, then P/R/F1

Skipped fields (absent from CORD ground truth):
  - date, taxes, discounts, payment
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FieldScore:
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    n_pred: int = 0
    n_gt: int = 0

    @classmethod
    def from_matches(cls, matches: int, n_pred: int, n_gt: int) -> "FieldScore":
        p = matches / n_pred if n_pred else 0.0
        r = matches / n_gt if n_gt else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        return cls(precision=p, recall=r, f1=f, n_pred=n_pred, n_gt=n_gt)


@dataclass
class ReceiptScore:
    store_name: FieldScore = field(default_factory=FieldScore)
    total: FieldScore = field(default_factory=FieldScore)
    item_count: FieldScore = field(default_factory=FieldScore)
    item_name: FieldScore = field(default_factory=FieldScore)
    item_total: FieldScore = field(default_factory=FieldScore)
    item_qty: FieldScore = field(default_factory=FieldScore)

    def overall_f1(self) -> float:
        scores = [
            self.store_name.f1,
            self.total.f1,
            self.item_name.f1,
            self.item_total.f1,
            self.item_qty.f1,
        ]
        return sum(scores) / len(scores)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_numeric(value) -> Optional[float]:
    """Convert a value to float, stripping currency symbols. Returns None on failure."""
    if value is None:
        return None
    try:
        return float(str(value).replace(",", "").replace("$", "").replace("£", "").strip())
    except (ValueError, TypeError):
        return None


def _numeric_match(pred_val, gt_val, tolerance: float = 0.02) -> bool:
    """Return True if pred is within ±tolerance fraction of gt."""
    p = _parse_numeric(pred_val)
    g = _parse_numeric(gt_val)
    if p is None or g is None:
        return False
    if g == 0:
        return p == 0
    return abs(p - g) / abs(g) <= tolerance


# ---------------------------------------------------------------------------
# Field scorers
# ---------------------------------------------------------------------------

def score_store_name(pred: dict, gt: dict) -> FieldScore:
    """Fuzzy string match on store.name (token_sort_ratio >= 80 → match)."""
    from rapidfuzz import fuzz

    pred_name = (pred.get("store") or {}).get("name") or pred.get("store_name")
    gt_name = (gt.get("store") or {}).get("name") or gt.get("store_name")

    if not gt_name:
        # GT missing — nothing to score
        return FieldScore(n_pred=1 if pred_name else 0, n_gt=0)
    if not pred_name:
        return FieldScore.from_matches(0, n_pred=0, n_gt=1)

    ratio = fuzz.token_sort_ratio(str(pred_name).lower(), str(gt_name).lower())
    matched = 1 if ratio >= 80 else 0
    return FieldScore.from_matches(matched, n_pred=1, n_gt=1)


def score_total(pred: dict, gt: dict) -> FieldScore:
    """Numeric total within ±2% → match."""
    pred_total = pred.get("total")
    gt_total = gt.get("total")

    if gt_total is None:
        return FieldScore(n_pred=1 if pred_total is not None else 0, n_gt=0)
    if pred_total is None:
        return FieldScore.from_matches(0, n_pred=0, n_gt=1)

    matched = 1 if _numeric_match(pred_total, gt_total) else 0
    return FieldScore.from_matches(matched, n_pred=1, n_gt=1)


def score_items(
    pred: dict, gt: dict
) -> tuple[FieldScore, FieldScore, FieldScore, FieldScore]:
    """
    Score line items.

    Returns (item_count, item_name, item_total, item_qty).

    Matching strategy:
      - For each predicted item, find the best unmatched GT item by name fuzzy score.
      - item_name match: token_sort_ratio >= 70
      - item_total match: numeric ±2%
      - item_qty match: exact string match (after stripping whitespace)
    """
    from rapidfuzz import fuzz

    pred_items: list[dict] = pred.get("line_items") or pred.get("items") or []
    gt_items: list[dict] = gt.get("line_items") or gt.get("items") or []

    n_pred = len(pred_items)
    n_gt = len(gt_items)

    # item_count: binary — 1 if counts match, else 0
    if n_gt == 0 and n_pred == 0:
        item_count_score = FieldScore(precision=1.0, recall=1.0, f1=1.0, n_pred=0, n_gt=0)
    else:
        count_match = 1 if n_pred == n_gt else 0
        item_count_score = FieldScore.from_matches(count_match, n_pred=1, n_gt=1)

    if n_gt == 0:
        empty = FieldScore.from_matches(0, n_pred=n_pred, n_gt=0)
        return item_count_score, empty, empty, empty

    if n_pred == 0:
        empty = FieldScore.from_matches(0, n_pred=0, n_gt=n_gt)
        return item_count_score, empty, empty, empty

    # Greedy matching by best name fuzzy score
    gt_matched = [False] * n_gt
    name_matches = 0
    total_matches = 0
    qty_matches = 0

    for p_item in pred_items:
        p_name = str(p_item.get("name") or "")
        best_score = -1
        best_idx = -1

        for j, g_item in enumerate(gt_items):
            if gt_matched[j]:
                continue
            g_name = str(g_item.get("name") or "")
            ratio = fuzz.token_sort_ratio(p_name.lower(), g_name.lower())
            if ratio > best_score:
                best_score = ratio
                best_idx = j

        if best_idx == -1:
            continue

        g_item = gt_items[best_idx]

        # Name match
        if best_score >= 70:
            name_matches += 1
            gt_matched[best_idx] = True  # consume only on name match

            # Total match (only score if name matched)
            if _numeric_match(p_item.get("total") or p_item.get("unit_price"), g_item.get("total") or g_item.get("unit_price")):
                total_matches += 1

            # Qty match (only score if name matched)
            p_qty = str(p_item.get("quantity") or "").strip()
            g_qty = str(g_item.get("quantity") or "").strip()
            if p_qty and g_qty and p_qty == g_qty:
                qty_matches += 1
        else:
            # No name match — don't consume this GT slot
            pass

    item_name_score = FieldScore.from_matches(name_matches, n_pred=n_pred, n_gt=n_gt)
    item_total_score = FieldScore.from_matches(total_matches, n_pred=n_pred, n_gt=n_gt)
    item_qty_score = FieldScore.from_matches(qty_matches, n_pred=n_pred, n_gt=n_gt)

    return item_count_score, item_name_score, item_total_score, item_qty_score


def score_receipt(pred: dict, gt: dict) -> ReceiptScore:
    """Score all CORD-compatible fields for a single receipt."""
    item_count, item_name, item_total, item_qty = score_items(pred, gt)
    return ReceiptScore(
        store_name=score_store_name(pred, gt),
        total=score_total(pred, gt),
        item_count=item_count,
        item_name=item_name,
        item_total=item_total,
        item_qty=item_qty,
    )


def aggregate_scores(scores: list[ReceiptScore]) -> ReceiptScore:
    """Return mean ReceiptScore over all samples."""
    if not scores:
        return ReceiptScore()

    def _mean_field(field_name: str) -> FieldScore:
        fields: list[FieldScore] = [getattr(s, field_name) for s in scores]
        n = len(fields)
        return FieldScore(
            precision=sum(f.precision for f in fields) / n,
            recall=sum(f.recall for f in fields) / n,
            f1=sum(f.f1 for f in fields) / n,
            n_pred=sum(f.n_pred for f in fields),
            n_gt=sum(f.n_gt for f in fields),
        )

    return ReceiptScore(
        store_name=_mean_field("store_name"),
        total=_mean_field("total"),
        item_count=_mean_field("item_count"),
        item_name=_mean_field("item_name"),
        item_total=_mean_field("item_total"),
        item_qty=_mean_field("item_qty"),
    )
