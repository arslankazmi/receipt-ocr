"""
Unit tests for eval/metrics.py.

No real images are needed — tests operate entirely on dicts.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Allow running from project root or tests/ directory
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from eval.metrics import (
    FieldScore,
    ReceiptScore,
    aggregate_scores,
    score_items,
    score_receipt,
    score_store_name,
    score_total,
)


# ---------------------------------------------------------------------------
# store_name
# ---------------------------------------------------------------------------

class TestScoreStoreName:
    def test_exact_match(self):
        pred = {"store": {"name": "Whole Foods Market"}}
        gt = {"store": {"name": "Whole Foods Market"}}
        score = score_store_name(pred, gt)
        assert score.f1 == pytest.approx(1.0)
        assert score.precision == pytest.approx(1.0)
        assert score.recall == pytest.approx(1.0)

    def test_fuzzy_match_above_threshold(self):
        """Minor variation should still match (token_sort_ratio >= 80)."""
        pred = {"store": {"name": "Whole Foods Mkt"}}
        gt = {"store": {"name": "Whole Foods Market"}}
        score = score_store_name(pred, gt)
        # Should be a match — both contain "Whole Foods" + similar tokens
        assert score.f1 > 0.0

    def test_no_match_completely_different(self):
        pred = {"store": {"name": "XYZZY Corp"}}
        gt = {"store": {"name": "Whole Foods Market"}}
        score = score_store_name(pred, gt)
        assert score.f1 == pytest.approx(0.0)

    def test_missing_pred_name(self):
        pred = {}
        gt = {"store": {"name": "Whole Foods Market"}}
        score = score_store_name(pred, gt)
        assert score.f1 == pytest.approx(0.0)
        assert score.recall == pytest.approx(0.0)

    def test_missing_gt_name(self):
        pred = {"store": {"name": "Some Store"}}
        gt = {}
        score = score_store_name(pred, gt)
        # GT absent — n_gt=0, no meaningful score
        assert score.n_gt == 0

    def test_flat_store_name_key(self):
        """Support flat store_name key as fallback."""
        pred = {"store_name": "Target"}
        gt = {"store_name": "Target"}
        score = score_store_name(pred, gt)
        assert score.f1 == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# total
# ---------------------------------------------------------------------------

class TestScoreTotal:
    def test_within_tolerance(self):
        """1% difference → match."""
        pred = {"total": 10.10}
        gt = {"total": 10.00}
        score = score_total(pred, gt)
        assert score.f1 == pytest.approx(1.0)

    def test_exact_match(self):
        pred = {"total": 42.50}
        gt = {"total": 42.50}
        score = score_total(pred, gt)
        assert score.f1 == pytest.approx(1.0)

    def test_outside_tolerance(self):
        """5% difference → no match."""
        pred = {"total": 10.50}
        gt = {"total": 10.00}
        score = score_total(pred, gt)
        assert score.f1 == pytest.approx(0.0)

    def test_string_numeric(self):
        """Total given as string with currency symbol."""
        pred = {"total": "$9.99"}
        gt = {"total": "9.99"}
        score = score_total(pred, gt)
        assert score.f1 == pytest.approx(1.0)

    def test_missing_pred_total(self):
        pred = {}
        gt = {"total": 20.00}
        score = score_total(pred, gt)
        assert score.f1 == pytest.approx(0.0)

    def test_missing_gt_total(self):
        pred = {"total": 20.00}
        gt = {}
        score = score_total(pred, gt)
        assert score.n_gt == 0

    def test_zero_gt_zero_pred(self):
        pred = {"total": 0.0}
        gt = {"total": 0.0}
        score = score_total(pred, gt)
        assert score.f1 == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# items
# ---------------------------------------------------------------------------

class TestScoreItems:
    def _make_items(self, items: list[dict]) -> dict:
        return {"line_items": items}

    def test_perfect_match(self):
        pred = self._make_items([
            {"name": "Apple", "quantity": "2", "total": "1.50"},
            {"name": "Milk", "quantity": "1", "total": "2.99"},
        ])
        gt = self._make_items([
            {"name": "Apple", "quantity": "2", "total": "1.50"},
            {"name": "Milk", "quantity": "1", "total": "2.99"},
        ])
        item_count, item_name, item_total, item_qty = score_items(pred, gt)
        assert item_count.f1 == pytest.approx(1.0)
        assert item_name.f1 == pytest.approx(1.0)
        assert item_total.f1 == pytest.approx(1.0)
        assert item_qty.f1 == pytest.approx(1.0)

    def test_empty_pred(self):
        pred = {"line_items": []}
        gt = self._make_items([
            {"name": "Apple", "quantity": "2", "total": "1.50"},
        ])
        item_count, item_name, item_total, item_qty = score_items(pred, gt)
        # item_count: 0 ≠ 1 → no match
        assert item_count.f1 == pytest.approx(0.0)
        # item fields: pred has no items → recall = 0
        assert item_name.recall == pytest.approx(0.0)
        assert item_total.recall == pytest.approx(0.0)
        assert item_qty.recall == pytest.approx(0.0)

    def test_empty_gt_and_pred(self):
        pred = {"line_items": []}
        gt = {"line_items": []}
        item_count, item_name, item_total, item_qty = score_items(pred, gt)
        assert item_count.f1 == pytest.approx(1.0)

    def test_extra_pred_items(self):
        """Pred has more items than GT → precision drops."""
        pred = self._make_items([
            {"name": "Apple", "quantity": "2", "total": "1.50"},
            {"name": "Ghost Item", "quantity": "1", "total": "9.99"},
        ])
        gt = self._make_items([
            {"name": "Apple", "quantity": "2", "total": "1.50"},
        ])
        item_count, item_name, item_total, item_qty = score_items(pred, gt)
        assert item_count.f1 == pytest.approx(0.0)  # 2 ≠ 1
        assert item_name.precision < 1.0
        assert item_name.recall == pytest.approx(1.0)

    def test_item_total_outside_tolerance(self):
        pred = self._make_items([{"name": "Bread", "quantity": "1", "total": "5.00"}])
        gt = self._make_items([{"name": "Bread", "quantity": "1", "total": "4.00"}])
        _, _, item_total, _ = score_items(pred, gt)
        assert item_total.f1 == pytest.approx(0.0)

    def test_item_qty_mismatch(self):
        pred = self._make_items([{"name": "Eggs", "quantity": "12", "total": "3.00"}])
        gt = self._make_items([{"name": "Eggs", "quantity": "6", "total": "3.00"}])
        _, _, _, item_qty = score_items(pred, gt)
        assert item_qty.f1 == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# score_receipt (integration)
# ---------------------------------------------------------------------------

class TestScoreReceipt:
    def test_full_receipt_perfect(self):
        receipt = {
            "store": {"name": "Costco"},
            "total": 55.25,
            "line_items": [
                {"name": "Olive Oil", "quantity": "1", "total": "12.99"},
                {"name": "Paper Towels", "quantity": "2", "total": "18.49"},
            ],
        }
        score = score_receipt(receipt, receipt)
        assert score.store_name.f1 == pytest.approx(1.0)
        assert score.total.f1 == pytest.approx(1.0)
        assert score.item_name.f1 == pytest.approx(1.0)
        assert score.overall_f1() == pytest.approx(1.0)

    def test_empty_pred(self):
        gt = {
            "store": {"name": "Costco"},
            "total": 55.25,
            "line_items": [{"name": "Olive Oil", "quantity": "1", "total": "12.99"}],
        }
        score = score_receipt({}, gt)
        assert score.store_name.f1 == pytest.approx(0.0)
        assert score.total.f1 == pytest.approx(0.0)
        assert score.item_name.f1 == pytest.approx(0.0)

    def test_overall_f1_average(self):
        """overall_f1 is mean of 5 specific fields (excludes item_count)."""
        receipt = {
            "store": {"name": "Trader Joe's"},
            "total": 30.00,
            "line_items": [{"name": "Salsa", "quantity": "1", "total": "3.99"}],
        }
        score = score_receipt(receipt, receipt)
        expected = (
            score.store_name.f1
            + score.total.f1
            + score.item_name.f1
            + score.item_total.f1
            + score.item_qty.f1
        ) / 5
        assert score.overall_f1() == pytest.approx(expected)


# ---------------------------------------------------------------------------
# aggregate_scores
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# ModelRunner interface
# ---------------------------------------------------------------------------

class TestModelRunnerInterface:
    def test_model_runner_interface(self):
        """Verify ModelRunner ABC is importable and has correct interface."""
        from eval.models.base import ModelRunner
        import inspect
        assert inspect.isabstract(ModelRunner)
        assert hasattr(ModelRunner, "extract")


# ---------------------------------------------------------------------------
# aggregate_scores
# ---------------------------------------------------------------------------

class TestAggregateScores:
    def test_empty_list(self):
        agg = aggregate_scores([])
        assert agg.store_name.f1 == pytest.approx(0.0)

    def test_single_score(self):
        s = ReceiptScore()
        s.store_name = FieldScore(precision=1.0, recall=1.0, f1=1.0, n_pred=1, n_gt=1)
        agg = aggregate_scores([s])
        assert agg.store_name.f1 == pytest.approx(1.0)

    def test_mean_of_two(self):
        s1 = ReceiptScore()
        s1.total = FieldScore(precision=1.0, recall=1.0, f1=1.0, n_pred=1, n_gt=1)
        s2 = ReceiptScore()
        s2.total = FieldScore(precision=0.0, recall=0.0, f1=0.0, n_pred=1, n_gt=1)
        agg = aggregate_scores([s1, s2])
        assert agg.total.f1 == pytest.approx(0.5)
        assert agg.total.precision == pytest.approx(0.5)
        assert agg.total.recall == pytest.approx(0.5)
