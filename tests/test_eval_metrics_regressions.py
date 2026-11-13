from eval.ir_metrics import (
    calculate_precision_at_k, calculate_recall_at_k,
    calculate_reciprocal_rank, calculate_ndcg_at_k,
)
from src.agent.registry import RegulatoryObligationRegistry


def test_ir_metrics_deduplicate_ranked_documents_and_handle_empty_relevance():
    assert calculate_precision_at_k(["a", "a", "b"], {"a"}, 2) == 0.5
    assert calculate_recall_at_k(["a", "a", "b"], {"a", "b"}, 3) == 1.0
    assert calculate_reciprocal_rank(["x", "a", "a"], {"a"}) == 0.5
    assert calculate_precision_at_k(["a"], set(), 4) == 0.0
    assert calculate_recall_at_k(["a"], set(), 4) == 0.0
    assert calculate_ndcg_at_k(["a"], {}, 4) == 0.0


def test_ndcg_respects_graded_relevance():
    ideal = calculate_ndcg_at_k(["high", "low"], {"high": 3, "low": 1}, 2)
    reversed_order = calculate_ndcg_at_k(["low", "high"], {"high": 3, "low": 1}, 2)
    assert ideal == 1.0
    assert 0 < reversed_order < ideal


def test_official_2025_cooling_off_threshold_and_fee_exception():
    reg = RegulatoryObligationRegistry()
    assert reg.evaluate_clause("14-day loan has a 2-day cooling-off period") is None
    assert reg.evaluate_clause("5-day loan has a 1-day cooling-off period") is None
    assert reg.evaluate_clause("14-day loan has a 6-hour cooling-off period")["status"] == "FAIL"
    assert reg.evaluate_clause("1-day cooling-off; reasonable one-time processing fee disclosed upfront") is None
    assert reg.evaluate_clause("1-day cooling-off subject to a 2% foreclosure penalty")["status"] == "FAIL"


def test_dlg_cap_boundary_and_correct_denominator_description():
    reg = RegulatoryObligationRegistry()
    assert reg.evaluate_clause("DLG cover is exactly 5 percent of amount disbursed") is None
    assert reg.evaluate_clause("DLG cover is 5.01 percent of amount disbursed")["status"] == "FAIL"
    obl = reg.get_obligation("OBL_DL_006")
    assert "total amount disbursed" in obl["statutory_rule"].lower()
