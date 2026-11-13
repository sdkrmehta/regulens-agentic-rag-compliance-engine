"""Phase 4 Verification & Quality Gate Tests (tests/test_cache.py)"""

from pathlib import Path
import csv
import pytest

from src.retrieval.semantic_cache import EntityScopedSemanticCache
from src.schemas import CitedResponse, CitationItem


def test_gate_assertion_1_ablation_superiority():
    """
    Gate Assertion 1 (Ablation Superiority):
    Confirm in eval/benchmark_results.csv that the V3.0 Agentic Hybrid pipeline
    outperforms the Naive Baseline in Context Precision and Context Recall.
    """
    csv_path = Path("eval/benchmark_results.csv")
    assert csv_path.exists(), f"Benchmark results not found at {csv_path}"

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    assert len(reader) >= 20, f"Expected at least 20 evaluation records, found {len(reader)}"

    avg_b_precision = sum(float(r["baseline_precision"]) for r in reader) / len(reader)
    avg_a_precision = sum(float(r["agentic_precision"]) for r in reader) / len(reader)

    avg_b_recall = sum(float(r["baseline_recall"]) for r in reader) / len(reader)
    avg_a_recall = sum(float(r["agentic_recall"]) for r in reader) / len(reader)

    assert avg_a_precision >= avg_b_precision, (
        f"Agentic precision ({avg_a_precision:.3f}) should outperform or match baseline ({avg_b_precision:.3f})"
    )
    assert avg_a_recall > avg_b_recall, (
        f"Agentic recall ({avg_a_recall:.3f}) must outperform baseline ({avg_b_recall:.3f})"
    )


def test_gate_assertion_2_entity_safe_cache_isolation():
    """
    Gate Assertion 2 (Entity-Safe Cache Isolation):
    Query 'What is the periodic KYC update frequency for High-Risk customers?'
    followed immediately by 'What is the periodic KYC update frequency for Low-Risk customers?'
    and assert the second query registers cache_hit == False despite >0.94 vector cosine similarity.
    """
    cache = EntityScopedSemanticCache()
    cache.init_cache_collection(recreate=True)

    high_risk_q = "What is the periodic KYC update frequency for High-Risk customers?"
    low_risk_q = "What is the periodic KYC update frequency for Low-Risk customers?"

    high_risk_resp = CitedResponse(
        answer="High-Risk customer KYC must be updated at least once every 2 years under Clause 38.",
        citations=[
            CitationItem(
                circular_name="RBI Master Direction on KYC",
                section_clause="Clause 38",
                page_number=30,
                verbatim_quote="once in every two years",
                verified=True
            )
        ],
        risk_level="HIGH"
    )

    # 1. Store High-Risk response in cache
    stored = cache.store_cache(
        question=high_risk_q,
        query_type="compliance_qa",
        extracted_entities=["High-Risk"],
        response=high_risk_resp
    )
    assert stored is True, "Failed to store High-Risk response in semantic cache"

    # 2. Query High-Risk again -> Must be a Cache Hit
    hit_high = cache.check_cache(
        question=high_risk_q,
        query_type="compliance_qa",
        extracted_entities=["High-Risk"]
    )
    assert hit_high is not None, "Expected cache hit for identical High-Risk query"
    assert hit_high["cache_hit"] is True

    # 3. Query Low-Risk -> Entity Guard MUST prevent collision (cache_hit == False / None)
    hit_low = cache.check_cache(
        question=low_risk_q,
        query_type="compliance_qa",
        extracted_entities=["Low-Risk"]
    )
    assert hit_low is None, (
        "Entity Guard failed! Low-Risk query incorrectly collided with High-Risk cached response."
    )


def test_cache_multi_dimensional_isolation_all_five_dimensions():
    """
    Verify strict cache isolation across all 5 dimensions:
    1. regulatory category
    2. entity type
    3. effective date
    4. historical mode
    5. corpus version
    """
    cache = EntityScopedSemanticCache()
    cache.init_cache_collection(recreate=True)

    base_q = "What are the record retention requirements for customer transaction logs?"
    base_resp = CitedResponse(
        answer="Records of customer transactions and identification must be retained for at least 5 years.",
        citations=[
            CitationItem(
                circular_name="Master Direction - Know Your Customer (KYC) Direction, 2016",
                section_clause="Section 38",
                page_number=35,
                verbatim_quote="shall be maintained for at least five years from the date of transaction",
                verified=True,
                verification_status="VERIFIED",
            )
        ],
        risk_level="HIGH",
    )

    # Store baseline entry with all 5 dimensions explicitly defined
    stored = cache.store_cache(
        question=base_q,
        query_type="compliance_qa",
        extracted_entities=["kyc"],
        response=base_resp,
        category_filter="KYC",
        entity_type="NBFC",
        effective_date="2024-01-01",
        historical_mode=False,
        corpus_version="3.0",
    )
    assert stored is True

    # 1. Exact match on all 5 dimensions -> CACHE HIT
    hit = cache.check_cache(
        question=base_q,
        query_type="compliance_qa",
        extracted_entities=["kyc"],
        category_filter="KYC",
        entity_type="NBFC",
        effective_date="2024-01-01",
        historical_mode=False,
        corpus_version="3.0",
    )
    assert hit is not None, "Expected cache hit on identical 5-dimensional query parameters"

    # 2. Divergence on category_filter -> CACHE MISS
    hit_diff_cat = cache.check_cache(
        question=base_q,
        query_type="compliance_qa",
        extracted_entities=["kyc"],
        category_filter="DIGITAL_LENDING",
        entity_type="NBFC",
        effective_date="2024-01-01",
        historical_mode=False,
        corpus_version="3.0",
    )
    assert hit_diff_cat is None, "Cache isolation failed: different regulatory category must trigger cache miss"

    # 3. Divergence on entity_type -> CACHE MISS
    hit_diff_entity = cache.check_cache(
        question=base_q,
        query_type="compliance_qa",
        extracted_entities=["kyc"],
        category_filter="KYC",
        entity_type="Commercial Bank",
        effective_date="2024-01-01",
        historical_mode=False,
        corpus_version="3.0",
    )
    assert hit_diff_entity is None, "Cache isolation failed: different entity type must trigger cache miss"

    # 4. Divergence on effective_date -> CACHE MISS
    hit_diff_date = cache.check_cache(
        question=base_q,
        query_type="compliance_qa",
        extracted_entities=["kyc"],
        category_filter="KYC",
        entity_type="NBFC",
        effective_date="2025-05-08",
        historical_mode=False,
        corpus_version="3.0",
    )
    assert hit_diff_date is None, "Cache isolation failed: different effective date must trigger cache miss"

    # 5. Divergence on historical_mode -> CACHE MISS
    hit_diff_hist = cache.check_cache(
        question=base_q,
        query_type="compliance_qa",
        extracted_entities=["kyc"],
        category_filter="KYC",
        entity_type="NBFC",
        effective_date="2024-01-01",
        historical_mode=True,
        corpus_version="3.0",
    )
    assert hit_diff_hist is None, "Cache isolation failed: historical mode switch must trigger cache miss"

    # 6. Divergence on corpus_version -> CACHE MISS
    hit_diff_ver = cache.check_cache(
        question=base_q,
        query_type="compliance_qa",
        extracted_entities=["kyc"],
        category_filter="KYC",
        entity_type="NBFC",
        effective_date="2024-01-01",
        historical_mode=False,
        corpus_version="2.0",
    )
    assert hit_diff_ver is None, "Cache isolation failed: differing corpus version must trigger cache miss"
