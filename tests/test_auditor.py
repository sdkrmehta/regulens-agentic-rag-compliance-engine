"""Phase 5 Quality Gate: Multi-Violation Contract Auditing & Latency SLA Verification."""

import time
import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.agent.auditor import audit_fintech_contract
from src.retrieval.semantic_cache import EntityScopedSemanticCache
from src.retrieval.vector_store import RBIHybridVectorStore
from src.schemas import CitedResponse, CitationItem


def test_gate_assertion_1_multi_violation_detection():
    """
    Gate Assertion 1 (Multi-Violation Detection):
    Submit sample non-compliant digital lending clause into audit pipeline and assert
    that ContractAuditReport independently flags all three statutory violations:
    (1) prohibited contact/gallery access
    (2) invalid 1-day cooling-off window vs. mandatory minimum look-up period
    (3) prohibited prepayment penalty during cooling-off
    """
    sample_clause = (
        "The borrower irrevocably consents that the DLA may access phone contacts, call logs, "
        "and photo gallery for credit scoring, and agrees to a 1-day cooling-off period subject to a 5% foreclosure fee."
    )

    # Test via direct module
    report = audit_fintech_contract(sample_clause)

    assert report.overall_verdict == "NON_COMPLIANT", f"Expected NON_COMPLIANT but got {report.overall_verdict}"
    assert len(report.clause_findings) >= 3, f"Expected at least 3 findings, got {len(report.clause_findings)}"

    fail_findings = [f for f in report.clause_findings if f.compliance_status == "FAIL"]
    assert len(fail_findings) >= 3, f"Expected 3 FAIL violations, got {len(fail_findings)}"

    # Violation 1: Prohibited contacts / gallery access
    v1_matches = [
        f for f in fail_findings
        if any(w in f.contract_excerpt.lower() for w in ["contact", "call log", "gallery", "photo"])
        or any(w in (f.violated_clause or "").lower() for w in ["collection", "sharing of data", "data", "10"])
    ]
    assert len(v1_matches) >= 1, "Violation 1 (Prohibited contact/gallery access) was not flagged."

    # Violation 2: Invalid 1-day cooling-off window
    v2_matches = [
        f for f in fail_findings
        if any(w in f.contract_excerpt.lower() for w in ["1-day", "cooling-off", "cooling off"])
        and ("8." in (f.violated_clause or "") or "cooling" in (f.violated_clause or "").lower())
    ]
    assert len(v2_matches) >= 1, "Violation 2 (Invalid 1-day cooling-off window) was not flagged."

    # Violation 3: Prohibited foreclosure / prepayment penalty
    v3_matches = [
        f for f in fail_findings
        if any(w in f.contract_excerpt.lower() for w in ["foreclosure", "fee", "penalty", "prepayment"])
        and ("penalty" in (f.verbatim_rbi_rule or "").lower() or "8." in (f.violated_clause or ""))
    ]
    assert len(v3_matches) >= 1, "Violation 3 (Prohibited prepayment penalty during cooling-off) was not flagged."


def test_gate_assertion_1_fastapi_endpoint():
    """Verify the /audit-contract endpoint works."""
    from src.api.main import audit_contract_endpoint
    from src.schemas import ContractAuditRequest
    sample_clause = (
        "The borrower irrevocably consents that the DLA may access phone contacts, call logs, "
        "and photo gallery for credit scoring, and agrees to a 1-day cooling-off period subject to a 5% foreclosure fee."
    )
    report = audit_contract_endpoint(ContractAuditRequest(contract_text=sample_clause))
    data = report.model_dump()
    assert data["overall_verdict"] == "NON_COMPLIANT"
    assert len(data["clause_findings"]) >= 3


def test_gate_assertion_2_latency_slas():
    """
    Gate Assertion 2 (Latency SLAs):
    Verify P95 latency is <= 2.2s for cold single-pass queries and <= 0.08s (80ms) for warm semantic cache hits.
    """
    # 1. Warm semantic cache hits test (10 iterations)
    cache = EntityScopedSemanticCache()
    test_q = "What is the mandatory cooling-off period in digital lending?"
    dummy_resp = CitedResponse(
        answer="A cooling-off period of not less than three days is mandatory.",
        citations=[CitationItem(
            circular_name="RBI Guidelines on Digital Lending",
            section_clause="8. Cooling off/look-up period",
            page_number=6,
            verbatim_quote="not be less than three days for loans having tenor of seven days or more",
            verified=True
        )],
        risk_level="HIGH"
    )
    cache.store_cache(test_q, "compliance_qa", ["cooling-off period"], dummy_resp)

    # Prime / warm-up lookup
    _ = cache.check_cache(test_q, "compliance_qa", ["cooling-off period"])

    warm_latencies = []
    for _ in range(10):
        t0 = time.time()
        hit = cache.check_cache(test_q, "compliance_qa", ["cooling-off period"])
        t1 = time.time()
        assert hit is not None
        warm_latencies.append(t1 - t0)

    p95_warm = float(np.percentile(warm_latencies, 95))
    print(f"\nWarm Cache P95 Latency: {p95_warm*1000:.2f} ms")
    assert p95_warm <= 0.080, f"Warm cache P95 latency exceeded SLA of 80ms: {p95_warm*1000:.2f}ms"

    # 2. Cold single-pass query retrieval test
    vs = RBIHybridVectorStore()
    cold_queries = [
        "What is the minimum cooling-off period for digital loans?",
        "What are V-CIP video encryption rules?",
        "How many hours to report cyber incident to CERT-In?",
    ]
    cold_latencies = []
    for q in cold_queries:
        t0 = time.time()
        docs = vs.hybrid_search_and_rerank(q, top_k=4)
        t1 = time.time()
        assert len(docs) > 0
        cold_latencies.append(t1 - t0)

    p95_cold = float(np.percentile(cold_latencies, 95))
    print(f"Cold Retrieval P95 Latency: {p95_cold*1000:.2f} ms")
    assert p95_cold <= 2.2, f"Cold query retrieval P95 latency exceeded SLA of 2.2s: {p95_cold:.2f}s"
