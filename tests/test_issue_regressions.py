"""Regressions for historical workflow, date filtering, cache isolation and provenance."""

from pathlib import Path

from src.agent import nodes
from src.agent.graph import build_compliance_graph
from src.agent.verifier import independent_page_verification, verify_citation_against_source
from src.retrieval.semantic_cache import EntityScopedSemanticCache, extract_regulatory_entities
from src.retrieval.vector_store import RBIHybridVectorStore, _effective_date_allows
from src.schemas import CitationItem


def test_explicit_historical_mode_survives_graph_and_grading(monkeypatch):
    old_doc = {
        "parent_id": "old-guidance", "document_id": "RBI_DL_2022",
        "text": "Digital lending first loss default guarantee guidelines historical regulatory provision",
        "circular_name": "Guidelines on Digital Lending (2022)",
        "section_clause": "Clause 1", "page_number": 1,
        "document_status": "REPEALED", "rrf_score": 1.0,
    }

    class Store:
        def hybrid_search_and_rerank(self, **kwargs):
            assert kwargs["historical_mode"] is True
            return [old_doc]

    monkeypatch.setattr(nodes, "get_vector_store", lambda: Store())
    monkeypatch.setattr(nodes, "get_llm", lambda *args, **kwargs: None)
    result = build_compliance_graph().invoke({
        "question": "Show prior rules on first loss guarantee",
        "historical_query": True,
        "search_query": "first loss default guarantee historical",
        "retry_count": 0,
    })
    assert result["historical_query"] is True
    assert [d["document_id"] for d in result["graded_parents"]] == ["RBI_DL_2022"]
    assert result["final_response"].applicable_status == "REPEALED"


def test_repealed_documents_are_removed_in_active_mode():
    doc = {"parent_id": "old", "document_id": "RBI_DL_2022", "text": "digital lending first loss default guarantee",
           "document_status": "REPEALED", "rrf_score": 1.0}
    assert nodes.grader_node({"question": "digital lending first loss", "retrieved_parents": [doc],
                              "historical_query": False})["graded_parents"] == []


def test_effective_date_filter_rejects_missing_invalid_and_future_dates():
    assert _effective_date_allows("2024-01-01", "2024-01-01")
    assert not _effective_date_allows(None, "2024-01-01")
    assert not _effective_date_allows("not-a-date", "2024-01-01")
    assert not _effective_date_allows("2025-05-08", "2024-01-01")
    assert _effective_date_allows(None, None)


def test_cache_uses_one_canonical_entity_dimension_in_all_modes():
    cache = EntityScopedSemanticCache()
    cache.clear_cache()
    entities = extract_regulatory_entities("What is the high-risk KYC update period?")
    assert cache.store_cache("What is the high-risk KYC update period?", "compliance_qa", ["router-output"],
                             _verified_response(), category_filter="KYC", historical_mode=True,
                             corpus_version="test")
    assert cache.check_cache("What is the high-risk KYC update period?", "compliance_qa", [],
                             category_filter="KYC", historical_mode=True, corpus_version="test")
    assert entities
    assert cache.check_cache("What is the high-risk KYC update period?", "compliance_qa", [],
                             category_filter="KYC", historical_mode=False, corpus_version="test") is None


def _verified_response():
    from src.schemas import CitedResponse
    citation = CitationItem(circular_name="RBI KYC", section_clause="Clause 38", page_number=None,
                            verbatim_quote="High risk customers update their KYC records once every two years",
                            document_id="RBI_KYC_2016", verified=True, verification_status="VERIFIED")
    return CitedResponse(answer="Verified", citations=[citation], risk_level="LOW")


def test_citation_verification_fails_closed_on_empty_registry(monkeypatch):
    monkeypatch.setattr("src.agent.verifier.get_verified_registry", lambda: {})
    citation = CitationItem(circular_name="Unknown RBI direction", section_clause="1", page_number=None,
                            verbatim_quote="This is a sufficiently long quotation for the verification test",
                            document_id="RBI_FAKE")
    ok, status, reason = verify_citation_against_source(citation, [])
    assert not ok and status == "UNVERIFIED"
    assert "registry" in reason.lower()


def test_page_verification_requires_genuine_source_page_markers():
    # The HTML gazette has no authentic PDF page boundaries; its page remains null.
    ok, reason = independent_page_verification("not used", "RBI_DL_2025", 12)
    assert not ok
    assert "unavailable" in reason.lower() or "cannot be verified" in reason.lower()


def test_rbi_metadata_matches_official_publication_references():
    import json
    registry = json.loads(Path("data/regulatory_registry.json").read_text(encoding="utf-8"))
    docs = {doc["document_id"]: doc for doc in registry["documents"]}
    assert "Id=12848" in docs["RBI_DL_2025"]["source_url"]
    assert "DOR.STR.REC.19/21.07.001/2025-26" in docs["RBI_DL_2025"]["circular_number"]
    assert "August 14, 2025" in docs["RBI_KYC_2016"]["version"]
    assert docs["RBI_CYBER_2024"]["circular_number"].startswith("RBI/DPSS/2024-25/123")
