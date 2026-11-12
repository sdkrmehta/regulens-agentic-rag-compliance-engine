"""Phase 2 Verification & Quality Gate Tests (tests/test_retrieval.py)"""

import pytest
from src.retrieval.vector_store import RBIHybridVectorStore


@pytest.fixture(scope="module")
def vector_store():
    return RBIHybridVectorStore()


def test_gate_assertion_1_acronym_and_clause_precision(vector_store):
    """
    Gate Assertion 1 (Acronym & Clause Precision):
    Execute 5 targeted test queries containing dense banking acronyms and clause numbers.
    Assert that for all 5 queries, the ground-truth Parent Chunk appears within Rank 1 or Rank 2
    after FlashRank reranking.
    """
    test_cases = [
        {
            "query": "What is the mandatory cooling-off period in a KFS under Digital Lending Directions?",
            "expected_terms": ["cooling", "look-up", "apr", "penalty"],
            "expected_doc_category": "DIGITAL_LENDING",
            "name": "Cooling-off / KFS"
        },
        {
            "query": "What are the V-CIP video encryption and geo-tagging requirements under KYC?",
            "expected_terms": ["v-cip", "encryption", "video"],
            "expected_doc_category": "KYC",
            "name": "V-CIP technical specifications"
        },
        {
            "query": "Within how many hours must a cybersecurity incident be reported to CERT-In and RBI?",
            "expected_terms": ["incident", "cert-in", "response", "hours"],
            "expected_doc_category": "CYBER_RESILIENCE",
            "name": "Cyber incident reporting window"
        },
        {
            "query": "What is the periodic KYC update schedule for High-Risk vs Low-Risk customers?",
            "expected_terms": ["updation", "periodic", "risk", "years"],
            "expected_doc_category": "KYC",
            "name": "Periodic KYC update schedule"
        },
        {
            "query": "Are Digital Lending Apps (DLAs) allowed to access mobile phone contacts and photo gallery?",
            "expected_terms": ["collection", "data", "third parties", "dla", "access"],
            "expected_doc_category": "DIGITAL_LENDING",
            "name": "DLA mobile resource access restrictions"
        }
    ]

    for tc in test_cases:
        results = vector_store.hybrid_search_and_rerank(query=tc["query"], top_k=4)
        assert len(results) >= 2, f"Expected at least 2 results for {tc['name']}, got {len(results)}"

        # Check that Rank 1 or Rank 2 contains the ground-truth concepts
        top_2_results = results[:2]
        matched = False
        matched_details = []

        for rank_idx, doc in enumerate(top_2_results, 1):
            doc_text = doc["text"].lower()
            doc_cat = doc.get("doc_category")
            clause = doc.get("section_clause", "")

            # Check if matching category and expected domain terms
            has_cat = doc_cat == tc["expected_doc_category"]
            term_matches = [term for term in tc["expected_terms"] if term in doc_text]

            matched_details.append(f"Rank {rank_idx} ({doc['parent_id']}): {len(term_matches)}/{len(tc['expected_terms'])} terms")
            if has_cat and len(term_matches) >= 2:
                matched = True
                break

        assert matched, (
            f"Ground-truth parent chunk did not appear in Rank 1 or 2 for '{tc['name']}'. Details: {matched_details}"
        )


def test_gate_assertion_2_parent_deduplication(vector_store):
    """
    Gate Assertion 2 (Parent Deduplication):
    Assert that hybrid_search_and_rerank() never returns two items with the same parent_id.
    """
    queries = [
        "What is the cooling-off period?",
        "V-CIP facial recognition and encryption rules",
        "Cybersecurity controls and incident escalation to RBI",
        "High risk customer periodic KYC update requirements",
        "Key Fact Statement disclosures and APR"
    ]

    for q in queries:
        results = vector_store.hybrid_search_and_rerank(query=q, top_k=6)
        parent_ids = [doc["parent_id"] for doc in results]
        unique_ids = set(parent_ids)
        assert len(parent_ids) == len(unique_ids), (
            f"Duplicate parent_ids returned for query '{q}': {parent_ids}"
        )
