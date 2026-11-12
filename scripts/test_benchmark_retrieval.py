"""Verification script for the 5 benchmark retrieval queries."""

from src.retrieval.vector_store import RBIHybridVectorStore

store = RBIHybridVectorStore()

test_queries = [
    ("Query 1 (Cooling-off / KFS)", "What is the mandatory cooling-off period in a KFS under Digital Lending Directions?"),
    ("Query 2 (V-CIP)", "What are the V-CIP video encryption and geo-tagging requirements under KYC?"),
    ("Query 3 (6-hour Incident Reporting)", "Within how many hours must a cybersecurity incident be reported to CERT-In and RBI?"),
    ("Query 4 (Periodic KYC Schedule)", "What is the periodic KYC update schedule for High-Risk vs Low-Risk customers?"),
    ("Query 5 (Phone Contacts / DLA)", "Are Digital Lending Apps (DLAs) allowed to access mobile phone contacts and photo gallery?")
]

for label, q in test_queries:
    results = store.hybrid_search_and_rerank(query=q, top_k=4)
    print(f"\n=== {label} ===")
    for i, r in enumerate(results[:3], 1):
        parent_id = r["parent_id"]
        score = r.get("rerank_score", 0.0)
        clause = r.get("section_clause", "")
        text_snippet = r["text"][:120].replace("\n", " ")
        print(f"Rank {i}: [{parent_id}] (Score: {score:.4f}) Clause: {clause}\n  Snippet: {text_snippet}...")
