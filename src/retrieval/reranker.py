"""Cross-Encoder Reranking module using FlashRank (ms-marco-MiniLM-L-12-v2)."""

from typing import Dict, List, Any, Optional
from flashrank import Ranker, RerankRequest

from src.config import settings

_RANKER: Optional[Ranker] = None


def get_ranker() -> Ranker:
    global _RANKER
    if _RANKER is None:
        _RANKER = Ranker(
            model_name=settings.RERANK_MODEL,
            cache_dir="./.model_cache"
        )
    return _RANKER


class FlashRankReranker:
    def __init__(self, top_k: Optional[int] = None, ranker: Optional[Ranker] = None):
        self.top_k = top_k or settings.RERANK_TOP_K
        self.ranker = ranker or get_ranker()

    def rerank(
        self,
        query: str,
        parent_documents: List[Dict[str, Any]],
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Rerank deduplicated parent documents using FlashRank cross-encoder.
        """
        if not parent_documents:
            return []

        limit = top_k or self.top_k

        # Prepare passages for FlashRank
        passages = []
        doc_map = {}
        for doc in parent_documents:
            doc_id = doc["parent_id"]
            # Deduplicate by id if passed multiple times
            if doc_id in doc_map:
                continue
            doc_map[doc_id] = doc
            passages.append({
                "id": doc_id,
                "text": doc.get("text", ""),
                "meta": {
                    "circular_name": doc.get("circular_name", ""),
                    "section_clause": doc.get("section_clause", ""),
                    "page_number": doc.get("page_number", 1),
                    "has_table": doc.get("has_table", False),
                }
            })

        rerank_request = RerankRequest(query=query, passages=passages)
        ranked_passages = self.ranker.rerank(rerank_request)

        # Attach rerank score and return top_k
        reranked_docs = []
        for r_item in ranked_passages[:limit]:
            doc_id = r_item["id"]
            orig_doc = dict(doc_map[doc_id])
            orig_doc["rerank_score"] = float(r_item.get("score", 0.0))
            reranked_docs.append(orig_doc)

        return reranked_docs
