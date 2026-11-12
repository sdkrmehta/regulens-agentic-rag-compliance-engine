"""Hybrid Vector Engine combining Dense (bge-small-en-v1.5) and Sparse (BM25) via Qdrant RRF."""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional

from fastembed import TextEmbedding, SparseTextEmbedding
from qdrant_client import QdrantClient, models

from src.config import settings
from src.retrieval.reranker import FlashRankReranker

# Global singletons for in-memory caching
_DENSE_MODEL: Optional[TextEmbedding] = None
_SPARSE_MODEL: Optional[SparseTextEmbedding] = None
_PARENT_DOCSTORE: Optional[Dict[str, Dict[str, Any]]] = None
_QDRANT_CLIENT: Optional[QdrantClient] = None


def get_dense_model() -> TextEmbedding:
    global _DENSE_MODEL
    if _DENSE_MODEL is None:
        _DENSE_MODEL = TextEmbedding(model_name=settings.DENSE_EMBED_MODEL)
    return _DENSE_MODEL


def get_sparse_model() -> SparseTextEmbedding:
    global _SPARSE_MODEL
    if _SPARSE_MODEL is None:
        _SPARSE_MODEL = SparseTextEmbedding(model_name=settings.SPARSE_EMBED_MODEL)
    return _SPARSE_MODEL


def get_parent_docstore(docstore_path: str = "data/parent_docstore.json") -> Dict[str, Dict[str, Any]]:
    global _PARENT_DOCSTORE
    if _PARENT_DOCSTORE is None:
        path = Path(docstore_path)
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                _PARENT_DOCSTORE = json.load(f)
        else:
            _PARENT_DOCSTORE = {}
    return _PARENT_DOCSTORE


def get_qdrant_client() -> QdrantClient:
    global _QDRANT_CLIENT
    if _QDRANT_CLIENT is None:
        # Check if Qdrant Cloud or remote URL is specified
        if settings.QDRANT_API_KEY and settings.QDRANT_API_KEY.get_secret_value():
            _QDRANT_CLIENT = QdrantClient(
                url=settings.QDRANT_URL,
                api_key=settings.QDRANT_API_KEY.get_secret_value()
            )
        else:
            # Try connecting to local Docker/service endpoint
            try:
                import urllib.request
                urllib.request.urlopen(f"{settings.QDRANT_URL}/collections", timeout=1)
                _QDRANT_CLIENT = QdrantClient(url=settings.QDRANT_URL)
            except Exception:
                # Seamless fallback to embedded on-disk Qdrant storage
                storage_path = Path("./qdrant_storage")
                storage_path.mkdir(parents=True, exist_ok=True)
                _QDRANT_CLIENT = QdrantClient(path=str(storage_path))
    return _QDRANT_CLIENT


class RBIHybridVectorStore:
    def __init__(
        self,
        collection_name: Optional[str] = None,
        client: Optional[QdrantClient] = None,
        docstore_path: str = "data/parent_docstore.json",
    ):
        self.collection_name = collection_name or settings.QDRANT_COLLECTION
        self.client = client or get_qdrant_client()
        self.dense_model = get_dense_model()
        self.sparse_model = get_sparse_model()
        self.docstore = get_parent_docstore(docstore_path)

    def init_collection(self, recreate: bool = False):
        """Create or recreate the hybrid collection with dual named vector spaces and payload indexes."""
        exists = self.client.collection_exists(self.collection_name)
        if exists and recreate:
            self.client.delete_collection(self.collection_name)
            exists = False

        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": models.VectorParams(size=384, distance=models.Distance.COSINE)
                },
                sparse_vectors_config={
                    "bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)
                }
            )
            # Create keyword payload indexes for accelerated metadata filtering
            for field in ["doc_category", "circular_name", "parent_id"]:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field,
                    field_schema=models.PayloadSchemaType.KEYWORD
                )
            print(f"[SUCCESS] Initialized collection '{self.collection_name}' with dual vector spaces.")

    def embed_texts(self, texts: List[str], batch_size: int = 64) -> tuple[List[List[float]], List[models.SparseVector]]:
        """Generate dense and sparse embeddings on CPU."""
        dense_generator = self.dense_model.embed(texts, batch_size=batch_size)
        dense_vectors = [vec.tolist() for vec in dense_generator]

        sparse_generator = self.sparse_model.embed(texts, batch_size=batch_size)
        sparse_vectors = [
            models.SparseVector(indices=sv.indices.tolist(), values=sv.values.tolist())
            for sv in sparse_generator
        ]
        return dense_vectors, sparse_vectors

    def index_child_chunks(self, child_chunks: List[Dict[str, Any]], batch_size: int = 64):
        """Idempotently batch-upsert child chunks using deterministic UUIDv5 child_id keys."""
        self.init_collection(recreate=False)
        total_chunks = len(child_chunks)
        print(f"Indexing {total_chunks} child chunks into '{self.collection_name}'...")

        for i in range(0, total_chunks, batch_size):
            batch = child_chunks[i : i + batch_size]
            texts = [c["text"] for c in batch]
            dense_vectors, sparse_vectors = self.embed_texts(texts, batch_size=batch_size)

            points = []
            for chunk, dense_vec, sparse_vec in zip(batch, dense_vectors, sparse_vectors):
                points.append(
                    models.PointStruct(
                        id=chunk["child_id"],
                        vector={
                            "dense": dense_vec,
                            "bm25": sparse_vec
                        },
                        payload={
                            "child_id": chunk["child_id"],
                            "parent_id": chunk["parent_id"],
                            "text": chunk["text"],
                            "circular_name": chunk["circular_name"],
                            "doc_category": chunk["doc_category"],
                            "section_clause": chunk["section_clause"],
                            "page_number": chunk["page_number"],
                            "has_table": chunk.get("has_table", False),
                            "child_index": chunk.get("child_index", 0)
                        }
                    )
                )

            self.client.upsert(collection_name=self.collection_name, points=points)
            print(f"Upserted points {i + 1} to {min(i + batch_size, total_chunks)} / {total_chunks}")

        print(f"[SUCCESS] Completed indexing of {total_chunks} chunks.")

    def hybrid_search(
        self,
        query: str,
        doc_category: Optional[str] = None,
        top_k: int = 15
    ) -> List[Dict[str, Any]]:
        """
        Execute single-roundtrip hybrid prefetch via Qdrant's Reciprocal Rank Fusion (RRF),
        followed by child-to-parent resolution and deduplication.
        """
        dense_vec, sparse_vec = self.embed_texts([query])
        dense_q = dense_vec[0]
        sparse_q = sparse_vec[0]

        query_filter = None
        if doc_category:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="doc_category",
                        match=models.MatchValue(value=doc_category)
                    )
                ]
            )

        # Single-roundtrip Qdrant Query API with RRF fusion
        results = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                models.Prefetch(query=dense_q, using="dense", limit=top_k, filter=query_filter),
                models.Prefetch(query=sparse_q, using="bm25", limit=top_k, filter=query_filter),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=top_k,
            with_payload=True
        )

        # Child-to-Parent resolution and deduplication
        deduped_parents: Dict[str, Dict[str, Any]] = {}
        matched_scores: Dict[str, float] = {}

        for hit in results.points:
            payload = hit.payload or {}
            parent_id = payload.get("parent_id")
            score = hit.score or 0.0

            if not parent_id:
                continue

            # Record max RRF score for this parent
            if parent_id not in matched_scores or score > matched_scores[parent_id]:
                matched_scores[parent_id] = score

            if parent_id not in deduped_parents:
                parent_entry = self.docstore.get(parent_id)
                if parent_entry:
                    doc_copy = dict(parent_entry)
                    doc_copy["rrf_score"] = score
                    doc_copy["matched_child_id"] = payload.get("child_id")
                    deduped_parents[parent_id] = doc_copy
                else:
                    # Fallback if parent missing from store
                    deduped_parents[parent_id] = {
                        "parent_id": parent_id,
                        "text": payload.get("text", ""),
                        "circular_name": payload.get("circular_name", ""),
                        "doc_category": payload.get("doc_category", ""),
                        "section_clause": payload.get("section_clause", ""),
                        "page_number": payload.get("page_number", 1),
                        "has_table": payload.get("has_table", False),
                        "rrf_score": score
                    }

        # Assign highest child RRF score and sort descending
        parent_list = []
        for p_id, p_doc in deduped_parents.items():
            p_doc["rrf_score"] = matched_scores[p_id]
            parent_list.append(p_doc)

        parent_list.sort(key=lambda x: x["rrf_score"], reverse=True)
        return parent_list

    def hybrid_search_and_rerank(
        self,
        query: str,
        doc_category: Optional[str] = None,
        prefetch_limit: Optional[int] = None,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Execute full two-tier hybrid retrieval:
        1. Single-roundtrip Dense + Sparse BM25 RRF prefetch (top 15)
        2. Child-to-Parent resolution and deduplication
        3. FlashRank Cross-Encoder reranking down to top 4 parent documents
        """
        limit = prefetch_limit or settings.RRF_PREFETCH_LIMIT
        deduped_parents = self.hybrid_search(query=query, doc_category=doc_category, top_k=limit)
        reranker = FlashRankReranker(top_k=top_k)
        return reranker.rerank(query=query, parent_documents=deduped_parents, top_k=top_k)


if __name__ == "__main__":
    import sys
    from scripts.index_corpus import main as run_indexing

    if "--upsert-cloud" in sys.argv:
        print("[MIGRATION] Running Qdrant Cloud index migration using configured QDRANT_URL & QDRANT_API_KEY...")
        run_indexing()
        print("[MIGRATION] Qdrant Cloud vector index upsert completed successfully.")
    else:
        print("Usage: python -m src.retrieval.vector_store [--upsert-cloud]")
