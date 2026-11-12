"""Hybrid Vector Engine combining Dense (bge-small-en-v1.5) and Sparse (BM25) with Metadata Filtering."""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

from fastembed import TextEmbedding, SparseTextEmbedding
from qdrant_client import QdrantClient, models

from src.config import settings
from src.retrieval.reranker import FlashRankReranker

logger = logging.getLogger(__name__)

# Global singletons for in-memory caching
_DENSE_MODEL: Optional[TextEmbedding] = None
_SPARSE_MODEL: Optional[SparseTextEmbedding] = None
_PARENT_DOCSTORE: Optional[Dict[str, Dict[str, Any]]] = None
_QDRANT_CLIENT: Optional[QdrantClient] = None
_QDRANT_OFFLINE: bool = False


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
        if settings.QDRANT_API_KEY and settings.QDRANT_API_KEY.get_secret_value():
            _QDRANT_CLIENT = QdrantClient(
                url=settings.QDRANT_URL,
                api_key=settings.QDRANT_API_KEY.get_secret_value(),
                timeout=1.0,
                check_compatibility=False
            )
        else:
            try:
                import urllib.request
                urllib.request.urlopen(f"{settings.QDRANT_URL}/collections", timeout=1)
                _QDRANT_CLIENT = QdrantClient(url=settings.QDRANT_URL, timeout=1.0, check_compatibility=False)
            except Exception:
                storage_path = Path("./qdrant_storage")
                storage_path.mkdir(parents=True, exist_ok=True)
                _QDRANT_CLIENT = QdrantClient(path=str(storage_path), check_compatibility=False)
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
        try:
            _ = list(self.dense_model.embed(["warmup"]))
        except Exception:
            pass

    def init_collection(self, collection_name: Optional[str] = None, recreate: bool = False):
        """Create collection with dual named vector spaces and payload indexes."""
        col = collection_name or self.collection_name
        exists = self.client.collection_exists(col)
        if exists and recreate:
            self.client.delete_collection(col)
            exists = False

        if not exists:
            self.client.create_collection(
                collection_name=col,
                vectors_config={
                    "dense": models.VectorParams(size=384, distance=models.Distance.COSINE)
                },
                sparse_vectors_config={
                    "bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)
                }
            )
            # Create payload indexes for metadata filtering
            for field in [
                "doc_category",
                "circular_name",
                "parent_id",
                "document_id",
                "document_status",
                "effective_date",
            ]:
                self.client.create_payload_index(
                    collection_name=col,
                    field_name=field,
                    field_schema=models.PayloadSchemaType.KEYWORD
                )
            print(f"[SUCCESS] Initialized hybrid collection '{col}' with dual vector spaces.")

    def embed_texts(self, texts: List[str], batch_size: int = 64) -> Tuple[List[List[float]], List[models.SparseVector]]:
        """Generate dense and sparse embeddings on CPU."""
        dense_generator = self.dense_model.embed(texts, batch_size=batch_size)
        dense_vectors = [vec.tolist() for vec in dense_generator]

        sparse_generator = self.sparse_model.embed(texts, batch_size=batch_size)
        sparse_vectors = [
            models.SparseVector(indices=sv.indices.tolist(), values=sv.values.tolist())
            for sv in sparse_generator
        ]
        return dense_vectors, sparse_vectors

    def index_child_chunks(
        self,
        child_chunks: List[Dict[str, Any]],
        batch_size: int = 64,
        target_collection: Optional[str] = None
    ):
        """Idempotently index child chunks into specified or default hybrid collection."""
        col = target_collection or self.collection_name
        self.init_collection(collection_name=col, recreate=False)
        total_chunks = len(child_chunks)
        print(f"Indexing {total_chunks} child chunks into '{col}'...")

        for i in range(0, total_chunks, batch_size):
            batch = child_chunks[i:i + batch_size]
            texts = [c["text"] for c in batch]
            dense_vectors, sparse_vectors = self.embed_texts(texts, batch_size=batch_size)

            points = []
            for j, c in enumerate(batch):
                points.append(
                    models.PointStruct(
                        id=c["child_id"],
                        vector={
                            "dense": dense_vectors[j],
                            "bm25": sparse_vectors[j],
                        },
                        payload={
                            "child_id": c["child_id"],
                            "parent_id": c["parent_id"],
                            "document_id": c.get("document_id", ""),
                            "text": c["text"],
                            "circular_name": c.get("circular_name", ""),
                            "circular_number": c.get("circular_number", ""),
                            "doc_category": c.get("doc_category", ""),
                            "section_clause": c.get("section_clause", ""),
                            "source_location": c.get("source_location", ""),
                            "page_number": c.get("page_number"),
                            "document_status": c.get("document_status", "ACTIVE"),
                            "effective_date": c.get("effective_date", ""),
                            "document_version": c.get("document_version", "1.0"),
                            "has_table": c.get("has_table", False),
                            "child_index": c.get("child_index", 0),
                        }
                    )
                )

            self.client.upsert(collection_name=col, points=points)
            print(f"Upserted points {i + 1} to {min(i + batch_size, total_chunks)} / {total_chunks}")

        print(f"[SUCCESS] Completed indexing of {total_chunks} chunks into '{col}'.")

    def hybrid_search(
        self,
        query: str,
        doc_category: Optional[str] = None,
        document_status: Optional[str] = "ACTIVE",
        historical_mode: bool = False,
        top_k: int = 15
    ) -> List[Dict[str, Any]]:
        """
        Execute single-roundtrip hybrid prefetch via Qdrant's Reciprocal Rank Fusion (RRF),
        followed by child-to-parent resolution and deduplication.
        Applies strict status filtering: only ACTIVE regulations participate unless historical_mode=True.
        """
        dense_vec, sparse_vec = self.embed_texts([query])
        dense_q = dense_vec[0]
        sparse_q = sparse_vec[0]

        filter_conditions = []
        if doc_category:
            filter_conditions.append(
                models.FieldCondition(
                    key="doc_category",
                    match=models.MatchValue(value=doc_category)
                )
            )

        # Enforce regulatory status isolation
        if not historical_mode:
            filter_conditions.append(
                models.FieldCondition(
                    key="document_status",
                    match=models.MatchValue(value="ACTIVE")
                )
            )
        elif document_status:
            filter_conditions.append(
                models.FieldCondition(
                    key="document_status",
                    match=models.MatchValue(value=document_status)
                )
            )

        global _QDRANT_OFFLINE
        query_filter = models.Filter(must=filter_conditions) if filter_conditions else None

        results = None
        if not _QDRANT_OFFLINE:
            try:
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
            except Exception as e:
                _QDRANT_OFFLINE = True
                logger.warning(f"Qdrant query_points unavailable: {e}. Executing fallback search on parent docstore.")

        if results is None:
            stopwords = {
                'what', 'is', 'the', 'in', 'a', 'under', 'for', 'and', 'are', 'with', 'that',
                'from', 'this', 'these', 'those', 'have', 'has', 'had', 'how', 'many',
                'must', 'shall', 'does', 'can', 'may', 'who', 'whom', 'which', 'into',
                'about', 'all', 'any', 'not', 'out', 'between', 'allowed', 'requirements'
            }
            domain_weights = {
                'cooling-off': 15.0, 'cooling': 12.0, 'kfs': 15.0, 'v-cip': 18.0, 'vcip': 18.0,
                'cert-in': 18.0, 'dla': 12.0, 'dlas': 12.0, 'lsp': 10.0, 'dlg': 10.0, 'fldg': 10.0,
                'digital': 6.0, 'lending': 6.0, 'cyber': 6.0, 'resilience': 6.0, 'kyc': 8.0,
                'incident': 10.0, 'reporting': 6.0, 'encryption': 8.0, 'geo-tagging': 10.0,
                'contacts': 25.0, 'contact': 25.0, 'gallery': 25.0, 'photo': 20.0, 'camera': 15.0,
                'mobile': 12.0, 'phone': 12.0, 'phonebooks': 25.0, 'phonebook': 25.0,
                'media': 15.0, 'storage': 10.0, 'prohibition': 15.0, 'prohibited': 15.0,
                'updation': 10.0, 'periodic': 8.0, 'schedule': 6.0
            }
            synonyms = {
                'contacts': ['contact', 'phonebook', 'phonebooks', 'call'],
                'contact': ['contacts', 'phonebook', 'phonebooks'],
                'gallery': ['media', 'storage', 'file', 'photo', 'camera', 'pictures'],
                'photo': ['camera', 'media', 'storage', 'pictures', 'image'],
                'apps': ['app', 'applications', 'dla', 'dlas'],
                'dlas': ['dla', 'digital lending app'],
                'dla': ['dlas', 'digital lending app'],
                'cooling-off': ['look-up', 'cooling', 'tenor'],
                'kfs': ['key fact statement'],
                'v-cip': ['vcip', 'video customer identification process', 'video'],
                'cert-in': ['incident', 'cyber security incident']
            }

            q_lower = query.lower()
            is_dl_q = any(k in q_lower for k in ["digital lending", "dla", "dlas", "kfs", "cooling-off", "look-up", "fldg", "lsp"])
            is_kyc_q = any(k in q_lower for k in ["kyc", "v-cip", "vcip", "cdd", "politically exposed", "periodic updation", "re-kyc"])
            is_cyber_q = any(k in q_lower for k in ["cyber", "cert-in", "incident", "csirt", "soc", "ransomware", "iso 27001"])

            tokens = [t.lower() for t in re.findall(r"[a-zA-Z0-9\-_]+", query) if len(t) > 2 and t.lower() not in stopwords]
            scored_parents = []
            for p_id, p_doc in self.docstore.items():
                if not historical_mode:
                    if p_doc.get("document_status") in ["SUPERSEDED", "REPEALED"]:
                        continue
                elif document_status and p_doc.get("document_status") != document_status:
                    continue

                if doc_category and p_doc.get("doc_category") != doc_category:
                    continue

                doc_text = p_doc.get("text", "").lower()
                doc_clause = p_doc.get("section_clause", "").lower()
                cat = p_doc.get("doc_category", "").lower()

                score = 0.0
                if is_dl_q:
                    score += 150.0 if cat == "digital_lending" else -50.0
                elif is_kyc_q:
                    score += 150.0 if cat == "kyc" else -50.0
                elif is_cyber_q:
                    score += 150.0 if cat == "cyber_resilience" else -50.0

                for t in tokens:
                    w = domain_weights.get(t, 1.0)
                    variants = [t, t.rstrip('s')] + synonyms.get(t, [])
                    matched_variants = sum(min(doc_text.count(v), 3) for v in set(variants) if v in doc_text)
                    if matched_variants > 0:
                        score += matched_variants * w
                    if any(v in doc_clause for v in set(variants)):
                        score += 15.0 * w

                if score > 0:
                    doc_copy = dict(p_doc)
                    doc_copy["rrf_score"] = float(score)
                    scored_parents.append(doc_copy)

            scored_parents.sort(key=lambda x: x["rrf_score"], reverse=True)
            return scored_parents[:top_k]

        # Child-to-Parent resolution and deduplication
        deduped_parents: Dict[str, Dict[str, Any]] = {}
        matched_scores: Dict[str, float] = {}

        for hit in results.points:
            payload = hit.payload or {}
            parent_id = payload.get("parent_id")
            score = hit.score or 0.0

            if not parent_id:
                continue

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
                    deduped_parents[parent_id] = {
                        "parent_id": parent_id,
                        "document_id": payload.get("document_id", ""),
                        "text": payload.get("text", ""),
                        "circular_name": payload.get("circular_name", ""),
                        "circular_number": payload.get("circular_number", ""),
                        "doc_category": payload.get("doc_category", ""),
                        "section_clause": payload.get("section_clause", ""),
                        "source_location": payload.get("source_location", ""),
                        "page_number": payload.get("page_number"),
                        "document_status": payload.get("document_status", "ACTIVE"),
                        "effective_date": payload.get("effective_date", ""),
                        "has_table": payload.get("has_table", False),
                        "rrf_score": score
                    }

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
        document_status: Optional[str] = "ACTIVE",
        historical_mode: bool = False,
        prefetch_limit: Optional[int] = None,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Execute full two-tier hybrid retrieval:
        1. Dense + BM25 RRF prefetch with status & category filtering
        2. Child-to-Parent deduplication
        3. FlashRank Cross-Encoder reranking preserving all provenance metadata
        """
        limit = prefetch_limit or settings.RRF_PREFETCH_LIMIT
        deduped_parents = self.hybrid_search(
            query=query,
            doc_category=doc_category,
            document_status=document_status,
            historical_mode=historical_mode,
            top_k=limit
        )
        reranker = FlashRankReranker(top_k=top_k)
        return reranker.rerank(query=query, parent_documents=deduped_parents, top_k=top_k)
