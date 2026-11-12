"""Entity-Safe Semantic Caching Layer using Qdrant (cosine similarity >= 0.93 + Entity Guard)."""

import uuid
import logging
from typing import List, Dict, Any, Optional, Set
import numpy as np
from qdrant_client import QdrantClient, models

from src.config import settings
from src.schemas import CitedResponse
from src.retrieval.vector_store import get_dense_model, get_qdrant_client

logger = logging.getLogger(__name__)

REGULATORY_KEY_ENTITIES = [
    "high-risk", "medium-risk", "low-risk", "high risk", "low risk", "medium risk",
    "v-cip", "vcip", "cooling-off", "look-up", "kfs", "dla", "lsp",
    "cert-in", "cert in", "c-kyc", "ckyc", "ovd", "aadhaar", "pan", "lei",
    "biometric", "foreclosure", "escrow", "direct disbursal"
]


def extract_regulatory_entities(text: str) -> List[str]:
    """Extract recognized regulatory entities/risk modifiers from query text."""
    text_lower = text.lower()
    return [ent for ent in REGULATORY_KEY_ENTITIES if ent in text_lower]


class EntityScopedSemanticCache:
    """
    Hardened Entity-Safe Semantic Cache.
    Guarantees zero cross-entity collisions (e.g. High-Risk vs Low-Risk KYC frequencies)
    and strictly enforces verified write-through policies.
    """

    def __init__(
        self,
        collection_name: Optional[str] = None,
        client: Optional[QdrantClient] = None,
        similarity_threshold: Optional[float] = None,
    ):
        self.collection_name = collection_name or settings.CACHE_COLLECTION
        self.client = client or get_qdrant_client()
        self.dense_model = get_dense_model()
        self.similarity_threshold = similarity_threshold or settings.SEMANTIC_CACHE_THRESHOLD
        self._memory_cache: Dict[str, Dict[str, Any]] = {}
        self._collection_initialized: bool = False
        self._qdrant_disabled: bool = False
        self._hits: int = 0
        self._misses: int = 0
        self._writes: int = 0
        self._rejections: int = 0

    def init_cache_collection(self, recreate: bool = False):
        """Initialize the semantic cache collection with 384-d cosine vectors."""
        if self._collection_initialized and not recreate:
            return

        if recreate:
            self._memory_cache.clear()
            self._collection_initialized = False

        self._collection_initialized = True

        if self._qdrant_disabled and not recreate:
            return

        try:
            exists = self.client.collection_exists(self.collection_name)
            if exists and recreate:
                self.client.delete_collection(self.collection_name)
                exists = False

            if not exists:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=384,
                        distance=models.Distance.COSINE
                    )
                )
        except Exception as e:
            self._qdrant_disabled = True
            logger.warning(f"Could not connect to Qdrant for semantic cache collection '{self.collection_name}': {e}. Using resilient in-memory cache.")

    def check_cache(
        self,
        question: str,
        query_type: str = "compliance_qa",
        extracted_entities: Optional[List[str]] = None,
        category_filter: Optional[str] = None,
        entity_type: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Dual-Condition Cache Lookup:
        1. Cosine similarity >= threshold on dense query embedding.
        2. Entity Guard: exact match on query_type AND extracted regulatory entity set.
        """
        self.init_cache_collection(recreate=False)
        if not extracted_entities:
            extracted_entities = extract_regulatory_entities(question)
        extracted_set = set(e.strip().lower() for e in extracted_entities)

        # Generate query embedding vector
        vector = list(self.dense_model.embed([question]))[0].tolist()
        vector_np = np.array(vector, dtype=np.float32)
        norm_v = np.linalg.norm(vector_np)
        if norm_v > 0:
            vector_np = vector_np / norm_v

        # Try Qdrant search first if not disabled
        if not self._qdrant_disabled:
            try:
                search_results = self.client.query_points(
                    collection_name=self.collection_name,
                    query=vector,
                    limit=1,
                    score_threshold=self.similarity_threshold,
                    with_payload=True,
                )
                points = search_results.points
                if points:
                    hit = points[0]
                    payload = hit.payload or {}
                    cached_query_type = payload.get("query_type", "")
                    cached_entities = set(e.strip().lower() for e in payload.get("extracted_entities", []))

                    # Entity Guard check
                    if cached_query_type == query_type and cached_entities == extracted_set:
                        cached_response = payload.get("response")
                        self._hits += 1
                        return {
                            "response": cached_response,
                            "cache_hit": True,
                            "similarity_score": hit.score,
                            "matched_question": payload.get("question"),
                        }
            except Exception as e:
                self._qdrant_disabled = True
                logger.debug(f"Qdrant cache lookup unavailable: {e}. Switching to in-memory cache.")

        # Fallback in-memory search
        best_hit = None
        best_score = -1.0

        for key, entry in self._memory_cache.items():
            entry_qtype = entry.get("query_type", "")
            entry_entities = entry.get("extracted_entities_set", set())

            # Entity guard
            if entry_qtype != query_type or entry_entities != extracted_set:
                continue

            entry_vec = entry.get("vector_norm")
            if entry_vec is not None:
                sim = float(np.dot(vector_np, entry_vec))
                if sim >= self.similarity_threshold and sim > best_score:
                    best_score = sim
                    best_hit = entry

        if best_hit:
            self._hits += 1
            return {
                "response": best_hit["response"],
                "cache_hit": True,
                "similarity_score": best_score,
                "matched_question": best_hit["question"],
            }

        self._misses += 1
        return None

    def store_cache(
        self,
        question: str,
        query_type: str,
        extracted_entities: List[str],
        response: CitedResponse,
        category_filter: Optional[str] = None,
        entity_type: Optional[str] = None,
        corpus_version: str = "3.0",
    ) -> bool:
        """
        Verified Write-Through Policy:
        Strict eligibility gates:
        1. risk_level != 'UNVERIFIED'
        2. len(citations) > 0 (Never cache zero-citation responses)
        3. all(c.verified for c in response.citations) (Zero unverified citations allowed)
        """
        # Eligibility Gate 1: Rejection of unverified responses
        if response.risk_level == "UNVERIFIED":
            self._rejections += 1
            return False

        # Eligibility Gate 2: Rejection of empty citation responses
        if not response.citations or len(response.citations) == 0:
            self._rejections += 1
            return False

        # Eligibility Gate 3: All citations must be strictly verified
        if not all(c.verified for c in response.citations):
            self._rejections += 1
            return False

        if not extracted_entities:
            extracted_entities = extract_regulatory_entities(question)

        extracted_set = set(e.strip().lower() for e in extracted_entities)

        self.init_cache_collection(recreate=False)
        vector = list(self.dense_model.embed([question]))[0].tolist()
        vector_np = np.array(vector, dtype=np.float32)
        norm_v = np.linalg.norm(vector_np)
        if norm_v > 0:
            vector_np = vector_np / norm_v

        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"cache_{question.strip().lower()}"))

        payload = {
            "question": question,
            "query_type": query_type,
            "extracted_entities": list(extracted_entities),
            "category_filter": category_filter,
            "entity_type": entity_type,
            "corpus_version": corpus_version,
            "response": response.model_dump(),
        }

        # Store in in-memory cache for fast sub-millisecond retrieval and offline fallback
        self._memory_cache[point_id] = {
            "question": question,
            "query_type": query_type,
            "extracted_entities_set": extracted_set,
            "vector_norm": vector_np,
            "category_filter": category_filter,
            "entity_type": entity_type,
            "corpus_version": corpus_version,
            "response": response.model_dump(),
        }

        # Upsert into Qdrant if available
        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    models.PointStruct(
                        id=point_id,
                        vector=vector,
                        payload=payload
                    )
                ]
            )
        except Exception as e:
            logger.debug(f"Could not upsert into Qdrant cache collection: {e}. Memory cache updated.")

        self._writes += 1
        return True

    def get_telemetry(self) -> Dict[str, Any]:
        """Return cache hit ratio and operational telemetry."""
        total_lookups = self._hits + self._misses
        hit_ratio = (self._hits / total_lookups) if total_lookups > 0 else 0.0
        return {
            "hits": self._hits,
            "misses": self._misses,
            "writes": self._writes,
            "rejections": self._rejections,
            "total_lookups": total_lookups,
            "hit_ratio": round(hit_ratio, 4),
            "cached_entries": len(self._memory_cache),
        }

    def clear_cache(self):
        """Purge all cached entries from memory and vector store."""
        self._memory_cache.clear()
        try:
            self.init_cache_collection(recreate=True)
        except Exception:
            pass
