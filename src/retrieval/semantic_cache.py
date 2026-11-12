"""Entity-Safe Semantic Caching Layer using Qdrant (cosine similarity >= 0.93 + Entity Guard)."""

import uuid
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient, models

from src.config import settings
from src.schemas import CitedResponse
from src.retrieval.vector_store import get_dense_model, get_qdrant_client


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

    def init_cache_collection(self, recreate: bool = False):
        """Initialize the semantic cache collection with 384-d cosine vectors."""
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
            print(f"[SUCCESS] Initialized semantic cache collection '{self.collection_name}'.")

    def check_cache(
        self,
        question: str,
        query_type: str = "compliance_qa",
        extracted_entities: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Dual-Condition Cache Lookup:
        1. Cosine similarity >= 0.93 on query embedding.
        2. V3.0 Entity Guard: exact match on query_type AND extracted regulatory entity set.
        """
        self.init_cache_collection(recreate=False)
        if not extracted_entities:
            extracted_entities = extract_regulatory_entities(question)
        extracted_set = set(e.strip().lower() for e in extracted_entities)

        # Generate dense query embedding
        vector = list(self.dense_model.embed([question]))[0].tolist()

        search_results = self.client.query_points(
            collection_name=self.collection_name,
            query=vector,
            limit=1,
            score_threshold=self.similarity_threshold,
            with_payload=True,
        )

        points = search_results.points
        if not points:
            return None

        hit = points[0]
        payload = hit.payload or {}

        # V3.0 Entity Guard: prevent collision between queries differing by regulatory modifier
        cached_query_type = payload.get("query_type", "")
        cached_entities = set(e.strip().lower() for e in payload.get("extracted_entities", []))

        if cached_query_type != query_type:
            return None

        if cached_entities != extracted_set:
            return None

        # Verified cache hit: deserialize response
        cached_response = payload.get("response")
        return {
            "response": cached_response,
            "cache_hit": True,
            "similarity_score": hit.score,
            "matched_question": payload.get("question"),
        }

    def store_cache(
        self,
        question: str,
        query_type: str,
        extracted_entities: List[str],
        response: CitedResponse,
    ) -> bool:
        """
        Verified Write-Through Policy:
        Write to cache only when risk_level != 'UNVERIFIED' and at least one citation is verified.
        """
        if response.risk_level == "UNVERIFIED":
            return False

        has_verified_citation = any(c.verified for c in response.citations)
        if not has_verified_citation and response.citations:
            return False

        if not extracted_entities:
            extracted_entities = extract_regulatory_entities(question)

        self.init_cache_collection(recreate=False)
        vector = list(self.dense_model.embed([question]))[0].tolist()
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"cache_{question.strip().lower()}"))

        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "question": question,
                        "query_type": query_type,
                        "extracted_entities": extracted_entities,
                        "response": response.model_dump(),
                    }
                )
            ]
        )
        return True
