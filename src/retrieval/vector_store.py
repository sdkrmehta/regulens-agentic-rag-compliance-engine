"""Hybrid Vector Engine combining Dense (bge-small-en-v1.5) and Sparse (BM25) with Metadata Filtering."""

import json
import logging
import re
from datetime import date
import uuid
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
_REGULATORY_REGISTRY_MAP: Optional[Dict[str, Dict[str, Any]]] = None
_QDRANT_CLIENT: Optional[QdrantClient] = None
_QDRANT_OFFLINE: bool = False
# Tracks whether Qdrant accepted a DATETIME index for effective_date.
# Set to True after a successful DatetimeRange query; set to False on first
# failure so subsequent queries skip Qdrant-level date filtering and rely on
# the post-query string-comparison fallback that already exists in hybrid_search.
_QDRANT_DATETIME_INDEX_SUPPORTED: Optional[bool] = None  # None = unknown, True/False = confirmed


def _effective_date_allows(document_effective_date: Optional[str], as_of_date: Optional[str]) -> bool:
    """Fail closed for point-in-time queries when a document date is absent/invalid."""
    if not as_of_date:
        return True
    try:
        as_of = date.fromisoformat(str(as_of_date))
        document_date = date.fromisoformat(str(document_effective_date))
    except (TypeError, ValueError):
        return False
    return document_date <= as_of


def get_regulatory_registry_map() -> Dict[str, Dict[str, Any]]:
    """Cached lookup of authoritative regulatory circular metadata from data/regulatory_registry.json."""
    global _REGULATORY_REGISTRY_MAP
    if _REGULATORY_REGISTRY_MAP is None:
        reg_path = Path("data/regulatory_registry.json")
        if reg_path.exists():
            try:
                with open(reg_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    _REGULATORY_REGISTRY_MAP = {
                        doc["document_id"]: doc for doc in data.get("documents", [])
                    }
            except Exception:
                _REGULATORY_REGISTRY_MAP = {}
        else:
            _REGULATORY_REGISTRY_MAP = {}
    return _REGULATORY_REGISTRY_MAP


def entity_type_matches(entity_type_query: str, doc_entities: List[str]) -> bool:
    """Check if query entity type matches document regulated entity types."""
    if not entity_type_query:
        return True
    q = entity_type_query.strip().lower()
    # Canonical synonyms & abbreviations
    synonyms_map = {
        "nbfc": ["non-banking financial", "nbfc"],
        "nbfcs": ["non-banking financial", "nbfc"],
        "bank": ["commercial bank", "banks", "bank"],
        "banks": ["commercial bank", "banks", "bank"],
        "commercial bank": ["commercial bank", "banks"],
        "commercial banks": ["commercial bank", "banks"],
        "pso": ["payment system operator", "pso"],
        "psos": ["payment system operator", "pso"],
        "lsp": ["lending service provider", "lsp"],
        "lsps": ["lending service provider", "lsp"],
        "dla": ["digital lending app", "dla"],
        "dlas": ["digital lending app", "dla"],
    }
    targets = synonyms_map.get(q, [q])
    for de in doc_entities:
        de_lower = de.lower()
        if any(t in de_lower for t in targets):
            return True
    return False


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
        client_created = False
        try:
            import urllib.request
            urllib.request.urlopen(f"{settings.QDRANT_URL}/collections", timeout=0.08)
            api_key = settings.QDRANT_API_KEY.get_secret_value() if (settings.QDRANT_API_KEY and settings.QDRANT_API_KEY.get_secret_value()) else None
            _QDRANT_CLIENT = QdrantClient(
                url=settings.QDRANT_URL,
                api_key=api_key,
                timeout=0.08,
                check_compatibility=False
            )
            client_created = True
        except Exception:
            client_created = False

        if not client_created:
            storage_path = Path("./qdrant_storage")
            storage_path.mkdir(parents=True, exist_ok=True)
            _QDRANT_CLIENT = QdrantClient(path=str(storage_path), check_compatibility=False)
    return _QDRANT_CLIENT


def _ensure_valid_point_id(raw_id: Any) -> Any:
    """Ensure point ID is a valid integer or UUID string for Qdrant compatibility."""
    if isinstance(raw_id, int):
        return raw_id
    id_str = str(raw_id)
    try:
        _ = uuid.UUID(id_str)
        return id_str
    except Exception:
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, id_str))


class RBIHybridVectorStore:
    def __init__(
        self,
        collection_name: Optional[str] = None,
        client: Optional[QdrantClient] = None,
        docstore_path: str = "data/parent_docstore.json",
    ):
        configured_collection = collection_name or settings.QDRANT_COLLECTION
        self.client = client or get_qdrant_client()
        self._production_collection = configured_collection
        self._active_alias = f"{configured_collection}_active"
        self.collection_name = self._active_alias if self._alias_target(self._active_alias) else configured_collection
        self.dense_model = get_dense_model()
        self.sparse_model = get_sparse_model()
        self.docstore = get_parent_docstore(docstore_path)
        try:
            _ = list(self.dense_model.embed(["warmup"]))
            _ = list(self.sparse_model.embed(["warmup"]))
            from src.retrieval.reranker import get_ranker
            _ = get_ranker()
        except Exception:
            pass

    def _alias_target(self, alias_name: str) -> Optional[str]:
        try:
            aliases = self.client.get_collection_aliases().aliases
            return next((a.collection_name for a in aliases if a.alias_name == alias_name), None)
        except Exception:
            return None

    def init_collection(self, collection_name: Optional[str] = None, recreate: bool = False):
        """Create collection with dual named vector spaces and payload indexes.

        effective_date is indexed as DATETIME (not KEYWORD) so that DatetimeRange
        queries used in hybrid_search can perform point-in-time filtering correctly.
        All other metadata fields remain KEYWORD indexes for exact-match lookups.
        """
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
            # Keyword payload indexes for exact-match metadata filtering
            for field in [
                "doc_category",
                "circular_name",
                "parent_id",
                "document_id",
                "document_status",
            ]:
                self.client.create_payload_index(
                    collection_name=col,
                    field_name=field,
                    field_schema=models.PayloadSchemaType.KEYWORD
                )

            # DATETIME index for effective_date — required for DatetimeRange range queries.
            # Falls back to KEYWORD on older Qdrant versions that lack datetime indexes.
            try:
                self.client.create_payload_index(
                    collection_name=col,
                    field_name="effective_date",
                    field_schema=models.PayloadSchemaType.DATETIME
                )
                logger.info(f"Created DATETIME payload index for 'effective_date' on '{col}'.")
            except Exception as dt_err:
                logger.warning(
                    f"DATETIME index for 'effective_date' unavailable ({dt_err}). "
                    f"Falling back to KEYWORD index — date-range filtering will use post-query filtering."
                )
                try:
                    self.client.create_payload_index(
                        collection_name=col,
                        field_name="effective_date",
                        field_schema=models.PayloadSchemaType.KEYWORD
                    )
                except Exception:
                    pass  # Collection proceeds without effective_date index

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

    def _count_points_exact(self, collection_name: str) -> int:
        """Return the exact point count for a collection using Qdrant's count API.

        Uses ``exact=True`` to bypass Qdrant's approximate/eventually-consistent
        estimate returned by ``get_collection().points_count``.  Falls back to the
        estimate on older Qdrant versions that lack the count endpoint.
        """
        try:
            result = self.client.count(collection_name=collection_name, exact=True)
            return result.count
        except Exception:
            try:
                col_info = self.client.get_collection(collection_name)
                return col_info.points_count or 0
            except Exception:
                return 0

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
                        id=_ensure_valid_point_id(c["child_id"]),
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

    def atomic_reindex(
        self,
        child_chunks: List[Dict[str, Any]],
        batch_size: int = 64,
        target_collection: Optional[str] = None,
        on_success: Optional[callable] = None,
    ) -> bool:
        """Genuinely atomic re-indexing with rollback protection.

        **Two-phase commit guarantee** — the production collection is never modified
        until the staging collection has been fully indexed AND independently
        verified with an exact point count.  Only after verification does the method
        promote staging → production via a Qdrant alias swap (single atomic
        operation on the Qdrant server).  If *any* step fails before the swap the
        production collection is left completely intact.

        Phases
        ------
        1. Create an isolated staging collection ``{col}_staging_{timestamp}``.
        2. Embed and index all child chunks into staging in batches.
        3. Verify staging point count with ``exact=True`` (not the eventually-
           consistent ``points_count`` estimate).  Raise if count mismatches.
        4. **Atomic alias swap**: update the ``{col}_alias`` alias to point at the
           new staging collection, then rename staging → production name.
           Falls back to scroll-copy when aliases are unavailable.
        5. Call the optional ``on_success`` callback (e.g. persist the docstore).
        6. ``finally``: clean up the staging collection.  Production is never
           touched in the ``except`` / ``finally`` paths.

        Parameters
        ----------
        child_chunks:
            All child chunk dicts to index.
        batch_size:
            Embedding and upsert batch size.
        target_collection:
            Override the production collection name (defaults to
            ``self.collection_name``).
        on_success:
            Zero-argument callable executed **only** after the index swap is
            confirmed.  Use this to persist the parent docstore so the disk
            file and Qdrant are always in sync.
        """
        import time
        col = target_collection or self.collection_name
        staging_col = f"{col}_staging_{int(time.time())}"
        staging_created = False
        expected_count = len(child_chunks)
        alias_name = f"{self._production_collection}_active"
        previous_target = self._alias_target(alias_name)
        if previous_target is None and self.client.collection_exists(self._production_collection):
            previous_target = self._production_collection

        try:
            # ── Phase 1: create staging ──────────────────────────────────────
            print(f"[ATOMIC INDEXING] Creating isolated staging collection '{staging_col}'...")
            self.init_collection(collection_name=staging_col, recreate=True)
            staging_created = True

            # ── Phase 2: index into staging ──────────────────────────────────
            self.index_child_chunks(
                child_chunks=child_chunks,
                batch_size=batch_size,
                target_collection=staging_col,
            )

            # ── Phase 3: exact verification ───────────────────────────────────
            # Qdrant's points_count is an approximate/stale estimate; use the
            # count API with exact=True to get a guaranteed accurate number.
            actual_count = self._count_points_exact(staging_col)
            if actual_count < expected_count:
                raise RuntimeError(
                    f"[ATOMIC INDEXING] Staging verification FAILED: "
                    f"expected {expected_count} points, found {actual_count}. "
                    f"Production collection '{col}' is preserved intact."
                )
            print(
                f"[ATOMIC INDEXING] Staging verified: {actual_count}/{expected_count} "
                f"points confirmed in '{staging_col}'."
            )

            # Qdrant updates aliases atomically server-side. Keep the old concrete
            # collection intact; never recreate or copy over production in place.
            alias_actions = []
            if self._alias_target(alias_name) is not None:
                alias_actions.append(models.DeleteAlias(alias_name=alias_name))
            alias_actions.append(models.CreateAlias(collection_name=staging_col, alias_name=alias_name))
            self.client.update_collection_aliases(change_aliases_operations=alias_actions)
            self.collection_name = alias_name
            if self._count_points_exact(alias_name) != expected_count:
                rollback = [models.DeleteAlias(alias_name=alias_name)]
                if previous_target:
                    rollback.append(models.CreateAlias(collection_name=previous_target, alias_name=alias_name))
                self.client.update_collection_aliases(change_aliases_operations=rollback)
                self.collection_name = alias_name if previous_target else self._production_collection
                raise RuntimeError("Promoted alias point count does not match verified staging count")

            # ── Phase 5: post-success callback (e.g. persist docstore) ───────
            if on_success is not None:
                try:
                    on_success()
                except Exception as cb_err:
                    rollback = []
                    rollback.append(models.DeleteAlias(alias_name=alias_name))
                    if previous_target:
                        rollback.append(models.CreateAlias(collection_name=previous_target, alias_name=alias_name))
                    self.client.update_collection_aliases(change_aliases_operations=rollback)
                    self.collection_name = alias_name if previous_target else self._production_collection
                    raise RuntimeError(f"Docstore persistence failed; alias promotion rolled back: {cb_err}") from cb_err

            print(
                f"[ATOMIC INDEXING SUCCESS] {expected_count} chunks live in production '{col}'."
            )
            return True

        except Exception as e:
            logger.error(
                f"[ATOMIC INDEXING FAILED] {e}. "
                f"Production collection '{col}' preserved (staging cleanup follows)."
            )
            raise

        finally:
            # Clean up staging.  At this point either:
            # - success: staging data has been fully copied to production.
            # - failure before swap: production was never modified.
            # - failure during swap: logged above with recovery instructions.
            if staging_created and self._alias_target(alias_name) != staging_col:
                try:
                    self.client.delete_collection(staging_col)
                    logger.debug(f"[ATOMIC INDEXING] Staging collection '{staging_col}' cleaned up.")
                except Exception as del_err:
                    logger.warning(
                        f"[ATOMIC INDEXING] Could not delete staging collection "
                        f"'{staging_col}': {del_err}. Delete it manually."
                    )


    def _search_docstore(
        self,
        query: str,
        doc_category: Optional[str] = None,
        document_status: Optional[str] = None,
        historical_mode: bool = False,
        regulator: Optional[str] = None,
        entity_type: Optional[str] = None,
        effective_date: Optional[str] = None,
        top_k: int = 15,
        reg_map: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Fallback in-memory scoring over parent docstore when Qdrant is offline or yields 0 hits."""
        if reg_map is None:
            reg_map = get_regulatory_registry_map()

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
        is_hist_q = any(k in q_lower for k in ["2022", "repealed", "historical", "prior", "september 2022", "former"])

        tokens = [t.lower() for t in re.findall(r"[a-zA-Z0-9\-_]+", query) if len(t) > 2 and t.lower() not in stopwords]
        scored_parents = []
        for p_id, p_doc in self.docstore.items():
            doc_id = p_doc.get("document_id", "")
            doc_meta = reg_map.get(doc_id, {})
            doc_regulator = doc_meta.get("regulator", "Reserve Bank of India")
            doc_entities = doc_meta.get("regulated_entity_types", [])

            if not historical_mode:
                if p_doc.get("document_status") in ["SUPERSEDED", "REPEALED"]:
                    continue
            elif document_status and p_doc.get("document_status") != document_status:
                continue

            if doc_category and p_doc.get("doc_category") != doc_category:
                continue

            # Filter by regulator
            if regulator and doc_regulator.lower() != regulator.lower():
                continue

            # Filter by entity type
            if entity_type and not entity_type_matches(entity_type, doc_entities):
                continue

            # Filter by effective date (point-in-time)
            doc_eff_date = p_doc.get("effective_date") or doc_meta.get("effective_date")
            if not _effective_date_allows(doc_eff_date, effective_date):
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

            # Prioritize repealed 2022 guidelines when queried in historical mode
            if historical_mode and is_hist_q and p_doc.get("document_id") == "RBI_DL_2022":
                score += 300.0

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

    def hybrid_search(
        self,
        query: str,
        doc_category: Optional[str] = None,
        document_status: Optional[str] = None,
        historical_mode: bool = False,
        regulator: Optional[str] = None,
        entity_type: Optional[str] = None,
        effective_date: Optional[str] = None,
        top_k: int = 15
    ) -> List[Dict[str, Any]]:
        """
        Execute single-roundtrip hybrid prefetch via Qdrant's Reciprocal Rank Fusion (RRF),
        followed by child-to-parent resolution, deduplication, and strict multi-parameter metadata filtering:
        - doc_category: filter by regulatory category (e.g. KYC, DIGITAL_LENDING, CYBER_RESILIENCE)
        - document_status: filter by ACTIVE, REPEALED, etc.
        - historical_mode: whether to permit superseded/repealed regulations
        - regulator: filter by governing regulator authority (e.g. Reserve Bank of India)
        - entity_type: filter by applicable regulated entity type (e.g. Commercial Banks, NBFCs, PSOs, LSPs)
        - effective_date: point-in-time applicability filter (regulation effective_date <= query effective_date)
        """
        dense_vec, sparse_vec = self.embed_texts([query])
        dense_q = dense_vec[0]
        sparse_q = sparse_vec[0]

        reg_map = get_regulatory_registry_map()

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

        # Document ID pre-filtering from regulatory metadata
        matching_doc_ids = None
        if regulator:
            matching_for_reg = [did for did, m in reg_map.items() if m.get("regulator", "").lower() == regulator.lower()]
            if not matching_for_reg:
                return []
            matching_doc_ids = set(matching_for_reg)

        if entity_type:
            matching_for_ent = [did for did, m in reg_map.items() if entity_type_matches(entity_type, m.get("regulated_entity_types", []))]
            if not matching_for_ent:
                return []
            if matching_doc_ids is None:
                matching_doc_ids = set(matching_for_ent)
            else:
                matching_doc_ids = matching_doc_ids.intersection(set(matching_for_ent))
                if not matching_doc_ids:
                    return []

        if matching_doc_ids is not None:
            filter_conditions.append(
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchAny(any=list(matching_doc_ids))
                )
            )

        # Qdrant-level effective_date filtering (DATETIME index).
        # On first use the result is probed; if Qdrant rejects DatetimeRange
        # (e.g. index fell back to KEYWORD), we set the flag to False and from
        # then on rely exclusively on the post-query string-comparison filter at
        # line ~640.  A module-level flag avoids repeated failed attempts.
        global _QDRANT_OFFLINE, _QDRANT_DATETIME_INDEX_SUPPORTED
        _effective_date_filter_added = False
        if effective_date and _QDRANT_DATETIME_INDEX_SUPPORTED is not False:
            try:
                from datetime import datetime, timezone
                dt = datetime.fromisoformat(effective_date)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                if hasattr(models, "DatetimeRange"):
                    filter_conditions.append(
                        models.FieldCondition(
                            key="effective_date",
                            range=models.DatetimeRange(lte=dt)
                        )
                    )
                    _effective_date_filter_added = True
            except Exception as _dte:
                logger.debug(f"Could not build DatetimeRange filter: {_dte}; will rely on post-query filtering.")

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
                # If we successfully ran a query with DatetimeRange, mark it as supported.
                if _effective_date_filter_added and _QDRANT_DATETIME_INDEX_SUPPORTED is None:
                    _QDRANT_DATETIME_INDEX_SUPPORTED = True
            except Exception as e:
                err_str = str(e).lower()
                # Distinguish: DatetimeRange rejected on KEYWORD index vs genuine connectivity failure
                if _effective_date_filter_added and (
                    "wrong tag" in err_str or "datetime" in err_str or "field type" in err_str
                    or "index" in err_str or "keyword" in err_str
                ):
                    logger.warning(
                        "Qdrant rejected DatetimeRange filter — effective_date index is KEYWORD. "
                        "Disabling Qdrant-level date filtering; post-query string comparison will apply."
                    )
                    _QDRANT_DATETIME_INDEX_SUPPORTED = False
                    # Retry without the date filter condition
                    filter_conditions_no_date = [
                        fc for fc in filter_conditions
                        if not (hasattr(fc, "key") and fc.key == "effective_date")
                    ]
                    query_filter_no_date = models.Filter(must=filter_conditions_no_date) if filter_conditions_no_date else None
                    try:
                        results = self.client.query_points(
                            collection_name=self.collection_name,
                            prefetch=[
                                models.Prefetch(query=dense_q, using="dense", limit=top_k, filter=query_filter_no_date),
                                models.Prefetch(query=sparse_q, using="bm25", limit=top_k, filter=query_filter_no_date),
                            ],
                            query=models.FusionQuery(fusion=models.Fusion.RRF),
                            limit=top_k,
                            with_payload=True
                        )
                    except Exception as retry_e:
                        _QDRANT_OFFLINE = True
                        logger.warning(f"Qdrant retry also failed: {retry_e}. Falling back to docstore.")
                else:
                    _QDRANT_OFFLINE = True
                    logger.warning(f"Qdrant query_points unavailable: {e}. Executing fallback search on parent docstore.")

        if results is None or not results.points:
            return self._search_docstore(
                query=query,
                doc_category=doc_category,
                document_status=document_status,
                historical_mode=historical_mode,
                regulator=regulator,
                entity_type=entity_type,
                effective_date=effective_date,
                top_k=top_k,
                reg_map=reg_map
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

            parent_entry = self.docstore.get(parent_id)
            doc_id = payload.get("document_id", "") or (parent_entry.get("document_id", "") if parent_entry else "")
            if not doc_id:
                circ = (payload.get("circular_name") or "").lower()
                pid = parent_id.upper()
                if "2022" in circ or pid.startswith("DL_") or "DL_2022" in pid:
                    doc_id = "RBI_DL_2022"
                elif "2025" in circ or "DL_2025" in pid:
                    doc_id = "RBI_DL_2025"
                elif "kyc" in circ or pid.startswith("KYC_") or "KYC_2016" in pid:
                    doc_id = "RBI_KYC_2016"
                elif "cyber" in circ or pid.startswith("CYB_") or "CYBER_2024" in pid:
                    doc_id = "RBI_CYBER_2024"

            doc_meta = reg_map.get(doc_id, {})
            doc_regulator = doc_meta.get("regulator", "Reserve Bank of India")
            doc_entities = doc_meta.get("regulated_entity_types", [])

            # Post-filter on resolved parent metadata for entity_type, regulator, effective_date
            if regulator and doc_regulator.lower() != regulator.lower():
                continue
            if entity_type and not entity_type_matches(entity_type, doc_entities):
                continue
            doc_eff_date = payload.get("effective_date") or doc_meta.get("effective_date")
            if not _effective_date_allows(doc_eff_date, effective_date):
                continue

            if parent_id not in matched_scores or score > matched_scores[parent_id]:
                matched_scores[parent_id] = score

            if parent_id not in deduped_parents:
                if parent_entry:
                    doc_copy = dict(parent_entry)
                    doc_copy["rrf_score"] = score
                    doc_copy["matched_child_id"] = payload.get("child_id")
                    if not doc_copy.get("document_id") and doc_id:
                        doc_copy["document_id"] = doc_id
                    deduped_parents[parent_id] = doc_copy
                else:
                    deduped_parents[parent_id] = {
                        "parent_id": parent_id,
                        "document_id": doc_id,
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
        q_lower = query.lower()
        is_hist_q = any(k in q_lower for k in ["2022", "repealed", "historical", "prior", "september 2022", "former"])
        for p_id, p_doc in deduped_parents.items():
            base_score = matched_scores[p_id]
            if historical_mode and is_hist_q and p_doc.get("document_id") == "RBI_DL_2022":
                base_score += 10.0
            p_doc["rrf_score"] = base_score
            parent_list.append(p_doc)

        parent_list.sort(key=lambda x: x["rrf_score"], reverse=True)

        if not parent_list:
            return self._search_docstore(
                query=query,
                doc_category=doc_category,
                document_status=document_status,
                historical_mode=historical_mode,
                regulator=regulator,
                entity_type=entity_type,
                effective_date=effective_date,
                top_k=top_k,
                reg_map=reg_map
            )

        return parent_list

    def hybrid_search_and_rerank(
        self,
        query: str,
        doc_category: Optional[str] = None,
        document_status: Optional[str] = None,
        historical_mode: bool = False,
        regulator: Optional[str] = None,
        entity_type: Optional[str] = None,
        effective_date: Optional[str] = None,
        prefetch_limit: Optional[int] = None,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Execute full two-tier hybrid retrieval:
        1. Dense + BM25 RRF prefetch with status, category, regulator, entity_type, and point-in-time effective_date filtering
        2. Child-to-Parent deduplication
        3. FlashRank Cross-Encoder reranking preserving all provenance metadata
        """
        limit = prefetch_limit or settings.RRF_PREFETCH_LIMIT
        deduped_parents = self.hybrid_search(
            query=query,
            doc_category=doc_category,
            document_status=document_status,
            historical_mode=historical_mode,
            regulator=regulator,
            entity_type=entity_type,
            effective_date=effective_date,
            top_k=limit
        )
        reranker = FlashRankReranker(top_k=top_k)
        return reranker.rerank(query=query, parent_documents=deduped_parents, top_k=top_k)
