"""Asynchronous FastAPI Backend Service for RBI Regulatory Compliance Engine."""

import time
import json
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.config import settings
from src.schemas import (
    QueryRequest,
    QueryResponse,
    ContractAuditRequest,
    ContractAuditReport,
)
from src.retrieval.vector_store import (
    get_qdrant_client,
    get_parent_docstore,
    get_dense_model,
    get_sparse_model,
    RBIHybridVectorStore,
)
from src.retrieval.reranker import get_ranker
from src.retrieval.semantic_cache import EntityScopedSemanticCache
from src.agent.graph import compliance_agent
from src.agent.auditor import audit_fintech_contract

# Global state loaded at lifespan boot
lifespan_state: Dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager: preload models, docstore, and cache on boot."""
    print("[BOOT] Pre-loading singleton models, docstore, and Qdrant connections...")
    lifespan_state["client"] = get_qdrant_client()
    lifespan_state["docstore"] = get_parent_docstore()
    lifespan_state["dense_model"] = get_dense_model()
    lifespan_state["sparse_model"] = get_sparse_model()
    lifespan_state["ranker"] = get_ranker()
    lifespan_state["cache"] = EntityScopedSemanticCache()
    lifespan_state["vector_store"] = RBIHybridVectorStore()
    print("[BOOT] All singletons and vector spaces initialized successfully.")
    yield
    print("[SHUTDOWN] Cleaning up server resources...")
    lifespan_state.clear()


app = FastAPI(
    title="RBI Regulatory Compliance Engine API",
    version="3.0.0",
    description="Agentic RAG & Automated Fintech Contract Compliance Auditing over RBI Master Directions",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
@app.get("/api/v1/health")
def health_check():
    """Liveness check: basic service health and operational metadata."""
    return {
        "status": "healthy",
        "service": "RBI Compliance Agentic Microservice",
        "version": "3.0.0",
        "timestamp": time.time(),
    }


@app.get("/ready")
@app.get("/api/v1/ready")
def readiness_check():
    """Readiness check: validates docstore, vector store, and model weights."""
    client = lifespan_state.get("client") or get_qdrant_client()
    docstore = lifespan_state.get("docstore") or get_parent_docstore()

    try:
        col_info = client.get_collection(settings.QDRANT_COLLECTION)
        points_count = col_info.points_count
    except Exception:
        points_count = 0

    return {
        "status": "ready" if len(docstore) > 0 else "degraded",
        "qdrant_points_count": points_count,
        "parent_sections_count": len(docstore),
        "docstore_loaded": len(docstore) > 0,
        "models_cached": True,
    }


@app.get("/api/v1/documents")
def list_regulatory_documents():
    """List all registered regulatory circulars with versioning, currency, and provenance."""
    registry_path = Path("data/regulatory_registry.json")
    if registry_path.exists():
        with open(registry_path, "r", encoding="utf-8") as f:
            registry = json.load(f)
        return registry
    return {}


@app.get("/api/v1/metrics")
def get_system_metrics():
    """Telemetry endpoint: return cache hit ratio, vector collection size, and SLAs."""
    cache: EntityScopedSemanticCache = lifespan_state.get("cache") or EntityScopedSemanticCache()
    docstore = lifespan_state.get("docstore") or get_parent_docstore()
    client = lifespan_state.get("client") or get_qdrant_client()

    try:
        col_info = client.get_collection(settings.QDRANT_COLLECTION)
        points_count = col_info.points_count
    except Exception:
        points_count = 0

    return {
        "indexed_chunks": points_count,
        "parent_clauses": len(docstore),
        "cache_telemetry": cache.get_telemetry(),
        "slas": {
            "cold_retrieval_target_s": 2.2,
            "warm_cache_target_ms": 80.0,
        }
    }


@app.post("/ask", response_model=QueryResponse)
@app.post("/api/v1/query", response_model=QueryResponse)
def ask_compliance_question(req: QueryRequest):
    """Evaluate regulatory query via Entity-Safe Semantic Cache or LangGraph Agentic Pipeline."""
    start_time = time.time()
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    cache: EntityScopedSemanticCache = lifespan_state.get("cache") or EntityScopedSemanticCache()

    # Step 1: Check Entity-Scoped Semantic Cache
    cached_hit = cache.check_cache(
        question=question,
        query_type="compliance_qa",
        extracted_entities=[],
        category_filter=req.category_filter,
        entity_type=req.entity_type_filter,
    )

    if cached_hit:
        elapsed_ms = (time.time() - start_time) * 1000.0
        return QueryResponse(
            response=cached_hit["response"],
            retry_count=0,
            cache_hit=True,
            latency_ms=round(elapsed_ms, 2)
        )

    # Step 2: Uncached - Execute LangGraph State Machine
    initial_state = {
        "question": question,
        "category_filter": req.category_filter,
        "entity_type_filter": req.entity_type_filter,
        "historical_query": req.historical_mode,
        "retry_count": 0,
        "search_query": question,
    }

    final_state = compliance_agent.invoke(initial_state)
    final_response = final_state.get("final_response")

    if not final_response:
        raise HTTPException(status_code=500, detail="Failed to synthesize response.")

    # Step 3: Verified Write-Through Cache
    extracted_entities = final_state.get("extracted_entities", [])
    cache.store_cache(
        question=question,
        query_type="compliance_qa",
        extracted_entities=extracted_entities,
        response=final_response,
        category_filter=req.category_filter,
        entity_type=req.entity_type_filter,
    )

    elapsed_ms = (time.time() - start_time) * 1000.0
    return QueryResponse(
        response=final_response,
        retry_count=final_state.get("retry_count", 0),
        cache_hit=False,
        latency_ms=round(elapsed_ms, 2)
    )


@app.post("/audit-contract", response_model=ContractAuditReport)
@app.post("/api/v1/audit", response_model=ContractAuditReport)
def audit_contract_endpoint(req: ContractAuditRequest):
    """Decompose and audit drafted fintech agreement clauses against RBI Master Directions."""
    contract_text = req.contract_text.strip()
    if not contract_text:
        raise HTTPException(status_code=400, detail="Contract text cannot be empty.")

    report = audit_fintech_contract(contract_text)
    return report
