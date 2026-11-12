"""LangGraph Node implementations for RBI Compliance Agent with Source-Specific Verification."""

import re
import time
from typing import Dict, Any, List, Optional

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq

from src.config import settings
from src.schemas import (
    RouterOutput,
    BatchGradeOutput,
    DocumentGrade,
    CitedResponse,
    CitationItem,
)
from src.agent.state import AgentState
from src.agent.prompts import (
    BANKING_GLOSSARY,
    ROUTER_SYSTEM_PROMPT,
    BATCH_GRADER_PROMPT,
    QUERY_REWRITER_PROMPT,
    GENERATOR_SYSTEM_PROMPT,
)
from src.agent.verifier import verify_citation_against_source
from src.retrieval.vector_store import RBIHybridVectorStore

# Shared vector store singleton
_VECTOR_STORE = None


def get_vector_store() -> RBIHybridVectorStore:
    global _VECTOR_STORE
    if _VECTOR_STORE is None:
        _VECTOR_STORE = RBIHybridVectorStore()
    return _VECTOR_STORE


def get_llm(model_name: str, temperature: float = 0.0, max_tokens: int = 1024) -> Optional[ChatGroq]:
    api_key = settings.GROQ_API_KEY.get_secret_value() if settings.GROQ_API_KEY else ""
    if not api_key or api_key.startswith("gsk_your_groq"):
        return None
    return ChatGroq(
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=api_key,
        request_timeout=4.0,
        max_retries=1,
    )


def sanitize_input_prompt(query: str) -> str:
    """Sanitize user query against obvious prompt injection and jailbreak attempts."""
    injection_patterns = [
        r"ignore\s+(?:all\s+)?(?:previous\s+)?(?:regulatory\s+|system\s+)?instructions?",
        r"system\s+prompt",
        r"you\s+are\s+now\s+an\s+unrestricted",
        r"disregard\s+(?:the\s+)?above",
        r"output\s+your\s+instructions",
    ]
    for pat in injection_patterns:
        if re.search(pat, query, re.IGNORECASE):
            return "Prompt injection pattern detected. Provide standard compliance inquiry."
    return query[:1500].strip()


def router_node(state: AgentState) -> Dict[str, Any]:
    """Classify user intent, extract acronyms/entities, and generate expanded query."""
    raw_question = state.get("question", "")
    question = sanitize_input_prompt(raw_question)
    llm = get_llm(settings.FAST_LLM_MODEL, temperature=0.0)

    # Detect explicit historical intent
    q_lower = question.lower()
    is_historical = any(k in q_lower for k in ["2022 guidelines", "old guidelines", "repealed", "historical", "prior to 2025", "earlier circular"])

    if llm:
        structured_llm = llm.with_structured_output(RouterOutput)
        messages = [
            SystemMessage(content=ROUTER_SYSTEM_PROMPT),
            HumanMessage(content=f"User Query: {question}\nGlossary Reference: {BANKING_GLOSSARY}"),
        ]
        try:
            output: RouterOutput = structured_llm.invoke(messages)
            return {
                "query_type": output.query_type,
                "extracted_entities": output.extracted_entities,
                "target_category": output.target_category,
                "historical_query": is_historical or output.historical_query,
                "search_query": output.search_query or question,
                "retry_count": state.get("retry_count", 0),
            }
        except Exception as e:
            print(f"[WARNING] Router LLM call failed: {e}. Using deterministic fallback.")

    # Rule-based fallback router
    out_of_scope_keywords = ["weather", "cricket", "movie", "recipe", "stock market tips", "personal tax advice"]
    if any(k in q_lower for k in out_of_scope_keywords):
        return {
            "query_type": "out_of_scope",
            "extracted_entities": [],
            "search_query": question,
            "historical_query": False,
            "retry_count": state.get("retry_count", 0),
        }

    audit_keywords = ["borrower agrees", "cooling-off period subject to", "loan agreement", "hereby consents", "foreclosure fee"]
    if any(k in q_lower for k in audit_keywords) and len(question.split()) > 15:
        return {
            "query_type": "contract_audit",
            "extracted_entities": ["Contract Clause"],
            "search_query": question,
            "historical_query": False,
            "retry_count": state.get("retry_count", 0),
        }

    # Extract common acronyms from glossary
    found_entities = [acronym for acronym in BANKING_GLOSSARY if re.search(r"\b" + re.escape(acronym) + r"\b", question, re.IGNORECASE)]

    return {
        "query_type": "compliance_qa",
        "extracted_entities": found_entities,
        "search_query": question,
        "historical_query": is_historical,
        "retry_count": state.get("retry_count", 0),
    }


def retrieve_node(state: AgentState) -> Dict[str, Any]:
    """Execute hybrid dense + sparse search and cross-encoder reranking."""
    search_query = state.get("search_query") or state.get("question", "")
    category_filter = state.get("category_filter") or state.get("target_category")
    historical_mode = state.get("historical_query", False)

    try:
        store = get_vector_store()
        parents = store.hybrid_search_and_rerank(
            query=search_query,
            doc_category=category_filter,
            historical_mode=historical_mode,
            prefetch_limit=settings.RRF_PREFETCH_LIMIT,
            top_k=settings.RERANK_TOP_K,
        )
    except Exception as e:
        print(f"[WARNING] Vector store retrieval failed: {e}. Falling back to empty parents.")
        parents = []

    return {"retrieved_parents": parents}


def grader_node(state: AgentState) -> Dict[str, Any]:
    """Single-call batch grader: evaluate relevance and authority of retrieved parents."""
    retrieved = state.get("retrieved_parents", [])
    if not retrieved:
        return {"graded_parents": []}

    question = state.get("question", "")
    llm = get_llm(settings.FAST_LLM_MODEL, temperature=0.0)

    if llm:
        structured_llm = llm.with_structured_output(BatchGradeOutput)
        docs_text = "\n\n".join([
            f"--- Parent Document ID: {p['parent_id']} | Status: {p.get('document_status', 'ACTIVE')} ---\n{p['text']}"
            for p in retrieved
        ])
        messages = [
            SystemMessage(content=BATCH_GRADER_PROMPT),
            HumanMessage(content=f"Query: {question}\n\nCandidate Documents:\n{docs_text}"),
        ]
        try:
            output: BatchGradeOutput = structured_llm.invoke(messages)
            relevant_ids = {g.parent_id for g in output.grades if g.is_relevant and g.is_authoritative}
            graded = [p for p in retrieved if p["parent_id"] in relevant_ids]
            return {"graded_parents": graded}
        except Exception as e:
            print(f"[WARNING] Grader LLM call failed: {e}. Using deterministic heuristic.")

    # Rule-based relevance check
    q_words = set(re.findall(r"\w+", question.lower())) - {"what", "is", "the", "under", "for", "and", "in", "to", "how", "are", "of"}
    graded = []
    for p in retrieved:
        p_text = p.get("text", "").lower()
        if "cryptocurrency mining" in question.lower() or "crypto mining" in question.lower():
            continue
        overlap = sum(1 for w in q_words if w in p_text)
        if overlap >= 2 or p.get("rrf_score", 0.0) > 0.015:
            graded.append(p)

    return {"graded_parents": graded}


def rewriter_node(state: AgentState) -> Dict[str, Any]:
    """Acronym-aware query rewriter node preserving statutory scope and intent."""
    question = state.get("question", "")
    search_query = state.get("search_query", question)
    current_retries = state.get("retry_count", 0)
    llm = get_llm(settings.FAST_LLM_MODEL, temperature=0.0)

    new_query = search_query
    if llm:
        messages = [
            SystemMessage(content=QUERY_REWRITER_PROMPT),
            HumanMessage(
                content=f"Original Query: {question}\nPrevious Failed Search: {search_query}\nGlossary: {BANKING_GLOSSARY}"
            ),
        ]
        try:
            resp = llm.invoke(messages)
            new_query = resp.content.strip()
        except Exception as e:
            print(f"[WARNING] Rewriter LLM call failed: {e}")

    # Fallback keyword expansion
    if new_query == search_query:
        expanded_terms = []
        for acronym, full_text in BANKING_GLOSSARY.items():
            if re.search(r"\b" + re.escape(acronym) + r"\b", question, re.IGNORECASE):
                expanded_terms.append(full_text)
        if expanded_terms:
            new_query = f"{question} {' '.join(expanded_terms[:2])}"

    return {
        "search_query": new_query,
        "retry_count": current_retries + 1,
    }


def generator_node(state: AgentState) -> Dict[str, Any]:
    """
    Flagship Cited Answer Generator paired with source-specific quote verification.
    Never fabricates fallback citations or passes unverified answers as verified.
    """
    question = state.get("question", "")
    graded_parents = state.get("graded_parents", [])

    if not graded_parents:
        return fallback_node(state)

    llm = get_llm(settings.SYNTHESIS_LLM_MODEL, temperature=0.0, max_tokens=2048)
    if llm:
        structured_llm = llm.with_structured_output(CitedResponse)
        context_str = "\n\n".join([
            f"=== Document ID: {p.get('document_id')} | Title: {p.get('circular_name')} | Clause: {p.get('section_clause')} | Page: {p.get('page_number')} | Location: {p.get('source_location')} ===\n{p.get('text')}"
            for p in graded_parents
        ])
        messages = [
            SystemMessage(content=GENERATOR_SYSTEM_PROMPT),
            HumanMessage(content=f"Question: {question}\n\nAuthoritative Regulatory Context:\n{context_str}"),
        ]
        try:
            resp: CitedResponse = structured_llm.invoke(messages)

            # Strict source-specific verification on every cited quote
            verified_citations = []
            has_unverified = False

            for cit in resp.citations:
                is_v, v_status, reason = verify_citation_against_source(cit, graded_parents)
                cit.verified = is_v
                cit.verification_status = v_status
                verified_citations.append(cit)
                if not is_v:
                    has_unverified = True

            resp.citations = verified_citations

            # Recompute risk and evidence sufficiency based on verification results
            if not verified_citations:
                resp.evidence_sufficiency = "INSUFFICIENT"
                resp.risk_level = "UNVERIFIED"
            elif has_unverified:
                resp.evidence_sufficiency = "PARTIAL"
                if resp.risk_level == "LOW":
                    resp.risk_level = "MEDIUM"

            return {"final_response": resp}
        except Exception as e:
            print(f"[WARNING] Generator 70B call failed: {e}. Returning unverified fallback state.")

    # Honest unverified response when LLM is unavailable (No fabricated quotations!)
    top_doc = graded_parents[0]
    honest_answer = (
        f"Retrieved authoritative regulatory provision from {top_doc.get('circular_name')} "
        f"({top_doc.get('section_clause')}). Automated LLM synthesis is currently unavailable. "
        f"Please inspect the underlying statutory section directly."
    )

    resp = CitedResponse(
        answer=honest_answer,
        citations=[],
        risk_level="UNVERIFIED",
        evidence_sufficiency="PARTIAL",
        applicable_status=top_doc.get("document_status", "ACTIVE")
    )
    return {"final_response": resp}


def out_of_scope_node(state: AgentState) -> Dict[str, Any]:
    """Handle out-of-scope non-regulatory questions."""
    resp = CitedResponse(
        answer="This query falls outside the scope of Reserve Bank of India (RBI) regulatory compliance. "
               "The compliance engine is dedicated strictly to RBI Master Directions on KYC, Digital Lending, and Cyber Resilience.",
        citations=[],
        risk_level="INFO",
        evidence_sufficiency="INSUFFICIENT",
    )
    return {"final_response": resp}


def fallback_node(state: AgentState) -> Dict[str, Any]:
    """Deterministic zero-evidence fallback when search retries are exhausted."""
    resp = CitedResponse(
        answer="Insufficient regulatory evidence found in the indexed RBI Master Directions "
               "(KYC, Digital Lending, Cyber Resilience) to verify this query.",
        citations=[],
        risk_level="UNVERIFIED",
        evidence_sufficiency="INSUFFICIENT",
    )
    return {"final_response": resp}
