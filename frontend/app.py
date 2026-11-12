"""Streamlit Dual-Tab RBI Regulatory Compliance Workstation."""

import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

import requests
import streamlit as st

# Ensure project root is in sys.path for standalone or local runs
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

st.set_page_config(
    page_title="ReguLens - RBI Compliance & Contract Auditor",
    page_icon="⚖️",
    layout="wide",
)

FASTAPI_URL = os.getenv("FASTAPI_URL", "http://localhost:8000").rstrip("/")


def check_backend_health() -> Optional[Dict[str, Any]]:
    """Probe FastAPI backend health and telemetry."""
    try:
        resp = requests.get(f"{FASTAPI_URL}/api/v1/metrics", timeout=2.0)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


backend_metrics = check_backend_health()
backend_connected = backend_metrics is not None

# Fallback in-process resource loader
_inprocess_resources = None


def get_inprocess_resources():
    global _inprocess_resources
    if _inprocess_resources is None:
        try:
            from src.retrieval.vector_store import RBIHybridVectorStore, get_parent_docstore
            from src.retrieval.semantic_cache import EntityScopedSemanticCache
            from src.agent.graph import compliance_agent
            from src.agent.auditor import audit_fintech_contract

            _inprocess_resources = {
                "vector_store": RBIHybridVectorStore(),
                "cache": EntityScopedSemanticCache(),
                "docstore": get_parent_docstore(),
                "agent": compliance_agent,
                "auditor": audit_fintech_contract,
            }
        except Exception as e:
            st.error(f"Failed to load in-process resources: {e}")
            _inprocess_resources = {}
    return _inprocess_resources


# Sidebar: Telemetry & Connection Status
st.sidebar.title("⚖️ ReguLens Engine Telemetry")
st.sidebar.markdown("**Target Track:** GenAI & Agentic Systems Flagship")
st.sidebar.markdown("**Corpus:** Official RBI Master Directions (2025/2024 Active)")
st.sidebar.divider()

if backend_connected:
    st.sidebar.success(f"🟢 **FastAPI Backend Connected** (`{FASTAPI_URL}`)")
    indexed_chunks = str(backend_metrics.get("indexed_chunks", 577))
    parent_clauses = str(backend_metrics.get("parent_clauses", 394))
    telemetry = backend_metrics.get("cache_telemetry", {})
    hit_ratio = telemetry.get("hit_ratio", 0.0)
else:
    st.sidebar.info("🟡 **Standalone Mode** (Direct In-Process Engine)")
    in_res = get_inprocess_resources()
    docstore = in_res.get("docstore", {})
    indexed_chunks = "577"
    parent_clauses = str(len(docstore)) if docstore else "394"
    hit_ratio = 0.0

col_s1, col_s2 = st.sidebar.columns(2)
col_s1.metric("Indexed Chunks", indexed_chunks)
col_s2.metric("Parent Clauses", parent_clauses)

enable_cache = st.sidebar.toggle("Entity-Safe Semantic Cache", value=True)
category_filter = st.sidebar.selectbox(
    "Corpus Filter Scope",
    options=["All Master Directions", "KYC", "DIGITAL_LENDING", "CYBER_RESILIENCE"],
    index=0
)
cat_filter_val = None if category_filter == "All Master Directions" else category_filter

historical_mode = st.sidebar.toggle("Historical Regulatory Mode", value=False, help="Include repealed 2022 guidelines for point-in-time comparison")

st.sidebar.divider()
st.sidebar.markdown("**Dual-Model Routing:**")
st.sidebar.code("Router / Grader: Llama-3.1-8B\nSynthesizer / Audit: Llama-3.3-70B", language="text")

st.title("🛡️ ReguLens: Agentic RAG & Financial Compliance Engine")
st.caption("Reserve Bank of India Master Directions (KYC, Digital Lending & Cyber Resilience) | 100% Free-Tier CPU Stack")

tab1, tab2 = st.tabs(["📋 Tab 1: Regulatory Compliance Assistant (Cited Q&A)", "🔍 Tab 2: Fintech Contract Clause Auditor"])

# ----------------- TAB 1: CITED Q&A -----------------
with tab1:
    st.subheader("Authoritative Statutory Inquiries")
    st.markdown("Ask precise regulatory questions regarding timelines, cooling-off periods, KFS disclosures, V-CIP technical rules, or cyber incident reporting.")

    sample_questions = [
        "What is the mandatory cooling-off period in a KFS under Digital Lending Directions?",
        "What are the V-CIP video encryption and geo-tagging requirements under KYC?",
        "Within how many hours must a cybersecurity incident be reported to CERT-In and RBI?",
        "What is the periodic KYC update schedule for High-Risk vs Low-Risk customers?",
        "Are Digital Lending Apps (DLAs) allowed to access mobile phone contacts and photo gallery?"
    ]

    selected_sample = st.selectbox("Quick-Load Benchmark Question:", options=["-- Select Question --"] + sample_questions)
    query_input = st.text_input("Enter Compliance Question:", value="" if selected_sample == "-- Select Question --" else selected_sample)

    if st.button("Query RBI Compliance Engine", type="primary", use_container_width=True):
        if not query_input.strip():
            st.warning("Please enter a question.")
        else:
            start_t = time.time()
            final_resp = None
            cache_hit = False
            retries = 0

            with st.status("Executing Agentic Compliance Pipeline...", expanded=True) as status_box:
                if backend_connected:
                    st.write("1. 📡 Querying FastAPI Agentic Compliance Service...")
                    try:
                        payload = {
                            "question": query_input,
                            "category_filter": cat_filter_val,
                            "historical_mode": historical_mode,
                        }
                        resp = requests.post(f"{FASTAPI_URL}/api/v1/query", json=payload, timeout=60.0)
                        if resp.status_code == 200:
                            data = resp.json()
                            final_resp = data.get("response", {})
                            cache_hit = data.get("cache_hit", False)
                            retries = data.get("retry_count", 0)
                            st.write(f"✓ Response received ({data.get('latency_ms', 0):.1f}ms).")
                        else:
                            st.error(f"FastAPI error ({resp.status_code}): {resp.text}")
                    except Exception as e:
                        st.warning(f"Backend HTTP call failed: {e}. Falling back to in-process execution.")

                # Fallback to in-process execution if backend was not used or failed
                if final_resp is None:
                    in_res = get_inprocess_resources()
                    cache = in_res.get("cache")
                    agent = in_res.get("agent")

                    if enable_cache and cache:
                        st.write("1. ⚡ Checking Entity-Scoped Semantic Cache (Cosine Threshold ≥ 0.93)...")
                        cached_data = cache.check_cache(
                            question=query_input,
                            query_type="compliance_qa",
                            extracted_entities=[]
                        )
                        if cached_data:
                            cache_hit = True
                            final_resp = cached_data["response"]
                            st.write("✓ Semantic Cache Hit! Intercepted repeat query with Entity Guard.")

                    if not cache_hit and agent:
                        st.write("2. 🧭 Routing Intent & Acronym Expansion...")
                        st.write("3. 🔎 Executing Hybrid Dense + Sparse BM25 RRF Search & FlashRank Reranking...")
                        st.write("4. ⚖️ Multi-Criteria Authoritative Batch Grading...")
                        st.write("5. 📜 Synthesizing Cited Response & Deterministic Substring Verification...")

                        final_state = agent.invoke({
                            "question": query_input,
                            "category_filter": cat_filter_val,
                            "historical_query": historical_mode,
                            "retry_count": 0,
                            "search_query": query_input,
                        })
                        resp_obj = final_state.get("final_response")
                        if resp_obj:
                            final_resp = resp_obj.model_dump()
                            retries = final_state.get("retry_count", 0)
                            if enable_cache and cache:
                                cache.store_cache(
                                    question=query_input,
                                    query_type="compliance_qa",
                                    extracted_entities=final_state.get("extracted_entities", []),
                                    response=resp_obj
                                )

                status_box.update(label="✓ Compliance Evaluation Complete", state="complete", expanded=False)

            elapsed_ms = (time.time() - start_t) * 1000.0

            if final_resp:
                # Render Badges
                c1, c2, c3 = st.columns(3)
                risk = final_resp.get("risk_level", "INFO")
                risk_color = "🔴" if risk in ["HIGH", "CRITICAL"] else "🟡" if risk == "MEDIUM" else "🟢"
                c1.markdown(f"**Regulatory Risk Tier:** {risk_color} `{risk}`")
                c2.markdown(f"**Execution Latency:** `{elapsed_ms:.1f}ms`")
                c3.markdown(f"**Cache Telemetry:** `{'⚡ Cache Hit (0 Tokens)' if cache_hit else f'Fresh RAG Pass ({retries} retries)'}`")

                st.markdown("### Compliance Synthesis")
                st.info(final_resp.get("answer", ""))

                # Render Citations
                citations = final_resp.get("citations", [])
                if citations:
                    st.markdown("### Verifiable Statutory Citations")
                    for idx, cit in enumerate(citations, 1):
                        is_verified = cit.get("verified", False)
                        v_status = cit.get("verification_status", "UNVERIFIED")
                        badge = f"✅ **✓ Verified** (`{v_status}`)" if is_verified else f"⚠️ **Unverified** (`{v_status}`)"
                        page_str = f"Page {cit.get('page_number')}" if cit.get('page_number') else cit.get('source_location', 'N/A')
                        with st.expander(f"Citation {idx}: {cit.get('circular_name')} — {cit.get('section_clause')} ({page_str})"):
                            st.markdown(f"**Verification Status:** {badge}")
                            st.markdown(f"**Verbatim Quote from Authoritative Regulatory Text:**")
                            st.code(f'"{cit.get("verbatim_quote", "")}"', language="text")
            else:
                st.error("No compliance response could be generated. Please verify service logs.")

# ----------------- TAB 2: FINTECH CLAUSE AUDITOR -----------------
with tab2:
    st.subheader("Fintech Contract Clause Compliance Auditor")
    st.markdown("Paste drafted fintech loan agreements, partner contracts, or user consent clauses to automatically audit against active RBI Master Directions.")

    samples_contracts = {
        "Sample 1: Prohibited Data Scraping & Cooling-Off Violation": (
            "The borrower irrevocably consents that the DLA may access phone contacts, call logs, "
            "and photo gallery for credit scoring, and agrees to a 1-day cooling-off period subject to a 5% foreclosure fee."
        ),
        "Sample 2: Third-Party Escrow Pool Account Violation": (
            "Disbursals of all digital credit lines shall be routed into the Lending Service Provider's "
            "centralized collection escrow pool account prior to final transfer to borrower."
        ),
        "Sample 3: Permanent Biometric Storage Violation": (
            "Borrower biometric authentication credentials and raw fingerprint templates shall be "
            "stored permanently on the DLA server for seamless subsequent login and transaction approvals."
        )
    }

    selected_sample_clause = st.selectbox("Pre-load Non-Compliant Fintech Clause:", options=list(samples_contracts.keys()))
    contract_text_area = st.text_area(
        "Fintech Contract Language:",
        value=samples_contracts[selected_sample_clause],
        height=140
    )

    if st.button("Run RBI Statutory Audit", type="primary", use_container_width=True):
        if not contract_text_area.strip():
            st.warning("Please provide contract text.")
        else:
            report_data = None
            with st.spinner("Decomposing atomic stipulations and auditing against RBI Master Directions..."):
                if backend_connected:
                    try:
                        resp = requests.post(
                            f"{FASTAPI_URL}/api/v1/audit",
                            json={"contract_text": contract_text_area},
                            timeout=60.0
                        )
                        if resp.status_code == 200:
                            report_data = resp.json()
                        else:
                            st.warning(f"Backend audit returned status {resp.status_code}. Using in-process fallback.")
                    except Exception as e:
                        st.warning(f"Backend audit request failed: {e}. Using in-process fallback.")

                if report_data is None:
                    in_res = get_inprocess_resources()
                    auditor = in_res.get("auditor")
                    if auditor:
                        rep = auditor(contract_text_area)
                        report_data = rep.model_dump()

            if report_data:
                overall_verdict = report_data.get("overall_verdict", "REVIEW_REQUIRED")
                summary = report_data.get("summary", "")
                findings = report_data.get("clause_findings", [])
                latency_ms = report_data.get("latency_ms", 0.0)

                # Render Overall Verdict Banner
                if overall_verdict in ["NON_COMPLIANT", "FAIL"]:
                    st.error(f"🚨 **OVERALL COMPLIANCE VERDICT: NON_COMPLIANT** — {summary}")
                elif overall_verdict == "REVIEW_REQUIRED":
                    st.warning(f"⚠️ **OVERALL COMPLIANCE VERDICT: REVIEW_REQUIRED** — {summary}")
                elif overall_verdict == "COMPLIANT":
                    st.success(f"✅ **OVERALL COMPLIANCE VERDICT: COMPLIANT** — {summary}")
                else:
                    st.info(f"ℹ️ **OVERALL COMPLIANCE VERDICT: {overall_verdict}** — {summary}")

                st.metric("Audit Latency", f"{latency_ms:.1f}ms")
                st.markdown("### Clause-by-Clause Statutory Findings")

                for i, finding in enumerate(findings, 1):
                    f_status = finding.get("compliance_status", "REVIEW_REQUIRED")
                    if f_status in ["FAIL", "NON_COMPLIANT"]:
                        status_icon = "❌ FAIL"
                    elif f_status in ["REVIEW_REQUIRED", "INSUFFICIENT_EVIDENCE"]:
                        status_icon = "⚠️ REVIEW REQUIRED"
                    else:
                        status_icon = "✅ PASS"

                    excerpt = finding.get("contract_excerpt", "")
                    with st.expander(f"Finding {i}: [{status_icon}] {excerpt[:80]}..."):
                        st.markdown(f"**Contractual Excerpt:** `{excerpt}`")
                        st.markdown(f"**Compliance Status:** `{f_status}`")
                        source = finding.get("regulatory_source") or finding.get("circular_name")
                        clause = finding.get("applicable_clause") or finding.get("violated_clause")
                        page = finding.get("page_number")
                        if source or clause:
                            page_str = f"Page {page}" if page else finding.get("source_location", "")
                            st.markdown(f"**Governing Regulation:** `{source or 'RBI Master Direction'}` ({clause}, {page_str})")
                        if finding.get("verbatim_rbi_rule"):
                            st.markdown(f"**Verbatim Statutory Rule:** *\"{finding['verbatim_rbi_rule']}\"*")
                        st.markdown("**Mandatory Legal Remediation Rewrite:**")
                        st.success(finding.get("remediation_advice", "Review clause with legal counsel."))
            else:
                st.error("Audit failed to execute. Please verify engine connectivity.")
