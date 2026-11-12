"""Streamlit Dual-Tab RBI Regulatory Compliance Workstation."""

import sys
from pathlib import Path

# Ensure project root is in sys.path so 'src' can always be imported
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import time
import streamlit as st

st.set_page_config(
    page_title="ReguLens - RBI Compliance & Contract Auditor",
    page_icon="⚖️",
    layout="wide",
)

from src.retrieval.vector_store import RBIHybridVectorStore, get_parent_docstore
from src.retrieval.semantic_cache import EntityScopedSemanticCache
from src.agent.graph import compliance_agent
from src.agent.auditor import audit_fintech_contract


@st.cache_resource
def load_resources():
    vector_store = RBIHybridVectorStore()
    cache = EntityScopedSemanticCache()
    docstore = get_parent_docstore()
    return vector_store, cache, docstore


vector_store, cache, docstore = load_resources()

# Sidebar: System Telemetry & Controls
st.sidebar.title("⚖️ ReguLens Engine Telemetry")
st.sidebar.markdown("**Target Track:** GenAI & Agentic Systems Flagship")
st.sidebar.markdown("**Corpus:** Official RBI Master Directions")
st.sidebar.divider()

col_s1, col_s2 = st.sidebar.columns(2)
col_s1.metric("Indexed Chunks", "561")
col_s2.metric("Parent Clauses", str(len(docstore)))

enable_cache = st.sidebar.toggle("Entity-Safe Semantic Cache", value=True)
category_filter = st.sidebar.selectbox(
    "Corpus Filter Scope",
    options=["All Master Directions", "KYC", "DIGITAL_LENDING", "CYBER_RESILIENCE"],
    index=0
)
cat_filter_val = None if category_filter == "All Master Directions" else category_filter

st.sidebar.divider()
st.sidebar.markdown("**Dual-Model Routing:**")
st.sidebar.code("Router / Grader: Llama-3.1-8B\nSynthesizer / Audit: Llama-3.3-70B", language="text")

st.title("🛡️ ReguLens: Agentic RAG & Financial Compliance Engine")
st.caption("Reserve Bank of India Master Directions (KYC, Digital Lending & Cyber Resilience) | 100% Free-Tier CPU Stack")

tab1, tab2 = st.tabs(["📋 Tab 1: Regulatory Compliance Assistant (Cited Q&A)", "🔍 Tab 2: Fintech Contract Clause Auditor"])

# ----------------- TAB 1: CITED Q&A -----------------
with tab1:
    st.subheader("Authoritative Statutory Inquiries")
    st.markdown("Ask precise regulatory questions regarding timelines, cooling-off periods, KFS disclosures, V-CIP technical rules, or cyber escalation.")

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
            cache_hit = False
            cached_data = None

            with st.status("Executing Agentic Compliance Pipeline...", expanded=True) as status_box:
                # Step 1: Cache check
                if enable_cache:
                    st.write("1. ⚡ Checking Entity-Scoped Semantic Cache (Cosine Threshold ≥ 0.93)...")
                    cached_data = cache.check_cache(
                        question=query_input,
                        query_type="compliance_qa",
                        extracted_entities=[]
                    )
                    if cached_data:
                        cache_hit = True
                        st.write("✓ Semantic Cache Hit! Intercepted repeat query with Entity Guard.")

                if not cache_hit:
                    st.write("2. 🧭 Routing Intent & Extracting Acronyms via Llama-3.1-8B-Instant...")
                    st.write("3. 🔎 Executing Hybrid Dense (bge-small) + Sparse (BM25) RRF Search & FlashRank Cross-Encoder Reranking...")
                    st.write("4. ⚖️ Single-Call Batch Grading 4 Parent Sections via Llama-3.1-8B...")
                    st.write("5. 📜 Synthesizing Cited Response & Deterministic Substring Verification via Llama-3.3-70B...")

                    final_state = compliance_agent.invoke({
                        "question": query_input,
                        "category_filter": cat_filter_val,
                        "retry_count": 0,
                        "search_query": query_input
                    })

                    final_resp = final_state.get("final_response")
                    retries = final_state.get("retry_count", 0)

                    # Write-through cache
                    if enable_cache and final_resp:
                        cache.store_cache(
                            question=query_input,
                            query_type="compliance_qa",
                            extracted_entities=final_state.get("extracted_entities", []),
                            response=final_resp
                        )
                else:
                    final_resp_dict = cached_data["response"]
                    from src.schemas import CitedResponse
                    final_resp = CitedResponse(**final_resp_dict)
                    retries = 0

                status_box.update(label="✓ Compliance Evaluation Complete", state="complete", expanded=False)

            elapsed_ms = (time.time() - start_t) * 1000.0

            # Render Badges
            c1, c2, c3 = st.columns(3)
            risk_color = "🔴" if final_resp.risk_level == "HIGH" else "🟡" if final_resp.risk_level == "MEDIUM" else "🟢"
            c1.markdown(f"**Regulatory Risk Tier:** {risk_color} `{final_resp.risk_level}`")
            c2.markdown(f"**Execution Latency:** `{elapsed_ms:.1f}ms`")
            c3.markdown(f"**Cache Telemetry:** `{'⚡ Cache Hit (0 Tokens)' if cache_hit else f'Fresh RAG Pass ({retries} retries)'}`")

            st.markdown("### Compliance Synthesis")
            st.info(final_resp.answer)

            # Render Citations
            if final_resp.citations:
                st.markdown("### Verifiable Statutory Citations")
                for idx, cit in enumerate(final_resp.citations, 1):
                    badge = "✅ **✓ Verbatim Substring Verified**" if cit.verified else "⚠️ Substring Unverified"
                    with st.expander(f"Citation {idx}: {cit.circular_name} - {cit.section_clause} (Page {cit.page_number})"):
                        st.markdown(f"**Verification Status:** {badge}")
                        st.markdown(f"**Verbatim Quote from Statutory Source:**")
                        st.code(f'"{cit.verbatim_quote}"', language="text")

# ----------------- TAB 2: FINTECH CLAUSE AUDITOR -----------------
with tab2:
    st.subheader("Fintech Contract Clause Compliance Auditor")
    st.markdown("Paste drafted fintech loan agreements, partner contracts, or user consent clauses to automatically audit against RBI Master Directions.")

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
            with st.spinner("Decomposing atomic stipulations and auditing against RBI Master Directions..."):
                report = audit_fintech_contract(contract_text_area)

            # Render Overall Verdict Banner
            if report.overall_verdict == "NON_COMPLIANT":
                st.error(f"🚨 **OVERALL COMPLIANCE VERDICT: NON_COMPLIANT** — {report.summary}")
            elif report.overall_verdict == "PARTIAL_RISK":
                st.warning(f"⚠️ **OVERALL COMPLIANCE VERDICT: PARTIAL_RISK** — {report.summary}")
            else:
                st.success(f"✅ **OVERALL COMPLIANCE VERDICT: COMPLIANT** — {report.summary}")

            st.metric("Audit Latency", f"{report.latency_ms:.1f}ms")
            st.markdown("### Clause-by-Clause Statutory Findings")

            for i, finding in enumerate(report.clause_findings, 1):
                status_icon = "❌ FAIL" if finding.compliance_status == "FAIL" else "⚠️ WARNING" if finding.compliance_status == "WARNING" else "✅ PASS"
                with st.expander(f"Finding {i}: [{status_icon}] {finding.contract_excerpt[:80]}..."):
                    st.markdown(f"**Contractual Excerpt:** `{finding.contract_excerpt}`")
                    st.markdown(f"**Compliance Status:** `{finding.compliance_status}`")
                    if finding.violated_circular:
                        st.markdown(f"**Governing Regulation:** `{finding.violated_circular}` ({finding.violated_clause}, Page {finding.page_number})")
                    if finding.verbatim_rbi_rule:
                        st.markdown(f"**Verbatim Statutory Rule:** *\"{finding.verbatim_rbi_rule}\"*")
                    st.markdown(f"**Mandatory Legal Remediation Rewrite:**")
                    st.success(finding.remediation_advice)
