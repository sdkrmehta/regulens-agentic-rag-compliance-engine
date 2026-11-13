"""Multi-Clause Fintech Contract Compliance Auditor with Fail-Closed Obligation Registry."""

import time
import re
from typing import List, Dict, Any, Optional

from langchain_core.messages import SystemMessage, HumanMessage

from src.config import settings
from src.schemas import (
    ClauseAuditItem,
    ContractAuditReport,
    AuditFindingStatus,
    CitationVerificationStatus,
)
from src.agent.nodes import get_llm, get_vector_store
from src.agent.prompts import CLAUSE_DECOMPOSER_PROMPT, CONTRACT_AUDITOR_PROMPT
from src.agent.verifier import normalize_statutory_text
from src.agent.registry import get_obligation_registry, extract_percentages, extract_durations

# Dynamic link to structured, version-controlled obligation registry
_registry = get_obligation_registry()
REGULATORY_OBLIGATION_REGISTRY = _registry.get_all_obligations()


def decompose_contract_clauses(contract_text: str) -> List[str]:
    """Decompose drafted contract paragraphs into distinct atomic legal obligations."""
    llm = get_llm(settings.FAST_LLM_MODEL, temperature=0.0)
    if llm:
        messages = [
            SystemMessage(content=CLAUSE_DECOMPOSER_PROMPT),
            HumanMessage(content=f"Drafted Contract Text to decompose:\n{contract_text}"),
        ]
        try:
            resp = llm.invoke(messages)
            raw_lines = [line.strip().lstrip("1234567890.-*•) ") for line in resp.content.split("\n") if line.strip()]
            obligations = [l for l in raw_lines if len(l.split()) >= 3]
            if len(obligations) >= 2:
                return obligations[:5]
        except Exception as e:
            print(f"[WARNING] Clause decomposition LLM call failed: {e}")

    # Fallback rule-based clause decomposition
    raw_parts = re.split(r"[;\n]+", contract_text)
    obligations = []
    for p in raw_parts:
        subparts = re.split(
            r",?\s+(?:and\s+)?(?:agrees|consents|undertakes)\s+to\s+|,?\s+(?:and\s+is\s+)?subject\s+to\s+|,?\s+and\s+(?:imposes|requires|demands)\s+",
            p.strip(),
            flags=re.IGNORECASE
        )
        for sp in subparts:
            sp_clean = sp.strip().rstrip(".")
            if len(sp_clean.split()) >= 3:
                obligations.append(sp_clean)

    return obligations[:5] if obligations else [contract_text.strip()]


def audit_contract_obligation(
    obligation: str,
    retrieved_parents: List[Dict[str, Any]],
) -> ClauseAuditItem:
    """
    Evaluate an atomic contract obligation against the RBI Obligation Registry and retrieved context.
    Enforces FAIL-CLOSED logic: if evidence is absent or inconclusive, status is REVIEW_REQUIRED, never PASS!
    """
    ob_lower = obligation.lower()

    # Step 1: Match against the Version-Controlled RBI Obligation Registry
    reg_eval = _registry.evaluate_clause(obligation)
    if reg_eval:
        return ClauseAuditItem(
            obligation_id=reg_eval["obligation_id"],
            contract_excerpt=obligation,
            compliance_status=reg_eval.get("status", "FAIL"),
            regulatory_source=reg_eval["circular_name"],
            applicable_clause=reg_eval["clause"],
            violated_clause=reg_eval["clause"],
            page_number=reg_eval.get("page_number"),
            source_location=reg_eval["clause"],
            verbatim_rbi_rule=reg_eval["statutory_rule"],
            remediation_advice=reg_eval["remediation_advice"],
            risk_level=reg_eval.get("risk_level", "HIGH"),
            verification_status="VERIFIED"
        )

    # Step 2: If LLM is available, invoke structured auditor prompt
    llm = get_llm(settings.SYNTHESIS_LLM_MODEL, temperature=0.0, max_tokens=1024)
    if llm and retrieved_parents:
        context_str = "\n\n".join([
            f"=== Circular: {p.get('circular_name')} | Clause: {p.get('section_clause')} ===\n{p.get('text')}"
            for p in retrieved_parents
        ])
        messages = [
            SystemMessage(content=CONTRACT_AUDITOR_PROMPT),
            HumanMessage(content=f"Contract Obligation:\n{obligation}\n\nRBI Regulations:\n{context_str}"),
        ]
        try:
            structured_llm = llm.with_structured_output(ClauseAuditItem)
            item: ClauseAuditItem = structured_llm.invoke(messages)
            # Prevent LLM from returning COMPLIANT without verified evidence
            if item.compliance_status == "COMPLIANT" and not item.verbatim_rbi_rule:
                item.compliance_status = "REVIEW_REQUIRED"
                item.remediation_advice = "Manual compliance review required: LLM suggested compliance but cited no verified statutory text."
            return item
        except Exception as e:
            print(f"[WARNING] Audit synthesis LLM call failed: {e}")

    # Step 3: FAIL-CLOSED DEFAULT
    # Never return PASS / COMPLIANT when evidence is absent or unverified!
    if retrieved_parents:
        top_doc = retrieved_parents[0]
        return ClauseAuditItem(
            obligation_id="UNRESOLVED_OBLIGATION",
            contract_excerpt=obligation,
            compliance_status="REVIEW_REQUIRED",
            regulatory_source=top_doc.get("circular_name"),
            applicable_clause=top_doc.get("section_clause"),
            page_number=top_doc.get("page_number"),
            source_location=top_doc.get("source_location"),
            verbatim_rbi_rule=None,
            remediation_advice="Stipulation does not match known violation patterns, but requires formal legal review to confirm compliance against RBI guidelines.",
            risk_level="MEDIUM",
            verification_status="INSUFFICIENT_EVIDENCE"
        )

    return ClauseAuditItem(
        obligation_id="UNRESOLVED_OBLIGATION",
        contract_excerpt=obligation,
        compliance_status="INSUFFICIENT_EVIDENCE",
        regulatory_source=None,
        applicable_clause=None,
        page_number=None,
        source_location=None,
        verbatim_rbi_rule=None,
        remediation_advice="No governing regulatory provisions found in indexed corpus. Manual legal inspection mandatory.",
        risk_level="UNKNOWN",
        verification_status="INSUFFICIENT_EVIDENCE"
    )


def audit_fintech_contract(contract_text: str) -> ContractAuditReport:
    """End-to-end multi-clause fintech contract audit pipeline."""
    start_time = time.time()
    vector_store = get_vector_store()

    obligations = decompose_contract_clauses(contract_text)
    findings: List[ClauseAuditItem] = []

    non_compliant_count = 0
    review_required_count = 0
    verified_count = 0

    for ob in obligations:
        retrieved_parents = vector_store.hybrid_search_and_rerank(
            query=ob,
            document_status="ACTIVE",
            top_k=3
        )
        finding = audit_contract_obligation(ob, retrieved_parents)
        findings.append(finding)

        if finding.compliance_status in ["NON_COMPLIANT", "FAIL"]:
            non_compliant_count += 1
        elif finding.compliance_status in ["REVIEW_REQUIRED", "PARTIALLY_COMPLIANT", "INSUFFICIENT_EVIDENCE"]:
            review_required_count += 1

        if finding.verification_status == "VERIFIED":
            verified_count += 1

    # Defensible Overall Verdict
    if non_compliant_count > 0:
        overall_verdict: AuditFindingStatus = "NON_COMPLIANT"
    elif review_required_count > 0:
        overall_verdict = "REVIEW_REQUIRED"
    elif verified_count == len(obligations) and verified_count > 0:
        overall_verdict = "COMPLIANT"
    else:
        overall_verdict = "INSUFFICIENT_EVIDENCE"

    summary = (
        f"Fintech Contract Audit completed across {len(obligations)} atomic stipulations. "
        f"Overall Verdict: {overall_verdict}. Flagged {non_compliant_count} statutory violations, "
        f"{review_required_count} items requiring manual review."
    )

    elapsed_ms = (time.time() - start_time) * 1000.0

    return ContractAuditReport(
        overall_verdict=overall_verdict,
        summary=summary,
        clause_findings=findings,
        verified_obligations_count=verified_count,
        flagged_violations_count=non_compliant_count,
        latency_ms=round(elapsed_ms, 2)
    )
