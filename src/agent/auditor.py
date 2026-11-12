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


# Authoritative Statutory Obligation Registry (RBI Active Regulations)
REGULATORY_OBLIGATION_REGISTRY = [
    {
        "obligation_id": "OBL_DL_001",
        "category": "DIGITAL_LENDING",
        "circular_name": "Reserve Bank of India (Digital Lending) Directions, 2025",
        "circular_number": "RBI/2025-26/36",
        "clause": "Clause 10.2: Collection, Usage and Sharing of Borrower Data",
        "page_number": None,
        "keywords": ["contact", "phonebook", "call log", "gallery", "media", "telephony", "file and media"],
        "prohibition": True,
        "statutory_rule": "DLAs are explicitly prohibited from accessing mobile phone resources, including file and media storage, contact lists, phonebooks, call logs, and telephony functions.",
        "remediation_advice": "Remove all clauses granting access to borrower contacts, call logs, or media files. Limit mobile permissions strictly to one-time camera/microphone/location access for onboarding and KYC with prior explicit consent.",
        "risk_level": "CRITICAL"
    },
    {
        "obligation_id": "OBL_DL_002",
        "category": "DIGITAL_LENDING",
        "circular_name": "Reserve Bank of India (Digital Lending) Directions, 2025",
        "circular_number": "RBI/2025-26/36",
        "clause": "Clause 8.2: Cooling-Off and Look-Up Period",
        "page_number": None,
        "keywords": ["cooling-off", "cooling off", "look-up", "lookup"],
        "prohibition": False,
        "invalid_patterns": [r"\b1[\s-]day\b", r"\b24[\s-]hours?\b", r"\b2[\s-]days?\b", r"\b48[\s-]hours?\b"],
        "statutory_rule": "The cooling-off period shall not be less than three (3) calendar days for loans having tenor of seven days or more, and one (1) calendar day for loans having tenor of less than seven days.",
        "remediation_advice": "Amend cooling-off look-up period to a mandatory minimum of three (3) calendar days for loans having tenor of 7 days or more.",
        "risk_level": "HIGH"
    },
    {
        "obligation_id": "OBL_DL_003",
        "category": "DIGITAL_LENDING",
        "circular_name": "Reserve Bank of India (Digital Lending) Directions, 2025",
        "circular_number": "RBI/2025-26/36",
        "clause": "Clause 8.1: Prepayment Penalty Prohibition during Cooling-Off",
        "page_number": None,
        "keywords": ["prepayment", "foreclosure", "exit penalty", "pre-payment", "penalty fee", "foreclosure fee"],
        "prohibition": True,
        "statutory_rule": "A borrower shall be given an explicit, unconditional option to exit a digital loan by paying the principal and the proportionate APR without any penalty, foreclosure fee, or exit charge during the cooling-off period.",
        "remediation_advice": "Remove all foreclosure fees and prepayment penalty stipulations during the statutory cooling-off look-up period.",
        "risk_level": "CRITICAL"
    },
    {
        "obligation_id": "OBL_DL_004",
        "category": "DIGITAL_LENDING",
        "circular_name": "Reserve Bank of India (Digital Lending) Directions, 2025",
        "circular_number": "RBI/2025-26/36",
        "clause": "Clause 4.1 & 4.2: Direct Disbursals & Repayments (No Pool Accounts)",
        "page_number": None,
        "keywords": ["pool account", "escrow pool", "pass-through account", "lsp account", "centralized collection account"],
        "prohibition": True,
        "statutory_rule": "REs shall ensure that all loan disbursals and repayments are executed directly into the bank account of the borrower without any pass-through or pool account of any third party or LSP.",
        "remediation_advice": "Eliminate routing through third-party or LSP escrow/collection pool accounts. Disburse loan funds directly into borrower bank account.",
        "risk_level": "CRITICAL"
    },
    {
        "obligation_id": "OBL_DL_005",
        "category": "DIGITAL_LENDING",
        "circular_name": "Reserve Bank of India (Digital Lending) Directions, 2025",
        "circular_number": "RBI/2025-26/36",
        "clause": "Clause 11.2: Biometric Data Storage Prohibition",
        "page_number": None,
        "keywords": ["biometric", "fingerprint", "face template", "biometric template", "store biometric"],
        "prohibition": True,
        "statutory_rule": "REs shall ensure that no biometric data or raw fingerprint/facial templates are stored or collected on the servers or systems of DLAs or LSPs.",
        "remediation_advice": "Remove all clauses permitting storage or retention of borrower biometric data on DLA or partner servers.",
        "risk_level": "CRITICAL"
    },
    {
        "obligation_id": "OBL_DL_006",
        "category": "DIGITAL_LENDING",
        "circular_name": "Reserve Bank of India (Digital Lending) Directions, 2025",
        "circular_number": "RBI/2025-26/36",
        "clause": "Clause 13.1: Default Loss Guarantee (DLG) 5% Cap",
        "page_number": None,
        "keywords": ["dlg", "fldg", "default loss guarantee", "first loss default", "loss sharing"],
        "prohibition": False,
        "invalid_patterns": [r"\b(?:[6-9]|[1-9]\d+)%\b"],  # >5%
        "statutory_rule": "Total Default Loss Guarantee (DLG) cover across all contractual arrangements shall not exceed five percent (5%) of the total outstanding loan portfolio.",
        "remediation_advice": "Cap total contractual DLG / FLDG cover to a maximum of 5% of the underlying loan portfolio and back with statutory liquid collateral.",
        "risk_level": "HIGH"
    },
    {
        "obligation_id": "OBL_KYC_001",
        "category": "KYC",
        "circular_name": "Master Direction - Know Your Customer (KYC) Direction, 2016",
        "circular_number": "RBI/DBR/2015-16/18",
        "clause": "Clause 18 & 27: V-CIP Live Video and Geo-Tagging Standards",
        "page_number": None,
        "keywords": ["v-cip", "vcip", "video kyc", "geo-tagging", "liveness check"],
        "prohibition": False,
        "statutory_rule": "V-CIP live video interactions must include real-time liveness checks, AES-256 encrypted channels, and verifiable geo-tagging confirming customer physical presence within Indian territory.",
        "remediation_advice": "Incorporate explicit contractual stipulations mandating AES-256 video encryption, active liveness validation, and Indian geo-tagging checks during V-CIP.",
        "risk_level": "HIGH"
    },
    {
        "obligation_id": "OBL_CYB_001",
        "category": "CYBER_RESILIENCE",
        "circular_name": "Master Directions on Cyber Resilience and Digital Payment Security Controls",
        "circular_number": "RBI/DPSS/2024-25/123",
        "clause": "Clause 14: Cybersecurity Incident Reporting Timeline",
        "page_number": None,
        "keywords": ["incident report", "cert-in", "breach notification", "cyber incident"],
        "prohibition": False,
        "invalid_patterns": [r"\b(?:[7-9]|[1-9]\d+)\s*hours?\b", r"\b(?:[2-9]|\d{2,})\s*days?\b"],  # > 6 hours
        "statutory_rule": "Payment System Operators must report any cybersecurity incident to CERT-In and RBI within six (6) hours of detection.",
        "remediation_advice": "Amend cybersecurity incident reporting SLA to a maximum of six (6) hours from detection.",
        "risk_level": "CRITICAL"
    }
]


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

    # Step 1: Match against the Authoritative RBI Obligation Registry
    for obl in REGULATORY_OBLIGATION_REGISTRY:
        # Check keyword presence
        has_keywords = any(k in ob_lower for k in obl["keywords"])
        if not has_keywords:
            continue

        # Check context keywords if defined
        if obl.get("context_keywords"):
            if not any(ck in ob_lower for ck in obl["context_keywords"]):
                continue

        # 1. Prohibitory clauses (e.g., accessing contacts/gallery, pool accounts, biometric storage)
        if obl.get("prohibition"):
            return ClauseAuditItem(
                obligation_id=obl["obligation_id"],
                contract_excerpt=obligation,
                compliance_status="FAIL",
                regulatory_source=obl["circular_name"],
                applicable_clause=obl["clause"],
                violated_clause=obl["clause"],
                page_number=obl.get("page_number"),
                source_location=obl["clause"],
                verbatim_rbi_rule=obl["statutory_rule"],
                remediation_advice=obl["remediation_advice"],
                risk_level=obl.get("risk_level", "HIGH"),
                verification_status="VERIFIED"
            )

        # 2. Invalid condition patterns (e.g., 1-day cooling off, >6h incident reporting, >5% DLG)
        if obl.get("invalid_patterns"):
            for pat in obl["invalid_patterns"]:
                if re.search(pat, ob_lower):
                    return ClauseAuditItem(
                        obligation_id=obl["obligation_id"],
                        contract_excerpt=obligation,
                        compliance_status="FAIL",
                        regulatory_source=obl["circular_name"],
                        applicable_clause=obl["clause"],
                        violated_clause=obl["clause"],
                        page_number=obl.get("page_number"),
                        source_location=obl["clause"],
                        verbatim_rbi_rule=obl["statutory_rule"],
                        remediation_advice=obl["remediation_advice"],
                        risk_level=obl.get("risk_level", "HIGH"),
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
