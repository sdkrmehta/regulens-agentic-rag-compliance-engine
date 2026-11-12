"""Pydantic V2 Data Models and Schemas for RBI Compliance Engine."""

from typing import List, Literal, Optional, Dict, Any
from pydantic import BaseModel, Field


# Regulatory Document Statuses
DocumentStatus = Literal[
    "ACTIVE",
    "AMENDED",
    "SUPERSEDED",
    "REPEALED",
    "HISTORICAL",
    "UNDER_REVIEW"
]

# Citation Verification Statuses
CitationVerificationStatus = Literal[
    "VERIFIED",
    "UNVERIFIED",
    "PARTIALLY_VERIFIED",
    "SOURCE_UNAVAILABLE",
    "OUTDATED_SOURCE",
    "INSUFFICIENT_EVIDENCE"
]

# Contract Audit Finding Statuses
AuditFindingStatus = Literal[
    "COMPLIANT",
    "NON_COMPLIANT",
    "PARTIALLY_COMPLIANT",
    "REVIEW_REQUIRED",
    "INSUFFICIENT_EVIDENCE",
    "FAIL",
    "PASS"
]


class RegulatoryDocumentMetadata(BaseModel):
    document_id: str = Field(..., description="Unique canonical identifier e.g. RBI_DL_2025")
    document_title: str = Field(..., description="Official gazette title")
    regulator: str = Field(default="Reserve Bank of India", description="Issuing authority")
    regulation_category: str = Field(..., description="Category: KYC, DIGITAL_LENDING, CYBER_RESILIENCE")
    regulated_entity_types: List[str] = Field(default_factory=list, description="Applicable entities")
    circular_number: str = Field(..., description="Official notification number")
    publication_date: str = Field(..., description="Date published (YYYY-MM-DD)")
    effective_date: str = Field(..., description="Date effective (YYYY-MM-DD)")
    amendment_date: Optional[str] = Field(None, description="Date of cumulative amendment if applicable")
    superseded_date: Optional[str] = Field(None, description="Date when repealed/superseded")
    document_status: DocumentStatus = Field(default="ACTIVE", description="Current regulatory validity")
    version: str = Field(default="1.0", description="Version or update release string")
    source_url: str = Field(..., description="Authoritative RBI gazette URL")
    source_type: str = Field(default="official_gazette", description="Source medium")
    source_hash: Optional[str] = Field(None, description="SHA-256 hash of authoritative document text")
    downloaded_at: Optional[str] = None
    last_verified_at: str = Field(..., description="Timestamp of official regulatory verification")
    notes: Optional[str] = None


class CitationItem(BaseModel):
    circular_name: str = Field(..., description="Official title of the RBI Master Direction")
    section_clause: str = Field(..., description="Exact section, paragraph, or annex clause number")
    page_number: Optional[int] = Field(None, description="Source PDF page number (if available)")
    source_location: Optional[str] = Field(None, description="Section anchor or heading breadcrumb")
    verbatim_quote: str = Field(..., description="Exact 10-25 word continuous substring from the source text")
    verified: bool = Field(default=False, description="Deterministic verification flag")
    verification_status: CitationVerificationStatus = Field(
        default="UNVERIFIED",
        description="Detailed verification outcome"
    )
    document_id: Optional[str] = Field(None, description="Canonical document ID e.g. RBI_DL_2025")
    source_url: Optional[str] = Field(None, description="URL of the governing regulatory circular")


class CitedResponse(BaseModel):
    answer: str = Field(..., description="Comprehensive compliance synthesis grounded strictly in graded sources")
    citations: List[CitationItem] = Field(default_factory=list, description="Supporting statutory citations")
    risk_level: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL", "INFO", "UNVERIFIED"] = Field(
        ..., description="Regulatory risk tier"
    )
    evidence_sufficiency: Literal["SUFFICIENT", "PARTIAL", "INSUFFICIENT"] = Field(
        default="SUFFICIENT", description="Evaluation of retrieved evidence coverage"
    )
    applicable_status: DocumentStatus = Field(
        default="ACTIVE", description="Regulatory currency of supporting sources"
    )


class DocumentGrade(BaseModel):
    parent_id: str = Field(..., description="ID of the evaluated parent chunk")
    is_relevant: bool = Field(..., description="True if chunk contains concrete RBI rules addressing query")
    is_authoritative: bool = Field(default=True, description="True if document is active and unrepealed")
    is_applicable: bool = Field(default=True, description="True if applicable to requested entity type")
    reason: str = Field(..., description="1-sentence justification for relevance decision")


class BatchGradeOutput(BaseModel):
    grades: List[DocumentGrade] = Field(..., description="Relevance evaluations for all retrieved parent chunks")


class RouterOutput(BaseModel):
    query_type: Literal[
        "compliance_qa",
        "contract_audit",
        "historical_comparison",
        "clause_explanation",
        "out_of_scope"
    ] = Field(..., description="Intent classification of the user input")
    extracted_entities: List[str] = Field(
        default_factory=list, description="Extracted banking acronyms, regulations, and risk tiers"
    )
    target_category: Optional[str] = Field(
        None, description="Identified regulatory domain: KYC, DIGITAL_LENDING, CYBER_RESILIENCE"
    )
    target_entity_type: Optional[str] = Field(
        None, description="Target entity type e.g. Commercial Banks, NBFCs, PSOs"
    )
    historical_query: bool = Field(
        default=False, description="Whether query inquires about superseded/historical regulatory provisions"
    )
    search_query: str = Field(
        ..., description="Acronym-expanded statutory search query for hybrid retrieval"
    )


class ClauseAuditItem(BaseModel):
    obligation_id: str = Field(default="", description="Unique obligation reference ID")
    contract_excerpt: str = Field(..., description="Atomic obligation extracted from user contract clause")
    compliance_status: AuditFindingStatus = Field(..., description="Defensible compliance verdict")
    regulatory_source: Optional[str] = Field(None, description="RBI Master Direction title")
    applicable_clause: Optional[str] = Field(None, description="Specific statutory clause or paragraph")
    violated_clause: Optional[str] = Field(None, description="Alias or violated statutory clause for findings")
    page_number: Optional[int] = Field(None, description="Source page number in the RBI document")
    source_location: Optional[str] = Field(None, description="Heading breadcrumb or section anchor")
    verbatim_rbi_rule: Optional[str] = Field(None, description="Verbatim statutory rule")
    remediation_advice: str = Field(..., description="Concrete legal/technical rewrite required to comply")
    risk_level: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL", "UNKNOWN"] = Field(default="MEDIUM")
    verification_status: CitationVerificationStatus = Field(default="UNVERIFIED")


class ContractAuditReport(BaseModel):
    overall_verdict: AuditFindingStatus
    summary: str
    clause_findings: List[ClauseAuditItem]
    verified_obligations_count: int = 0
    flagged_violations_count: int = 0
    latency_ms: float = 0.0


# API Request/Response Models
class QueryRequest(BaseModel):
    question: str = Field(..., description="Compliance query to evaluate")
    category_filter: Optional[str] = Field(
        None, description="Optional corpus filter (KYC, DIGITAL_LENDING, CYBER_RESILIENCE)"
    )
    entity_type_filter: Optional[str] = Field(
        None, description="Optional entity type filter (e.g. Commercial Banks, NBFCs, PSOs)"
    )
    historical_mode: bool = Field(
        default=False, description="Whether to include superseded/repealed regulations in search"
    )
    effective_date: Optional[str] = Field(
        None, description="Target effective date for point-in-time regulatory evaluation (YYYY-MM-DD)"
    )


class QueryResponse(BaseModel):
    response: CitedResponse
    retry_count: int = 0
    cache_hit: bool = False
    latency_ms: float = 0.0


class ContractAuditRequest(BaseModel):
    contract_text: str = Field(..., description="Raw text of drafted contract or fintech clause to audit")
    entity_type: Optional[str] = Field(default="NBFC", description="Regulated Entity type (Bank, NBFC, LSP, PSO)")
