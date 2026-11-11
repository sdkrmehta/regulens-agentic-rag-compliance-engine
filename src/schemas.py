from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class CitationItem(BaseModel):
    circular_name: str = Field(..., description="Official title of the RBI Master Direction")
    section_clause: str = Field(..., description="Exact section, paragraph, or annex clause number")
    page_number: int = Field(..., description="Source PDF page number")
    verbatim_quote: str = Field(..., description="Exact 10-25 word verbatim substring from the source text")
    verified: bool = Field(default=False, description="Set deterministically by Python substring verifier")


class CitedResponse(BaseModel):
    answer: str = Field(..., description="Comprehensive compliance synthesis grounded strictly in graded sources")
    citations: List[CitationItem] = Field(default_factory=list, description="Supporting statutory citations")
    risk_level: Literal["LOW", "MEDIUM", "HIGH", "INFO", "UNVERIFIED"] = Field(..., description="Regulatory risk tier")


class DocumentGrade(BaseModel):
    parent_id: str = Field(..., description="ID of the evaluated parent chunk")
    is_relevant: bool = Field(..., description="True if chunk contains concrete RBI rules addressing the query")
    reason: str = Field(..., description="1-sentence justification for relevance decision")


class BatchGradeOutput(BaseModel):
    grades: List[DocumentGrade] = Field(..., description="Relevance evaluations for all retrieved parent chunks")


class RouterOutput(BaseModel):
    query_type: Literal["compliance_qa", "contract_audit", "out_of_scope"] = Field(
        ..., description="Intent classification of the user input"
    )
    extracted_entities: List[str] = Field(
        default_factory=list, description="Extracted banking acronyms, regulations, and risk tiers"
    )
    search_query: str = Field(
        ..., description="Initial acronym-expanded search query for hybrid retrieval"
    )


class ClauseAuditItem(BaseModel):
    contract_excerpt: str = Field(..., description="Atomic obligation extracted from the user's contract clause")
    compliance_status: Literal["PASS", "FAIL", "WARNING"] = Field(..., description="RBI compliance verdict")
    violated_circular: Optional[str] = Field(None, description="RBI Master Direction governing this obligation")
    violated_clause: Optional[str] = Field(None, description="Specific statutory clause or paragraph")
    page_number: Optional[int] = Field(None, description="Source page number in the RBI PDF")
    verbatim_rbi_rule: Optional[str] = Field(None, description="Verbatim quote from the RBI regulation")
    remediation_advice: str = Field(..., description="Concrete legal/technical rewrite required to achieve compliance")


class ContractAuditReport(BaseModel):
    overall_verdict: Literal["COMPLIANT", "NON_COMPLIANT", "PARTIAL_RISK"]
    summary: str
    clause_findings: List[ClauseAuditItem]
    latency_ms: float = 0.0


# API Request/Response Models
class QueryRequest(BaseModel):
    question: str = Field(..., description="Compliance query to evaluate")
    category_filter: Optional[str] = Field(None, description="Optional corpus filter (KYC, DIGITAL_LENDING, CYBER_RESILIENCE)")


class QueryResponse(BaseModel):
    response: CitedResponse
    retry_count: int = 0
    cache_hit: bool = False
    latency_ms: float = 0.0


class ContractAuditRequest(BaseModel):
    contract_text: str = Field(..., description="Raw text of drafted contract or fintech clause to audit")
