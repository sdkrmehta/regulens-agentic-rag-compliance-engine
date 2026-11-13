from typing import TypedDict, List, Dict, Any, Optional
from src.schemas import CitedResponse, ClauseAuditItem


class AgentState(TypedDict, total=False):
    question: str
    query_type: str  # "compliance_qa", "contract_audit", "out_of_scope"
    category_filter: Optional[str]
    entity_type_filter: Optional[str]
    target_entity_type: Optional[str]
    regulator: Optional[str]
    effective_date: Optional[str]
    historical_query: bool
    extracted_entities: List[str]
    search_query: str
    retrieved_parents: List[Dict[str, Any]]
    graded_parents: List[Dict[str, Any]]
    retry_count: int
    final_response: Optional[CitedResponse]
    contract_findings: Optional[List[ClauseAuditItem]]
    latency_ms: float
