"""LangGraph Cyclic State Machine for RBI Regulatory Compliance Engine."""

from typing import Literal
from langgraph.graph import StateGraph, START, END

from src.agent.state import AgentState
from src.agent.nodes import (
    router_node,
    retrieve_node,
    grader_node,
    rewriter_node,
    generator_node,
    out_of_scope_node,
    fallback_node,
)


def route_by_intent(state: AgentState) -> Literal["out_of_scope_node", "retrieve_node"]:
    """Conditional edge routing based on classified intent."""
    query_type = state.get("query_type", "compliance_qa")
    if query_type == "out_of_scope":
        return "out_of_scope_node"
    return "retrieve_node"


def decide_to_generate_or_rewrite(
    state: AgentState,
) -> Literal["generator_node", "rewriter_node", "fallback_node"]:
    """
    Conditional edge deciding between answer synthesis, acronym query rewrite,
    or terminating via fallback when retries exceed the limit.
    """
    graded_parents = state.get("graded_parents", [])
    retry_count = state.get("retry_count", 0)

    if len(graded_parents) > 0:
        return "generator_node"

    # Self-correction loop: max 2 retry attempts
    if retry_count < 2:
        return "rewriter_node"

    return "fallback_node"


def build_compliance_graph() -> StateGraph:
    """Construct and assemble the compiled LangGraph compliance workflow."""
    workflow = StateGraph(AgentState)

    # Add all agent nodes
    workflow.add_node("router_node", router_node)
    workflow.add_node("out_of_scope_node", out_of_scope_node)
    workflow.add_node("retrieve_node", retrieve_node)
    workflow.add_node("grader_node", grader_node)
    workflow.add_node("rewriter_node", rewriter_node)
    workflow.add_node("generator_node", generator_node)
    workflow.add_node("fallback_node", fallback_node)

    # Wire edges
    workflow.add_edge(START, "router_node")

    workflow.add_conditional_edges(
        "router_node",
        route_by_intent,
        {
            "out_of_scope_node": "out_of_scope_node",
            "retrieve_node": "retrieve_node",
        },
    )

    workflow.add_edge("out_of_scope_node", END)
    workflow.add_edge("retrieve_node", "grader_node")

    workflow.add_conditional_edges(
        "grader_node",
        decide_to_generate_or_rewrite,
        {
            "generator_node": "generator_node",
            "rewriter_node": "rewriter_node",
            "fallback_node": "fallback_node",
        },
    )

    # Cyclic feedback loop
    workflow.add_edge("rewriter_node", "retrieve_node")
    workflow.add_edge("generator_node", END)
    workflow.add_edge("fallback_node", END)

    return workflow.compile()


# Compiled singleton instance
compliance_agent = build_compliance_graph()
