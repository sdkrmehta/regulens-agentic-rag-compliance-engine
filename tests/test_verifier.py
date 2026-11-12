"""Phase 3 Verification & Quality Gate Tests (tests/test_verifier.py)"""

import pytest
from src.agent.graph import compliance_agent
from src.agent.verifier import verify_citation_quote, normalize_text


def test_gate_assertion_1_cycle_termination():
    """
    Gate Assertion 1 (Cycle Termination):
    Test a deliberately unanswerable financial query and assert that retry_count
    increments to 2, terminates cleanly via fallback_node, and never enters an infinite loop.
    """
    unanswerable_query = "What is the RBI capital adequacy ratio for cryptocurrency mining pools?"
    initial_state = {
        "question": unanswerable_query,
        "retry_count": 0,
        "search_query": unanswerable_query,
    }

    final_state = compliance_agent.invoke(initial_state)

    assert final_state is not None, "Workflow invocation returned None"
    assert final_state.get("retry_count") == 2, (
        f"Expected retry_count to terminate at 2, but got {final_state.get('retry_count')}"
    )

    resp = final_state.get("final_response")
    assert resp is not None, "Missing final_response in final state"
    assert resp.risk_level == "UNVERIFIED", (
        f"Expected risk_level to be UNVERIFIED, got {resp.risk_level}"
    )
    assert "Insufficient regulatory evidence" in resp.answer, (
        f"Expected zero-evidence fallback text, got: {resp.answer}"
    )


def test_gate_assertion_2_table_quote_verification():
    """
    Gate Assertion 2 (Table Quote Verification):
    Pass a real quote extracted from a Markdown table normalized by an LLM into single spaces
    into verify_citation_quote() and assert it returns True, while a fabricated quote returns False.
    """
    # Sample parent document containing a multi-column table
    mock_parent_with_table = {
        "parent_id": "KYC_030_P0078",
        "text": (
            "### 38. Updation / Periodic Updation of KYC\n\n"
            "| Customer Risk Category | Minimum Updation Frequency |\n"
            "|---|---|\n"
            "| High Risk Customers | At least once in every 2 years |\n"
            "| Medium Risk Customers | At least once in every 8 years |\n"
            "| Low Risk Customers | At least once in every 10 years |\n"
        ),
        "circular_name": "RBI Master Direction on KYC"
    }

    graded_parents = [mock_parent_with_table]

    # Real table quote with pipes and newlines collapsed into single spaces
    real_llm_normalized_quote = "High Risk Customers At least once in every 2 years"
    assert verify_citation_quote(real_llm_normalized_quote, graded_parents) is True, (
        "Failed to verify real LLM-normalized table quote"
    )

    # Another valid row
    real_low_risk_quote = "Low Risk Customers At least once in every 10 years"
    assert verify_citation_quote(real_low_risk_quote, graded_parents) is True, (
        "Failed to verify Low Risk customer table row"
    )

    # Fabricated / hallucinated quote
    fabricated_quote = "Cryptocurrency mining pools must maintain 50 percent statutory liquidity ratio daily"
    assert verify_citation_quote(fabricated_quote, graded_parents) is False, (
        "Fabricated quote was incorrectly verified as True"
    )

    # Short quote under minimum word count threshold (< 4 words)
    too_short_quote = "High Risk"
    assert verify_citation_quote(too_short_quote, graded_parents) is False, (
        "Sub-threshold short quote was incorrectly verified as True"
    )
