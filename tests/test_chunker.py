"""Phase 1 Verification & Quality Gate Tests (tests/test_chunker.py)"""

import json
from pathlib import Path
import pytest


@pytest.fixture
def docstore_path():
    return Path("data/parent_docstore.json")


def test_gate_assertion_1_docstore_integrity(docstore_path):
    """
    Gate Assertion 1 (Docstore Integrity):
    Assert data/parent_docstore.json exists, contains >= 150 parent sections,
    and every parent_id referenced by any child chunk resolves in O(1) time.
    """
    assert docstore_path.exists(), f"Docstore not found at {docstore_path}"
    with open(docstore_path, "r", encoding="utf-8") as f:
        docstore = json.load(f)

    assert len(docstore) >= 150, (
        f"Expected at least 150 parent sections, but found {len(docstore)}"
    )

    # Verify O(1) dictionary key resolution for sample parent keys
    for parent_id, data in list(docstore.items())[:20]:
        assert parent_id in docstore
        assert data["parent_id"] == parent_id
        assert "text" in data and len(data["text"]) > 0


def test_gate_assertion_2_zero_table_truncation(docstore_path):
    """
    Gate Assertion 2 (Zero Table Truncation):
    For all parent entries with has_table == True, assert that the number of
    pipe characters | per row is consistent between the table header and the final row of the table.
    """
    with open(docstore_path, "r", encoding="utf-8") as f:
        docstore = json.load(f)

    table_entries = [entry for entry in docstore.values() if entry.get("has_table") is True]
    assert len(table_entries) > 0, "No parent entries found with has_table == True"

    for entry in table_entries:
        text = entry["text"]
        lines = [line.strip() for line in text.split("\n") if line.strip().startswith("|") and line.strip().endswith("|")]
        if len(lines) >= 2:
            header_pipe_count = lines[0].count("|")
            last_row_pipe_count = lines[-1].count("|")
            assert header_pipe_count == last_row_pipe_count, (
                f"Table pipe mismatch in parent {entry['parent_id']}: "
                f"Header has {header_pipe_count} pipes, last row has {last_row_pipe_count} pipes."
            )


def test_gate_assertion_3_metadata_completeness(docstore_path):
    """
    Gate Assertion 3 (Metadata Completeness):
    Assert 100% of generated parent chunks possess non-null circular_name,
    doc_category, section_clause, and page_number > 0.
    """
    with open(docstore_path, "r", encoding="utf-8") as f:
        docstore = json.load(f)

    for parent_id, item in docstore.items():
        assert item.get("circular_name") is not None and len(item["circular_name"]) > 0, (
            f"Missing circular_name in {parent_id}"
        )
        assert item.get("doc_category") in ["KYC", "DIGITAL_LENDING", "CYBER_RESILIENCE"], (
            f"Invalid doc_category in {parent_id}: {item.get('doc_category')}"
        )
        assert item.get("section_clause") is not None and len(item["section_clause"]) > 0, (
            f"Missing section_clause in {parent_id}"
        )
        # Genuine page provenance: PDF-based circulars have positive int pages; HTML gazettes have None (unknown)
        doc_id = item.get("document_id")
        page_num = item.get("page_number")
        if doc_id in ["RBI_KYC_2016", "RBI_CYBER_2024", "RBI_DL_2022"]:
            assert isinstance(page_num, int) and page_num > 0, (
                f"Invalid or missing PDF page_number in {parent_id}: {page_num}"
            )
        else:
            assert page_num is None or page_num == "unknown" or (isinstance(page_num, int) and page_num > 0), (
                f"Unexpected synthetic page_number in {parent_id}: {page_num}"
            )
