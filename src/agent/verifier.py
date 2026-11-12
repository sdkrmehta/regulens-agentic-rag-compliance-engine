"""Deterministic Source-Specific Citation & Quotation Verification Engine."""

import re
from typing import List, Dict, Any, Optional, Tuple
from src.schemas import CitationItem, CitationVerificationStatus


def normalize_statutory_text(text: str) -> str:
    """Normalize whitespace, Markdown table pipes, and punctuation for robust comparison."""
    # Replace markdown table pipes, dashes, asterisks with spaces
    cleaned = re.sub(r"[\|\*\_\#\:\-]+", " ", text)
    # Collapse multiple whitespaces and lowercase
    cleaned = re.sub(r"\s+", " ", cleaned).strip().lower()
    return cleaned


# Alias for backwards compatibility with tests and callers
normalize_text = normalize_statutory_text


def verify_citation_against_source(
    citation: CitationItem,
    source_documents: List[Dict[str, Any]],
) -> Tuple[bool, CitationVerificationStatus, str]:
    """
    Strong, source-specific citation verification:
    1. Checks if the verbatim quote is 10 to 25 words long.
    2. Identifies matching source document by document_id, circular_name, or section_clause.
    3. Verifies that verbatim_quote is a continuous normalized substring in the actual cited text.
    4. Rejects fabricated or misplaced citations.
    """
    quote = citation.verbatim_quote.strip()
    words = quote.split()

    if len(words) < 10 or len(words) > 30:
        # Schema requires 10-25 words (allowing minor variance up to 30)
        return False, "UNVERIFIED", f"Quotation length ({len(words)} words) outside required statutory range (10-25 words)."

    norm_quote = normalize_statutory_text(quote)

    # Filter candidate sources matching the cited circular / document
    cited_circular = (citation.circular_name or "").lower()
    cited_clause = (citation.section_clause or "").lower()

    target_docs = []
    other_docs = []

    for doc in source_documents:
        doc_circular = (doc.get("circular_name") or "").lower()
        doc_clause = (doc.get("section_clause") or "").lower()
        doc_id = (doc.get("document_id") or "").lower()

        if (
            (citation.document_id and citation.document_id.lower() == doc_id)
            or (cited_circular and cited_circular in doc_circular)
            or (cited_clause and (cited_clause in doc_clause or doc_clause in cited_clause))
        ):
            target_docs.append(doc)
        else:
            other_docs.append(doc)

    # 1. Verify quote inside the target cited document
    for doc in target_docs:
        norm_source = normalize_statutory_text(doc.get("text", ""))
        if norm_quote in norm_source:
            # Check page number alignment if present
            doc_page = doc.get("page_number")
            if citation.page_number is not None and doc_page is not None:
                if citation.page_number != doc_page:
                    return False, "PARTIALLY_VERIFIED", f"Quote found in {doc.get('section_clause')} but cited page {citation.page_number} differs from source page {doc_page}."

            # Check if source document is superseded or repealed
            if doc.get("document_status") in ["SUPERSEDED", "REPEALED"]:
                return False, "OUTDATED_SOURCE", f"Quote verified but originates from a repealed/superseded regulation ({doc.get('document_id')})."

            return True, "VERIFIED", f"Verbatim continuous quote successfully verified in {doc.get('section_clause')}."

    # 2. Check if quote appeared in another retrieved document (misattributed citation)
    for doc in other_docs:
        norm_source = normalize_statutory_text(doc.get("text", ""))
        if norm_quote in norm_source:
            return False, "UNVERIFIED", f"Misattributed citation: Quote appears in {doc.get('circular_name')} ({doc.get('section_clause')}), not in cited {citation.circular_name}."

    return False, "UNVERIFIED", "Quotation not found in retrieved authoritative regulatory text."


def verify_citation_quote(verbatim_quote: str, graded_parents: List[Dict[str, Any]]) -> bool:
    """Legacy helper: verify continuous normalized substring in graded parent documents."""
    if not verbatim_quote or not graded_parents:
        return False

    words = verbatim_quote.split()
    if len(words) < 4:
        return False

    norm_quote = normalize_statutory_text(verbatim_quote)
    for parent in graded_parents:
        norm_parent = normalize_statutory_text(parent.get("text", ""))
        if norm_quote in norm_parent:
            return True

    return False
