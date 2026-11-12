"""Whitespace-Normalized Deterministic Substring & SequenceMatcher Quote Verifier."""

import re
from difflib import SequenceMatcher
from typing import List, Dict, Any


def normalize_text(text: str) -> str:
    """Collapse pipes, markdown symbols, newlines, and multiple spaces into single spaces."""
    return re.sub(r"[\s\|\*\_\#\:\-]+", " ", text).strip().lower()


def verify_citation_quote(verbatim_quote: str, graded_parents: List[Dict[str, Any]]) -> bool:
    """
    Verify that verbatim_quote appears in at least one graded parent document.
    Handles LLM normalization of Markdown table pipes, newlines, and whitespace,
    with a SequenceMatcher token-overlap fallback (>= 0.90) for table rows.
    """
    if not verbatim_quote or not graded_parents:
        return False

    norm_quote = normalize_text(verbatim_quote)
    if len(norm_quote.split()) < 4:
        return False

    for parent in graded_parents:
        parent_text = parent.get("text", "")
        norm_parent = normalize_text(parent_text)

        # 1. Exact normalized substring match
        if norm_quote in norm_parent:
            return True

        # 2. SequenceMatcher token-overlap fallback for table rows and multi-line breaks
        matcher = SequenceMatcher(None, norm_quote, norm_parent)
        match = matcher.find_longest_match(0, len(norm_quote), 0, len(norm_parent))
        if match.size / max(len(norm_quote), 1) >= 0.90:
            return True

    return False
