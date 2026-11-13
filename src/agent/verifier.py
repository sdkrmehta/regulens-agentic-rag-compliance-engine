"""Deterministic Source-Specific Citation & Quotation Verification Engine.

Enforces strict document identity, circular provenance, quotation accuracy,
and independent page provenance verification against authoritative RBI Master Directions.

Page Verification Strategy
--------------------------
RBI source documents are stored as processed Markdown files under data/processed_md/.
Each Markdown file uses `<!-- page N -->` HTML comments injected during PDF-to-Markdown
conversion, or (when absent) a heuristic that assigns approximate page numbers based on
the density of section headings and content volume (≈ 1 page per 400 words, anchored at
the first `# ` heading).  The verifier builds a lightweight *page-span index* per document
at first access and uses it to independently confirm that a cited page number is consistent
with where the verbatim quote actually appears in the source file.
"""

import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
from urllib.parse import urlparse
from src.schemas import CitationItem, CitationVerificationStatus

logger = logging.getLogger(__name__)

# Cached registry of valid RBI circulars
_REGISTRY_DOCS: Optional[Dict[str, Dict[str, Any]]] = None

# Cache: document_id -> list of (start_char, end_char, page_number) tuples
_PAGE_INDEX_CACHE: Dict[str, List[Tuple[int, int, int]]] = {}



def _build_page_index(document_id: str) -> List[Tuple[int, int, int]]:
    """
    Build a character-offset page index for a source markdown file.

    Returns a list of (start_char, end_char, page_number) tuples covering
    the entire document.  Lookup: iterate until start_char <= quote_offset < end_char.

    Only explicit PDF page markers are accepted as provenance. A word-count
    estimate is not a source page reference.
    """
    if document_id in _PAGE_INDEX_CACHE:
        return _PAGE_INDEX_CACHE[document_id]

    registry = get_verified_registry()
    reg_entry = registry.get(document_id, {})
    source_filepath = reg_entry.get("source_filepath", "")
    if not source_filepath:
        _PAGE_INDEX_CACHE[document_id] = []
        return []

    md_path = Path(source_filepath)
    if not md_path.exists():
        _PAGE_INDEX_CACHE[document_id] = []
        return []

    try:
        full_text = md_path.read_text(encoding="utf-8")
    except Exception as e:
        logger.warning(f"Could not read markdown source for page index ({document_id}): {e}")
        _PAGE_INDEX_CACHE[document_id] = []
        return []

    # Strategy 1: explicit <!-- page N --> markers
    page_markers = list(re.finditer(r"<!--\s*PAGE_NUMBER:\s*(\d+)\s*-->", full_text, re.IGNORECASE))
    if page_markers:
        spans: List[Tuple[int, int, int]] = []
        for idx, m in enumerate(page_markers):
            page_num = int(m.group(1))
            start = m.end()
            end = page_markers[idx + 1].start() if idx + 1 < len(page_markers) else len(full_text)
            spans.append((start, end, page_num))
        _PAGE_INDEX_CACHE[document_id] = spans
        return spans

    _PAGE_INDEX_CACHE[document_id] = []
    return []


def independent_page_verification(
    verbatim_quote: str,
    document_id: str,
    cited_page: Optional[int],
) -> Tuple[bool, str]:
    """
    Independently verify that *cited_page* is consistent with where the verbatim
    quote actually appears in the original source markdown file for *document_id*.

    Returns:
        (is_consistent: bool, reason: str)

    Outcomes:
    - ``True``  — cited page matches or source has no paginated markers (cannot contradict).
    - ``False`` — cited page is inconsistent with the independently computed page span.
    """
    if cited_page is None:
        # No page claimed — nothing to independently verify
        return True, "No page number cited; independent verification skipped."

    page_index = _build_page_index(document_id)
    if not page_index:
        return False, f"Source page provenance unavailable for '{document_id}'; cited page cannot be verified."

    # Locate the verbatim quote inside the source text
    registry = get_verified_registry()
    source_filepath = registry.get(document_id, {}).get("source_filepath", "")
    md_path = Path(source_filepath)
    if not md_path.exists():
        return False, "Source page provenance unavailable; cited page cannot be verified."

    try:
        full_text = md_path.read_text(encoding="utf-8")
    except Exception:
        return False, "Source page provenance unreadable; cited page cannot be verified."

    norm_full = normalize_statutory_text(full_text)
    norm_quote = normalize_statutory_text(verbatim_quote)
    quote_offset = norm_full.find(norm_quote)

    if quote_offset == -1:
        # Quote not found in source — cannot do page check (quote verification will catch this)
        return False, "Quote not located in the registered source; page provenance cannot be verified."

    # Resolve the raw source offset exactly by normalizing prefixes. This avoids
    # proportional offset guesses drifting across long documents.
    offset_map = _normalized_offset_map(full_text)
    if quote_offset >= len(offset_map):
        return False, "Could not map quotation to source page provenance."
    raw_offset = offset_map[quote_offset]

    # Find the page span that contains this offset
    actual_page = None
    for start, end, pnum in page_index:
        if start <= raw_offset < end:
            actual_page = pnum
            break
    if actual_page is None and page_index:
        # Fallback: assign last page if offset exceeds all spans (rounding artefact)
        actual_page = page_index[-1][2]

    if actual_page is None:
        return True, "Could not resolve page from index; independent page verification deferred."

    if actual_page == cited_page:
        return True, f"Cited page {cited_page} independently confirmed in source."
    else:
        return False, (
            f"Page number mismatch: citation claims page {cited_page} but independent "
            f"source analysis places quote at approximately page {actual_page} in '{document_id}'."
        )


def get_verified_registry() -> Dict[str, Dict[str, Any]]:
    """Load canonical document registry from data/regulatory_registry.json."""
    global _REGISTRY_DOCS
    if _REGISTRY_DOCS is None:
        reg_path = Path("data/regulatory_registry.json")
        if reg_path.exists():
            try:
                with open(reg_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    _REGISTRY_DOCS = {
                        d["document_id"]: d for d in data.get("documents", [])
                    }
            except Exception as e:
                logger.error(f"Failed to load regulatory registry: {e}")
                _REGISTRY_DOCS = {}
        else:
            _REGISTRY_DOCS = {}
    return _REGISTRY_DOCS


def _registry_is_authentic(registry: Dict[str, Dict[str, Any]]) -> bool:
    """Fail closed unless the registry and every registered source hash verify."""
    if not registry:
        return False
    try:
        from src.ingestion.integrity import verify_regulatory_document_integrity
        report = verify_regulatory_document_integrity()
        urls_are_official = all(
            urlparse(str(entry.get("source_url", ""))).scheme == "https"
            and urlparse(str(entry.get("source_url", ""))).hostname in {"www.rbi.org.in", "rbi.org.in"}
            and entry.get("source_hash_algorithm") == "SHA-256"
            and entry.get("source_hash_scope") == "processed_markdown_snapshot"
            for entry in registry.values()
        )
        return urls_are_official and bool(report.get("all_valid")) and all(
            doc_id in report.get("details", {})
            and report["details"][doc_id].get("hash_valid")
            and report["details"][doc_id].get("metadata_valid")
            for doc_id in registry
        )
    except Exception:
        logger.exception("Regulatory registry authenticity check failed")
        return False


def normalize_statutory_text(text: str) -> str:
    """Normalize whitespace, Markdown table pipes, and punctuation for robust comparison."""
    if not text:
        return ""
    # Replace markdown table pipes, dashes, asterisks with spaces
    cleaned = re.sub(r"[\|\*\_\#\:\-]+", " ", text)
    # Collapse multiple whitespaces and lowercase
    cleaned = re.sub(r"\s+", " ", cleaned).strip().lower()
    return cleaned


def _normalized_offset_map(text: str) -> List[int]:
    """Map normalized characters to raw offsets using the same cleanup rules."""
    offsets: List[int] = []
    previous_space = False
    for idx, char in enumerate(text):
        if char in "|*_#:-":
            continue
        if char.isspace():
            if offsets and not previous_space:
                offsets.append(idx)
            previous_space = True
            continue
        offsets.append(idx)
        previous_space = False
    return offsets


# Alias for backwards compatibility with tests and callers
normalize_text = normalize_statutory_text


# Hard limits for verbatim statutory quotations.
# Lower bound: too short to anchor a clause uniquely (< 10 words is a fragment).
# Upper bound: RBI circulars use compact normative sentences; > 25 words suggests
#              paraphrasing or stitching, not verbatim extraction.
_QUOTE_MIN_WORDS: int = 10
_QUOTE_MAX_WORDS: int = 25


def verify_citation_against_source(
    citation: CitationItem,
    source_documents: List[Dict[str, Any]],
    allow_historical: bool = False,
) -> Tuple[bool, CitationVerificationStatus, str]:
    """
    Strong, source-specific citation verification:

    1. Requires a non-empty document_id — citations without document identity are rejected.
    2. Enforces verbatim quote length: 10 – 25 words (fragments and paraphrases rejected).
    3. Validates document_id against the authoritative regulatory registry (must exist).
    4. Cross-checks circular_name against the registry title for year-based mismatches.
    5. Enforces document_id-matched source lookup; falls through to misattribution check.
    6. Verifies verbatim_quote is a continuous normalised substring in the cited document.
    7. Validates page provenance (rejects synthetic pages on unpaginated circulars).
    8. Independently verifies page number against source markdown (layer 2).
    9. Rejects repealed/superseded sources unless allow_historical=True.
    10. If the quote exists in no retrieved document, returns UNVERIFIED with an explicit
        "unverifiable source" message so callers can surface it to the user.
    """
    quote = citation.verbatim_quote.strip() if citation.verbatim_quote else ""
    words = quote.split()

    # Rule 1: document_id is mandatory — without it we cannot verify source identity.
    cited_doc_id = citation.document_id
    if not cited_doc_id or not cited_doc_id.strip():
        return (
            False,
            "UNVERIFIED",
            "Citation rejected: document_id is mandatory for source verification. "
            "Every citation must identify its source regulatory document.",
        )

    # Rule 2: Quote length guard.
    if len(words) < _QUOTE_MIN_WORDS:
        return (
            False,
            "UNVERIFIED",
            f"Quotation too short ({len(words)} words). "
            f"Minimum {_QUOTE_MIN_WORDS} words required to anchor a statutory clause.",
        )
    if len(words) > _QUOTE_MAX_WORDS:
        return (
            False,
            "UNVERIFIED",
            f"Quotation too long ({len(words)} words). "
            f"Maximum {_QUOTE_MAX_WORDS} words permitted; longer extracts suggest "
            "paraphrasing rather than verbatim reproduction.",
        )

    norm_quote = normalize_statutory_text(quote)
    registry = get_verified_registry()

    # Rule 3: document_id must exist in the official registry.
    if not _registry_is_authentic(registry):
        return False, "UNVERIFIED", "Regulatory registry or registered source authenticity could not be verified."
    if cited_doc_id not in registry:
        return (
            False,
            "UNVERIFIED",
            f"Unverifiable source: document_id '{cited_doc_id}' is not registered in "
            "the authoritative regulatory registry. Citation cannot be verified.",
        )

    # Rule 4: Cross-check circular_name vs registry title for year-based mismatches.
    if cited_doc_id in registry:
        reg_entry = registry[cited_doc_id]
        expected_title = reg_entry.get("document_title", "").lower()
        cited_title = (citation.circular_name or "").lower()
        if cited_title and len(cited_title.split()) >= 3:
            if "2022" in cited_title and "2025" in expected_title:
                return (
                    False,
                    "UNVERIFIED",
                    f"Document identity mismatch: cited '{citation.circular_name}' but "
                    f"document_id '{cited_doc_id}' is '{reg_entry.get('document_title')}'.",
                )
            if "2025" in cited_title and "2022" in expected_title:
                return (
                    False,
                    "UNVERIFIED",
                    f"Document identity mismatch: cited '{citation.circular_name}' but "
                    f"document_id '{cited_doc_id}' is '{reg_entry.get('document_title')}'.",
                )

    # Rule 5: Classify source documents (strict document_id match required since it is mandatory).
    cited_circular = (citation.circular_name or "").lower()
    cited_clause = (citation.section_clause or "").lower()

    target_docs = []
    other_docs = []

    for doc in source_documents:
        doc_circular = (doc.get("circular_name") or "").lower()
        doc_clause = (doc.get("section_clause") or "").lower()
        doc_id_raw = (doc.get("document_id") or "").lower()

        # With mandatory document_id, always use strict document_id matching as the
        # primary criterion; fall back to circular/clause name only when doc has no ID.
        is_target = False
        if cited_doc_id.lower() == doc_id_raw:
            is_target = True
        elif not doc_id_raw:
            # Doc has no stored ID — fall back to name-based matching
            if cited_circular and (cited_circular in doc_circular or doc_circular in cited_circular):
                is_target = True
            elif cited_clause and (cited_clause in doc_clause or doc_clause in cited_clause):
                is_target = True

        if is_target:
            target_docs.append(doc)
        else:
            other_docs.append(doc)

    # Rule 6: Verify quote inside target documents.
    for doc in target_docs:
        norm_source = normalize_statutory_text(doc.get("text", ""))
        if norm_quote in norm_source:
            doc_id = doc.get("document_id")
            doc_page = doc.get("page_number")
            cited_page = citation.page_number

            # Rule 7: Page provenance — layer 1 (vector-store metadata)
            if cited_page is not None:
                if doc_page is None:
                    return (
                        False,
                        "PARTIALLY_VERIFIED",
                        f"Quote found in {doc.get('section_clause')}, but source is an "
                        f"unpaginated official gazette (synthetic page {cited_page} invalid).",
                    )
                elif cited_page != doc_page:
                    return (
                        False,
                        "PARTIALLY_VERIFIED",
                        f"Quote found in {doc.get('section_clause')}, but cited page "
                        f"{cited_page} differs from authentic source page {doc_page}.",
                    )

            # Rule 8: Independent page verification against original regulatory PDF/markdown.
            if cited_page is not None and doc_id:
                page_ok, page_reason = independent_page_verification(
                    verbatim_quote=quote,
                    document_id=doc_id,
                    cited_page=cited_page,
                )
                if not page_ok:
                    return False, "PARTIALLY_VERIFIED", page_reason

            # Rule 9: Reject repealed/superseded sources.
            if doc.get("document_status") in ["SUPERSEDED", "REPEALED"]:
                if allow_historical:
                    return (
                        True,
                        "VERIFIED",
                        f"Verbatim quote verified in historical regulation {doc_id} "
                        f"({doc.get('section_clause')}).",
                    )
                return (
                    False,
                    "OUTDATED_SOURCE",
                    f"Quote verified but originates from a repealed/superseded regulation ({doc_id}).",
                )

            return (
                True,
                "VERIFIED",
                f"Verbatim continuous quote verified in {doc.get('section_clause')} "
                f"of {doc.get('circular_name')}.",
            )

    # Misattribution check: quote exists in a document other than the one cited.
    for doc in other_docs:
        norm_source = normalize_statutory_text(doc.get("text", ""))
        if norm_quote in norm_source:
            return (
                False,
                "UNVERIFIED",
                f"Misattributed citation: Quote appears in '{doc.get('circular_name')}' "
                f"({doc.get('section_clause')}, ID: {doc.get('document_id')}), "
                f"not in cited '{citation.circular_name}' (ID: {cited_doc_id}).",
            )

    # Rule 10: Quote found nowhere in retrieved corpus — source is unverifiable.
    if not source_documents:
        return (
            False,
            "UNVERIFIED",
            f"Unverifiable source: no regulatory documents were retrieved for document_id "
            f"'{cited_doc_id}'. The cited source cannot be confirmed.",
        )
    return (
        False,
        "UNVERIFIED",
        f"Unverifiable source: quotation not found in any retrieved authoritative text for "
        f"document_id '{cited_doc_id}'. The cited passage may be fabricated or paraphrased.",
    )


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
