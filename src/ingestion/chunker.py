"""Two-Tier Hierarchical Parent-Child Chunker with Regulatory Metadata & Provenance Preservation."""

import json
import re
import uuid
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import tiktoken
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

CLAUSE_REGEX = re.compile(
    r"(?:Section|Para(?:graph)?|Clause|Annex(?:ure)?|Chapter|Schedule)\s*[\dIVX]+(?:\.[\d]+)*(?:\s*\([a-zA-Z0-9]+\))*|\bClause\s+\d+\b",
    re.IGNORECASE,
)

PAGE_MARKER_REGEX = re.compile(r"<!--\s*PAGE_NUMBER:\s*(\d+)\s*-->")
DOC_METADATA_REGEX = re.compile(r"<!--\s*DOCUMENT_METADATA:\s*(\{.*?\})\s*-->")


def extract_table_header(text: str) -> Optional[str]:
    """Extract table header and separator rows if text contains a Markdown table."""
    lines = text.strip().split("\n")
    for i in range(len(lines) - 1):
        line1 = lines[i].strip()
        line2 = lines[i + 1].strip()
        if line1.startswith("|") and line1.endswith("|") and line2.startswith("|") and line2.endswith("|"):
            if "---" in line2 or "--" in line2:
                return f"{line1}\n{line2}"
    return None


def is_inside_table(text: str) -> bool:
    """Check if the text represents or contains Markdown table rows."""
    lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
    table_lines = [line for line in lines if line.startswith("|") and line.endswith("|")]
    return len(table_lines) >= 2


def extract_section_clause(header_text: str, body_lead: str) -> str:
    """Extract statutory clause using CLAUSE_REGEX from header or body lead-in."""
    candidate = f"{header_text} {body_lead[:250]}"
    match = CLAUSE_REGEX.search(candidate)
    if match:
        return match.group(0).strip()
    first_line = header_text.strip().lstrip("#").strip()
    return first_line if first_line else "General Regulatory Provisions"


class TwoTierChunker:
    def __init__(
        self,
        parent_token_limit: int = 1500,
        child_token_size: int = 256,
        child_token_overlap: int = 32,
    ):
        self.parent_token_limit = parent_token_limit
        self.child_token_size = child_token_size
        self.child_token_overlap = child_token_overlap
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

        self.header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "h1"),
                ("##", "h2"),
                ("###", "h3"),
                ("####", "h4"),
            ],
            strip_headers=False,
        )

        self.child_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=self.child_token_size,
            chunk_overlap=self.child_token_overlap,
        )

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def split_parent_section(self, section_text: str) -> List[str]:
        """Sub-split section exceeding parent token limit without slicing tables mid-row."""
        if self.count_tokens(section_text) <= self.parent_token_limit:
            return [section_text]

        paragraphs = section_text.split("\n\n")
        chunks = []
        current_chunk = []
        current_tokens = 0
        in_table_block = False

        for para in paragraphs:
            para_tokens = self.count_tokens(para)
            is_table = is_inside_table(para)

            if (current_tokens + para_tokens > self.parent_token_limit) and current_chunk and not in_table_block:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = [para]
                current_tokens = para_tokens
            else:
                current_chunk.append(para)
                current_tokens += para_tokens

            if is_table:
                in_table_block = True
            elif in_table_block and not is_table:
                in_table_block = False

        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        return chunks

    def process_document(
        self,
        markdown_text: str,
        doc_category: str,
        doc_metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Process a single Markdown document into Parent chunks (docstore) and Child chunks.
        Preserves true page provenance, statutory section breadcrumbs, and regulatory versioning.
        """
        # Parse embedded document metadata if present
        meta_dict = dict(doc_metadata or {})
        doc_meta_match = DOC_METADATA_REGEX.search(markdown_text)
        if doc_meta_match:
            try:
                embedded = json.loads(doc_meta_match.group(1))
                meta_dict.update(embedded)
            except Exception:
                pass

        doc_id = meta_dict.get("document_id", f"RBI_{doc_category}")
        circular_name = meta_dict.get("title") or meta_dict.get("circular_name") or f"RBI {doc_category} Direction"
        circular_number = meta_dict.get("circular_number", "")
        document_status = meta_dict.get("status") or meta_dict.get("document_status", "ACTIVE")
        effective_date = meta_dict.get("effective_date") or meta_dict.get("date")
        if effective_date:
            from datetime import date
            try:
                effective_date = date.fromisoformat(str(effective_date)[:10]).isoformat()
            except ValueError:
                effective_date = None
        doc_version = meta_dict.get("version", "1.0")

        # Split into structural sections by heading
        header_sections = self.header_splitter.split_text(markdown_text)

        parents: Dict[str, Dict[str, Any]] = {}
        children: List[Dict[str, Any]] = []

        parent_idx = 0
        current_page: Optional[int] = None

        for section in header_sections:
            content = section.page_content

            # Detect real page marker if present in markdown
            page_matches = PAGE_MARKER_REGEX.findall(content)
            if page_matches:
                current_page = int(page_matches[-1])

            # Extract header hierarchy breadcrumb
            h1 = section.metadata.get("h1", "")
            h2 = section.metadata.get("h2", "")
            h3 = section.metadata.get("h3", "")
            breadcrumb = " > ".join(filter(None, [h1, h2, h3])) or "General Provisions"

            # Sub-split oversized parent sections safely
            sub_sections = self.split_parent_section(content)

            for sub_text in sub_sections:
                parent_idx += 1
                if current_page and current_page > 0:
                    page_num = int(current_page)
                    page_tag = f"_P{page_num:03d}"
                else:
                    page_num = None
                    page_tag = "_Punknown"
                parent_id = f"{doc_id}{page_tag}_S{parent_idx:04d}"

                has_table = is_inside_table(sub_text)
                table_header = extract_table_header(sub_text) if has_table else None
                section_clause = extract_section_clause(breadcrumb, sub_text)

                parent_record = {
                    "parent_id": parent_id,
                    "document_id": doc_id,
                    "text": sub_text,
                    "circular_name": circular_name,
                    "circular_number": circular_number,
                    "doc_category": doc_category,
                    "section_clause": section_clause,
                    "source_location": breadcrumb,
                    "page_number": page_num,
                    "document_status": document_status,
                    "effective_date": effective_date,
                    "document_version": doc_version,
                    "has_table": has_table,
                }
                parents[parent_id] = parent_record

                # Tier 2: Generate Child Chunks
                child_texts = self.child_splitter.split_text(sub_text)
                for child_idx, child_text in enumerate(child_texts):
                    final_child_text = child_text
                    if has_table and table_header and "|" in child_text:
                        if table_header.split("\n")[0] not in child_text:
                            final_child_text = f"[{breadcrumb}]\n{table_header}\n{child_text}"
                        else:
                            final_child_text = f"[{breadcrumb}]\n{child_text}"
                    else:
                        final_child_text = f"[{breadcrumb} | {section_clause}]\n{child_text}"

                    child_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{parent_id}_child_{child_idx}"))

                    child_record = {
                        "child_id": child_id,
                        "parent_id": parent_id,
                        "document_id": doc_id,
                        "text": final_child_text,
                        "circular_name": circular_name,
                        "circular_number": circular_number,
                        "doc_category": doc_category,
                        "section_clause": section_clause,
                        "source_location": breadcrumb,
                        "page_number": page_num,
                        "document_status": document_status,
                        "effective_date": effective_date,
                        "document_version": doc_version,
                        "has_table": has_table,
                        "child_index": child_idx,
                    }
                    children.append(child_record)

        return parents, children

    def save_parent_docstore(self, parents: Dict[str, Dict[str, Any]], filepath: str = "data/parent_docstore.json"):
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_name(path.name + ".tmp")
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(parents, f, indent=2, ensure_ascii=False)
            f.flush()
            import os
            os.fsync(f.fileno())
        os.replace(temp_path, path)
        print(f"[SUCCESS] Saved {len(parents)} parent sections to {path}")
