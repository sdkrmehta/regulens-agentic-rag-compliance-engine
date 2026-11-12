"""RBI Gazette HTML-to-Markdown Converter with Table Preservation and Genuine Structural Provenance."""

import re
from pathlib import Path
from bs4 import BeautifulSoup, Tag

CLAUSE_HEADER_RE = re.compile(
    r"^(?:(?:Section|Para(?:graph)?|Clause|Chapter|Annex(?:ure)?|Schedule)\s*[\dIVX]+(?:\.[\d]+)*(?:\s*\([a-zA-Z0-9]+\))*|\d+\.\s+[A-Za-z]|\d+\.\d+\s+[A-Za-z]|\([a-z]\)\s+[A-Za-z])",
    re.IGNORECASE,
)


def table_to_markdown(table_tag: Tag) -> str:
    """Convert an HTML table to a clean Markdown pipe table preserving all columns and headers."""
    rows = []
    max_cols = 0

    tr_tags = table_tag.find_all("tr")
    for tr in tr_tags:
        cells = tr.find_all(["th", "td"])
        if not cells:
            continue
        row = [cell.get_text(separator=" ", strip=True).replace("|", "\\|").replace("\n", " ") for cell in cells]
        if any(row):
            rows.append(row)
            if len(row) > max_cols:
                max_cols = len(row)

    if not rows or max_cols == 0:
        return ""

    normalized_rows = []
    for row in rows:
        if len(row) < max_cols:
            row = row + [""] * (max_cols - len(row))
        normalized_rows.append(row)

    md_lines = []
    header = normalized_rows[0]
    md_lines.append("| " + " | ".join(header) + " |")
    md_lines.append("| " + " | ".join(["---"] * max_cols) + " |")
    for r in normalized_rows[1:]:
        md_lines.append("| " + " | ".join(r) + " |")

    return "\n" + "\n".join(md_lines) + "\n"


def convert_rbi_html_to_markdown(html_path: Path, doc_category: str) -> str:
    """
    Parse RBI Gazette HTML document into structured Markdown.
    Preserves headings, clause hierarchies, and multi-row tables without artificial character-based page splitting.
    """
    with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f, "html.parser")

    container = soup.find("table", class_="tablebg")
    if not container:
        container = soup.find("div", id="accordion") or soup.body

    if not container:
        return ""

    # Remove superscript footnotes to prevent breaking numbered clause headers (e.g., '18. 62REs' -> '18. REs')
    for sup in container.find_all("sup"):
        sup.decompose()

    blocks = []
    current_section = 1

    for el in container.find_all(["h1", "h2", "h3", "h4", "p", "table"]):
        if el.name in ["h1", "h2", "h3", "h4"]:
            text = el.get_text(strip=True)
            if text:
                level = el.name
                prefix = "#" if level == "h1" else "##" if level == "h2" else "###"
                blocks.append(f"\n\n{prefix} {text}\n\n")
        elif el.name == "table" and el.get("class") != ["tablebg"]:
            # Only convert leaf tables
            if not el.find("table"):
                tbl_md = table_to_markdown(el)
                if tbl_md.strip():
                    blocks.append(f"\n\n{tbl_md}\n\n")
        elif el.name == "p":
            if el.find("table"):
                continue
            text = el.get_text(separator=" ", strip=True)
            if text and len(text) > 5:
                is_clause_header = bool(CLAUSE_HEADER_RE.match(text))
                if not is_clause_header and el.get("class") and "head" in el.get("class"):
                    is_clause_header = True

                if is_clause_header:
                    blocks.append(f"\n\n### {text}\n\n")
                else:
                    blocks.append(f"\n\n{text}\n\n")

    return "".join(blocks).strip()
