"""RBI Gazette HTML-to-Markdown Converter with Table Preservation and Page Marker Injection."""

import re
from pathlib import Path
from bs4 import BeautifulSoup, Tag


CLAUSE_HEADER_RE = re.compile(
    r"^(?:(?:Section|Para(?:graph)?|Clause|Chapter|Annex(?:ure)?|Schedule)\s*[\dIVX]+(?:\.[\d]+)*(?:\s*\([a-zA-Z0-9]+\))*|\d+\.\s+[A-Z]|\d+\.\d+\s+[A-Z])",
    re.IGNORECASE,
)


def table_to_markdown(table_tag: Tag) -> str:
    """Convert an HTML table to a clean Markdown pipe table."""
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
    with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f, "html.parser")

    container = soup.find("table", class_="tablebg")
    if not container:
        container = soup.find("div", id="accordion") or soup.body

    blocks = []
    chars_per_page = 2200
    current_char_count = 0
    current_page = 1

    blocks.append(f"<!-- PAGE_NUMBER: {current_page} -->\n")

    # Iterate through child paragraphs and tables
    for el in container.find_all(["h1", "h2", "h3", "h4", "p", "table"]):
        if el.name in ["h1", "h2", "h3", "h4"]:
            text = el.get_text(strip=True)
            if text:
                level = el.name
                prefix = "#" if level == "h1" else "##" if level == "h2" else "###"
                block = f"\n\n{prefix} {text}\n\n"
                blocks.append(block)
                current_char_count += len(block)
        elif el.name == "table" and el.get("class") != ["tablebg"]:
            # If not nested
            if not el.find("table"):
                tbl_md = table_to_markdown(el)
                if tbl_md.strip():
                    blocks.append(f"\n\n{tbl_md}\n\n")
                    current_char_count += len(tbl_md)
        elif el.name == "p":
            if el.find("table"):
                continue
            text = el.get_text(separator=" ", strip=True)
            if text and len(text) > 5:
                # If matches statutory clause header
                if CLAUSE_HEADER_RE.match(text):
                    block = f"\n\n### {text}\n\n"
                else:
                    block = f"\n\n{text}\n\n"
                blocks.append(block)
                current_char_count += len(block)

        # Page-number boundary injection
        if current_char_count >= chars_per_page:
            current_page += 1
            blocks.append(f"\n<!-- PAGE_NUMBER: {current_page} -->\n")
            current_char_count = 0

    full_md = "".join(blocks)
    full_md = re.sub(r"\n{3,}", "\n\n", full_md)
    return full_md
