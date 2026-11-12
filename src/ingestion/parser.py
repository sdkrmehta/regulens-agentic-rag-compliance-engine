"""Layout-aware Document Parsing Engine supporting IBM Docling, PyPDFium2, and Structured HTML."""

import re
from pathlib import Path
from typing import Optional, Dict, Any, List


class RBIDocumentParser:
    def __init__(self, do_ocr: bool = False):
        self.do_ocr = do_ocr
        self._converter = None

    def _get_converter(self):
        if self._converter is None:
            try:
                from docling.document_converter import DocumentConverter, PdfFormatOption
                from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
                from docling.datamodel.base_models import InputFormat

                pipeline_options = PdfPipelineOptions()
                pipeline_options.do_table_structure = True
                pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
                pipeline_options.do_ocr = self.do_ocr

                self._converter = DocumentConverter(
                    format_options={
                        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                    }
                )
            except Exception as e:
                print(f"[INFO] Docling neural engine not initialized ({e}); utilizing high-fidelity layout fallback.")
                self._converter = "FALLBACK"
        return self._converter

    def sanitize_markdown(self, markdown_text: str) -> str:
        """Strip repetitive running headers, footers, and noise while strictly preserving tables and headings."""
        lines = markdown_text.split("\n")
        cleaned_lines = []

        header_patterns = [
            re.compile(r"^Reserve Bank of India\s*[-–]\s*Master Direction.*$", re.IGNORECASE),
            re.compile(r"^भारतीय\s+रिज़र्व\s+बैंक.*$", re.IGNORECASE),
            re.compile(r"^RESERVE BANK OF INDIA.*$", re.IGNORECASE),
            re.compile(r"^Page\s+\d+\s+of\s+\d+$", re.IGNORECASE),
            re.compile(r"^\d+\s*$", re.IGNORECASE),
        ]

        for line in lines:
            stripped = line.strip()
            # Preserve table rows, markdown headings, and structural markers
            if stripped.startswith("|") or stripped.startswith("#") or stripped.startswith("<!--"):
                cleaned_lines.append(line)
                continue

            is_noise = any(pat.match(stripped) for pat in header_patterns)
            if not is_noise:
                cleaned_lines.append(line)

        sanitized = "\n".join(cleaned_lines)
        sanitized = re.sub(r"\n{3,}", "\n\n", sanitized)
        return sanitized

    def parse_pdf(self, pdf_path: str | Path, output_md_path: Optional[str | Path] = None) -> str:
        """Parse a PDF document with genuine page provenance tracking."""
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found at: {pdf_path}")

        converter = self._get_converter()

        if converter != "FALLBACK":
            try:
                print(f"[DOCLING] Parsing {pdf_path.name} with IBM Docling TableFormer...")
                conv_res = converter.convert(str(pdf_path))
                doc = conv_res.document

                pages_dict: Dict[int, List[str]] = {}
                for item, level in doc.iterate_items():
                    page_no = 1
                    if hasattr(item, "prov") and item.prov:
                        page_no = item.prov[0].page_no

                    if page_no not in pages_dict:
                        pages_dict[page_no] = []

                    if hasattr(item, "export_to_markdown"):
                        pages_dict[page_no].append(item.export_to_markdown())
                    elif hasattr(item, "text"):
                        pages_dict[page_no].append(item.text)

                assembled_parts = []
                for p_no in sorted(pages_dict.keys()):
                    assembled_parts.append(f"\n<!-- PAGE_NUMBER: {p_no} -->\n")
                    assembled_parts.append("\n".join(pages_dict[p_no]))

                raw_markdown = "\n\n".join(assembled_parts)
                if not raw_markdown.strip():
                    raw_markdown = doc.export_to_markdown()
            except Exception as e:
                print(f"[WARNING] Docling conversion encountered error ({e}); falling back to PyPDFium2 layout parser.")
                raw_markdown = self._fallback_parse_pdf(pdf_path)
        else:
            raw_markdown = self._fallback_parse_pdf(pdf_path)

        sanitized_md = self.sanitize_markdown(raw_markdown)

        if output_md_path:
            out_p = Path(output_md_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            with open(out_p, "w", encoding="utf-8") as f:
                f.write(sanitized_md)
            print(f"[SUCCESS] Exported processed markdown to: {out_p}")

        return sanitized_md

    def _fallback_parse_pdf(self, pdf_path: Path) -> str:
        """Extract PDF text with true 1-based page boundary injection using PyPDFium2."""
        import pypdfium2 as pdfium
        pdf = pdfium.PdfDocument(str(pdf_path))
        md_pages = []

        for page_idx, page in enumerate(pdf):
            page_num = page_idx + 1
            textpage = page.get_textpage()
            text = textpage.get_text_range()
            md_pages.append(f"\n<!-- PAGE_NUMBER: {page_num} -->\n\n{text}")

        return "\n\n".join(md_pages)

    def parse_document(
        self,
        file_path: str | Path,
        doc_category: str,
        output_md_path: Optional[str | Path] = None
    ) -> str:
        """Unified parser entrypoint handling PDF and HTML formats with provenance."""
        path = Path(file_path)
        suffix = path.suffix.lower()

        if suffix == ".pdf":
            return self.parse_pdf(path, output_md_path)
        elif suffix in [".html", ".htm"]:
            from src.ingestion.html_to_markdown import convert_rbi_html_to_markdown
            md_text = convert_rbi_html_to_markdown(path, doc_category)
            sanitized_md = self.sanitize_markdown(md_text)
            if output_md_path:
                out_p = Path(output_md_path)
                out_p.parent.mkdir(parents=True, exist_ok=True)
                with open(out_p, "w", encoding="utf-8") as f:
                    f.write(sanitized_md)
            return sanitized_md
        elif suffix == ".md":
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            return self.sanitize_markdown(text)
        else:
            raise ValueError(f"Unsupported document format: {suffix}")
