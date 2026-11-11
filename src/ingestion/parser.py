"""Layout-aware PDF-to-Markdown Extraction Engine using IBM Docling with table preservation."""

import re
from pathlib import Path
from typing import Optional


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
                print(f"[WARNING] Could not initialize full Docling pipeline: {e}")
                self._converter = "FALLBACK"
        return self._converter

    def sanitize_markdown(self, markdown_text: str) -> str:
        """Strip repetitive page headers, footers, and noise while preserving tables and headings."""
        lines = markdown_text.split("\n")
        cleaned_lines = []

        # Patterns for repetitive headers and footers
        header_patterns = [
            re.compile(r"^Reserve Bank of India\s*[-–]\s*Master Direction.*$", re.IGNORECASE),
            re.compile(r"^भारतीय\s+रिज़र्व\s+बैंक.*$", re.IGNORECASE),
            re.compile(r"^RESERVE BANK OF INDIA.*$", re.IGNORECASE),
            re.compile(r"^Page\s+\d+\s+of\s+\d+$", re.IGNORECASE),
            re.compile(r"^\d+\s*$", re.IGNORECASE),  # standalone page number
        ]

        for line in lines:
            stripped = line.strip()
            # Do not strip table rows or markdown headings
            if stripped.startswith("|") or stripped.startswith("#") or stripped.startswith("<!--"):
                cleaned_lines.append(line)
                continue

            # Check if line matches repetitive headers
            is_noise = any(pat.match(stripped) for pat in header_patterns)
            if not is_noise:
                cleaned_lines.append(line)

        # Collapse excess empty lines
        sanitized = "\n".join(cleaned_lines)
        sanitized = re.sub(r"\n{3,}", "\n\n", sanitized)
        return sanitized

    def parse_pdf(self, pdf_path: str | Path, output_md_path: Optional[str | Path] = None) -> str:
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found at: {pdf_path}")

        converter = self._get_converter()

        if converter != "FALLBACK":
            print(f"Parsing {pdf_path.name} with IBM Docling TableFormer...")
            conv_res = converter.convert(str(pdf_path))
            doc = conv_res.document

            # Page-Number Boundary Injection
            # Iterate through Docling items or export markdown with page provenance
            pages_dict: dict[int, list[str]] = {}
            for item, level in doc.iterate_items():
                page_no = 1
                if hasattr(item, "prov") and item.prov:
                    page_no = item.prov[0].page_no
                
                # Check for table or text export
                if page_no not in pages_dict:
                    pages_dict[page_no] = []
                
                # Get item text representation
                if hasattr(item, "export_to_markdown"):
                    pages_dict[page_no].append(item.export_to_markdown())
                elif hasattr(item, "text"):
                    pages_dict[page_no].append(item.text)

            # Assemble with explicit page markers
            assembled_parts = []
            for p_no in sorted(pages_dict.keys()):
                assembled_parts.append(f"\n<!-- PAGE_NUMBER: {p_no} -->\n")
                assembled_parts.append("\n".join(pages_dict[p_no]))

            raw_markdown = "\n\n".join(assembled_parts)
            if not raw_markdown.strip():
                # Fallback to direct export if iterate_items didn't populate
                raw_markdown = doc.export_to_markdown()
        else:
            # Layout-aware fallback using pypdfium2 / pdfplumber if Docling unavailable
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
        """Robust fallback extractor when full Docling neural models are not initialized."""
        import pypdfium2 as pdfium
        pdf = pdfium.PdfDocument(str(pdf_path))
        md_pages = []

        for page_idx, page in enumerate(pdf):
            page_num = page_idx + 1
            textpage = page.get_textpage()
            text = textpage.get_text_range()
            md_pages.append(f"\n<!-- PAGE_NUMBER: {page_num} -->\n\n{text}")

        return "\n\n".join(md_pages)
