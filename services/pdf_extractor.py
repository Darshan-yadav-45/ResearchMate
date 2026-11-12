"""PDF extraction service using PyMuPDF for academic papers."""

import re
import logging
from pathlib import Path
from typing import Dict, Any, List

import pymupdf

logger = logging.getLogger(__name__)


class PDFExtractionError(Exception):
    """Raised when PDF extraction fails or document is unreadable."""
    pass


class PDFExtractorService:
    """Extracts text, metadata, and page structures from academic PDF files."""

    def extract_document(self, file_path: str) -> Dict[str, Any]:
        """Extracts text, metadata, and per-page content from a PDF file.

        Args:
            file_path: Absolute or relative path to PDF.

        Returns:
            Dict with metadata, full_text, pages (list of {page_number, text}), and page_count.
        """
        path = Path(file_path)
        if not path.exists():
            raise PDFExtractionError(f"PDF file not found at: {file_path}")

        try:
            doc = pymupdf.open(file_path)
        except Exception as e:
            logger.error(f"PyMuPDF failed to open PDF {file_path}: {e}")
            raise PDFExtractionError(f"Failed to open PDF document: {str(e)}")

        try:
            if doc.is_encrypted:
                raise PDFExtractionError(
                    "The uploaded PDF is password-protected and cannot be analyzed."
                )

            page_count = len(doc)
            if page_count == 0:
                raise PDFExtractionError("The PDF document contains 0 pages.")

            # Extract standard PDF metadata
            meta = doc.metadata or {}
            pdf_title = meta.get("title") or path.stem.replace("_", " ").replace("-", " ")
            pdf_author = meta.get("author") or "Unknown Author"

            pages_data: List[Dict[str, Any]] = []
            full_text_parts: List[str] = []

            for page_idx in range(page_count):
                page = doc[page_idx]
                page_num = page_idx + 1
                raw_text = page.get_text("text") or ""
                cleaned_text = self._clean_text(raw_text)

                pages_data.append(
                    {
                        "page_number": page_num,
                        "text": cleaned_text,
                        "char_count": len(cleaned_text),
                    }
                )
                if cleaned_text:
                    full_text_parts.append(
                        f"--- PAGE {page_num} ---\n{cleaned_text}"
                    )

            full_text = "\n\n".join(full_text_parts)

            if len(full_text.strip()) < 50:
                logger.warning(
                    f"PDF {file_path} yielded very little text ({len(full_text)} chars). Likely scanned or image-only."
                )

            # Heuristically detect title from first page if PDF metadata title is generic
            refined_title = self._detect_title_from_first_page(
                pages_data[0]["text"] if pages_data else "", pdf_title
            )

            return {
                "title": refined_title,
                "author": pdf_author,
                "page_count": page_count,
                "pages": pages_data,
                "full_text": full_text,
                "total_chars": len(full_text),
                "total_words": len(full_text.split()),
            }

        except Exception as e:
            if isinstance(e, PDFExtractionError):
                raise e
            logger.error(f"Error parsing PDF content: {e}")
            raise PDFExtractionError(f"Error extracting text from PDF: {str(e)}")
        finally:
            doc.close()

    def _clean_text(self, text: str) -> str:
        """Cleans and standardizes extracted academic text."""
        if not text:
            return ""

        # Normalize carriage returns and weird whitespace
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Reconnect words split across lines by hyphens (e.g. "meth-\nodology" -> "methodology")
        text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)

        # Replace excessive whitespace while keeping paragraph breaks
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()

    def _detect_title_from_first_page(self, first_page_text: str, default_title: str) -> str:
        """Heuristically extracts paper title from the top of the first page."""
        if not first_page_text:
            return default_title

        lines = [line.strip() for line in first_page_text.split("\n") if line.strip()]
        # Skip common headers like "IEEE TRANSACTIONS", "arXiv:...", "Vol. 12"
        candidate_lines = []
        for line in lines[:10]:
            if re.search(r"(?i)^(arxiv|proceedings|journal|volume|issue|issn|isbn|doi)", line):
                continue
            if len(line) > 15:
                candidate_lines.append(line)
            if len(" ".join(candidate_lines)) > 150:
                break

        if candidate_lines:
            potential_title = " ".join(candidate_lines[:2])
            # If default_title is generic or filename-like, use candidate
            if (
                len(default_title) < 5
                or default_title.endswith(".pdf")
                or "untitled" in default_title.lower()
            ):
                return potential_title

        return default_title
