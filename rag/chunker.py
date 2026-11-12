"""Academic document text chunking module."""

import re
from typing import List, Dict, Any


class TextChunker:
    """Splits academic text into overlapping chunks while preserving page numbers and sections."""

    def __init__(self, chunk_size: int = 900, chunk_overlap: int = 150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_page_texts(
        self, pages_data: List[Dict[str, Any]], paper_id: int = 0
    ) -> List[Dict[str, Any]]:
        """Splits page-by-page extracted text into indexed chunks with page attribution.

        Args:
            pages_data: List of dicts with 'page_number' and 'text'.
            paper_id: ID of the paper being processed.

        Returns:
            List of chunk dicts: {paper_id, chunk_index, content, page_number, section_hint}
        """
        chunks = []
        chunk_idx = 0

        for page in pages_data:
            page_num = page.get("page_number", 1)
            text = page.get("text", "").strip()

            if not text:
                continue

            # Detect section headings in page text if available
            current_section = self._detect_section(text)

            # Split text by paragraphs / sentences
            page_chunks = self._split_text(text)

            for chunk_content in page_chunks:
                if len(chunk_content.strip()) < 40:
                    continue  # Ignore tiny headers/footers

                chunks.append(
                    {
                        "paper_id": paper_id,
                        "chunk_index": chunk_idx,
                        "content": chunk_content.strip(),
                        "page_number": page_num,
                        "section_hint": current_section,
                    }
                )
                chunk_idx += 1

        return chunks

    def _split_text(self, text: str) -> List[str]:
        """Splits a single text block into overlapping chunks."""
        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = start + self.chunk_size
            if end >= text_len:
                chunks.append(text[start:])
                break

            # Try to break at paragraph or sentence boundary
            breakpoint = text.rfind("\n\n", start, end)
            if breakpoint == -1 or breakpoint <= start + 100:
                breakpoint = text.rfind(". ", start, end)
                if breakpoint != -1:
                    breakpoint += 1
            if breakpoint == -1 or breakpoint <= start + 100:
                breakpoint = text.rfind("\n", start, end)
            if breakpoint == -1 or breakpoint <= start + 100:
                breakpoint = text.rfind(" ", start, end)

            if breakpoint == -1 or breakpoint <= start:
                breakpoint = end

            chunk = text[start:breakpoint].strip()
            if chunk:
                chunks.append(chunk)

            start = max(start + 1, breakpoint - self.chunk_overlap)

        return chunks

    def _detect_section(self, text: str) -> str:
        """Heuristic check for common academic paper section headers."""
        patterns = [
            (r"(?i)\babstract\b", "Abstract"),
            (r"(?i)\b1\.?\s*introduction\b", "Introduction"),
            (r"(?i)\b2\.?\s*(related\s+work|literature\s+review)\b", "Related Work"),
            (r"(?i)\b3\.?\s*(methodology|methods|proposed\s+approach|system\s+design)\b", "Methodology"),
            (r"(?i)\b4\.?\s*(experiments?|evaluation|results?)\b", "Results"),
            (r"(?i)\b5\.?\s*(discussion|limitations)\b", "Discussion"),
            (r"(?i)\b6\.?\s*(conclusion|concluding\s+remarks)\b", "Conclusion"),
            (r"(?i)\breferences\b", "References"),
        ]
        for pattern, section_name in patterns:
            if re.search(pattern, text[:400]):
                return section_name
        return "Body"
