"""Unit tests for PyMuPDF PDF extraction and text cleaning."""

import os
import pymupdf
import pytest
from services.pdf_extractor import PDFExtractorService, PDFExtractionError


@pytest.fixture
def sample_pdf(tmp_path):
    """Creates a sample multi-page academic PDF file for testing."""
    pdf_path = tmp_path / "test_paper.pdf"
    doc = pymupdf.open()

    # Page 1: Abstract & Intro
    page1 = doc.new_page()
    page1.insert_text(
        (50, 72),
        "Efficient Deep Learning on Embedded Microcontrollers\n"
        "Abstract\n"
        "We propose an adaptive pruning algorithm that reduces memory footprint by 45%.\n"
        "1. Introduction\n"
        "Modern deep neural networks require significant memory bandwidth.",
        fontsize=11,
    )

    # Page 2: Methodology & Results
    page2 = doc.new_page()
    page2.insert_text(
        (50, 72),
        "3. Methodology\n"
        "Our method quantizes weights to int8.\n"
        "4. Experimental Results\n"
        "Evaluation on CIFAR-10 demonstrates 92.4% accuracy with 3.2x speedup.",
        fontsize=11,
    )

    doc.set_metadata({"title": "Efficient Deep Learning on Embedded Microcontrollers", "author": "Dr. Jane Smith"})
    doc.save(str(pdf_path))
    doc.close()

    return str(pdf_path)


def test_pdf_extraction(sample_pdf):
    extractor = PDFExtractorService()
    result = extractor.extract_document(sample_pdf)

    assert result["page_count"] == 2
    assert "Efficient Deep Learning" in result["title"]
    assert len(result["pages"]) == 2
    assert "Abstract" in result["pages"][0]["text"]
    assert "Methodology" in result["pages"][1]["text"]
    assert result["total_words"] > 20


def test_pdf_extraction_non_existent():
    extractor = PDFExtractorService()
    with pytest.raises(PDFExtractionError):
        extractor.extract_document("non_existent_file.pdf")
