"""Unit tests for academic reference and citation formatting."""

import pytest
from services.reference_formatter import ReferenceFormatterService


def test_apa_formatting():
    service = ReferenceFormatterService()
    ref_data = {
        "title": "Attention Is All You Need",
        "authors": "Ashish Vaswani, Noam Shazeer, Niki Parmar",
        "year": 2017,
        "journal": "Advances in Neural Information Processing Systems",
        "volume": "30",
        "pages": "5998-6008",
        "doi": "10.5555/3295222.3295349",
    }
    apa = service.format_apa(ref_data)
    assert "Vaswani, A." in apa
    assert "(2017)" in apa
    assert "Attention Is All You Need." in apa
    assert "*Advances in Neural Information Processing Systems*" in apa
    assert "https://doi.org/10.5555/3295222.3295349" in apa


def test_ieee_formatting():
    service = ReferenceFormatterService()
    ref_data = {
        "title": "Deep Residual Learning for Image Recognition",
        "authors": "Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun",
        "year": 2016,
        "journal": "IEEE Conference on Computer Vision and Pattern Recognition",
        "pages": "770-778",
        "doi": "10.1109/CVPR.2016.90",
    }
    ieee = service.format_ieee(ref_data, citation_num=1)
    assert "[1]" in ieee
    assert '"Deep Residual Learning for Image Recognition,"' in ieee
    assert "2016" in ieee
    assert "doi: 10.1109/CVPR.2016.90" in ieee


def test_bibtex_formatting():
    service = ReferenceFormatterService()
    ref_data = {
        "title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
        "authors": "Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova",
        "year": 2019,
        "journal": "NAACL-HLT",
        "doi": "10.18653/v1/N19-1423",
    }
    bib = service.format_bibtex(ref_data)
    assert "@article{" in bib
    assert "author    =" in bib
    assert "Jacob Devlin and Ming-Wei Chang" in bib
    assert "title     = {{BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding}}" in bib
    assert "year      = {2019}" in bib


def test_ris_formatting():
    service = ReferenceFormatterService()
    ref_data = {
        "title": "Test RIS Publication",
        "authors": "John Doe, Jane Smith",
        "year": 2023,
        "journal": "Nature Machine Intelligence",
        "volume": "5",
        "issue": "2",
        "pages": "100-110",
        "doi": "10.1038/s42256-023-00000-0",
    }
    ris = service.format_ris(ref_data)
    assert "TY  - JOUR" in ris
    assert "TI  - Test RIS Publication" in ris
    assert "AU  - John Doe" in ris
    assert "PY  - 2023" in ris
    assert "SP  - 100" in ris
    assert "EP  - 110" in ris
    assert "ER  - " in ris
