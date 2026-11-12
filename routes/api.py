"""REST API routes for ResearchMate."""

import os
import json
import logging
from pathlib import Path
from werkzeug.utils import secure_filename
from flask import Blueprint, request, jsonify, send_from_directory, current_app

from config import Config
from database.db import db
from database.models import (
    Paper,
    PaperChunk,
    PaperSummary,
    LiteratureReview,
    ResearchGap,
    ReferenceItem,
    ReportDraft,
    ResearchHistory,
    Project,
)
from ai.watsonx_client import (
    WatsonxClient,
    WatsonxConfigurationError,
    WatsonxAPIError,
)
from ai.prompts import (
    build_question_analysis_prompt,
    build_paper_summary_prompt,
    build_literature_review_prompt,
    build_research_gap_prompt,
    build_report_draft_prompt,
)
from services.paper_search import AcademicPaperSearchService
from services.pdf_extractor import PDFExtractorService, PDFExtractionError
from services.reference_formatter import ReferenceFormatterService
from services.report_generator import ReportGeneratorService
from rag.chunker import TextChunker
from rag.rag_engine import RAGEngine, global_vector_store

logger = logging.getLogger(__name__)
api_bp = Blueprint("api", __name__, url_prefix="/api")

# Services instantiation
paper_search_service = AcademicPaperSearchService()
pdf_extractor_service = PDFExtractorService()
reference_formatter_service = ReferenceFormatterService()
report_generator_service = ReportGeneratorService()
text_chunker = TextChunker()
rag_engine = RAGEngine(global_vector_store)


def log_history(action_type: str, title: str, summary: str = "", metadata: dict = None):
    """Logs an action to the audit history."""
    try:
        entry = ResearchHistory(
            action_type=action_type,
            title=title,
            summary=summary,
            metadata_json=json.dumps(metadata or {}),
        )
        db.session.add(entry)
        db.session.commit()
    except Exception as e:
        logger.warning(f"Could not log research history: {e}")
        db.session.rollback()


# ==============================================================================
# 0. System & Settings Endpoints
# ==============================================================================


@api_bp.route("/status", methods=["GET"])
def get_status():
    """Returns application status and IBM watsonx configuration."""
    watsonx_status = Config.get_watsonx_status()
    counts = {
        "papers": Paper.query.count(),
        "literature_reviews": LiteratureReview.query.count(),
        "research_gaps": ResearchGap.query.count(),
        "references": ReferenceItem.query.count(),
        "reports": ReportDraft.query.count(),
    }
    return jsonify({"success": True, "watsonx": watsonx_status, "counts": counts})


@api_bp.route("/settings", methods=["POST"])
def update_settings():
    """Updates IBM watsonx.ai credentials in .env file and runtime."""
    data = request.get_json() or {}
    apikey = data.get("apikey", "").strip()
    project_id = data.get("project_id", "").strip()
    url = data.get("url", "https://us-south.ml.cloud.ibm.com").strip()
    model_id = data.get("model_id", "ibm/granite-13b-chat-v2").strip()

    if apikey:
        Config.WATSONX_APIKEY = apikey
    if project_id:
        Config.WATSONX_PROJECT_ID = project_id
    if url:
        Config.WATSONX_URL = url
    if model_id:
        Config.IBM_GRANITE_MODEL_ID = model_id

    # Persist to .env file if writable
    env_path = Config.BASE_DIR / ".env"
    try:
        lines = []
        if env_path.exists():
            lines = env_path.read_text(encoding="utf-8").splitlines()

        updated_keys = set()
        new_lines = []
        for line in lines:
            if line.startswith("WATSONX_APIKEY=") and apikey:
                new_lines.append(f"WATSONX_APIKEY={apikey}")
                updated_keys.add("WATSONX_APIKEY")
            elif line.startswith("WATSONX_PROJECT_ID=") and project_id:
                new_lines.append(f"WATSONX_PROJECT_ID={project_id}")
                updated_keys.add("WATSONX_PROJECT_ID")
            elif line.startswith("WATSONX_URL=") and url:
                new_lines.append(f"WATSONX_URL={url}")
                updated_keys.add("WATSONX_URL")
            elif line.startswith("IBM_GRANITE_MODEL_ID=") and model_id:
                new_lines.append(f"IBM_GRANITE_MODEL_ID={model_id}")
                updated_keys.add("IBM_GRANITE_MODEL_ID")
            else:
                new_lines.append(line)

        # Append any missing keys
        if "WATSONX_APIKEY" not in updated_keys and apikey:
            new_lines.append(f"WATSONX_APIKEY={apikey}")
        if "WATSONX_PROJECT_ID" not in updated_keys and project_id:
            new_lines.append(f"WATSONX_PROJECT_ID={project_id}")
        if "WATSONX_URL" not in updated_keys and url:
            new_lines.append(f"WATSONX_URL={url}")
        if "IBM_GRANITE_MODEL_ID" not in updated_keys and model_id:
            new_lines.append(f"IBM_GRANITE_MODEL_ID={model_id}")

        env_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
        logger.info("Updated .env file with new credentials.")
    except Exception as e:
        logger.warning(f"Could not persist settings to .env file: {e}")

    log_history("UPDATE_SETTINGS", "IBM watsonx.ai settings updated")
    return jsonify(
        {
            "success": True,
            "message": "Settings updated successfully",
            "status": Config.get_watsonx_status(),
        }
    )


@api_bp.route("/settings/test", methods=["POST"])
def test_watsonx():
    """Tests IBM watsonx.ai connection with provided or saved credentials."""
    data = request.get_json() or {}
    apikey = data.get("apikey", "").strip() or Config.WATSONX_APIKEY
    project_id = data.get("project_id", "").strip() or Config.WATSONX_PROJECT_ID
    url = data.get("url", "").strip() or Config.WATSONX_URL
    model_id = data.get("model_id", "").strip() or Config.IBM_GRANITE_MODEL_ID

    client = WatsonxClient(apikey=apikey, project_id=project_id, url=url, model_id=model_id)
    res = client.test_connection()
    return jsonify(res)


# ==============================================================================
# Module 1: Research Question Analyzer
# ==============================================================================


@api_bp.route("/analyze-question", methods=["POST"])
def analyze_question():
    """Module 1: Analyzes research topic using IBM Granite."""
    data = request.get_json() or {}
    topic = data.get("topic", "").strip()
    background = data.get("background", "").strip()

    if not topic:
        return jsonify({"success": False, "error": "Research topic or question is required."}), 400

    client = WatsonxClient()
    if not client.is_configured():
        return jsonify(
            {
                "success": False,
                "setup_required": True,
                "error": "IBM watsonx.ai credentials are required to use IBM Granite for Research Question Analysis.",
                "message": "Please configure your WATSONX_APIKEY and WATSONX_PROJECT_ID in the Settings page.",
            }
        ), 400

    prompt = build_question_analysis_prompt(topic=topic, background_context=background)

    try:
        granite_res = client.generate(
            prompt=prompt,
            max_new_tokens=1400,
            temperature=0.3,
        )
        output_text = granite_res.get("generated_text", "")

        log_history(
            "QUESTION_ANALYSIS",
            f"Analyzed topic: {topic[:60]}",
            summary=output_text[:150],
            metadata={"topic": topic, "model": granite_res.get("model_id")},
        )

        return jsonify(
            {
                "success": True,
                "topic": topic,
                "analysis": output_text,
                "model_used": granite_res.get("model_id"),
                "tokens": {
                    "input": granite_res.get("input_token_count", 0),
                    "output": granite_res.get("generated_token_count", 0),
                },
            }
        )

    except (WatsonxConfigurationError, WatsonxAPIError) as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ==============================================================================
# Module 2: Academic Paper Search (OpenAlex)
# ==============================================================================


@api_bp.route("/search-papers", methods=["GET"])
def search_papers():
    """Module 2: Searches legitimate academic papers from OpenAlex."""
    query = request.args.get("query", "").strip()
    from_year = request.args.get("from_year", type=int)
    to_year = request.args.get("to_year", type=int)
    page = request.args.get("page", 1, type=int)

    if not query:
        return jsonify({"success": False, "error": "Search query parameter is required."}), 400

    result = paper_search_service.search_papers(
        query=query, from_year=from_year, to_year=to_year, page=page
    )

    log_history(
        "PAPER_SEARCH",
        f"Searched: {query[:60]}",
        summary=f"Found {result.get('total_count', 0)} papers",
        metadata={"query": query, "filters": {"from_year": from_year, "to_year": to_year}},
    )

    return jsonify({"success": True, "data": result})


@api_bp.route("/save-paper", methods=["POST"])
def save_paper():
    """Saves a paper from OpenAlex or manual entry into database."""
    data = request.get_json() or {}
    title = data.get("title", "").strip()

    if not title:
        return jsonify({"success": False, "error": "Paper title is required."}), 400

    # Check if already saved
    existing = Paper.query.filter_by(title=title).first()
    if existing:
        return jsonify(
            {"success": True, "message": "Paper already exists in your library.", "paper": existing.to_dict()}
        )

    year_val = data.get("year")
    try:
        year_int = int(year_val) if year_val and str(year_val).isdigit() else None
    except Exception:
        year_int = None

    paper = Paper(
        title=title,
        authors=data.get("authors", "Unknown Authors"),
        year=year_int,
        abstract=data.get("abstract", ""),
        doi=data.get("doi", ""),
        url=data.get("url", ""),
        journal=data.get("journal", "Academic Publication"),
        source=data.get("source", "openalex"),
    )
    db.session.add(paper)
    db.session.commit()

    # Automatically create a ReferenceItem for it as well
    ref_formatted = reference_formatter_service.format_all(data)
    ref_item = ReferenceItem(
        paper_id=paper.id,
        title=title,
        authors=paper.authors,
        year=paper.year,
        journal=paper.journal,
        doi=paper.doi,
        url=paper.url,
        citation_apa=ref_formatted["apa"],
        citation_ieee=ref_formatted["ieee"],
        bibtex=ref_formatted["bibtex"],
    )
    db.session.add(ref_item)
    db.session.commit()

    log_history("SAVE_PAPER", f"Saved paper: {title[:60]}")
    return jsonify({"success": True, "message": "Paper saved to library", "paper": paper.to_dict()})


# ==============================================================================
# Module 3: Research Paper PDF Analyzer
# ==============================================================================


@api_bp.route("/upload-pdf", methods=["POST"])
def upload_pdf():
    """Module 3: Uploads PDF, extracts text using PyMuPDF, chunks, and indexes in vector store."""
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file part in upload request."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"success": False, "error": "No file selected."}), 400

    if not file.filename.lower().endswith(".pdf"):
        return jsonify({"success": False, "error": "Only genuine PDF documents are supported."}), 400

    upload_folder = Path(Config.UPLOAD_FOLDER)
    upload_folder.mkdir(parents=True, exist_ok=True)

    filename = secure_filename(file.filename)
    # Prefix timestamp to prevent collisions
    safe_name = f"{int(os.times().system * 1000)}_{filename}"
    file_path = upload_folder / safe_name
    file.save(str(file_path))

    # Extract text with PyMuPDF
    try:
        extracted = pdf_extractor_service.extract_document(str(file_path))
    except PDFExtractionError as e:
        if file_path.exists():
            file_path.unlink()
        return jsonify({"success": False, "error": str(e)}), 400

    # Save Paper record
    paper = Paper(
        title=extracted.get("title") or filename,
        authors=extracted.get("author") or "Unknown Author",
        source="pdf_upload",
        pdf_filename=filename,
        pdf_path=str(file_path),
        full_text=extracted.get("full_text", ""),
        page_count=extracted.get("page_count", 1),
    )
    db.session.add(paper)
    db.session.commit()

    # Chunk text for RAG
    chunks_data = text_chunker.chunk_page_texts(extracted.get("pages", []), paper_id=paper.id)

    db_chunks = []
    store_chunks = []
    for c in chunks_data:
        chunk_rec = PaperChunk(
            paper_id=paper.id,
            chunk_index=c["chunk_index"],
            content=c["content"],
            page_number=c["page_number"],
            section_hint=c["section_hint"],
        )
        db_chunks.append(chunk_rec)

        store_chunks.append(
            {
                "paper_id": paper.id,
                "paper_title": paper.title,
                "chunk_index": c["chunk_index"],
                "content": c["content"],
                "page_number": c["page_number"],
                "section_hint": c["section_hint"],
            }
        )

    db.session.bulk_save_objects(db_chunks)
    db.session.commit()

    # Add to RAG vector store
    global_vector_store.add_chunks(store_chunks)

    # Add reference item
    ref_formatted = reference_formatter_service.format_all(
        {
            "title": paper.title,
            "authors": paper.authors,
            "year": paper.year or 2024,
            "journal": "PDF Document",
        }
    )
    ref_item = ReferenceItem(
        paper_id=paper.id,
        title=paper.title,
        authors=paper.authors,
        year=paper.year,
        journal="Uploaded Research Paper",
        citation_apa=ref_formatted["apa"],
        citation_ieee=ref_formatted["ieee"],
        bibtex=ref_formatted["bibtex"],
    )
    db.session.add(ref_item)
    db.session.commit()

    log_history(
        "PDF_UPLOAD",
        f"Uploaded PDF: {paper.title[:60]}",
        summary=f"Extracted {extracted['page_count']} pages, created {len(db_chunks)} chunks.",
    )

    return jsonify(
        {
            "success": True,
            "message": "PDF uploaded and processed successfully",
            "paper": paper.to_dict(),
            "stats": {
                "pages": extracted["page_count"],
                "words": extracted["total_words"],
                "chunks_created": len(db_chunks),
            },
        }
    )


@api_bp.route("/summarize-paper/<int:paper_id>", methods=["POST"])
def summarize_paper(paper_id):
    """Module 3: Generates 6-section IBM Granite summary for a paper."""
    paper = Paper.query.get_or_404(paper_id)

    # Check if summary already exists
    if paper.summary:
        return jsonify(
            {
                "success": True,
                "message": "Retrieved existing summary",
                "summary": paper.summary.to_dict(),
                "model_used": paper.summary.model_used,
            }
        )

    client = WatsonxClient()
    if not client.is_configured():
        return jsonify(
            {
                "success": False,
                "setup_required": True,
                "error": "IBM watsonx.ai credentials are required to summarize papers using IBM Granite.",
                "message": "Configure your WATSONX_APIKEY and WATSONX_PROJECT_ID in Settings.",
            }
        ), 400

    # Build prompt with paper text
    paper_content = paper.full_text or paper.abstract
    if not paper_content:
        return jsonify({"success": False, "error": "This paper contains no extractable text or abstract."}), 400

    prompt = build_paper_summary_prompt(
        paper_title=paper.title,
        authors=paper.authors or "Unknown",
        paper_text=paper_content,
    )

    try:
        granite_res = client.generate(
            prompt=prompt,
            max_new_tokens=1500,
            temperature=0.2,
        )
        raw_text = granite_res.get("generated_text", "")

        # Parse sections
        parsed = _parse_summary_sections(raw_text)

        summary_rec = PaperSummary(
            paper_id=paper.id,
            research_objectives=parsed.get("objectives", ""),
            abstract_summary=parsed.get("abstract", ""),
            methodology=parsed.get("methodology", ""),
            key_findings=parsed.get("findings", ""),
            limitations=parsed.get("limitations", ""),
            future_directions=parsed.get("future_directions", ""),
            raw_response=raw_text,
            model_used=granite_res.get("model_id"),
        )
        paper.is_analyzed = True
        db.session.add(summary_rec)
        db.session.commit()

        log_history(
            "PAPER_SUMMARY",
            f"Summarized: {paper.title[:60]}",
            metadata={"paper_id": paper.id, "model": granite_res.get("model_id")},
        )

        return jsonify(
            {
                "success": True,
                "message": "Summary generated successfully with IBM Granite",
                "summary": summary_rec.to_dict(),
                "model_used": granite_res.get("model_id"),
                "tokens": {
                    "input": granite_res.get("input_token_count", 0),
                    "output": granite_res.get("generated_token_count", 0),
                },
            }
        )

    except (WatsonxConfigurationError, WatsonxAPIError) as e:
        return jsonify({"success": False, "error": str(e)}), 500


def _parse_summary_sections(text: str) -> dict:
    """Parses standard 6 sections from Granite summary response."""
    sections = {
        "objectives": "",
        "abstract": "",
        "methodology": "",
        "findings": "",
        "limitations": "",
        "future_directions": "",
    }
    # Pattern for ### 1. Research Objectives, etc.
    parts = re.split(r"###\s*\d*\.?\s*", text)
    for p in parts:
        p_strip = p.strip()
        if not p_strip:
            continue
        first_line = p_strip.split("\n")[0].lower()
        content = "\n".join(p_strip.split("\n")[1:]).strip()

        if "objective" in first_line:
            sections["objectives"] = content
        elif "abstract" in first_line:
            sections["abstract"] = content
        elif "method" in first_line:
            sections["methodology"] = content
        elif "finding" in first_line or "result" in first_line:
            sections["findings"] = content
        elif "limitation" in first_line:
            sections["limitations"] = content
        elif "future" in first_line or "direction" in first_line:
            sections["future_directions"] = content

    # Fallback if markdown split didn't find specific headers
    if not any(sections.values()):
        sections["abstract"] = text

    return sections


# ==============================================================================
# Module 4: Literature Review Assistant
# ==============================================================================


@api_bp.route("/literature-review", methods=["POST"])
def generate_literature_review():
    """Module 4: Generates structured multi-paper synthesis using IBM Granite."""
    data = request.get_json() or {}
    topic = data.get("topic", "").strip()
    paper_ids = data.get("paper_ids", [])

    if not topic:
        return jsonify({"success": False, "error": "Research topic is required."}), 400

    if not paper_ids or len(paper_ids) < 1:
        return jsonify({"success": False, "error": "Please select at least 1 research paper to synthesize."}), 400

    client = WatsonxClient()
    if not client.is_configured():
        return jsonify(
            {
                "success": False,
                "setup_required": True,
                "error": "IBM watsonx.ai credentials are required to synthesize literature reviews.",
                "message": "Configure your WATSONX_APIKEY and WATSONX_PROJECT_ID in Settings.",
            }
        ), 400

    # Retrieve selected papers
    papers = Paper.query.filter(Paper.id.in_(paper_ids)).all()
    papers_data = []
    for p in papers:
        content = ""
        if p.summary:
            content = f"Objectives: {p.summary.research_objectives}\nMethodology: {p.summary.methodology}\nFindings: {p.summary.key_findings}\nLimitations: {p.summary.limitations}"
        elif p.full_text:
            content = p.full_text[:3000]
        else:
            content = p.abstract or ""

        papers_data.append(
            {
                "id": p.id,
                "title": p.title,
                "authors": p.authors,
                "year": p.year,
                "content": content,
            }
        )

    prompt = build_literature_review_prompt(topic=topic, papers_data=papers_data)

    try:
        granite_res = client.generate(
            prompt=prompt,
            max_new_tokens=1800,
            temperature=0.3,
        )
        review_text = granite_res.get("generated_text", "")

        # Save to database
        lit_rec = LiteratureReview(
            title=f"Literature Review: {topic[:100]}",
            topic=topic,
            paper_ids=json.dumps(paper_ids),
            synthesized_review=review_text,
            model_used=granite_res.get("model_id"),
        )
        db.session.add(lit_rec)
        db.session.commit()

        log_history(
            "LITERATURE_REVIEW",
            f"Review generated on: {topic[:60]}",
            summary=review_text[:150],
            metadata={"paper_ids": paper_ids, "model": granite_res.get("model_id")},
        )

        return jsonify(
            {
                "success": True,
                "review": lit_rec.to_dict(),
                "model_used": granite_res.get("model_id"),
                "tokens": {
                    "input": granite_res.get("input_token_count", 0),
                    "output": granite_res.get("generated_token_count", 0),
                },
            }
        )

    except (WatsonxConfigurationError, WatsonxAPIError) as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ==============================================================================
# Module 5: Research Gap and Hypothesis Generator
# ==============================================================================


@api_bp.route("/research-gaps", methods=["POST"])
def generate_research_gaps():
    """Module 5: Analyzes papers to find evidence-grounded research gaps and testable hypotheses."""
    data = request.get_json() or {}
    topic = data.get("topic", "").strip()
    paper_ids = data.get("paper_ids", [])

    if not topic:
        return jsonify({"success": False, "error": "Research topic is required."}), 400

    if not paper_ids:
        return jsonify({"success": False, "error": "Select at least 1 paper to inspect for gaps."}), 400

    client = WatsonxClient()
    if not client.is_configured():
        return jsonify(
            {
                "success": False,
                "setup_required": True,
                "error": "IBM watsonx.ai credentials are required to analyze research gaps with IBM Granite.",
                "message": "Configure your WATSONX_APIKEY and WATSONX_PROJECT_ID in Settings.",
            }
        ), 400

    papers = Paper.query.filter(Paper.id.in_(paper_ids)).all()
    papers_data = []
    for p in papers:
        content = ""
        if p.summary:
            content = f"Findings: {p.summary.key_findings}\nLimitations: {p.summary.limitations}\nFuture Work: {p.summary.future_directions}"
        else:
            content = p.full_text[:3000] if p.full_text else (p.abstract or "")

        papers_data.append(
            {"id": p.id, "title": p.title, "year": p.year, "content": content}
        )

    prompt = build_research_gap_prompt(topic=topic, papers_data=papers_data)

    try:
        granite_res = client.generate(
            prompt=prompt,
            max_new_tokens=1600,
            temperature=0.3,
        )
        analysis_text = granite_res.get("generated_text", "")

        gap_rec = ResearchGap(
            topic=topic,
            paper_ids=json.dumps(paper_ids),
            gaps_json=json.dumps([{"raw": analysis_text}]),
            hypotheses_json=json.dumps([]),
            model_used=granite_res.get("model_id"),
        )
        db.session.add(gap_rec)
        db.session.commit()

        log_history(
            "RESEARCH_GAPS",
            f"Gap analysis on: {topic[:60]}",
            metadata={"paper_ids": paper_ids, "model": granite_res.get("model_id")},
        )

        return jsonify(
            {
                "success": True,
                "topic": topic,
                "analysis": analysis_text,
                "model_used": granite_res.get("model_id"),
                "tokens": {
                    "input": granite_res.get("input_token_count", 0),
                    "output": granite_res.get("generated_token_count", 0),
                },
            }
        )

    except (WatsonxConfigurationError, WatsonxAPIError) as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ==============================================================================
# Module 6: Reference Manager
# ==============================================================================


@api_bp.route("/references", methods=["GET"])
def get_references():
    """Module 6: Retrieves organized references."""
    refs = ReferenceItem.query.order_by(ReferenceItem.created_at.desc()).all()
    return jsonify({"success": True, "references": [r.to_dict() for r in refs]})


@api_bp.route("/references", methods=["POST"])
def add_reference():
    """Module 6: Adds a new reference item and generates APA/IEEE/BibTeX."""
    data = request.get_json() or {}
    title = data.get("title", "").strip()

    if not title:
        return jsonify({"success": False, "error": "Title is required for reference."}), 400

    formatted = reference_formatter_service.format_all(data)

    ref = ReferenceItem(
        title=title,
        authors=data.get("authors", ""),
        year=data.get("year"),
        journal=data.get("journal", ""),
        volume=data.get("volume", ""),
        issue=data.get("issue", ""),
        pages=data.get("pages", ""),
        doi=data.get("doi", ""),
        url=data.get("url", ""),
        citation_apa=formatted["apa"],
        citation_ieee=formatted["ieee"],
        bibtex=formatted["bibtex"],
    )
    db.session.add(ref)
    db.session.commit()

    log_history("ADD_REFERENCE", f"Added citation: {title[:60]}")
    return jsonify({"success": True, "reference": ref.to_dict()})


@api_bp.route("/references/export", methods=["GET"])
def export_references():
    """Module 6: Exports all references in requested format (apa, ieee, bibtex, ris)."""
    fmt = request.args.get("format", "bibtex").lower()
    refs = ReferenceItem.query.order_by(ReferenceItem.year.desc()).all()

    if not refs:
        return jsonify({"success": False, "error": "No references available to export."}), 400

    output_lines = []
    if fmt == "apa":
        for r in refs:
            output_lines.append(r.citation_apa or r.title)
        content = "\n\n".join(output_lines)
        ext = "txt"
    elif fmt == "ieee":
        for idx, r in enumerate(refs, 1):
            ieee_line = reference_formatter_service.format_ieee(r.to_dict(), citation_num=idx)
            output_lines.append(ieee_line)
        content = "\n\n".join(output_lines)
        ext = "txt"
    elif fmt == "ris":
        for r in refs:
            ris_entry = reference_formatter_service.format_ris(r.to_dict())
            output_lines.append(ris_entry)
        content = "\n\n".join(output_lines)
        ext = "ris"
    else:  # bibtex default
        for r in refs:
            bib_entry = r.bibtex or reference_formatter_service.format_bibtex(r.to_dict())
            output_lines.append(bib_entry)
        content = "\n\n".join(output_lines)
        ext = "bib"

    # Save to export folder
    export_dir = Path(Config.EXPORT_FOLDER)
    export_dir.mkdir(parents=True, exist_ok=True)
    filename = f"references_{fmt}.{ext}"
    file_path = export_dir / filename
    file_path.write_text(content, encoding="utf-8")

    return jsonify(
        {
            "success": True,
            "format": fmt,
            "content": content,
            "filename": filename,
            "download_url": f"/api/download-export/{filename}",
        }
    )


@api_bp.route("/references/<int:ref_id>", methods=["DELETE"])
def delete_reference(ref_id):
    """Deletes a reference item."""
    ref = ReferenceItem.query.get_or_404(ref_id)
    db.session.delete(ref)
    db.session.commit()
    return jsonify({"success": True, "message": "Reference deleted"})


# ==============================================================================
# Module 7: Research Report and Paper Draft Generator
# ==============================================================================


@api_bp.route("/generate-report", methods=["POST"])
def generate_report():
    """Module 7: Generates complete 13-section academic research paper draft using IBM Granite."""
    data = request.get_json() or {}
    topic = data.get("topic", "").strip()
    lit_review_id = data.get("literature_review_id")
    paper_ids = data.get("paper_ids", [])

    if not topic:
        return jsonify({"success": False, "error": "Research topic is required."}), 400

    client = WatsonxClient()
    if not client.is_configured():
        return jsonify(
            {
                "success": False,
                "setup_required": True,
                "error": "IBM watsonx.ai credentials are required to generate paper drafts with IBM Granite.",
                "message": "Configure your WATSONX_APIKEY and WATSONX_PROJECT_ID in Settings.",
            }
        ), 400

    # Retrieve context
    lit_text = ""
    if lit_review_id:
        rev = LiteratureReview.query.get(lit_review_id)
        if rev:
            lit_text = rev.synthesized_review

    papers_summary_text = ""
    if paper_ids:
        papers = Paper.query.filter(Paper.id.in_(paper_ids)).all()
        summaries = []
        for p in papers:
            if p.summary:
                summaries.append(f"{p.title}: {p.summary.key_findings}")
            elif p.abstract:
                summaries.append(f"{p.title}: {p.abstract[:300]}")
        papers_summary_text = "\n".join(summaries)

    prompt = build_report_draft_prompt(
        topic=topic,
        lit_review=lit_text,
        papers_summary=papers_summary_text,
    )

    try:
        granite_res = client.generate(
            prompt=prompt,
            max_new_tokens=2200,
            temperature=0.35,
        )
        markdown_text = granite_res.get("generated_text", "")

        # Parse into 13 sections
        parsed_sections = report_generator_service.parse_markdown_to_sections(markdown_text)

        draft = ReportDraft(
            title=f"Research Draft: {topic[:100]}",
            topic=topic,
            sections_json=json.dumps(parsed_sections),
            full_markdown=markdown_text,
            model_used=granite_res.get("model_id"),
        )
        db.session.add(draft)
        db.session.commit()

        log_history(
            "REPORT_GENERATION",
            f"Paper draft generated: {topic[:60]}",
            metadata={"draft_id": draft.id, "model": granite_res.get("model_id")},
        )

        return jsonify(
            {
                "success": True,
                "draft": draft.to_dict(),
                "model_used": granite_res.get("model_id"),
                "tokens": {
                    "input": granite_res.get("input_token_count", 0),
                    "output": granite_res.get("generated_token_count", 0),
                },
            }
        )

    except (WatsonxConfigurationError, WatsonxAPIError) as e:
        return jsonify({"success": False, "error": str(e)}), 500


@api_bp.route("/export-report", methods=["POST"])
def export_report():
    """Exports a draft report in Markdown, HTML, or Plain Text."""
    data = request.get_json() or {}
    report_id = data.get("report_id")
    export_format = data.get("format", "markdown").lower()
    custom_content = data.get("content")

    if report_id:
        draft = ReportDraft.query.get_or_404(report_id)
        title = draft.title
        topic = draft.topic
        content = custom_content or draft.full_markdown
    else:
        title = data.get("title", "Research Draft")
        topic = data.get("topic", "Academic Research")
        content = custom_content or ""

    if not content:
        return jsonify({"success": False, "error": "No content provided to export."}), 400

    export_result = report_generator_service.export_report(
        title=title, topic=topic, markdown_content=content, export_format=export_format
    )

    return jsonify(
        {
            "success": True,
            "filename": export_result["filename"],
            "download_url": f"/api/download-export/{export_result['filename']}",
            "format": export_format,
        }
    )


@api_bp.route("/download-export/<path:filename>", methods=["GET"])
def download_export(filename):
    """Serves generated export files securely."""
    export_dir = Path(Config.EXPORT_FOLDER).resolve()
    return send_from_directory(
        directory=str(export_dir), path=filename, as_attachment=True
    )


# ==============================================================================
# RAG: Grounded Question Answering
# ==============================================================================


@api_bp.route("/rag/query", methods=["POST"])
def rag_query():
    """RAG grounded question answering across indexed research papers."""
    data = request.get_json() or {}
    question = data.get("question", "").strip()
    paper_ids = data.get("paper_ids")
    top_k = data.get("top_k", 5)

    if not question:
        return jsonify({"success": False, "error": "Question is required."}), 400

    # Ensure vector store has chunks from database if currently empty
    if not global_vector_store.chunks:
        all_chunks = PaperChunk.query.all()
        if all_chunks:
            chunk_dicts = [
                {
                    "paper_id": c.paper_id,
                    "paper_title": c.paper.title if c.paper else "Document",
                    "chunk_index": c.chunk_index,
                    "content": c.content,
                    "page_number": c.page_number,
                    "section_hint": c.section_hint,
                }
                for c in all_chunks
            ]
            global_vector_store.set_chunks(chunk_dicts)

    rag_result = rag_engine.query(
        question=question, top_k=top_k, paper_ids=paper_ids
    )

    log_history("RAG_QUERY", f"RAG Query: {question[:60]}")
    return jsonify({"success": True, "result": rag_result})


# ==============================================================================
# Paper Management & History Endpoints
# ==============================================================================


@api_bp.route("/papers", methods=["GET"])
def list_papers():
    """List all saved and uploaded academic papers."""
    papers = Paper.query.order_by(Paper.created_at.desc()).all()
    return jsonify({"success": True, "papers": [p.to_dict() for p in papers]})


@api_bp.route("/papers/<int:paper_id>", methods=["GET"])
def get_paper(paper_id):
    """Get single paper with summary and chunk stats."""
    paper = Paper.query.get_or_404(paper_id)
    data = paper.to_dict(include_text=True)
    if paper.summary:
        data["summary"] = paper.summary.to_dict()
    data["chunks_count"] = len(paper.chunks)
    return jsonify({"success": True, "paper": data})


@api_bp.route("/papers/<int:paper_id>", methods=["DELETE"])
def delete_paper(paper_id):
    """Deletes paper and associated chunks, summary, and PDF file."""
    paper = Paper.query.get_or_404(paper_id)

    # Delete PDF file if exists
    if paper.pdf_path and os.path.exists(paper.pdf_path):
        try:
            os.remove(paper.pdf_path)
        except Exception as e:
            logger.warning(f"Could not remove PDF file: {e}")

    db.session.delete(paper)
    db.session.commit()

    # Rebuild vector store
    remaining_chunks = PaperChunk.query.all()
    chunk_dicts = [
        {
            "paper_id": c.paper_id,
            "paper_title": c.paper.title if c.paper else "Document",
            "chunk_index": c.chunk_index,
            "content": c.content,
            "page_number": c.page_number,
            "section_hint": c.section_hint,
        }
        for c in remaining_chunks
    ]
    global_vector_store.set_chunks(chunk_dicts)

    return jsonify({"success": True, "message": "Paper and chunks deleted successfully"})


@api_bp.route("/history", methods=["GET"])
def get_history():
    """Get audit trail of research history."""
    entries = ResearchHistory.query.order_by(ResearchHistory.created_at.desc()).limit(100).all()
    return jsonify({"success": True, "history": [e.to_dict() for e in entries]})


@api_bp.route("/history", methods=["DELETE"])
def clear_history():
    """Clear research history."""
    ResearchHistory.query.delete()
    db.session.commit()
    return jsonify({"success": True, "message": "Research history cleared"})
