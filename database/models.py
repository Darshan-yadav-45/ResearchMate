"""Database models for ResearchMate."""

import json
from datetime import datetime
from database.db import db


class Project(db.Model):
    """Represents a research project workspace."""

    __tablename__ = "projects"

    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False, default="Default Research Project")
    topic = db.Column(db.Text, nullable=True)
    description = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(
        db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    # Relationships
    papers = db.relationship("Paper", backref="project", lazy=True, cascade="all, delete-orphan")
    literature_reviews = db.relationship("LiteratureReview", backref="project", lazy=True, cascade="all, delete-orphan")
    research_gaps = db.relationship("ResearchGap", backref="project", lazy=True, cascade="all, delete-orphan")
    references = db.relationship("ReferenceItem", backref="project", lazy=True, cascade="all, delete-orphan")
    report_drafts = db.relationship("ReportDraft", backref="project", lazy=True, cascade="all, delete-orphan")

    def to_dict(self):
        return {
            "id": self.id,
            "title": self.title,
            "topic": self.topic,
            "description": self.description,
            "paper_count": len(self.papers),
            "created_at": self.created_at.strftime("%Y-%m-%d %H:%M"),
            "updated_at": self.updated_at.strftime("%Y-%m-%d %H:%M"),
        }


class Paper(db.Model):
    """Academic paper metadata and content."""

    __tablename__ = "papers"

    id = db.Column(db.Integer, primary_key=True)
    project_id = db.Column(db.Integer, db.ForeignKey("projects.id"), nullable=True)
    title = db.Column(db.Text, nullable=False)
    authors = db.Column(db.Text, nullable=True)  # Comma separated or string
    year = db.Column(db.Integer, nullable=True)
    abstract = db.Column(db.Text, nullable=True)
    doi = db.Column(db.String(255), nullable=True)
    url = db.Column(db.Text, nullable=True)
    journal = db.Column(db.String(255), nullable=True)
    source = db.Column(db.String(50), default="openalex")  # 'openalex', 'pdf_upload', 'manual'
    pdf_filename = db.Column(db.String(255), nullable=True)
    pdf_path = db.Column(db.String(500), nullable=True)
    full_text = db.Column(db.Text, nullable=True)
    page_count = db.Column(db.Integer, default=1)
    is_analyzed = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Relationships
    chunks = db.relationship("PaperChunk", backref="paper", lazy=True, cascade="all, delete-orphan")
    summary = db.relationship("PaperSummary", backref="paper", uselist=False, lazy=True, cascade="all, delete-orphan")

    def to_dict(self, include_text=False):
        data = {
            "id": self.id,
            "project_id": self.project_id,
            "title": self.title,
            "authors": self.authors or "Unknown Authors",
            "year": self.year or "N/A",
            "abstract": self.abstract or "",
            "doi": self.doi or "",
            "url": self.url or "",
            "journal": self.journal or "Academic Publication",
            "source": self.source,
            "pdf_filename": self.pdf_filename,
            "page_count": self.page_count,
            "is_analyzed": self.is_analyzed,
            "has_summary": self.summary is not None,
            "created_at": self.created_at.strftime("%Y-%m-%d %H:%M"),
        }
        if include_text:
            data["full_text"] = self.full_text or ""
        return data


class PaperChunk(db.Model):
    """Text chunks extracted from paper PDFs for RAG vector search."""

    __tablename__ = "paper_chunks"

    id = db.Column(db.Integer, primary_key=True)
    paper_id = db.Column(db.Integer, db.ForeignKey("papers.id"), nullable=False)
    chunk_index = db.Column(db.Integer, nullable=False)
    content = db.Column(db.Text, nullable=False)
    page_number = db.Column(db.Integer, default=1)
    section_hint = db.Column(db.String(100), nullable=True)

    def to_dict(self):
        return {
            "id": self.id,
            "paper_id": self.paper_id,
            "paper_title": self.paper.title if self.paper else "Unknown",
            "chunk_index": self.chunk_index,
            "content": self.content,
            "page_number": self.page_number,
            "section_hint": self.section_hint,
        }


class PaperSummary(db.Model):
    """IBM Granite structured summary for an academic paper."""

    __tablename__ = "paper_summaries"

    id = db.Column(db.Integer, primary_key=True)
    paper_id = db.Column(db.Integer, db.ForeignKey("papers.id"), unique=True, nullable=False)
    research_objectives = db.Column(db.Text, nullable=True)
    abstract_summary = db.Column(db.Text, nullable=True)
    methodology = db.Column(db.Text, nullable=True)
    key_findings = db.Column(db.Text, nullable=True)
    limitations = db.Column(db.Text, nullable=True)
    future_directions = db.Column(db.Text, nullable=True)
    raw_response = db.Column(db.Text, nullable=True)
    model_used = db.Column(db.String(100), default="ibm/granite-13b-chat-v2")
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "paper_id": self.paper_id,
            "paper_title": self.paper.title if self.paper else "",
            "research_objectives": self.research_objectives or "",
            "abstract_summary": self.abstract_summary or "",
            "methodology": self.methodology or "",
            "key_findings": self.key_findings or "",
            "limitations": self.limitations or "",
            "future_directions": self.future_directions or "",
            "model_used": self.model_used,
            "created_at": self.created_at.strftime("%Y-%m-%d %H:%M"),
        }


class LiteratureReview(db.Model):
    """Multi-paper synthesized literature review."""

    __tablename__ = "literature_reviews"

    id = db.Column(db.Integer, primary_key=True)
    project_id = db.Column(db.Integer, db.ForeignKey("projects.id"), nullable=True)
    title = db.Column(db.String(255), nullable=False)
    topic = db.Column(db.Text, nullable=False)
    paper_ids = db.Column(db.Text, nullable=False)  # JSON array of paper IDs
    synthesized_review = db.Column(db.Text, nullable=False)
    comparative_table_json = db.Column(db.Text, nullable=True)
    unsupported_conclusions = db.Column(db.Text, nullable=True)
    model_used = db.Column(db.String(100), default="ibm/granite-13b-chat-v2")
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "project_id": self.project_id,
            "title": self.title,
            "topic": self.topic,
            "paper_ids": json.loads(self.paper_ids) if self.paper_ids else [],
            "synthesized_review": self.synthesized_review,
            "comparative_table": json.loads(self.comparative_table_json) if self.comparative_table_json else [],
            "unsupported_conclusions": self.unsupported_conclusions or "",
            "model_used": self.model_used,
            "created_at": self.created_at.strftime("%Y-%m-%d %H:%M"),
        }


class ResearchGap(db.Model):
    """Research gap analysis and suggested hypotheses."""

    __tablename__ = "research_gaps"

    id = db.Column(db.Integer, primary_key=True)
    project_id = db.Column(db.Integer, db.ForeignKey("projects.id"), nullable=True)
    topic = db.Column(db.Text, nullable=False)
    paper_ids = db.Column(db.Text, nullable=False)  # JSON array
    gaps_json = db.Column(db.Text, nullable=False)  # JSON array of gap objects
    hypotheses_json = db.Column(db.Text, nullable=False)  # JSON array of hypotheses
    model_used = db.Column(db.String(100), default="ibm/granite-13b-chat-v2")
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "project_id": self.project_id,
            "topic": self.topic,
            "paper_ids": json.loads(self.paper_ids) if self.paper_ids else [],
            "gaps": json.loads(self.gaps_json) if self.gaps_json else [],
            "hypotheses": json.loads(self.hypotheses_json) if self.hypotheses_json else [],
            "model_used": self.model_used,
            "created_at": self.created_at.strftime("%Y-%m-%d %H:%M"),
        }


class ReferenceItem(db.Model):
    """Bibliographic citation record with APA, IEEE, and BibTeX formatting."""

    __tablename__ = "references"

    id = db.Column(db.Integer, primary_key=True)
    project_id = db.Column(db.Integer, db.ForeignKey("projects.id"), nullable=True)
    paper_id = db.Column(db.Integer, db.ForeignKey("papers.id"), nullable=True)
    title = db.Column(db.Text, nullable=False)
    authors = db.Column(db.Text, nullable=True)
    year = db.Column(db.Integer, nullable=True)
    journal = db.Column(db.String(255), nullable=True)
    volume = db.Column(db.String(50), nullable=True)
    issue = db.Column(db.String(50), nullable=True)
    pages = db.Column(db.String(50), nullable=True)
    doi = db.Column(db.String(255), nullable=True)
    url = db.Column(db.Text, nullable=True)
    citation_apa = db.Column(db.Text, nullable=True)
    citation_ieee = db.Column(db.Text, nullable=True)
    bibtex = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "project_id": self.project_id,
            "paper_id": self.paper_id,
            "title": self.title,
            "authors": self.authors or "",
            "year": self.year or "",
            "journal": self.journal or "",
            "volume": self.volume or "",
            "issue": self.issue or "",
            "pages": self.pages or "",
            "doi": self.doi or "",
            "url": self.url or "",
            "citation_apa": self.citation_apa or "",
            "citation_ieee": self.citation_ieee or "",
            "bibtex": self.bibtex or "",
            "created_at": self.created_at.strftime("%Y-%m-%d %H:%M"),
        }


class ReportDraft(db.Model):
    """13-section structured academic research paper draft."""

    __tablename__ = "report_drafts"

    id = db.Column(db.Integer, primary_key=True)
    project_id = db.Column(db.Integer, db.ForeignKey("projects.id"), nullable=True)
    title = db.Column(db.String(255), nullable=False)
    topic = db.Column(db.Text, nullable=False)
    sections_json = db.Column(db.Text, nullable=False)  # Dict of section names to content
    full_markdown = db.Column(db.Text, nullable=False)
    model_used = db.Column(db.String(100), default="ibm/granite-13b-chat-v2")
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(
        db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    def to_dict(self):
        return {
            "id": self.id,
            "project_id": self.project_id,
            "title": self.title,
            "topic": self.topic,
            "sections": json.loads(self.sections_json) if self.sections_json else {},
            "full_markdown": self.full_markdown,
            "model_used": self.model_used,
            "created_at": self.created_at.strftime("%Y-%m-%d %H:%M"),
            "updated_at": self.updated_at.strftime("%Y-%m-%d %H:%M"),
        }


class ResearchHistory(db.Model):
    """Audit log of user research interactions."""

    __tablename__ = "research_history"

    id = db.Column(db.Integer, primary_key=True)
    action_type = db.Column(db.String(100), nullable=False)
    title = db.Column(db.String(255), nullable=False)
    summary = db.Column(db.Text, nullable=True)
    metadata_json = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "action_type": self.action_type,
            "title": self.title,
            "summary": self.summary or "",
            "metadata": json.loads(self.metadata_json) if self.metadata_json else {},
            "created_at": self.created_at.strftime("%Y-%m-%d %H:%M:%S"),
        }
