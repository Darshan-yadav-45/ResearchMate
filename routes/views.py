"""Template rendering views for ResearchMate."""

from flask import Blueprint, render_template, current_app
from database.models import Paper, LiteratureReview, ResearchGap, ReferenceItem, ReportDraft, Project
from config import Config

views_bp = Blueprint("views", __name__)


@views_bp.route("/")
def dashboard():
    """Main application dashboard."""
    papers_count = Paper.query.count()
    reviews_count = LiteratureReview.query.count()
    gaps_count = ResearchGap.query.count()
    references_count = ReferenceItem.query.count()
    reports_count = ReportDraft.query.count()

    recent_papers = Paper.query.order_by(Paper.created_at.desc()).limit(5).all()
    watsonx_status = Config.get_watsonx_status()

    return render_template(
        "index.html",
        papers_count=papers_count,
        reviews_count=reviews_count,
        gaps_count=gaps_count,
        references_count=references_count,
        reports_count=reports_count,
        recent_papers=recent_papers,
        watsonx_status=watsonx_status,
        active_page="dashboard",
    )


@views_bp.route("/question-analyzer")
def question_analyzer():
    """Module 1: Research Question Analyzer."""
    return render_template("question_analyzer.html", active_page="question_analyzer")


@views_bp.route("/search-papers")
def search_papers():
    """Module 2: Academic Paper Search."""
    return render_template("search_papers.html", active_page="search_papers")


@views_bp.route("/analyze-pdf")
def analyze_pdf():
    """Module 3: Research Paper PDF Analyzer."""
    papers = Paper.query.order_by(Paper.created_at.desc()).all()
    return render_template("analyze_pdf.html", papers=papers, active_page="analyze_pdf")


@views_bp.route("/literature-review")
def literature_review():
    """Module 4: Literature Review Assistant."""
    papers = Paper.query.order_by(Paper.created_at.desc()).all()
    reviews = LiteratureReview.query.order_by(LiteratureReview.created_at.desc()).all()
    return render_template(
        "literature_review.html",
        papers=papers,
        reviews=reviews,
        active_page="literature_review",
    )


@views_bp.route("/research-gaps")
def research_gaps():
    """Module 5: Research Gap and Hypothesis Generator."""
    papers = Paper.query.order_by(Paper.created_at.desc()).all()
    gaps_records = ResearchGap.query.order_by(ResearchGap.created_at.desc()).all()
    return render_template(
        "research_gaps.html",
        papers=papers,
        gaps_records=gaps_records,
        active_page="research_gaps",
    )


@views_bp.route("/references")
def references():
    """Module 6: Reference Manager."""
    refs = ReferenceItem.query.order_by(ReferenceItem.created_at.desc()).all()
    papers = Paper.query.all()
    return render_template(
        "references.html", refs=refs, papers=papers, active_page="references"
    )


@views_bp.route("/report-generator")
def report_generator():
    """Module 7: Research Report and Paper Draft Generator."""
    papers = Paper.query.all()
    reviews = LiteratureReview.query.all()
    reports = ReportDraft.query.order_by(ReportDraft.created_at.desc()).all()
    return render_template(
        "report_generator.html",
        papers=papers,
        reviews=reviews,
        reports=reports,
        active_page="report_generator",
    )


@views_bp.route("/history")
def history():
    """Research Activity History."""
    return render_template("history.html", active_page="history")


@views_bp.route("/settings")
def settings():
    """IBM watsonx.ai & Granite configuration settings."""
    status = Config.get_watsonx_status()
    return render_template("settings.html", status=status, active_page="settings")
