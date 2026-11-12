"""Template rendering views for ResearchMate."""

from functools import wraps
from flask import Blueprint, render_template, current_app, flash, redirect, url_for
from flask_login import login_required, current_user
from database.models import Paper, LiteratureReview, ResearchGap, ReferenceItem, ReportDraft, Project
from config import Config

views_bp = Blueprint("views", __name__)


def admin_required(f):
    """Requires the logged-in user to have administrator privileges."""
    @wraps(f)
    @login_required
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or current_user.role != "admin":
            flash("Administrator privileges required to access IBM watsonx.ai & Granite Settings.", "danger")
            return redirect(url_for("views.dashboard"))
        return f(*args, **kwargs)
    return decorated_function


@views_bp.route("/")
@login_required
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
@login_required
def question_analyzer():
    """Module 1: Research Question Analyzer."""
    return render_template("question_analyzer.html", active_page="question_analyzer")


@views_bp.route("/search-papers")
@login_required
def search_papers():
    """Module 2: Academic Paper Search."""
    return render_template("search_papers.html", active_page="search_papers")


@views_bp.route("/analyze-pdf")
@login_required
def analyze_pdf():
    """Module 3: Research Paper PDF Analyzer."""
    papers = Paper.query.order_by(Paper.created_at.desc()).all()
    return render_template("analyze_pdf.html", papers=papers, active_page="analyze_pdf")


@views_bp.route("/literature-review")
@login_required
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
@login_required
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
@login_required
def references():
    """Module 6: Reference Manager."""
    refs = ReferenceItem.query.order_by(ReferenceItem.created_at.desc()).all()
    papers = Paper.query.all()
    return render_template(
        "references.html", refs=refs, papers=papers, active_page="references"
    )


@views_bp.route("/report-generator")
@login_required
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
@login_required
def history():
    """Research Activity History."""
    return render_template("history.html", active_page="history")


@views_bp.route("/profile")
@login_required
def profile():
    """User profile, account details, and JWT token access."""
    from services.jwt_auth import generate_jwt_token
    token = generate_jwt_token(current_user)
    papers_count = Paper.query.count()
    reviews_count = LiteratureReview.query.count()
    gaps_count = ResearchGap.query.count()
    reports_count = ReportDraft.query.count()
    return render_template(
        "profile.html",
        active_page="profile",
        jwt_token=token,
        papers_count=papers_count,
        reviews_count=reviews_count,
        gaps_count=gaps_count,
        reports_count=reports_count,
    )


@views_bp.route("/settings")
@admin_required
def settings():
    """IBM watsonx.ai & Granite configuration settings (Admin Only)."""
    status = Config.get_watsonx_status()
    return render_template("settings.html", status=status, active_page="settings")
