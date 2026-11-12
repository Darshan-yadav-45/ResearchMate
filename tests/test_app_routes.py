"""Functional integration tests for ResearchMate web views and API endpoints."""

import json
import pytest
from app import create_app, db
from database.models import Paper, ReferenceItem


@pytest.fixture
def client(tmp_path):
    test_db = tmp_path / "test_researchmate.db"
    app = create_app()
    app.config["TESTING"] = True
    app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{test_db}"

    with app.app_context():
        db.create_all()
        yield app.test_client()
        db.session.remove()
        db.drop_all()


def test_views_load(client):
    """Test that all frontend web routes render successfully for their authorized roles."""
    # Authenticate with seeded researcher account
    login_res = client.post(
        "/login",
        data={"identity": "researcher", "password": "Password123!"},
        follow_redirects=True,
    )
    assert login_res.status_code == 200

    researcher_endpoints = [
        "/",
        "/question-analyzer",
        "/search-papers",
        "/analyze-pdf",
        "/literature-review",
        "/research-gaps",
        "/references",
        "/report-generator",
        "/history",
        "/profile",
    ]
    for ep in researcher_endpoints:
        response = client.get(ep)
        assert response.status_code == 200
        assert b"ResearchMate" in response.data

    # Log in as admin to test the admin-only settings route
    client.get("/logout")
    admin_login = client.post(
        "/login",
        data={"identity": "admin", "password": "Admin123!"},
        follow_redirects=True,
    )
    assert admin_login.status_code == 200
    settings_res = client.get("/settings")
    assert settings_res.status_code == 200
    assert b"ResearchMate" in settings_res.data


def test_api_status(client):
    """Test system status API endpoint."""
    res = client.get("/api/status")
    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is True
    assert "watsonx" in data
    assert "counts" in data


def test_api_references_flow(client):
    """Test creating, listing, and exporting references."""
    ref_payload = {
        "title": "A Mathematical Theory of Communication",
        "authors": "Claude Shannon",
        "year": 1948,
        "journal": "Bell System Technical Journal",
        "volume": "27",
        "pages": "379-423",
    }
    post_res = client.post("/api/references", json=ref_payload)
    assert post_res.status_code == 200
    post_data = post_res.get_json()
    assert post_data["success"] is True
    assert "Shannon, C." in post_data["reference"]["citation_apa"]

    # List references
    get_res = client.get("/api/references")
    assert get_res.status_code == 200
    assert len(get_res.get_json()["references"]) >= 1

    # Export references
    export_res = client.get("/api/references/export?format=bibtex")
    assert export_res.status_code == 200
    assert export_res.get_json()["success"] is True


def test_api_save_paper(client):
    """Test saving a paper to the database library."""
    paper_payload = {
        "title": "Language Models are Few-Shot Learners",
        "authors": "Tom Brown et al.",
        "year": 2020,
        "abstract": "We demonstrate that scaling up language models greatly improves task-agnostic, few-shot performance.",
        "journal": "NeurIPS",
    }
    res = client.post("/api/save-paper", json=paper_payload)
    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is True
    assert data["paper"]["title"] == paper_payload["title"]


def test_api_export_report(client):
    """Test exporting a paper draft as Markdown and HTML."""
    export_payload = {
        "title": "Quantum Error Correction Survey",
        "topic": "Quantum Computing",
        "content": "# Quantum Error Correction Survey\n## 1. Abstract\nThis paper surveys topological codes.",
        "format": "markdown",
    }
    res = client.post("/api/export-report", json=export_payload)
    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is True
    assert data["filename"].endswith(".md")
