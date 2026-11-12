"""Unit and functional tests for user authentication."""

import pytest
from app import create_app, db
from database.models import User


@pytest.fixture
def auth_client(tmp_path):
    test_db = tmp_path / "test_auth_researchmate.db"
    app = create_app()
    app.config["TESTING"] = True
    app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{test_db}"

    with app.app_context():
        db.create_all()
        yield app.test_client()
        db.session.remove()
        db.drop_all()


def test_user_password_hashing():
    user = User(username="testuser", email="test@example.com")
    user.set_password("SecurePass123!")
    assert user.password_hash != "SecurePass123!"
    assert user.check_password("SecurePass123!")
    assert not user.check_password("WrongPassword")


def test_unauthenticated_redirect(auth_client):
    """Unauthenticated users should be redirected to /login for protected pages."""
    res = auth_client.get("/")
    assert res.status_code == 302
    assert "/login" in res.location


def test_login_page_renders(auth_client):
    res = auth_client.get("/login")
    assert res.status_code == 200
    assert b"Sign In to Workspace" in res.data
    assert b"Username or Email" in res.data


def test_register_page_renders(auth_client):
    res = auth_client.get("/register")
    assert res.status_code == 200
    assert b"Create Researcher Account" in res.data


def test_register_flow(auth_client):
    payload = {
        "username": "dr_curie",
        "email": "curie@radium.edu",
        "full_name": "Dr. Marie Curie",
        "institution": "Sorbonne University",
        "password": "Password123!",
        "confirm_password": "Password123!",
    }
    res = auth_client.post("/register", data=payload, follow_redirects=True)
    assert res.status_code == 200
    assert b"Dr. Marie Curie" in res.data or b"Dashboard" in res.data


def test_demo_login_flow(auth_client):
    """The default seeded researcher account should authenticate and access dashboard."""
    res = auth_client.post(
        "/login",
        data={"identity": "researcher", "password": "Password123!"},
        follow_redirects=True,
    )
    assert res.status_code == 200
    assert b"Research Workspace Dashboard" in res.data
    assert b"Sign Out" in res.data


def test_invalid_login(auth_client):
    res = auth_client.post(
        "/login",
        data={"identity": "researcher", "password": "WrongPassword"},
        follow_redirects=True,
    )
    assert res.status_code == 200
    assert b"Invalid username/email or password" in res.data


def test_logout_flow(auth_client):
    # First log in
    auth_client.post(
        "/login",
        data={"identity": "researcher", "password": "Password123!"},
        follow_redirects=True,
    )
    # Then log out
    res = auth_client.get("/logout", follow_redirects=True)
    assert res.status_code == 200
    assert b"Sign In to Workspace" in res.data

