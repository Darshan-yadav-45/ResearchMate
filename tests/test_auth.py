"""Unit and functional tests for user authentication, admin permissions, and JWT."""

import pytest
from app import create_app, db
from database.models import User
from services.jwt_auth import generate_jwt_token, decode_jwt_token


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


def test_researcher_login_flow(auth_client):
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


# ==============================================================================
# Admin & Settings Access Control Tests
# ==============================================================================

def test_admin_login_and_settings_access(auth_client):
    """Admin users should be able to access the /settings view."""
    # Login as admin
    login_res = auth_client.post(
        "/login",
        data={"identity": "admin", "password": "Admin123!"},
        follow_redirects=True,
    )
    assert login_res.status_code == 200

    # Access settings
    settings_res = auth_client.get("/settings")
    assert settings_res.status_code == 200
    assert b"IBM watsonx.ai & Granite Settings" in settings_res.data or b"Settings" in settings_res.data


def test_researcher_settings_access_denied(auth_client):
    """Researcher (non-admin) should be redirected away from /settings with flash message."""
    # Login as researcher
    auth_client.post(
        "/login",
        data={"identity": "researcher", "password": "Password123!"},
        follow_redirects=True,
    )

    # Attempt to access settings
    res = auth_client.get("/settings", follow_redirects=True)
    assert res.status_code == 200
    # Should redirect to dashboard with danger flash
    assert b"Administrator privileges required" in res.data


def test_api_settings_admin_only(auth_client):
    """API endpoints for watsonx settings must reject non-admins."""
    # 1. Without login or token -> 403
    res = auth_client.post("/api/settings", json={"apikey": "test"})
    assert res.status_code == 403

    # 2. Login as researcher -> 403
    auth_client.post("/login", data={"identity": "researcher", "password": "Password123!"})
    res = auth_client.post("/api/settings", json={"apikey": "test"})
    assert res.status_code == 403
    assert res.get_json()["error"].startswith("Forbidden")

    # 3. Login as admin -> 200
    auth_client.get("/logout")
    auth_client.post("/login", data={"identity": "admin", "password": "Admin123!"})
    res = auth_client.post("/api/settings", json={"apikey": "test-key-123"})
    assert res.status_code == 200
    assert res.get_json()["success"] is True


# ==============================================================================
# User Profile View & Actions Tests
# ==============================================================================

def test_user_profile_view(auth_client):
    """Logged in user can view profile page."""
    auth_client.post("/login", data={"identity": "researcher", "password": "Password123!"})
    res = auth_client.get("/profile")
    assert res.status_code == 200
    assert b"User Profile" in res.data
    assert b"Edit Profile" in res.data
    assert b"Dr. Alex Rivera" in res.data


def test_user_profile_update(auth_client):
    """Logged in user can update their full name and institution."""
    auth_client.post("/login", data={"identity": "researcher", "password": "Password123!"})
    res = auth_client.post(
        "/profile/update",
        data={
            "full_name": "Dr. Alexander Rivera, Ph.D.",
            "email": "researcher@researchmate.org",
            "institution": "MIT CSAIL",
        },
        follow_redirects=True,
    )
    assert res.status_code == 200
    assert b"Profile updated successfully" in res.data
    assert b"MIT CSAIL" in res.data


def test_user_password_change(auth_client):
    """User can change their password."""
    auth_client.post("/login", data={"identity": "researcher", "password": "Password123!"})
    res = auth_client.post(
        "/profile/password",
        data={
            "current_password": "Password123!",
            "new_password": "NewSecretPassword456!",
            "confirm_password": "NewSecretPassword456!",
        },
        follow_redirects=True,
    )
    assert res.status_code == 200
    assert b"Password updated successfully" in res.data


# ==============================================================================
# JWT Authentication Tests
# ==============================================================================

def test_jwt_generation_and_decoding():
    user = User(
        id=99,
        username="jwt_user",
        email="jwt@test.edu",
        full_name="JWT Researcher",
        institution="Lab of AI",
        role="researcher",
    )
    token = generate_jwt_token(user)
    assert isinstance(token, str)
    assert len(token) > 20

    payload = decode_jwt_token(token)
    assert payload is not None
    assert payload["username"] == "jwt_user"
    assert payload["user_id"] == 99
    assert payload["role"] == "researcher"


def test_api_auth_token_endpoint(auth_client):
    """POST /api/auth/token generates a valid JWT token."""
    # Valid credentials
    res = auth_client.post(
        "/api/auth/token",
        json={"username": "researcher", "password": "Password123!"},
    )
    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is True
    token = data["token"]
    assert token is not None

    # Invalid credentials
    bad_res = auth_client.post(
        "/api/auth/token",
        json={"username": "researcher", "password": "WrongPassword"},
    )
    assert bad_res.status_code == 401
    assert bad_res.get_json()["success"] is False


def test_api_auth_profile_with_jwt(auth_client):
    """GET /api/auth/profile validates JWT in Authorization header."""
    # Obtain token
    token_res = auth_client.post(
        "/api/auth/token",
        json={"username": "researcher", "password": "Password123!"},
    )
    token = token_res.get_json()["token"]

    # Call /api/auth/profile with Bearer token
    profile_res = auth_client.get(
        "/api/auth/profile",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert profile_res.status_code == 200
    profile_data = profile_res.get_json()
    assert profile_data["success"] is True
    assert profile_data["user"]["username"] == "researcher"

    # Call with invalid token -> 401
    bad_res = auth_client.get(
        "/api/auth/profile",
        headers={"Authorization": "Bearer invalid.token.signature"},
    )
    assert bad_res.status_code == 401
