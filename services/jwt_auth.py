"""JSON Web Token (JWT) Authentication Service for ResearchMate."""

import logging
from datetime import datetime, timedelta, timezone
from functools import wraps
from typing import Optional, Dict, Any

import jwt
from flask import request, jsonify, current_app
from flask_login import current_user

from config import Config
from database.db import db
from database.models import User

logger = logging.getLogger(__name__)

JWT_ALGORITHM = "HS256"
DEFAULT_EXPIRATION_HOURS = 24


def generate_jwt_token(user: User, expires_hours: int = DEFAULT_EXPIRATION_HOURS) -> str:
    """Generates a signed JWT access token for a User.

    Args:
        user: The User model instance.
        expires_hours: Token validity duration in hours.

    Returns:
        Encoded JWT token string.
    """
    secret = Config.SECRET_KEY or "researchmate-default-dev-secret-key-2026"
    now = datetime.now(timezone.utc)
    exp = now + timedelta(hours=expires_hours)

    payload = {
        "sub": str(user.id),
        "user_id": user.id,
        "username": user.username,
        "email": user.email,
        "role": user.role,
        "full_name": user.full_name,
        "institution": user.institution,
        "iat": now,
        "exp": exp,
        "iss": "ResearchMate-Auth",
    }

    token = jwt.encode(payload, secret, algorithm=JWT_ALGORITHM)
    return token


def decode_jwt_token(token: str) -> Optional[Dict[str, Any]]:
    """Decodes and validates a JWT token.

    Args:
        token: Encoded JWT string.

    Returns:
        Dict payload if valid, None if expired or invalid.
    """
    if not token:
        return None

    secret = Config.SECRET_KEY or "researchmate-default-dev-secret-key-2026"
    try:
        payload = jwt.decode(
            token,
            secret,
            algorithms=[JWT_ALGORITHM],
            issuer="ResearchMate-Auth",
        )
        return payload
    except jwt.ExpiredSignatureError:
        logger.warning("JWT token has expired.")
        return None
    except jwt.InvalidTokenError as e:
        logger.warning(f"Invalid JWT token: {e}")
        return None


def jwt_required(f):
    """Decorator for REST API endpoints requiring JWT Bearer authentication.
    Also accepts active Flask-Login session if available for browser calls.
    """
    @wraps(f)
    def decorated(*args, **kwargs):
        # 1. Check Authorization Bearer header
        auth_header = request.headers.get("Authorization", "")
        token = None

        if auth_header.startswith("Bearer "):
            token = auth_header[7:].strip()
        elif request.headers.get("X-API-Token"):
            token = request.headers.get("X-API-Token", "").strip()

        if token:
            payload = decode_jwt_token(token)
            if not payload:
                return jsonify({
                    "success": False,
                    "error": "Unauthorized: JWT token is invalid or expired.",
                }), 401

            user_id = payload.get("user_id")
            user = db.session.get(User, user_id)
            if not user:
                return jsonify({
                    "success": False,
                    "error": "Unauthorized: User associated with JWT token no longer exists.",
                }), 401

            request.jwt_user = user
            return f(*args, **kwargs)

        # 2. Fallback to active Flask-Login session for browser requests
        if current_user.is_authenticated:
            request.jwt_user = current_user
            return f(*args, **kwargs)

        return jsonify({
            "success": False,
            "error": "Unauthorized: Missing Authorization header (Format: 'Bearer <JWT_TOKEN>').",
        }), 401

    return decorated


def admin_jwt_required(f):
    """Decorator for REST API endpoints requiring Admin JWT Bearer authentication."""
    @wraps(f)
    def decorated(*args, **kwargs):
        # First authenticate via JWT or session
        auth_header = request.headers.get("Authorization", "")
        token = None
        if auth_header.startswith("Bearer "):
            token = auth_header[7:].strip()
        elif request.headers.get("X-API-Token"):
            token = request.headers.get("X-API-Token", "").strip()

        user = None
        if token:
            payload = decode_jwt_token(token)
            if payload and payload.get("user_id"):
                user = db.session.get(User, payload["user_id"])
        elif current_user.is_authenticated:
            user = current_user

        if not user:
            return jsonify({
                "success": False,
                "error": "Unauthorized: Valid Admin JWT Bearer token required.",
            }), 401

        if user.role != "admin":
            return jsonify({
                "success": False,
                "error": "Forbidden: Administrator role required.",
            }), 403

        request.jwt_user = user
        return f(*args, **kwargs)

    return decorated
