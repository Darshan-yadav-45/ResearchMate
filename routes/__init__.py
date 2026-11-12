"""Routes package."""
from routes.views import views_bp
from routes.api import api_bp
from routes.auth import auth_bp

__all__ = ["views_bp", "api_bp", "auth_bp"]
