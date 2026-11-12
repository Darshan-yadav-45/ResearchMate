"""Main Flask application entry point for ResearchMate."""

import os
import logging
from pathlib import Path
from flask import Flask, render_template, jsonify
from flask_cors import CORS
from flask_login import LoginManager

from config import Config
from database.db import db
from database.models import User, PaperChunk, Project
from routes.views import views_bp
from routes.api import api_bp
from routes.auth import auth_bp
from rag.rag_engine import global_vector_store

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
)
logger = logging.getLogger("ResearchMate")

login_manager = LoginManager()


def create_app(config_class=Config):
    """Application factory for ResearchMate."""
    app = Flask(__name__)
    app.config.from_object(config_class)

    # Enable CORS for local API access
    CORS(app)

    # Initialize extensions
    db.init_app(app)
    login_manager.init_app(app)
    login_manager.login_view = "auth.login"
    login_manager.login_message = "Please sign in to access your ResearchMate workspace."
    login_manager.login_message_category = "info"

    @login_manager.user_loader
    def load_user(user_id):
        return User.query.get(int(user_id))

    # Ensure required runtime folders exist
    Path(app.config["UPLOAD_FOLDER"]).mkdir(parents=True, exist_ok=True)
    Path(app.config["EXPORT_FOLDER"]).mkdir(parents=True, exist_ok=True)

    # Register blueprints
    app.register_blueprint(auth_bp)
    app.register_blueprint(views_bp)
    app.register_blueprint(api_bp)

    # Context processors for base template
    @app.context_processor
    def inject_global_context():
        return {
            "watsonx_configured": Config.is_watsonx_configured(),
            "granite_model_id": Config.IBM_GRANITE_MODEL_ID,
        }

    # Error Handlers
    @app.errorhandler(404)
    def not_found_error(error):
        return render_template("base.html", error_404=True), 404

    @app.errorhandler(413)
    def request_entity_too_large(error):
        return (
            jsonify(
                {
                    "success": False,
                    "error": "File size exceeds the 32MB upload limit.",
                }
            ),
            413,
        )

    @app.errorhandler(500)
    def internal_error(error):
        db.session.rollback()
        return (
            jsonify(
                {
                    "success": False,
                    "error": "An internal server error occurred.",
                }
            ),
            500,
        )

    # Initialize database tables and vector store
    with app.app_context():
        db.create_all()

        # Seed default project if empty
        if Project.query.count() == 0:
            default_proj = Project(
                title="Default Research Project",
                topic="AI-Powered Academic Research",
                description="Initial workspace for managing surveyed literature, RAG documents, and drafts.",
            )
            db.session.add(default_proj)
            db.session.commit()

        # Seed default academic researcher account if empty
        if User.query.count() == 0:
            demo_user = User(
                username="researcher",
                email="researcher@researchmate.org",
                full_name="Dr. Alex Rivera",
                institution="Institute of Advanced Research",
                role="Lead Researcher",
            )
            demo_user.set_password("Password123!")
            db.session.add(demo_user)
            db.session.commit()
            logger.info("Created default academic researcher account (researcher / Password123!)")

        # Populate vector store from existing chunks in database
        try:
            chunks = PaperChunk.query.all()
            if chunks:
                chunk_dicts = [
                    {
                        "paper_id": c.paper_id,
                        "paper_title": c.paper.title if c.paper else "Document",
                        "chunk_index": c.chunk_index,
                        "content": c.content,
                        "page_number": c.page_number,
                        "section_hint": c.section_hint,
                    }
                    for c in chunks
                ]
                global_vector_store.set_chunks(chunk_dicts)
                logger.info(
                    f"Initialized RAG vector store with {len(chunks)} chunks."
                )
        except Exception as e:
            logger.warning(f"Could not prime vector store at startup: {e}")

    logger.info("ResearchMate application successfully initialized.")
    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
