import os
from pathlib import Path
from dotenv import load_dotenv

# Base directory of the application
BASE_DIR = Path(__file__).resolve().parent

# Load environment variables from .env file
load_dotenv(BASE_DIR / ".env")


class Config:
    """Base application configuration."""

    # Secret key for session management and CSRF protection
    SECRET_KEY = os.getenv("SECRET_KEY", "researchmate-default-dev-secret-key-2026")

    # SQLite Database URI
    SQLALCHEMY_DATABASE_URI = os.getenv(
        "DATABASE_URI", f"sqlite:///{BASE_DIR / 'researchmate.db'}"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Uploads and Exports directories
    UPLOAD_FOLDER = BASE_DIR / os.getenv("UPLOAD_FOLDER", "uploads")
    EXPORT_FOLDER = BASE_DIR / os.getenv("EXPORT_FOLDER", "exports")
    MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH", 32 * 1024 * 1024))
    ALLOWED_EXTENSIONS = {"pdf"}

    # IBM watsonx.ai & Granite Model Configuration
    WATSONX_APIKEY = os.getenv("WATSONX_APIKEY", "").strip()
    WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID", "").strip()
    WATSONX_URL = os.getenv(
        "WATSONX_URL", "https://us-south.ml.cloud.ibm.com"
    ).rstrip("/")
    IBM_GRANITE_MODEL_ID = os.getenv(
        "IBM_GRANITE_MODEL_ID", "ibm/granite-13b-chat-v2"
    ).strip()

    # Academic Search API settings
    OPENALEX_API_URL = "https://api.openalex.org/works"
    USER_AGENT = "ResearchMate/1.0 (academic-research-agent@ibm-aicte.edu)"

    @classmethod
    def is_watsonx_configured(cls) -> bool:
        """Returns True if minimum required IBM watsonx credentials are set."""
        return bool(cls.WATSONX_APIKEY and cls.WATSONX_PROJECT_ID)

    @classmethod
    def get_watsonx_status(cls) -> dict:
        """Returns diagnostic status of IBM watsonx.ai configuration."""
        configured = cls.is_watsonx_configured()
        masked_key = (
            f"{cls.WATSONX_APIKEY[:4]}...{cls.WATSONX_APIKEY[-4:]}"
            if len(cls.WATSONX_APIKEY) >= 8
            else ("Set" if cls.WATSONX_APIKEY else "Missing")
        )
        return {
            "configured": configured,
            "apikey_status": masked_key,
            "project_id": cls.WATSONX_PROJECT_ID or "Missing",
            "service_url": cls.WATSONX_URL,
            "model_id": cls.IBM_GRANITE_MODEL_ID,
        }
