

import os
from dotenv import load_dotenv
from app.utils.errors import ServiceError

# Load environment variables from .env
load_dotenv()

# Provider keys are optional during import so health checks, docs, and
# unrelated modules remain available when a provider is not configured.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or None
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY") or None
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")


class ConfigurationError(ServiceError):
    """Raised when a feature is used without its required configuration."""

    def __init__(self, name: str) -> None:
        super().__init__(
            "missing_configuration",
            f"{name} is not configured. Set it in the environment or .env file.",
            503
        )


def require_api_key(name: str) -> str:
    """Return a configured provider key or raise a clear configuration error."""
    value = os.getenv(name)
    if not value:
        raise ConfigurationError(name)
    return value
