"""Helper utilities."""

import os
from pathlib import Path


def ensure_dirs(*dirs: str) -> None:
    """Create directories if they don't exist."""
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)


def get_project_root() -> Path:
    """Get project root directory."""
    return Path(__file__).parent.parent.parent


def get_data_dir() -> Path:
    """Get data directory."""
    return get_project_root() / "data"


def get_models_dir() -> Path:
    """Get models directory."""
    return get_project_root() / "models"


def get_logs_dir() -> Path:
    """Get logs directory."""
    return get_project_root() / "logs"
