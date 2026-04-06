"""
Classical Machine Learning Project
Production-ready ML system with centralized logging, config, and modular architecture.
"""

__version__ = "0.1.0"
__author__ = "ML Team"

from src.logger import get_logger
from src.config import Config

__all__ = ["get_logger", "Config"]
