"""Reproducibility utilities for ML projects."""

import os
import random
import numpy as np
import torch
from typing import Optional
from src.logger import get_logger

logger = get_logger(__name__)


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility across all libraries.
    
    Args:
        seed: Random seed value
    """
    # Python
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch (if available)
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except (ImportError, NameError):
        pass
    
    # Environment variable
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    logger.info(f"Random seed set to {seed}")


def get_seed() -> int:
    """Get current random seed."""
    return int(os.environ.get('PYTHONHASHSEED', 42))
