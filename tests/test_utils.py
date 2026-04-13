"""Basic test suite for project modules."""

import pytest
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.logger import get_logger
from src.config import Config
from src.data.loader import DataLoader
from src.utils.reproducibility import set_seed


class TestLogger:
    """Test logging module."""
    
    def test_get_logger(self):
        """Test logger creation."""
        logger = get_logger("test_module")
        assert logger is not None
        assert logger.name == "test_module"


class TestConfig:
    """Test configuration module."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = Config()
        assert config.project_name == "classical_ml"
        assert config.random_state == 42
    
    def test_config_to_dict(self):
        """Test config serialization."""
        config = Config()
        config_dict = config.to_dict()
        assert "data" in config_dict
        assert "model" in config_dict


class TestReproducibility:
    """Test reproducibility utilities."""
    
    def test_set_seed(self):
        """Test seed setting."""
        set_seed(42)
        
        # Generate random numbers
        val1 = np.random.random()
        
        # Reset seed
        set_seed(42)
        val2 = np.random.random()
        
        # Should be identical
        assert val1 == val2


class TestDataLoader:
    """Test data loader."""
    
    def test_loader_initialization(self):
        """Test DataLoader creation."""
        loader = DataLoader("data/raw")
        assert loader.data_dir.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
