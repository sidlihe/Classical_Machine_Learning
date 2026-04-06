"""
Data loading and management utilities.
Handles loading from various sources, validation, and caching.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any
import pickle
import json

from src.logger import get_logger

logger = get_logger(__name__)


class DataLoader:
    """Load and manage datasets from various sources."""
    
    def __init__(self, data_dir: str = "data/raw"):
        """
        Initialize DataLoader.
        
        Args:
            data_dir: Directory containing raw data files
        """
        self.data_dir = Path(data_dir)
        self.data_cache: Dict[str, pd.DataFrame] = {}
    
    def load_csv(
        self,
        filename: str,
        **kwargs
    ) -> pd.DataFrame:
        """
        Load CSV file with optional caching.
        
        Args:
            filename: Name of CSV file in data_dir
            **kwargs: Additional arguments passed to pd.read_csv()
            
        Returns:
            Loaded DataFrame
        """
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            logger.error(f"File not found: {filepath}")
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        logger.info(f"Loading CSV: {filename}")
        df = pd.read_csv(filepath, **kwargs)
        logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
        
        self.data_cache[filename] = df
        return df
    
    def load_parquet(self, filename: str, **kwargs) -> pd.DataFrame:
        """Load Parquet file."""
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            logger.error(f"File not found: {filepath}")
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        logger.info(f"Loading Parquet: {filename}")
        df = pd.read_parquet(filepath, **kwargs)
        logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
        
        self.data_cache[filename] = df
        return df
    
    def load_json(self, filename: str, **kwargs) -> Dict[str, Any]:
        """Load JSON file."""
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            logger.error(f"File not found: {filepath}")
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        logger.info(f"Loading JSON: {filename}")
        with open(filepath, 'r') as f:
            data = json.load(f, **kwargs)
        
        return data
    
    def save_processed(
        self,
        df: pd.DataFrame,
        filename: str,
        output_dir: str = "data/processed",
        format: str = "csv"
    ) -> Path:
        """
        Save processed data.
        
        Args:
            df: DataFrame to save
            filename: Output filename
            output_dir: Directory to save to
            format: File format (csv, parquet, pkl)
            
        Returns:
            Path to saved file
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        full_path = output_path / filename
        
        if format == "csv":
            df.to_csv(full_path, index=False)
        elif format == "parquet":
            df.to_parquet(full_path, index=False)
        elif format == "pkl":
            df.to_pickle(full_path)
        else:
            logger.error(f"Unsupported format: {format}")
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Saved data to: {full_path}")
        return full_path
    
    def get_cached(self, filename: str) -> Optional[pd.DataFrame]:
        """Get cached DataFrame if available."""
        return self.data_cache.get(filename)
    
    def clear_cache(self) -> None:
        """Clear data cache."""
        self.data_cache.clear()
        logger.info("Data cache cleared")
    
    def list_files(self, extension: str = "csv") -> List[str]:
        """List all files with given extension in data_dir."""
        files = list(self.data_dir.glob(f"*.{extension}"))
        logger.info(f"Found {len(files)} {extension} files")
        return [f.name for f in files]
