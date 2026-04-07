"""
Data validation and quality checks.
Ensures data meets expected requirements before processing.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from src.logger import get_logger

logger = get_logger(__name__)


class DataValidator:
    """Validate data quality and integrity."""
    
    def __init__(self, target_column: Optional[str] = None):
        """
        Initialize validator.
        
        Args:
            target_column: Name of target column for classification
        """
        self.target_column = target_column
        self.validation_report = {}
    
    def validate_structure(self, df: pd.DataFrame) -> Dict[str, bool]:
        """
        Validate basic dataframe structure.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dictionary of validation checks
        """
        checks = {}
        
        # Check if DataFrame is empty
        checks["not_empty"] = len(df) > 0
        if not checks["not_empty"]:
            logger.warning("DataFrame is empty")
        
        # Check for columns
        checks["has_columns"] = len(df.columns) > 0
        
        # Check for target column
        if self.target_column:
            checks["has_target"] = self.target_column in df.columns
            if not checks["has_target"]:
                logger.warning(f"Target column '{self.target_column}' not found")
        
        logger.info(f"Structure validation: {sum(checks.values())}/{len(checks)} passed")
        return checks
    
    def validate_missing_values(
        self,
        df: pd.DataFrame,
        max_missing_pct: float = 0.5
    ) -> Dict[str, float]:
        """
        Check for missing values.
        
        Args:
            df: DataFrame to validate
            max_missing_pct: Maximum allowed missing percentage (0-1)
            
        Returns:
            Dictionary of missing value percentages by column
        """
        missing_pct = (df.isna().sum() / len(df) * 100).sort_values(ascending=False)
        missing_pct = missing_pct[missing_pct > 0]
        
        issues = missing_pct[missing_pct > max_missing_pct * 100]
        
        if len(issues) > 0:
            logger.warning(f"Columns with >50% missing values:\n{issues}")
        
        logger.info(f"Missing values found in {len(missing_pct)} columns")
        return missing_pct.to_dict()
    
    def validate_data_types(self, df: pd.DataFrame) -> Dict[str, str]:
        """Validate and report data types."""
        dtypes = df.dtypes.to_dict()
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
        
        logger.info(f"Data types: {len(numeric_cols)} numeric, {len(categorical_cols)} categorical")
        return dtypes
    
    def validate_duplicates(self, df: pd.DataFrame) -> Tuple[int, float]:
        """
        Check for duplicate rows.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Tuple of (duplicate_count, duplicate_percentage)
        """
        n_duplicates = df.duplicated().sum()
        pct_duplicates = n_duplicates / len(df) * 100
        
        if n_duplicates > 0:
            logger.warning(f"Found {n_duplicates} duplicate rows ({pct_duplicates:.2f}%)")
        else:
            logger.info("No duplicate rows found")
        
        return n_duplicates, pct_duplicates
    
    def validate_target_distribution(self, df: pd.DataFrame) -> Dict[str, int]:
        """
        Check target variable distribution.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dictionary of target class counts
        """
        if not self.target_column or self.target_column not in df.columns:
            logger.warning("Target column not specified or not found")
            return {}
        
        distribution = df[self.target_column].value_counts().to_dict()
        
        # Check for class imbalance
        counts = list(distribution.values())
        if len(counts) > 1:
            imbalance_ratio = max(counts) / min(counts)
            logger.info(f"Target distribution: {distribution}")
            logger.info(f"Imbalance ratio: {imbalance_ratio:.2f}:1")
        
        return distribution
    
    def validate_outliers(
        self,
        df: pd.DataFrame,
        method: str = "iqr",
        threshold: float = 1.5
    ) -> Dict[str, int]:
        """
        Detect outliers in numeric columns.
        
        Args:
            df: DataFrame to validate
            method: Detection method (iqr, zscore)
            threshold: Threshold for outlier detection
            
        Returns:
            Dictionary of outlier counts by column
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outliers = {}
        
        for col in numeric_cols:
            if method == "iqr":
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                outlier_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            
            elif method == "zscore":
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outlier_count = (z_scores > threshold).sum()
            
            else:
                raise ValueError(f"Unknown method: {method}")
            
            if outlier_count > 0:
                outliers[col] = outlier_count
        
        if outliers:
            logger.warning(f"Outliers detected (using {method}): {outliers}")
        
        return outliers
    
    def full_validation(
        self,
        df: pd.DataFrame,
        max_missing_pct: float = 0.5
    ) -> Dict:
        """
        Run all validation checks.
        
        Args:
            df: DataFrame to validate
            max_missing_pct: Maximum allowed missing percentage
            
        Returns:
            Comprehensive validation report
        """
        logger.info("=" * 50)
        logger.info("STARTING FULL DATA VALIDATION")
        logger.info("=" * 50)
        
        report = {
            "structure": self.validate_structure(df),
            "missing_values": self.validate_missing_values(df, max_missing_pct),
            "data_types": self.validate_data_types(df),
            "duplicates": self.validate_duplicates(df),
            "target_distribution": self.validate_target_distribution(df),
            "outliers": self.validate_outliers(df),
        }
        
        logger.info("=" * 50)
        logger.info("VALIDATION COMPLETE")
        logger.info("=" * 50)
        
        self.validation_report = report
        return report
