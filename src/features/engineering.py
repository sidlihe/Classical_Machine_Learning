"""
Feature engineering utilities.
Create, transform, and select features for ML models.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    OneHotEncoder, LabelEncoder, PowerTransformer
)
from sklearn.decomposition import PCA
from sklearn.feature_selection import (
    SelectKBest, f_classif, f_regression,
    mutual_info_classif, mutual_info_regression,
    RFE, SequentialFeatureSelector
)
from src.logger import get_logger

logger = get_logger(__name__)


class FeatureEngineer:
    """Create and engineer features."""
    
    def __init__(self):
        """Initialize FeatureEngineer."""
        self.created_features = []
    
    def create_polynomial_features(
        self,
        df: pd.DataFrame,
        columns: List[str],
        degree: int = 2,
        include_bias: bool = False
    ) -> pd.DataFrame:
        """
        Create polynomial features.
        
        Args:
            df: Input DataFrame
            columns: Columns to create polynomials from
            degree: Polynomial degree
            include_bias: Include bias term
            
        Returns:
            DataFrame with polynomial features added
        """
        df_copy = df.copy()
        
        for col in columns:
            for d in range(2, degree + 1):
                new_col = f"{col}_pow_{d}"
                df_copy[new_col] = df[col] ** d
                self.created_features.append(new_col)
        
        logger.info(f"Created {len(self.created_features)} polynomial features")
        return df_copy
    
    def create_interaction_features(
        self,
        df: pd.DataFrame,
        columns: List[str]
    ) -> pd.DataFrame:
        """
        Create interaction features between columns.
        
        Args:
            df: Input DataFrame
            columns: Columns to create interactions from
            
        Returns:
            DataFrame with interaction features added
        """
        df_copy = df.copy()
        n_created = 0
        
        for i, col1 in enumerate(columns):
            for col2 in columns[i+1:]:
                new_col = f"{col1}_x_{col2}"
                df_copy[new_col] = df[col1] * df[col2]
                self.created_features.append(new_col)
                n_created += 1
        
        logger.info(f"Created {n_created} interaction features")
        return df_copy
    
    def create_ratio_features(
        self,
        df: pd.DataFrame,
        columns: List[Tuple[str, str]]
    ) -> pd.DataFrame:
        """
        Create ratio features.
        
        Args:
            df: Input DataFrame
            columns: List of (numerator, denominator) tuples
            
        Returns:
            DataFrame with ratio features added
        """
        df_copy = df.copy()
        
        for num_col, denom_col in columns:
            new_col = f"{num_col}_div_{denom_col}"
            # Avoid division by zero
            df_copy[new_col] = df[num_col] / (df[denom_col] + 1e-8)
            self.created_features.append(new_col)
        
        logger.info(f"Created {len(columns)} ratio features")
        return df_copy
    
    def create_binned_features(
        self,
        df: pd.DataFrame,
        columns: Dict[str, int]
    ) -> pd.DataFrame:
        """
        Create binned (discretized) features.
        
        Args:
            df: Input DataFrame
            columns: Dictionary of {column: n_bins}
            
        Returns:
            DataFrame with binned features added
        """
        df_copy = df.copy()
        
        for col, n_bins in columns.items():
            new_col = f"{col}_binned"
            df_copy[new_col] = pd.qcut(df[col], q=n_bins, labels=False, duplicates='drop')
            self.created_features.append(new_col)
        
        logger.info(f"Created {len(columns)} binned features")
        return df_copy
    
    def get_created_features(self) -> List[str]:
        """Get list of created features."""
        return self.created_features


class FeatureScaler:
    """Scale numerical features."""
    
    def __init__(self, method: str = "standard"):
        """
        Initialize scaler.
        
        Args:
            method: Scaling method (standard, minmax, robust)
        """
        self.method = method
        
        if method == "standard":
            self.scaler = StandardScaler()
        elif method == "minmax":
            self.scaler = MinMaxScaler()
        elif method == "robust":
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        self.fitted = False
        logger.info(f"Initialized {method} scaler")
    
    def fit_transform(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Fit scaler and transform data.
        
        Args:
            df: Input DataFrame
            columns: Columns to scale. If None, scales all numeric columns
            
        Returns:
            Scaled DataFrame
        """
        df_copy = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        self.scaler.fit(df[columns])
        df_copy[columns] = self.scaler.transform(df[columns])
        self.fitted = True
        
        logger.info(f"Fitted and transformed {len(columns)} columns using {self.method}")
        return df_copy
    
    def transform(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Transform data using fitted scaler.
        
        Args:
            df: Input DataFrame
            columns: Columns to scale
            
        Returns:
            Scaled DataFrame
        """
        if not self.fitted:
            raise ValueError("Scaler not fitted. Call fit_transform first.")
        
        df_copy = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        df_copy[columns] = self.scaler.transform(df[columns])
        return df_copy


class FeatureSelector:
    """Select important features."""
    
    def __init__(self, method: str = "SelectKBest", n_features: int = 10):
        """
        Initialize feature selector.
        
        Args:
            method: Selection method (SelectKBest, RFE, SequentialFeatureSelector)
            n_features: Number of features to select
        """
        self.method = method
        self.n_features = n_features
        self.selected_features = []
    
    def select_features(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        task: str = "classification"
    ) -> List[str]:
        """
        Select top features.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            task: Task type (classification, regression)
            
        Returns:
            List of selected feature names
        """
        score_func = f_classif if task == "classification" else f_regression
        
        selector = SelectKBest(
            score_func=score_func,
            k=min(self.n_features, X.shape[1])
        )
        
        selector.fit(X, y)
        
        # Get feature scores
        feature_scores = pd.DataFrame({
            'feature': X.columns,
            'score': selector.scores_
        }).sort_values('score', ascending=False)
        
        self.selected_features = feature_scores['feature'].head(self.n_features).tolist()
        
        logger.info(f"Selected {len(self.selected_features)} features")
        logger.info(f"Top features: {self.selected_features[:5]}")
        
        return self.selected_features
    
    def get_selected_features(self) -> List[str]:
        """Get list of selected features."""
        return self.selected_features
