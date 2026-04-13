"""
Validation strategies for ML models.
Implement various cross-validation and splitting techniques.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Generator
from sklearn.model_selection import (
    StratifiedKFold, KFold, TimeSeriesSplit,
    train_test_split
)
from src.logger import get_logger

logger = get_logger(__name__)


class ValidationStrategy:
    """Base validation strategy."""
    
    def __init__(self, random_state: int = 42):
        """Initialize validation strategy."""
        self.random_state = random_state
    
    def split(self, X: pd.DataFrame, y: pd.Series):
        """Split data into train/test folds. Override in subclasses."""
        raise NotImplementedError


class StratifiedKFoldValidation(ValidationStrategy):
    """Stratified K-Fold cross-validation."""
    
    def __init__(self, n_splits: int = 5, random_state: int = 42):
        """
        Initialize stratified K-fold.
        
        Args:
            n_splits: Number of folds
            random_state: Random seed
        """
        super().__init__(random_state)
        self.n_splits = n_splits
        self.skf = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state
        )
    
    def split(self, X, y):
        """Generate train/test indices for stratified folds."""
        for train_idx, test_idx in self.skf.split(X, y):
            logger.info(f"Fold: Train={len(train_idx)}, Test={len(test_idx)}")
            yield train_idx, test_idx


class TimeSeriesValidation(ValidationStrategy):
    """Time series cross-validation."""
    
    def __init__(self, n_splits: int = 5, random_state: int = 42):
        """
        Initialize time series validation.
        
        Args:
            n_splits: Number of folds
            random_state: Random seed
        """
        super().__init__(random_state)
        self.n_splits = n_splits
        self.tscv = TimeSeriesSplit(n_splits=n_splits)
    
    def split(self, X, y):
        """Generate train/test indices for time series folds."""
        for train_idx, test_idx in self.tscv.split(X):
            logger.info(f"Fold: Train={len(train_idx)}, Test={len(test_idx)}")
            yield train_idx, test_idx


class HoldOutValidation(ValidationStrategy):
    """Hold-out validation (train/val/test split)."""
    
    def __init__(
        self,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42
    ):
        """
        Initialize hold-out validation.
        
        Args:
            test_size: Test set fraction
            val_size: Validation set fraction
            random_state: Random seed
        """
        super().__init__(random_state)
        self.test_size = test_size
        self.val_size = val_size
    
    def split(self, X, y):
        """Generate train/val/test split."""
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
        )
        
        val_size_adj = self.val_size / (1 - self.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adj,
            random_state=self.random_state,
            stratify=y_temp
        )
        
        logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        yield {
            'train_idx': X_train.index.tolist() if hasattr(X_train, 'index') else list(range(len(X_train))),
            'val_idx': X_val.index.tolist() if hasattr(X_val, 'index') else list(range(len(X_val))),
            'test_idx': X_test.index.tolist() if hasattr(X_test, 'index') else list(range(len(X_test)))
        }
