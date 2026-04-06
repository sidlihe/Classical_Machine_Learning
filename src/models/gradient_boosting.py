"""
Gradient Boosting models: XGBoost, LightGBM, CatBoost.
"""

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from typing import Dict, Any

from src.logger import get_logger

logger = get_logger(__name__)


class GradientBoostingFactory:
    """Factory for gradient boosting models."""
    
    @staticmethod
    def create_xgboost(
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 5,
        random_state: int = 42,
        **kwargs
    ) -> XGBClassifier:
        """
        Create XGBoost classifier.
        
        Args:
            n_estimators: Number of boosting rounds
            learning_rate: Shrinkage (eta)
            max_depth: Maximum depth of trees
            random_state: Random seed
            **kwargs: Additional XGBoost parameters
            
        Returns:
            XGBClassifier instance
        """
        logger.info(f"Creating XGBoost with {n_estimators} estimators")
        return XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
            tree_method="hist",
            **kwargs
        )
    
    @staticmethod
    def create_lightgbm(
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 5,
        num_leaves: int = 31,
        random_state: int = 42,
        **kwargs
    ) -> LGBMClassifier:
        """
        Create LightGBM classifier.
        
        Args:
            n_estimators: Number of boosting rounds
            learning_rate: Learning rate (shrinkage)
            max_depth: Maximum depth of trees
            num_leaves: Number of leaves
            random_state: Random seed
            **kwargs: Additional LightGBM parameters
            
        Returns:
            LGBMClassifier instance
        """
        logger.info(f"Creating LightGBM with {n_estimators} estimators")
        return LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            num_leaves=num_leaves,
            random_state=random_state,
            n_jobs=-1,
            **kwargs
        )
    
    @staticmethod
    def create_catboost(
        iterations: int = 100,
        learning_rate: float = 0.1,
        depth: int = 5,
        random_state: int = 42,
        verbose: bool = False,
        **kwargs
    ) -> CatBoostClassifier:
        """
        Create CatBoost classifier.
        
        Args:
            iterations: Number of boosting iterations
            learning_rate: Learning rate
            depth: Tree depth
            random_state: Random seed
            verbose: Print progress
            **kwargs: Additional CatBoost parameters
            
        Returns:
            CatBoostClassifier instance
        """
        logger.info(f"Creating CatBoost with {iterations} iterations")
        return CatBoostClassifier(
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            random_state=random_state,
            verbose=verbose,
            **kwargs
        )
    
    @staticmethod
    def get_default_hparams(model_name: str) -> Dict[str, Any]:
        """Get default hyperparameters for a model."""
        defaults = {
            "xgboost": {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 5,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
            },
            "lightgbm": {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 5,
                "num_leaves": 31,
                "feature_fraction": 0.8,
            },
            "catboost": {
                "iterations": 100,
                "learning_rate": 0.1,
                "depth": 5,
                "subsample": 0.8,
            },
        }
        
        return defaults.get(model_name, {})
