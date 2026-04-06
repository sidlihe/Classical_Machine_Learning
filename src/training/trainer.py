"""
Training orchestrator for the complete ML pipeline.
Handles model training, validation, hyperparameter tuning, etc.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Any
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer

from src.logger import get_logger
from src.evaluation.metrics import ClassificationMetrics

logger = get_logger(__name__)


class Trainer:
    """Main training orchestrator."""
    
    def __init__(self, random_state: int = 42):
        """Initialize trainer."""
        self.random_state = random_state
        self.trained_models = {}
        self.best_model = None
        self.best_score = -np.inf
    
    def train_model(
        self,
        model,
        X_train: np.ndarray,
        y_train: np.ndarray,
        model_name: str = "model"
    ):
        """
        Train a single model.
        
        Args:
            model: Sklearn model
            X_train: Training features
            y_train: Training labels
            model_name: Name to store the model
        """
        logger.info(f"Training model: {model_name}")
        model.fit(X_train, y_train)
        self.trained_models[model_name] = model
        logger.info(f"Model trained: {model_name}")
        
        return model
    
    def hyperparameter_search(
        self,
        model,
        X_train: np.ndarray,
        y_train: np.ndarray,
        param_grid: Dict[str, list],
        cv: int = 5,
        scoring: str = "f1",
        search_type: str = "grid",
        **kwargs
    ):
        """
        Perform hyperparameter search.
        
        Args:
            model: Sklearn model
            X_train: Training features
            y_train: Training labels
            param_grid: Parameter grid for search
            cv: Number of CV folds
            scoring: Scoring metric
            search_type: "grid" or "random"
            **kwargs: Additional parameters for search
            
        Returns:
            Best model and search results
        """
        logger.info(f"Starting {search_type} hyperparameter search")
        logger.info(f"Parameters grid: {param_grid}")
        
        if search_type == "grid":
            search = GridSearchCV(
                model,
                param_grid,
                cv=cv,
                scoring=scoring,
                n_jobs=-1,
                verbose=1,
                **kwargs
            )
        elif search_type == "random":
            search = RandomizedSearchCV(
                model,
                param_grid,
                cv=cv,
                scoring=scoring,
                n_iter=10,
                n_jobs=-1,
                verbose=1,
                random_state=self.random_state,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown search type: {search_type}")
        
        search.fit(X_train, y_train)
        
        logger.info(f"Best parameters: {search.best_params_}")
        logger.info(f"Best score: {search.best_score_:.4f}")
        
        return search.best_estimator_, search
    
    def evaluate_on_multiple_metrics(
        self,
        model,
        X_test: np.ndarray,
        y_test: np.ndarray,
        metrics: list = None
    ) -> Dict[str, float]:
        """
        Evaluate model on multiple metrics.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            metrics: List of metrics to calculate
            
        Returns:
            Dictionary of metrics
        """
        if metrics is None:
            metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
        
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
        
        results = ClassificationMetrics.calculate_metrics(
            y_test, y_pred, y_pred_proba
        )
        
        return results
    
    def compare_models(
        self,
        models: Dict[str, object],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> pd.DataFrame:
        """
        Train and compare multiple models.
        
        Args:
            models: Dictionary of {name: model} pairs
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            
        Returns:
            DataFrame comparing model metrics
        """
        results = {}
        
        for model_name, model in models.items():
            logger.info(f"\n{'='*50}")
            logger.info(f"Training: {model_name}")
            logger.info(f"{'='*50}")
            
            # Train
            self.train_model(model, X_train, y_train, model_name)
            
            # Evaluate
            metrics = self.evaluate_on_multiple_metrics(model, X_test, y_test)
            results[model_name] = metrics
            
            # Track best model
            f1_score = metrics.get('f1', 0)
            if f1_score > self.best_score:
                self.best_score = f1_score
                self.best_model = model
        
        df = pd.DataFrame(results).T
        logger.info("\nMODEL COMPARISON:")
        logger.info(df.to_string())
        
        return df
    
    def get_best_model(self):
        """Get best trained model."""
        if self.best_model is None:
            logger.warning("No model trained yet")
        return self.best_model
