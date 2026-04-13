"""
Model training strategy using hyperparameter tuning.
"""

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from typing import Dict, Tuple, Optional, Any

from src.logger import get_logger

logger = get_logger(__name__)


class HyperparameterTuner:
    """Hyperparameter tuning utility."""
    
    def __init__(self, random_state: int = 42):
        """Initialize tuner."""
        self.random_state = random_state
    
    def grid_search(
        self,
        estimator,
        param_grid: Dict[str, list],
        X_train,
        y_train,
        cv: int = 5,
        scoring: str = "f1",
        **kwargs
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Perform grid search.
        
        Args:
            estimator: Sklearn estimator
            param_grid: Parameter grid
            X_train: Training features
            y_train: Training labels
            cv: Number of CV folds
            scoring: Scoring metric
            
        Returns:
            Tuple of (best_estimator, best_params)
        """
        logger.info("Starting Grid Search...")
        
        search = GridSearchCV(
            estimator,
            param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            verbose=1,
            **kwargs
        )
        
        search.fit(X_train, y_train)
        
        logger.info(f"Best score: {search.best_score_:.4f}")
        logger.info(f"Best params: {search.best_params_}")
        
        return search.best_estimator_, search.best_params_
    
    def random_search(
        self,
        estimator,
        param_distributions: Dict[str, list],
        X_train,
        y_train,
        n_iter: int = 10,
        cv: int = 5,
        scoring: str = "f1",
        **kwargs
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Perform random search.
        
        Args:
            estimator: Sklearn estimator
            param_distributions: Parameter distributions
            X_train: Training features
            y_train: Training labels
            n_iter: Number of iterations
            cv: Number of CV folds
            scoring: Scoring metric
            
        Returns:
            Tuple of (best_estimator, best_params)
        """
        logger.info("Starting Random Search...")
        
        search = RandomizedSearchCV(
            estimator,
            param_distributions,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            verbose=1,
            random_state=self.random_state,
            **kwargs
        )
        
        search.fit(X_train, y_train)
        
        logger.info(f"Best score: {search.best_score_:.4f}")
        logger.info(f"Best params: {search.best_params_}")
        
        return search.best_estimator_, search.best_params_
