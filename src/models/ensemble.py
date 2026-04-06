"""
Ensemble methods including Bagging, Boosting, Voting, and Stacking.
"""

from sklearn.ensemble import (
    RandomForestClassifier,
    BaggingClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from typing import List, Tuple

from src.logger import get_logger

logger = get_logger(__name__)


class EnsembleFactory:
    """Factory for creating ensemble models."""
    
    @staticmethod
    def create_bagging(
        base_estimator=None,
        n_estimators: int = 10,
        random_state: int = 42,
        **kwargs
    ):
        """Create bagging classifier."""
        if base_estimator is None:
            from sklearn.tree import DecisionTreeClassifier
            base_estimator = DecisionTreeClassifier(random_state=random_state)
        
        logger.info(f"Creating Bagging classifier with {n_estimators} estimators")
        return BaggingClassifier(
            estimator=base_estimator,
            n_estimators=n_estimators,
            random_state=random_state,
            **kwargs
        )
    
    @staticmethod
    def create_random_forest(
        n_estimators: int = 100,
        max_depth: int = None,
        random_state: int = 42,
        **kwargs
    ):
        """Create Random Forest classifier."""
        logger.info(f"Creating Random Forest with {n_estimators} estimators")
        return RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
            **kwargs
        )
    
    @staticmethod
    def create_adaboost(
        n_estimators: int = 50,
        learning_rate: float = 1.0,
        random_state: int = 42,
        **kwargs
    ):
        """Create AdaBoost classifier."""
        logger.info(f"Creating AdaBoost with {n_estimators} estimators")
        return AdaBoostClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state,
            **kwargs
        )
    
    @staticmethod
    def create_gradient_boosting(
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        random_state: int = 42,
        **kwargs
    ):
        """Create Gradient Boosting classifier."""
        logger.info(f"Creating Gradient Boosting with {n_estimators} estimators")
        return GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
            **kwargs
        )
    
    @staticmethod
    def create_voting(
        estimators: List[Tuple[str, object]],
        voting: str = "soft",
        **kwargs
    ):
        """Create Voting classifier."""
        logger.info(f"Creating Voting classifier with {len(estimators)} estimators")
        return VotingClassifier(
            estimators=estimators,
            voting=voting,
            **kwargs
        )
    
    @staticmethod
    def create_stacking(
        estimators: List[Tuple[str, object]],
        final_estimator=None,
        **kwargs
    ):
        """Create Stacking classifier."""
        if final_estimator is None:
            final_estimator = LogisticRegression(max_iter=1000, random_state=42)
        
        logger.info(f"Creating Stacking classifier with {len(estimators)} estimators")
        return StackingClassifier(
            estimators=estimators,
            final_estimator=final_estimator,
            **kwargs
        )
