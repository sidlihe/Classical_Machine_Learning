"""
Classical ML models with comprehensive implementations.
Includes Logistic Regression, SVM, KNN, Naive Bayes, and Decision Trees.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from typing import Dict, Any

from src.logger import get_logger

logger = get_logger(__name__)


class ModelFactory:
    """Factory for creating model instances."""
    
    _models = {
        "logistic_regression": LogisticRegression,
        "decision_tree": DecisionTreeClassifier,
        "knn": KNeighborsClassifier,
        "naive_bayes": GaussianNB,
        "svm": SVC,
        "dummy": DummyClassifier,
    }
    
    @classmethod
    def create(cls, model_name: str, **kwargs):
        """
        Create a model instance.
        
        Args:
            model_name: Name of the model
            **kwargs: Model parameters
            
        Returns:
            Model instance
        """
        if model_name not in cls._models:
            raise ValueError(f"Unknown model: {model_name}")
        
        model_class = cls._models[model_name]
        logger.info(f"Creating model: {model_name}")
        
        return model_class(**kwargs)
    
    @classmethod
    def get_available_models(cls) -> list:
        """Get list of available models."""
        return list(cls._models.keys())
    
    @classmethod
    def get_default_params(cls, model_name: str) -> Dict[str, Any]:
        """Get default parameters for a model."""
        defaults = {
            "logistic_regression": {
                "max_iter": 1000,
                "random_state": 42,
                "n_jobs": -1,
                "solver": "lbfgs",
            },
            "decision_tree": {
                "random_state": 42,
                "max_depth": 10,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
            },
            "knn": {
                "n_neighbors": 5,
                "n_jobs": -1,
            },
            "naive_bayes": {},
            "svm": {
                "kernel": "rbf",
                "C": 1.0,
                "gamma": "scale",
            },
            "dummy": {
                "strategy": "stratified",
            },
        }
        
        return defaults.get(model_name, {})


class BaselineClassifier:
    """Baseline classifier for comparison."""
    
    def __init__(self, strategy: str = "stratified"):
        """
        Initialize baseline classifier.
        
        Args:
            strategy: Strategy for baseline (stratified, most_frequent, uniform)
        """
        self.classifier = DummyClassifier(strategy=strategy)
        logger.info(f"Baseline classifier created with strategy: {strategy}")
    
    def fit(self, X, y):
        """Fit baseline classifier."""
        self.classifier.fit(X, y)
        return self
    
    def predict(self, X):
        """Make predictions."""
        return self.classifier.predict(X)
    
    def predict_proba(self, X):
        """Get probability predictions."""
        return self.classifier.predict_proba(X)
