"""
Evaluation metrics and reporting.
Comprehensive metrics calculation and visualization.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, matthews_corrcoef, cohen_kappa_score,
    confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)
from src.logger import get_logger

logger = get_logger(__name__)


class ClassificationMetrics:
    """Calculate classification metrics."""
    
    @staticmethod
    def calculate_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None,
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        Calculate comprehensive classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Probability predictions
            threshold: Classification threshold
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "mcc": matthews_corrcoef(y_true, y_pred),
            "kappa": cohen_kappa_score(y_true, y_pred),
            # "specificity": specificity_score(y_true, y_pred, zero_division=0),
        }
        
        # ROC-AUC (if probabilities available)
        if y_pred_proba is not None:
            if y_pred_proba.ndim == 2:
                # Binary or multiclass
                if y_pred_proba.shape[1] == 2:
                    metrics["roc_auc"] = roc_auc_score(y_true, y_pred_proba[:, 1])
                else:
                    metrics["roc_auc"] = roc_auc_score(
                        y_true, y_pred_proba, multi_class="ovr", zero_division=0
                    )
            else:
                metrics["roc_auc"] = roc_auc_score(y_true, y_pred_proba)
        
        return metrics
    
    @staticmethod
    def get_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Calculate confusion matrix."""
        return confusion_matrix(y_true, y_pred)
    
    @staticmethod
    def get_classification_report(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        output_dict: bool = False
    ):
        """Get detailed classification report."""
        return classification_report(y_true, y_pred, output_dict=output_dict, zero_division=0)


class RegressionMetrics:
    """Calculate regression metrics."""
    
    @staticmethod
    def calculate_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate comprehensive regression metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary of metrics
        """
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100 if np.any(y_true != 0) else 0
        
        return {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "mape": mape,
            "r2": r2,
        }


class ModelEvaluator:
    """Comprehensive model evaluation."""
    
    def __init__(self, task: str = "classification"):
        """
        Initialize evaluator.
        
        Args:
            task: Task type (classification, regression)
        """
        self.task = task
        self.evaluation_results = {}
    
    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None,
        dataset_name: str = "test"
    ) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            y_true: True labels/values
            y_pred: Predicted labels/values
            y_pred_proba: Probability predictions (classification)
            dataset_name: Name of dataset (train, val, test)
            
        Returns:
            Dictionary of metrics
        """
        if self.task == "classification":
            metrics = ClassificationMetrics.calculate_metrics(
                y_true, y_pred, y_pred_proba
            )
        else:
            metrics = RegressionMetrics.calculate_metrics(y_true, y_pred)
        
        self.evaluation_results[dataset_name] = metrics
        
        logger.info(f"\n{dataset_name.upper()} SET METRICS:")
        logger.info("=" * 50)
        for metric_name, metric_value in metrics.items():
            logger.info(f"{metric_name:20s}: {metric_value:.4f}")
        logger.info("=" * 50)
        
        return metrics
    
    def compare_models(self, results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """
        Compare metrics across models.
        
        Args:
            results: Dictionary of {model_name: {metric: value}}
            
        Returns:
            DataFrame comparing model metrics
        """
        df = pd.DataFrame(results).T
        logger.info("\nMODEL COMPARISON:")
        logger.info(df.to_string())
        
        return df
