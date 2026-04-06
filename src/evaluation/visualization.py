"""
Visualization utilities for ML model analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Tuple, Optional
from sklearn.metrics import (
    confusion_matrix, roc_curve, precision_recall_curve,
    ConfusionMatrixDisplay, RocCurveDisplay
)

from src.logger import get_logger

logger = get_logger(__name__)


class ModelVisualizer:
    """Visualization utilities for model evaluation."""
    
    @staticmethod
    def plot_confusion_matrix(
        y_true,
        y_pred,
        title: str = "Confusion Matrix",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved confusion matrix to {save_path}")
        
        return fig
    
    @staticmethod
    def plot_roc_curve(
        y_true,
        y_pred_proba,
        title: str = "ROC Curve",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot ROC curve.
        
        Args:
            y_true: True labels
            y_pred_proba: Probability predictions
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1] if y_pred_proba.ndim > 1 else y_pred_proba)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC={np.trapz(tpr, fpr):.3f})')
        ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved ROC curve to {save_path}")
        
        return fig
    
    @staticmethod
    def plot_precision_recall(
        y_true,
        y_pred_proba,
        title: str = "Precision-Recall Curve",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot precision-recall curve.
        
        Args:
            y_true: True labels
            y_pred_proba: Probability predictions
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        precision, recall, _ = precision_recall_curve(
            y_true, 
            y_pred_proba[:, 1] if y_pred_proba.ndim > 1 else y_pred_proba
        )
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(recall, precision, linewidth=2)
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved precision-recall curve to {save_path}")
        
        return fig
    
    @staticmethod
    def plot_feature_importance(
        feature_names,
        importances,
        top_n: int = 20,
        title: str = "Feature Importance",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot feature importance.
        
        Args:
            feature_names: Names of features
            importances: Importance scores
            top_n: Number of top features to show
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        # Sort by importance
        indices = np.argsort(importances)[::-1][:top_n]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(range(len(indices)), importances[indices])
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.set_xlabel('Importance')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved feature importance to {save_path}")
        
        return fig
