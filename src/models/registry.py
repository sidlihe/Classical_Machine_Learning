"""
Model factory and registry for managing ML models.
Create, save, load, and track models across experiments.
"""

import pickle
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import mlflow
from sklearn.base import BaseEstimator

from src.logger import get_logger

logger = get_logger(__name__)


class ModelRegistry:
    """Registry for managing models."""
    
    def __init__(self, registry_dir: str = "models"):
        """
        Initialize model registry.
        
        Args:
            registry_dir: Directory to store models
        """
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(exist_ok=True)
        self.metadata_file = self.registry_dir / "registry.json"
        self.models = self._load_registry()
    
    def _load_registry(self) -> Dict[str, Any]:
        """Load existing registry from file."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_registry(self) -> None:
        """Save registry to file."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.models, f, indent=4, default=str)
    
    def register_model(
        self,
        model: BaseEstimator,
        name: str,
        version: str = "1.0.0",
        metrics: Optional[Dict[str, float]] = None,
        tags: Optional[Dict[str, str]] = None,
        description: str = ""
    ) -> str:
        """
        Register and save a model.
        
        Args:
            model: Trained sklearn model
            name: Model name
            version: Model version
            metrics: Dictionary of evaluation metrics
            tags: Dictionary of tags
            description: Model description
            
        Returns:
            Model file path
        """
        timestamp = datetime.now().isoformat()
        model_filename = f"{name}_{version}_{timestamp.replace(':', '-')}.pkl"
        model_path = self.registry_dir / model_filename
        
        # Save model
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Register metadata
        model_id = f"{name}_{version}"
        self.models[model_id] = {
            "name": name,
            "version": version,
            "filename": model_filename,
            "filepath": str(model_path),
            "timestamp": timestamp,
            "metrics": metrics or {},
            "tags": tags or {},
            "description": description,
            "model_type": model.__class__.__name__,
        }
        
        self._save_registry()
        logger.info(f"Registered model: {model_id}")
        
        return str(model_path)
    
    def load_model(self, model_id: str) -> Optional[BaseEstimator]:
        """
        Load a registered model.
        
        Args:
            model_id: Model identifier (name_version)
            
        Returns:
            Loaded model or None if not found
        """
        if model_id not in self.models:
            logger.error(f"Model not found: {model_id}")
            return None
        
        model_info = self.models[model_id]
        model_path = model_info["filepath"]
        
        if not Path(model_path).exists():
            logger.error(f"Model file not found: {model_path}")
            return None
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        logger.info(f"Loaded model: {model_id}")
        return model
    
    def list_models(self) -> Dict[str, Any]:
        """List all registered models."""
        logger.info(f"Found {len(self.models)} registered models")
        return self.models
    
    def get_best_model(
        self,
        metric_name: str = "f1",
        name_filter: Optional[str] = None
    ) -> Optional[str]:
        """
        Get best model based on metric.
        
        Args:
            metric_name: Metric to use for ranking
            name_filter: Filter models by name pattern
            
        Returns:
            Best model ID or None
        """
        candidates = self.models
        
        if name_filter:
            candidates = {k: v for k, v in candidates.items() if name_filter in k}
        
        best_model_id = None
        best_score = -float('inf')
        
        for model_id, info in candidates.items():
            score = info.get('metrics', {}).get(metric_name, -float('inf'))
            if score > best_score:
                best_score = score
                best_model_id = model_id
        
        if best_model_id:
            logger.info(f"Best model: {best_model_id} (score: {best_score:.4f})")
        
        return best_model_id


class MLflowTracker:
    """Track experiments with MLflow."""
    
    def __init__(self, experiment_name: str = "default"):
        """
        Initialize MLflow tracker.
        
        Args:
            experiment_name: Name of the experiment
        """
        self.experiment_name = experiment_name
        mlflow.set_experiment(experiment_name)
        logger.info(f"Initialized MLflow experiment: {experiment_name}")
    
    def start_run(self, run_name: str) -> None:
        """Start a new MLflow run."""
        mlflow.start_run(run_name=run_name)
        logger.info(f"Started MLflow run: {run_name}")
    
    def end_run(self) -> None:
        """End current MLflow run."""
        mlflow.end_run()
        logger.info("Ended MLflow run")
    
    def log_param(self, key: str, value: Any) -> None:
        """Log parameter."""
        mlflow.log_param(key, value)
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """Log multiple parameters."""
        mlflow.log_params(params)
    
    def log_metric(self, key: str, value: float) -> None:
        """Log metric."""
        mlflow.log_metric(key, value)
    
    def log_metrics(self, metrics: Dict[str, float]) -> None:
        """Log multiple metrics."""
        mlflow.log_metrics(metrics)
    
    def log_model(self, model: BaseEstimator, artifact_path: str = "model") -> None:
        """Log model artifact."""
        mlflow.sklearn.log_model(model, artifact_path)
