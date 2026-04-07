"""
Configuration management for ML experiments.
Supports YAML-based configuration with environment variable overrides.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class DataConfig:
    """Data-related configuration."""
    raw_path: str = "data/raw"
    processed_path: str = "data/processed"
    splits_path: str = "data/splits"
    test_size: float = 0.2
    val_size: float = 0.1
    random_state: int = 42
    target_column: Optional[str] = None


@dataclass
class PreprocessingConfig:
    """Preprocessing-related configuration."""
    handle_missing: str = "mean"  # mean, median, drop, forward_fill
    scaling_method: str = "standard"  # standard, minmax, robust
    categorical_encoding: str = "onehot"  # onehot, label, target
    handle_outliers: bool = True
    outlier_method: str = "iqr"  # iqr, zscore
    handle_imbalance: bool = True
    imbalance_method: str = "smote"  # smote, adasyn, oversample, undersample


@dataclass
class ModelConfig:
    """Model training configuration."""
    model_type: str = "logistic_regression"
    hyperparameters: Dict[str, Any] = None
    random_state: int = 42
    n_jobs: int = -1
    
    def __post_init__(self):
        if self.hyperparameters is None:
            self.hyperparameters = {}


@dataclass
class TrainingConfig:
    """Training-related configuration."""
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    validation_strategy: str = "stratified_kfold"  # stratified_kfold, time_series
    n_splits: int = 5
    early_stopping_patience: int = 10
    metric_to_optimize: str = "f1"  # f1, auc, accuracy, precision, recall


@dataclass
class EvaluationConfig:
    """Evaluation-related configuration."""
    metrics: list = None
    create_plots: bool = True
    plot_types: list = None
    threshold: Optional[float] = None
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
        if self.plot_types is None:
            self.plot_types = ["confusion_matrix", "roc_curve", "precision_recall"]


@dataclass
class Config:
    """Main configuration class."""
    data: DataConfig = None
    preprocessing: PreprocessingConfig = None
    model: ModelConfig = None
    training: TrainingConfig = None
    evaluation: EvaluationConfig = None
    
    # General settings
    project_name: str = "classical_ml"
    experiment_name: str = "baseline"
    debug: bool = False
    random_state: int = 42
    
    # Paths
    config_path: str = "config"
    models_path: str = "models"
    logs_path: str = "logs"
    mlruns_path: str = "mlruns"
    
    def __post_init__(self):
        if self.data is None:
            self.data = DataConfig()
        if self.preprocessing is None:
            self.preprocessing = PreprocessingConfig()
        if self.model is None:
            self.model = ModelConfig()
        if self.training is None:
            self.training = TrainingConfig()
        if self.evaluation is None:
            self.evaluation = EvaluationConfig()
    
    @classmethod
    def from_yaml(cls, config_path: str) -> "Config":
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f) or {}

        # Override with environment variables (top-level keys only)
        for key, value in os.environ.items():
            if key.startswith('ML_'):
                config_key = key[3:].lower()
                if config_key in config_dict:
                    config_dict[config_key] = value

        # Build nested dataclass instances from nested dicts
        data_dict = config_dict.get('data', {}) or {}
        preprocessing_dict = config_dict.get('preprocessing', {}) or {}
        model_dict = config_dict.get('model', {}) or {}
        training_dict = config_dict.get('training', {}) or {}
        evaluation_dict = config_dict.get('evaluation', {}) or {}

        data = DataConfig(**data_dict)
        preprocessing = PreprocessingConfig(**preprocessing_dict)
        model = ModelConfig(**model_dict)
        training = TrainingConfig(**training_dict)
        evaluation = EvaluationConfig(**evaluation_dict)

        # Collect other top-level settings
        other_keys = {
            'project_name': config_dict.get('project_name', cls().project_name),
            'experiment_name': config_dict.get('experiment_name', cls().experiment_name),
            'debug': config_dict.get('debug', cls().debug),
            'random_state': config_dict.get('random_state', cls().random_state),
            'config_path': config_path,
        }

        return cls(
            data=data,
            preprocessing=preprocessing,
            model=model,
            training=training,
            evaluation=evaluation,
            **other_keys,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "data": asdict(self.data),
            "preprocessing": asdict(self.preprocessing),
            "model": asdict(self.model),
            "training": asdict(self.training),
            "evaluation": asdict(self.evaluation),
            "project_name": self.project_name,
            "experiment_name": self.experiment_name,
            "debug": self.debug,
            "random_state": self.random_state,
        }
    
    def to_yaml(self, output_path: str) -> None:
        """Save configuration to YAML file."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)


# Create default config instance
DEFAULT_CONFIG = Config()
