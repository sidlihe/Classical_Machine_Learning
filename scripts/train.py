"""
Main training script for ML pipeline.
Orchestrates data loading, preprocessing, training, and evaluation.
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
from src.logger import get_logger
from src.config import Config
from src.data.loader import DataLoader
from src.data.validator import DataValidator
from src.features.engineering import FeatureScaler
from src.training.validation import StratifiedKFoldValidation
from src.evaluation.metrics import ClassificationMetrics, ModelEvaluator
from src.models.registry import ModelRegistry, MLflowTracker
from src.utils.reproducibility import set_seed
from src.utils.helpers import ensure_dirs

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

logger = get_logger(__name__)


def main(config_path: str = "config/creditcard_config.yaml"):
    """
    Main training pipeline.
    
    Args:
        config_path: Path to configuration file
    """
    # Initialize
    logger.info("=" * 70)
    logger.info("STARTING ML TRAINING PIPELINE")
    logger.info("=" * 70)
    
    # Load configuration
    if Path(config_path).exists():
        config = Config.from_yaml(config_path)
    else:
        config = Config()
    
    logger.info(f"Configuration: {config.project_name}/{config.experiment_name}")
    
    # Set seed for reproducibility
    set_seed(config.random_state)
    
    # Ensure directories exist
    ensure_dirs(
        config.data.raw_path,
        config.data.processed_path,
        config.data.splits_path,
        config.models_path,
        config.logs_path,
        config.mlruns_path
    )
    
    # ===== STEP 1: LOAD DATA =====
    logger.info("\n" + "=" * 70)
    logger.info("STEP 1: DATA LOADING")
    logger.info("=" * 70)
    
    data_loader = DataLoader(config.data.raw_path)
    df = data_loader.load_csv("creditcard_fraud.csv")
    
    # ===== STEP 2: VALIDATE DATA =====
    logger.info("\n" + "=" * 70)
    logger.info("STEP 2: DATA VALIDATION")
    logger.info("=" * 70)
    
    validator = DataValidator(target_column=config.data.target_column)
    validation_report = validator.full_validation(df)
    
    # ===== STEP 3: PREPROCESSING =====
    logger.info("\n" + "=" * 70)
    logger.info("STEP 3: PREPROCESSING")
    logger.info("=" * 70)
    
    # Separate features and target
    X = df.drop(columns=[config.data.target_column])
    y = df[config.data.target_column]
    
    logger.info(f"Features shape: {X.shape}")
    logger.info(f"Target distribution:\n{y.value_counts()}")
    
    # Scale features
    scaler = FeatureScaler(method=config.preprocessing.scaling_method)
    X_scaled = scaler.fit_transform(X)
    
    # ===== STEP 4: TRAIN/TEST SPLIT =====
    logger.info("\n" + "=" * 70)
    logger.info("STEP 4: TRAIN/TEST SPLIT")
    logger.info("=" * 70)
    
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
        test_size=config.data.test_size,
        random_state=config.random_state,
        stratify=y
    )
    
    logger.info(f"Train set: {X_train.shape}")
    logger.info(f"Test set: {X_test.shape}")
    logger.info(f"Class balance - Train: {y_train.value_counts().to_dict()}")
    logger.info(f"Class balance - Test: {y_test.value_counts().to_dict()}")
    
    # ===== STEP 5: HANDLE IMBALANCE =====
    logger.info("\n" + "=" * 70)
    logger.info("STEP 5: HANDLING IMBALANCE")
    logger.info("=" * 70)
    
    if config.preprocessing.handle_imbalance:
        from imblearn.over_sampling import SMOTE
        
        smote = SMOTE(random_state=config.random_state, n_jobs=-1)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        
        logger.info(f"After SMOTE - Train: {y_train_balanced.value_counts().to_dict()}")
        X_train, y_train = X_train_balanced, y_train_balanced
    
    # ===== STEP 6: TRAIN MODELS =====
    logger.info("\n" + "=" * 70)
    logger.info("STEP 6: MODEL TRAINING")
    logger.info("=" * 70)
    
    # Initialize MLflow tracker
    mlflow_tracker = MLflowTracker(config.experiment_name)
    model_registry = ModelRegistry(config.models_path)
    
    # Initialize evaluator
    evaluator = ModelEvaluator(task="classification")
    
    # Define models
    models = {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            random_state=config.random_state,
            n_jobs=-1
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            random_state=config.random_state,
            n_jobs=-1
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=100,
            random_state=config.random_state
        ),
        "xgboost": XGBClassifier(
            n_estimators=100,
            random_state=config.random_state,
            n_jobs=-1
        ),
    }
    
    results = {}
    
    for model_name, model in models.items():
        logger.info(f"\nTraining {model_name}...")
        
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
        
        # Evaluate
        metrics = evaluator.evaluate(y_test, y_pred, y_pred_proba, dataset_name=model_name)
        results[model_name] = metrics
        
        # Register model
        model_registry.register_model(
            model=model,
            name=model_name,
            version="1.0.0",
            metrics=metrics,
            description=f"Trained on credit card fraud data"
        )
        
        # Log to MLflow
        mlflow_tracker.start_run(run_name=model_name)
        mlflow_tracker.log_params({"model_type": model_name})
        mlflow_tracker.log_metrics(metrics)
        mlflow_tracker.log_model(model)
        mlflow_tracker.end_run()
    
    # ===== STEP 7: MODEL COMPARISON =====
    logger.info("\n" + "=" * 70)
    logger.info("STEP 7: MODEL COMPARISON")
    logger.info("=" * 70)
    
    comparison_df = evaluator.compare_models(results)
    
    # Save comparison
    comparison_df.to_csv(f"{config.models_path}/model_comparison.csv")
    logger.info(f"Model comparison saved to {config.models_path}/model_comparison.csv")
    
    # ===== COMPLETE =====
    logger.info("\n" + "=" * 70)
    logger.info("ML TRAINING PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ML models")
    parser.add_argument(
        "--config",
        type=str,
        default="config/creditcard_config.yaml",
        help="Path to configuration file"
    )
    
    args = parser.parse_args()
    main(config_path=args.config)
