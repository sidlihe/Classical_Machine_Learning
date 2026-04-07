# Classical Machine Learning - Production-Ready System

A **state-of-the-art ML project** showcasing end-to-end implementation of classical algorithms with production-grade architecture, experiment tracking, CI/CD integration, and advanced validation strategies.

## 🎯 Project Overview

This project demonstrates:
- ✅ **Professional folder structure** with modular code organization
- ✅ **Centralized logging & configuration** management
- ✅ **Complete ML lifecycle**: Feature engineering, preprocessing, training, evaluation
- ✅ **Multiple algorithms**: Classical ML, ensemble methods, gradient boosting
- ✅ **Advanced validation**: Stratified K-fold, time series, hold-out splits
- ✅ **Model registry & versioning** with artifact management
- ✅ **Experiment tracking** with MLflow
- ✅ **CI/CD pipelines** with GitHub Actions
- ✅ **Reproducibility**: Seed management, config versioning
- ✅ **Scalable & extensible** architecture

## 📁 Project Structure

```
Classical_Machine_Learning/
├── .github/workflows/          # CI/CD pipelines (GitHub Actions)
├── .dvc/                       # Data version control
├── config/                     # YAML configuration files
│   └── creditcard_config.yaml
├── data/
│   ├── raw/                    # Raw datasets (DVC tracked)
│   ├── processed/              # Processed datasets
│   └── splits/                 # Train/val/test splits
├── src/                        # Main source code
│   ├── __init__.py
│   ├── logger.py               # Centralized logging
│   ├── config.py               # Configuration management
│   ├── data/
│   │   ├── loader.py           # Data loading & caching
│   │   └── validator.py        # Data quality checks
│   ├── features/
│   │   └── engineering.py      # Feature creation & selection
│   ├── models/
│   │   └── registry.py         # Model registry & MLflow integration
│   ├── training/
│   │   └── validation.py       # Cross-validation strategies
│   ├── evaluation/
│   │   └── metrics.py          # Metrics & reporting
│   └── utils/
│       ├── helpers.py          # Utility functions
│       └── reproducibility.py  # Seed management
├── scripts/
│   ├── train.py                # Main training orchestrator
│   ├── evaluate.py             # Evaluation script (coming)
│   └── predict.py              # Inference script (coming)
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_training.ipynb
├── tests/                      # Unit & integration tests (coming)
├── models/                     # Saved models & registry (DVC tracked)
├── logs/                       # Application logs
├── mlruns/                     # MLflow experiment tracking
├── pyproject.toml             # Modern Python packaging
├── setup.py                   # Package installation (coming)
├── requirements.txt           # Python dependencies
├── .gitignore                 # Git ignore patterns
├── .dvcignore                 # DVC ignore patterns
├── dvc.yaml                   # DVC pipeline definition (coming)
└── README.md                  # Project documentation
```

## 🚀 Quick Start

### 1. **Environment Setup**

```bash
# Clone repository
git clone <your-repo>
cd Classical_Machine_Learning

# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows)
.\venv\Scripts\Activate.ps1

# Activate virtual environment (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install jupyter black flake8 pytest
```

### 2. **Configure Your Experiment**

Edit `config/creditcard_config.yaml` to customize:
- Data paths and splits
- Preprocessing methods
- Model hyperparameters
- Training strategies
- Evaluation metrics

### 3. **Run Training Pipeline**

```bash
# Run with default config
python scripts/train.py

# Run with custom config
python scripts/train.py --config config/custom_config.yaml
```

### 4. **View Experiment Tracking**

```bash
# Launch MLflow UI
mlflow ui
```

Browse to `http://localhost:5000` to view experiments, metrics, and artifacts.

## 📊 Data Pipeline

### Step 1: Data Loading
```python
from src.data.loader import DataLoader

loader = DataLoader("data/raw")
df = loader.load_csv("creditcard_fraud.csv")
```

### Step 2: Data Validation
```python
from src.data.validator import DataValidator

validator = DataValidator(target_column="Class")
report = validator.full_validation(df)
```

### Step 3: Feature Engineering
```python
from src.features.engineering import FeatureEngineer, FeatureScaler

engineer = FeatureEngineer()
df = engineer.create_polynomial_features(df, ["Amount", "Time"], degree=2)

scaler = FeatureScaler(method="standard")
X_scaled = scaler.fit_transform(df)
```

### Step 4: Train/Test Split
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
```

### Step 5: Handle Imbalance
```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)
```

## 🤖 Supported Models

### Classical ML
- Logistic Regression
- Decision Trees
- K-Nearest Neighbors
- Naive Bayes
- Support Vector Machines

### Ensemble Methods
- Random Forest (Bagging)
- AdaBoost
- Gradient Boosting
- Voting Classifier
- Stacking Classifier

### Gradient Boosting
- XGBoost
- LightGBM
- CatBoost

## 📈 Validation Strategies

1. **Stratified K-Fold** - Maintains class distribution
2. **Time Series** - For sequential data
3. **Hold-out** - Train/Val/Test splitting

```python
from src.training.validation import StratifiedKFoldValidation

validation = StratifiedKFoldValidation(n_splits=5)
for train_idx, test_idx in validation.split(X, y):
    # Train on fold
    pass
```

## 📊 Evaluation & Metrics

Comprehensive metrics for classification:
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC, Matthews Correlation Coefficient
- Confusion Matrix, Classification Report
- Custom metric combinations

```python
from src.evaluation.metrics import ClassificationMetrics

metrics = ClassificationMetrics.calculate_metrics(y_true, y_pred, y_pred_proba)
```

## 🎛️ Configuration Management

**YAML-based configuration** with environment variable overrides:

```yaml
data:
  test_size: 0.2
  val_size: 0.1
  target_column: "Class"

preprocessing:
  scaling_method: "standard"
  handle_imbalance: true
  imbalance_method: "smote"

training:
  n_splits: 5
  metric_to_optimize: "f1"
```

**Override with environment variables:**
```bash
export ML_test_size=0.15
python scripts/train.py
```

## 🔧 Logging System

Centralized logging to **both file and console**:

```python
from src.logger import get_logger

logger = get_logger(__name__)
logger.info("Training started")
logger.warning("Potential issue detected")
logger.error("Critical error occurred")
```

Logs are saved to `logs/` directory with automatic rotation.

## 🏆 Model Registry

Track, version, and compare trained models:

```python
from src.models.registry import ModelRegistry

registry = ModelRegistry("models")
registry.register_model(
    model=model,
    name="logistic_regression",
    version="1.0.0",
    metrics={"f1": 0.95, "auc": 0.92}
)

# Load best model
best_model_id = registry.get_best_model(metric_name="f1")
model = registry.load_model(best_model_id)
```

## 📡 Experiment Tracking with MLflow

Automatically log experiments with MLflow:

```python
from src.models.registry import MLflowTracker

tracker = MLflowTracker(experiment_name="creditcard_fraud")
tracker.start_run("baseline_v1")
tracker.log_params({"n_estimators": 100})
tracker.log_metrics({"f1": 0.95, "auc": 0.92})
tracker.log_model(model)
tracker.end_run()
```

## 🔄 CI/CD Integration

GitHub Actions workflows automatically:
- Run tests on every push
- Execute training pipeline
- Track metrics & artifacts
- Deploy models

See `.github/workflows/` for pipeline definitions.

## 🧪 Testing

Run unit tests:

```bash
pytest tests/
```

## 📚 Datasets

### Credit Card Fraud Detection
- **Size**: ~284,807 transactions
- **Features**: Time, Amount, V1-V28 (PCA-transformed)
- **Target**: Binary (fraud/non-fraud)
- **Challenge**: Highly imbalanced dataset (0.1% fraud)

### Telco Customer Churn
- **Size**: ~7,043 customers
- **Features**: Account info, service usage, demographics
- **Target**: Binary (churn/no-churn)
- **Challenge**: Mixed data types, categorical features

## 🔮 Advanced Topics

### Custom Transformers
Create sklearn-compatible custom transformers:

```python
from sklearn.base import BaseEstimator, TransformerMixin

class CustomTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X
```

### Feature Selection
Multiple strategies for feature selection:

```python
from src.features.engineering import FeatureSelector

selector = FeatureSelector(method="SelectKBest", n_features=10)
selected = selector.select_features(X, y, task="classification")
```

### Hyperparameter Optimization
Grid search and random search:

```python
from sklearn.model_selection import GridSearchCV

params = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
search = GridSearchCV(SVC(), params, cv=5)
search.fit(X, y)
```

## 📖 Documentation

- **Code Documentation**: Comprehensive docstrings in all modules
- **Configuration Guide**: See `config/creditcard_config.yaml` example
- **Logging Format**: Timestamps, level, module name, message

## 🚦 Reproducibility

Ensure reproducibility across runs:

```python
from src.utils.reproducibility import set_seed

set_seed(42)  # Sets seed for Python, NumPy, PyTorch
```

## 🤝 Contributing

Areas for contribution:
- [ ] Add more ML algorithms
- [ ] Implement DVC pipeline (`dvc.yaml`)
- [ ] Add Docker containerization
- [ ] Extend test coverage
- [ ] Add API endpoint for predictions
- [ ] Implement model serving

## 📝 License

This project is open source and available under the MIT License.

## 🎓 Learning Resources

- Scikit-learn documentation: https://scikit-learn.org/
- XGBoost guide: https://xgboost.readthedocs.io/
- MLflow docs: https://mlflow.org/docs/
- Imbalanced-learn: https://imbalanced-learn.org/

## 📧 Support

For questions or issues:
1. Check existing documentation
2. Review code examples in notebooks/
3. Run tests to verify setup: `pytest tests/`
4. Check logs in logs/ directory

---

## OWN Conclusion
MODEL COMPARISON:
2026-04-07 14:38:42 - src.evaluation.metrics - INFO -                      accuracy  precision    recall        f1       mcc     kappa   roc_auc
logistic_regression  0.974141   0.057878  0.918367  0.108893  0.227009  0.105999  0.970988
random_forest        0.999491   0.870968  0.826531  0.848168  0.848204  0.847913  0.978255
gradient_boosting    0.986956   0.109091  0.918367  0.195016  0.314080  0.192533  0.981793
xgboost              0.999087   0.688525  0.857143  0.763636  0.767784  0.763184  0.983126

**Built with ❤️ for production-grade ML systems**
