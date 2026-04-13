# Getting Started Guide

## 🚀 Quick Setup

### 1. Activate Virtual Environment

```bash
# Windows
.\venv\Scripts\Activate.ps1

# Linux/Mac
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Verify Installation

```bash
# Check imports
python -c "import sklearn, xgboost, lightgbm, catboost, mlflow; print('✓ All imports successful')"

# Run basic test
pytest tests/test_utils.py -v
```

## 📊 Running Your First Training

### Option 1: Quick Start (Default Config)

```bash
cd Classical_Machine_Learning
python scripts/train.py
```

This will:
- Load credit card fraud dataset
- Validate data quality
- Scale features
- Train 4 models (Logistic Regression, Random Forest, Gradient Boosting, XGBoost)
- Compare metrics
- Save models to `models/`

### Option 2: Custom Configuration

1. Edit `config/creditcard_config.yaml`:
```yaml
data:
  test_size: 0.15
  target_column: "Class"

preprocessing:
  scaling_method: "robust"
  handle_imbalance: true

training:
  n_splits: 10
  metric_to_optimize: "roc_auc"
```

2. Run training:
```bash
python scripts/train.py --config config/creditcard_config.yaml
```

## 📈 View Results

### Check Training Logs

```bash
tail -f logs/*.log
```

### View Experiment Tracking

```bash
mlflow ui
```

Then open `http://localhost:5000` in your browser.

### List Trained Models

```bash
python -c "
from src.models.registry import ModelRegistry
registry = ModelRegistry('models')
print(registry.list_models())
"
```

## 🤖 Using the API

### Load and Use a Trained Model

```python
from src.models.registry import ModelRegistry
from src.data.loader import DataLoader

# Load registry and model
registry = ModelRegistry("models")
model = registry.load_model("logistic_regression_1.0.0")

# Load new data
loader = DataLoader("data/raw")
df = loader.load_csv("creditcard_fraud.csv")

# Make predictions
predictions = model.predict(df.drop("Class", axis=1))
probabilities = model.predict_proba(df.drop("Class", axis=1))
```

### Create Custom Configuration

```python
from src.config import Config, DataConfig, PreprocessingConfig

config = Config(
    data=DataConfig(target_column="Class"),
    preprocessing=PreprocessingConfig(scaling_method="robust"),
    project_name="my_project",
    experiment_name="exp_1"
)

# Convert to YAML
config.to_yaml("config/my_config.yaml")
```

### Feature Engineering

```python
from src.features.engineering import FeatureEngineer, FeatureScaler
import pandas as pd

df = pd.read_csv("data/raw/creditcard_fraud.csv")

# Create new features
engineer = FeatureEngineer()
df = engineer.create_polynomial_features(df, ["Amount", "Time"], degree=2)
df = engineer.create_interaction_features(df, ["V1", "V2", "V3"])

# Scale features
scaler = FeatureScaler("standard")
df_scaled = scaler.fit_transform(df)
```

### Data Validation

```python
from src.data.validator import DataValidator
import pandas as pd

df = pd.read_csv("data/raw/creditcard_fraud.csv")

validator = DataValidator(target_column="Class")
report = validator.full_validation(df)

print(report)
```

### Hyperparameter Tuning

```python
from src.training.hyperparameter import HyperparameterTuner
from src.models.classical import ModelFactory
from sklearn.model_selection import train_test_split

X, y = load_your_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create model
model = ModelFactory.create("logistic_regression")

# Tune hyperparameters
tuner = HyperparameterTuner()
best_model, best_params = tuner.grid_search(
    model,
    param_grid={"C": [0.001, 0.01, 0.1, 1, 10]},
    X_train=X_train,
    y_train=y_train,
    cv=5,
    scoring="f1"
)
```

## 🧪 Running Tests

```bash
# All tests
pytest tests/ -v

# Specific test file
pytest tests/test_utils.py -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

## 📚 Project Structure Review

```
Classical_Machine_Learning/
├── config/                    # ← Configuration files
├── data/
│   ├── raw/                  # ← Your raw datasets
│   ├── processed/            # ← Processed outputs
│   └── splits/               # ← Train/test splits
├── src/
│   ├── config.py             # ← Configuration management
│   ├── logger.py             # ← Logging setup
│   ├── data/                 # ← Data loading & validation
│   ├── features/             # ← Feature engineering
│   ├── models/               # ← Model factories & registry
│   ├── training/             # ← Training orchestration
│   ├── evaluation/           # ← Metrics & visualization
│   └── utils/                # ← Helpers & reproducibility
├── scripts/
│   ├── train.py              # ← Main training script
│   └── evaluate.py           # ← Evaluation script (coming)
├── notebooks/                # ← Jupyter notebooks
├── tests/                    # ← Unit tests
├── models/                   # ← Trained models
├── logs/                     # ← Application logs
├── mlruns/                   # ← MLflow artifacts
├── requirements.txt          # ← Dependencies
└── README.md                 # ← Documentation
```

## 🔧 Troubleshooting

### Import Errors

```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# Check Python path
python -c "import sys; print(sys.path)"
```

### Models Not Saving

```bash
# Check models directory exists
mkdir -p models/

# Check permissions
ls -la models/
```

### Data Not Found

```bash
# Verify data files
ls -la data/raw/

# Adjust paths in config
# Or pass custom path to DataLoader("custom/path")
```

## 🚀 Next Steps

1. **Load Your Data**: Place CSV files in `data/raw/`
2. **Configure**: Edit `config/creditcard_config.yaml`
3. **Train**: Run `python scripts/train.py`
4. **Monitor**: Use `mlflow ui` to view results
5. **Deploy**: Use registered models for predictions

## 📖 Documentation

- [Main README](README.md) - Comprehensive project overview
- [Configuration Guide](config/creditcard_config.yaml) - All config options
- [API Reference](#) - Module documentation (in progress)

## 💡 Tips

- Use `set_seed(42)` for reproducibility
- Config values can be overridden with environment variables (`ML_*`)
- Logs are automatically saved to `logs/` directory
- MLflow automatically tracks all experiments in `mlruns/`

## ❓ Common Issues

**Q: ModuleNotFoundError when importing src?**
A: Make sure you're running scripts from project root and virtual environment is activated

**Q: Memory issues with large datasets?**
A: Use `iterator=True` in data loader or process in batches

**Q: Models not loading?**
A: Check model path and ensure pickle file isn't corrupted

---

**Happy machine learning! 🎉**
