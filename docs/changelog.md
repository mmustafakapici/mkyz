# Changelog

All notable changes to MKYZ are documented here.

---

## [0.2.0] - 2024-01-15

### 🚀 New Features

#### Core Module (`mkyz.core`)
- **Config class** - Global configuration with `get_config()`, `set_config()`
- **Custom exceptions** - `MKYZError`, `DataValidationError`, `ModelNotTrainedError`, `PersistenceError`
- **Base classes** - `BaseEstimator`, `DataMixin`, `ModelMixin`

#### Persistence Module (`mkyz.persistence`)
- **Model saving** - `save_model()` with joblib/pickle support
- **Model loading** - `load_model()` with metadata retrieval
- **Pipeline export** - `export_pipeline()`, `import_pipeline()` for JSON configs

#### Evaluation Module (`mkyz.evaluation`)
- **Comprehensive metrics** - `classification_metrics()`, `regression_metrics()`, `clustering_metrics()`
- **10 CV strategies** - Including `StratifiedKFold`, `TimeSeriesSplit`, `GroupKFold`
- **ModelReport class** - Auto-generated HTML reports

#### Data Module (`mkyz.data`)
- **FeatureEngineer class** - Polynomial, interaction, datetime, lag, rolling features
- **Feature selection** - `select_features()` with mutual info and f-score
- **DataLoader class** - Support for CSV, Excel, JSON, Parquet, Feather
- **Validation** - `validate_dataset()`, `check_target_balance()`, `detect_data_leakage()`

#### Utils Module (`mkyz.utils`)
- **Colored logging** - `setup_logging()`, `get_logger()`
- **Parallel processing** - `parallel_map()`, `chunk_data()`

### 🔄 Changes
- Updated `__init__.py` to expose all new modules
- Backward compatible with v0.1 API
- Version bumped to 0.2.0

### 📁 New Files (17 total)
```
mkyz/
├── core/
│   ├── __init__.py
│   ├── base.py
│   ├── config.py
│   └── exceptions.py
├── data/
│   ├── __init__.py
│   ├── feature_engineering.py
│   ├── loading.py
│   ├── preprocessing.py
│   └── validation.py
├── evaluation/
│   ├── __init__.py
│   ├── cross_validation.py
│   ├── metrics.py
│   └── reports.py
├── persistence/
│   ├── __init__.py
│   └── serialization.py
└── utils/
    ├── __init__.py
    ├── logging.py
    └── parallel.py
```

---

## [0.1.1] - Previous Release

### Features
- Data preparation with `prepare_data()`
- Model training with `train()`, `predict()`, `evaluate()`
- AutoML with `auto_train()`
- Hyperparameter optimization
- Visualization with `visualize()`

### Supported Models
- Classification: rf, lr, svm, knn, dt, nb, gb, xgb, lgbm, catboost
- Regression: rf, lr, svm, knn, dt
- Clustering: kmeans, dbscan, agglomerative, gmm
- Dimensionality Reduction: pca, svd, nmf

---

## [0.1.0] - Initial Release

- Initial release of MKYZ library
- Basic data processing
- Model training pipeline
- Visualization support

---

## Upgrade Guide

### From 0.1.x to 0.2.0

No breaking changes. All existing code continues to work.

New features can be accessed via:

```python
import mkyz

# New: Save/Load models
mkyz.save_model(model, 'model.joblib')
model = mkyz.load_model('model.joblib')

# New: Feature engineering
fe = mkyz.FeatureEngineer()
df = fe.create_datetime_features(df, 'date')

# New: Cross-validation
results = mkyz.cross_validate(model, X, y, cv='stratified')

# New: Reports
report = mkyz.ModelReport(model, X_test, y_test)
report.export_html('report.html')

# New: Configuration
mkyz.set_config(random_state=123, n_jobs=4)
```

---

## Roadmap

### Planned for v0.3.0
- [ ] Deep learning integration (TensorFlow/PyTorch)
- [ ] Time series forecasting module
- [ ] NLP text processing
- [ ] MLflow integration for experiment tracking

### Planned for v0.4.0
- [ ] Model serving API (FastAPI)
- [ ] AutoML improvements
- [ ] GPU acceleration support
