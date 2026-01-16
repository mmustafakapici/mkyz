# MKYZ - Machine Learning Library

<p align="center">
  <img src="https://img.shields.io/badge/version-0.2.3-blue.svg" alt="Version">
  <img src="https://img.shields.io/badge/python-3.6+-green.svg" alt="Python">
  <img src="https://img.shields.io/badge/license-MIT-orange.svg" alt="License">
</p>

**MKYZ** is a comprehensive Python machine learning library designed to simplify data processing, model training, evaluation, and visualization tasks. Built on top of scikit-learn, it provides a unified API for common ML workflows.

## 📚 Examples

You can find comprehensive Jupyter notebooks in the [examples/](examples/) directory:

- [01_quickstart.ipynb](examples/01_quickstart.ipynb): End-to-end basic ML workflow.
- [02_data_profiling_and_eda.ipynb](examples/02_data_profiling_and_eda.ipynb): Data profiling and HTML reports.
- [03_feature_engineering.ipynb](examples/03_feature_engineering.ipynb): Advanced feature creation.
- [04_automl_and_optimization.ipynb](examples/04_automl_and_optimization.ipynb): AutoML and hyperparameter tuning.
- [05_model_persistence_and_reports.ipynb](examples/05_model_persistence_and_reports.ipynb): Saving models and rich reports.

---

## 🏗️ Architecture

### Core Capabilities
- **🔄 Data Preparation** - Automatic handling of missing values, outliers, and categorical encoding
- **🎯 Model Training** - Support for 20+ classification, regression, and clustering algorithms
- **📊 AutoML** - Automatic model selection and hyperparameter optimization
- **📈 Evaluation** - Comprehensive metrics with 10 cross-validation strategies
- **🎨 Visualization** - 40+ built-in plotting functions for EDA and model results

### New in v0.2.0
- **💾 Model Persistence** - Save and load models with metadata
- **🔧 Feature Engineering** - Polynomial, datetime, lag, and rolling features
- **📝 Auto Reports** - Generate HTML reports with one line of code
- **⚡ Parallel Processing** - Built-in utilities for faster training
- **🛡️ Robust Error Handling** - Custom exceptions for better debugging

## 📦 Installation

```bash
pip install mkyz
```

### From Source
```bash
git clone https://github.com/mmustafakapici/mkyz.git
cd mkyz
pip install -e .
```

### Dependencies
```
pandas, scikit-learn, numpy, matplotlib, seaborn, 
plotly, xgboost, lightgbm, catboost, rich, mlxtend
```

## 🎯 Choose Your API

MKYZ provides two APIs for different use cases:

### Basic API

For quick tasks and beginners. Simple functions that handle common ML workflows automatically.

**Best for:** Quick prototyping, learning, simple projects

**Key functions:** `prepare_data()`, `train()`, `predict()`, `evaluate()`, `auto_train()`, `visualize()`

### Modular API

For advanced users and production. Granular control over every step.

**Best for:** Production systems, custom workflows, advanced features

**Key modules:** `core.config`, `data`, `evaluation`, `persistence`, `utils`

---

## 📚 Basic API Guide

The Basic API provides simple functions for common ML tasks:

```python
import mkyz

# 1. Prepare data (automatic preprocessing)
data = mkyz.prepare_data('dataset.csv', target_column='price')
X_train, X_test, y_train, y_test, df, target, num_cols, cat_cols = data

# 2. Train a model
model = mkyz.train(data, task='classification', model='rf')

# 3. Make predictions
predictions = mkyz.predict(data, model, task='classification')

# 4. Evaluate
scores = mkyz.evaluate(data, predictions, task='classification')
print(scores)

# 5. Visualize
mkyz.visualize(data, graphics='histogram')
```

### AutoML with Basic API

```python
import mkyz

data = mkyz.prepare_data('dataset.csv', target_column='target')

# Automatically find the best model
best_model = mkyz.auto_train(
    data,
    task='classification',
    optimize_models=True,
    optimization_method='bayesian'
)
```

### Basic API Examples

| Example | Notebook | Description |
|---------|----------|-------------|
| Binary Classification | [examples/simple_api/01_binary_classification_heart.ipynb](examples/simple_api/01_binary_classification_heart.ipynb) | Heart disease prediction |
| Regression | [examples/simple_api/02_regression_exam.ipynb](examples/simple_api/02_regression_exam.ipynb) | Exam score prediction |
| Clustering | [examples/simple_api/03_clustering_customers.ipynb](examples/simple_api/03_clustering_customers.ipynb) | Customer segmentation |

---

## 🔧 Modular API Guide

The Modular API provides fine-grained control over your ML pipeline:

```python
import mkyz

# Configure globally
mkyz.set_config(random_state=42, n_jobs=-1, verbose=1)

# Load data flexibly
df = mkyz.load_data('data.csv')  # Supports CSV, Excel, JSON, Parquet

# Validate dataset
validation = mkyz.validate_dataset(df, target_column='target')
if not validation['is_valid']:
    print(validation['issues'])

# Feature Engineering
fe = mkyz.FeatureEngineer()
df = fe.create_datetime_features(df, 'date_column')
df = fe.create_polynomial_features(df, ['age', 'income'], degree=2)

# Select best features
selected = mkyz.select_features(X, y, k=10, method='mutual_info')

# Advanced Cross-Validation
results = mkyz.cross_validate(
    model, X, y,
    cv=mkyz.CVStrategy.STRATIFIED,
    n_splits=5,
    return_train_score=True
)
print(f"Mean accuracy: {results['mean_test_score']:.4f}")

# Save trained model with metadata
mkyz.save_model(model, 'models/my_model', metadata={'version': '1.0'})

# Generate comprehensive report
report = mkyz.ModelReport(model, X_test, y_test, task='classification')
report.generate()
report.export_html('reports/model_report.html')
```

### Modular API Examples

| Example | Notebook | Description |
|---------|----------|-------------|
| Binary Classification | [examples/modular_api/01_binary_classification_heart.ipynb](examples/modular_api/01_binary_classification_heart.ipynb) | Heart disease with validation, CV |
| Multiclass Classification | [examples/modular_api/02_multiclass_classification_wine.ipynb](examples/modular_api/02_multiclass_classification_wine.ipynb) | Wine quality classification |
| Univariate Regression | [examples/modular_api/03_univariate_regression_exam.ipynb](examples/modular_api/03_univariate_regression_exam.ipynb) | Exam score prediction |
| Multivariate Regression | [examples/modular_api/04_multivariate_regression_housing.ipynb](examples/modular_api/04_multivariate_regression_housing.ipynb) | Boston housing prices |
| Clustering | [examples/modular_api/05_clustering_customers.ipynb](examples/modular_api/05_clustering_customers.ipynb) | Customer segmentation |
| Dimensionality Reduction | [examples/modular_api/06_dimensionality_reduction_wine.ipynb](examples/modular_api/06_dimensionality_reduction_wine.ipynb) | Wine PCA visualization |
| Anomaly Detection | [examples/modular_api/07_anomaly_detection_creditcard.ipynb](examples/modular_api/07_anomaly_detection_creditcard.ipynb) | Credit card fraud |

## 📚 Documentation

### API Reference

| API | Description | Link |
|-----|-------------|------|
| **Basic API** | Simple functions for quick ML tasks | [docs/basic_api.md](docs/basic_api.md) |
| **Core Module** | Configuration and exceptions | [docs/api/core.md](docs/api/core.md) |
| **Data Module** | Data processing functions | [docs/api/data.md](docs/api/data.md) |
| **Evaluation Module** | Metrics and cross-validation | [docs/api/evaluation.md](docs/api/evaluation.md) |
| **Persistence Module** | Model serialization | [docs/api/persistence.md](docs/api/persistence.md) |
| **Utils Module** | Logging and parallel processing | [docs/api/utils.md](docs/api/utils.md) |

### Detailed Guides

- [Installation Guide](docs/installation.md)
- [Quick Start Tutorial](docs/quickstart.md)
- [Basic API Reference](docs/basic_api.md)
- [Metrics and Visualizations Guide](docs/guides/metrics_and_visualizations.md) - **NEW!** Choose the right metrics and plots
- [Data Preparation Guide](docs/guides/data_preparation.md)
- [Feature Engineering Guide](docs/guides/feature_engineering.md)
- [Model Training Guide](docs/guides/model_training.md)

## 🔧 Supported Models

### Classification
| Model | Key | Description |
|-------|-----|-------------|
| Random Forest | `rf` | Ensemble of decision trees |
| Logistic Regression | `lr` | Linear classification |
| SVM | `svm` | Support Vector Machine |
| KNN | `knn` | K-Nearest Neighbors |
| Decision Tree | `dt` | Single decision tree |
| Naive Bayes | `nb` | Probabilistic classifier |
| Gradient Boosting | `gb` | Boosted trees |
| XGBoost | `xgb` | Extreme Gradient Boosting |
| LightGBM | `lgbm` | Light Gradient Boosting |
| CatBoost | `catboost` | Categorical Boosting |

### Regression
| Model | Key | Description |
|-------|-----|-------------|
| Random Forest | `rf` | Ensemble regressor |
| Linear Regression | `lr` | OLS regression |
| SVR | `svm` | Support Vector Regression |
| KNN | `knn` | K-Nearest Neighbors |
| Decision Tree | `dt` | Single decision tree |

### Clustering
| Model | Key | Description |
|-------|-----|-------------|
| K-Means | `kmeans` | Centroid-based |
| DBSCAN | `dbscan` | Density-based |
| Agglomerative | `agglomerative` | Hierarchical |
| GMM | `gmm` | Gaussian Mixture |
| Mean Shift | `mean_shift` | Mode-seeking |

### Dimensionality Reduction
| Model | Key | Description |
|-------|-----|-------------|
| PCA | `pca` | Principal Component Analysis |
| SVD | `svd` | Truncated SVD |
| NMF | `nmf` | Non-negative Matrix Factorization |

## 📊 Cross-Validation Strategies

```python
from mkyz import cross_validate, CVStrategy

# Available strategies
strategies = [
    CVStrategy.KFOLD,              # Standard K-Fold
    CVStrategy.STRATIFIED,         # Stratified K-Fold (default)
    CVStrategy.TIME_SERIES,        # Time Series Split
    CVStrategy.GROUP,              # Group K-Fold
    CVStrategy.REPEATED,           # Repeated K-Fold
    CVStrategy.REPEATED_STRATIFIED,# Repeated Stratified
    CVStrategy.LEAVE_ONE_OUT,      # Leave-One-Out
    CVStrategy.SHUFFLE,            # Shuffle Split
    CVStrategy.STRATIFIED_SHUFFLE  # Stratified Shuffle
]

# Usage
results = cross_validate(model, X, y, cv=CVStrategy.TIME_SERIES, n_splits=5)
```

## 🔧 Configuration

```python
import mkyz

# View current config
print(mkyz.get_config().to_dict())

# Update config
mkyz.set_config(
    random_state=42,
    n_jobs=-1,
    cv_folds=5,
    verbose=1,
    dark_mode=True
)
```

### Available Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `random_state` | 42 | Random seed for reproducibility |
| `n_jobs` | -1 | Parallel jobs (-1 = all CPUs) |
| `cv_folds` | 5 | Default CV folds |
| `test_size` | 0.2 | Train/test split ratio |
| `verbose` | 1 | Verbosity level |
| `optimization_method` | 'grid_search' | 'grid_search' or 'bayesian' |
| `missing_value_strategy` | 'mean' | 'mean', 'median', 'mode', 'drop' |
| `outlier_strategy` | 'remove' | 'remove', 'cap', 'keep' |

## 🛡️ Error Handling

```python
from mkyz import (
    MKYZError,           # Base exception
    DataValidationError, # Data issues
    ModelNotTrainedError,# Model not fitted
    UnsupportedTaskError,# Invalid task type
    PersistenceError     # Save/load failures
)

try:
    model = mkyz.load_model('nonexistent.joblib')
except PersistenceError as e:
    print(f"Failed to load model: {e}")
```

## 📈 Visualization

```python
import mkyz

# EDA visualizations
mkyz.visualize(data, plot_type='histogram')
mkyz.visualize(data, plot_type='correlation')
mkyz.visualize(data, plot_type='boxplot')

# Available plot types:
# histogram, bar, box, violin, pie, scatter, line,
# heatmap, pair, swarm, strip, kde, ridge, density,
# joint, regression, residual, qq, ecdf, dendrogram...
```

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md).

```bash
# Clone the repository
git clone https://github.com/mmustafakapici/mkyz.git

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Mustafa Kapıcı**
- Email: m.mustafakapici@gmail.com
- GitHub: [@mmustafakapici](https://github.com/mmustafakapici)

## 🙏 Acknowledgments

- Built on top of [scikit-learn](https://scikit-learn.org/)
- Boosting models from [XGBoost](https://xgboost.ai/), [LightGBM](https://lightgbm.readthedocs.io/), [CatBoost](https://catboost.ai/)
- Visualization powered by [Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/), [Plotly](https://plotly.com/)

---

<p align="center">
  Made with ❤️ in Turkey
</p>
