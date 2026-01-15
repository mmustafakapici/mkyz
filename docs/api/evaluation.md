# API Reference - Evaluation Module

The evaluation module provides metrics, cross-validation, and reporting.

```python
from mkyz.evaluation import (
    classification_metrics, regression_metrics, clustering_metrics,
    cross_validate, CVStrategy,
    ModelReport
)
```

---

## Metrics

### classification_metrics

```python
mkyz.classification_metrics(
    y_true,
    y_pred,
    y_proba=None,
    average='weighted'
) -> dict
```

Calculate classification metrics.

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `y_true` | array | True labels |
| `y_pred` | array | Predicted labels |
| `y_proba` | array | Predicted probabilities (optional) |
| `average` | str | Averaging method: 'weighted', 'macro', 'micro' |

**Returns:**
```python
{
    'accuracy': 0.95,
    'precision': 0.94,
    'recall': 0.93,
    'f1_score': 0.94,
    'mcc': 0.89,           # Matthews Correlation Coefficient
    'cohen_kappa': 0.91,
    'roc_auc': 0.97        # If y_proba provided
}
```

**Example:**
```python
metrics = mkyz.classification_metrics(y_test, predictions, y_proba)
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"F1 Score: {metrics['f1_score']:.4f}")
```

---

### regression_metrics

```python
mkyz.regression_metrics(y_true, y_pred) -> dict
```

Calculate regression metrics.

**Returns:**
```python
{
    'mse': 10.5,              # Mean Squared Error
    'rmse': 3.24,             # Root Mean Squared Error
    'mae': 2.1,               # Mean Absolute Error
    'r2_score': 0.89,         # R-squared
    'explained_variance': 0.90,
    'max_error': 15.2,
    'mape': 0.05,             # Mean Absolute Percentage Error
    'msle': 0.02              # Mean Squared Log Error (if positive)
}
```

**Example:**
```python
metrics = mkyz.regression_metrics(y_test, predictions)
print(f"RMSE: {metrics['rmse']:.4f}")
print(f"R² Score: {metrics['r2_score']:.4f}")
```

---

### clustering_metrics

```python
mkyz.clustering_metrics(X, labels) -> dict
```

Calculate clustering quality metrics.

**Returns:**
```python
{
    'silhouette_score': 0.65,
    'calinski_harabasz': 150.3,
    'davies_bouldin': 0.45,
    'n_clusters': 5
}
```

---

## Cross-Validation

### CVStrategy

```python
from mkyz import CVStrategy
```

Enumeration of cross-validation strategies.

| Strategy | Value | Description |
|----------|-------|-------------|
| `KFOLD` | `'kfold'` | Standard K-Fold |
| `STRATIFIED` | `'stratified'` | Stratified K-Fold (preserves class ratios) |
| `TIME_SERIES` | `'time_series'` | Time Series Split |
| `GROUP` | `'group'` | Group K-Fold |
| `REPEATED` | `'repeated'` | Repeated K-Fold |
| `REPEATED_STRATIFIED` | `'repeated_stratified'` | Repeated Stratified K-Fold |
| `LEAVE_ONE_OUT` | `'loo'` | Leave-One-Out |
| `LEAVE_ONE_GROUP_OUT` | `'logo'` | Leave-One-Group-Out |
| `SHUFFLE` | `'shuffle'` | Shuffle Split |
| `STRATIFIED_SHUFFLE` | `'stratified_shuffle'` | Stratified Shuffle Split |

---

### cross_validate

```python
mkyz.cross_validate(
    model,
    X,
    y,
    cv='stratified',
    n_splits=5,
    scoring=None,
    return_train_score=False,
    return_estimator=False,
    n_jobs=-1,
    groups=None,
    verbose=0
) -> dict
```

Perform cross-validation with multiple metrics.

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | estimator | - | Sklearn-compatible model |
| `X` | array | - | Feature matrix |
| `y` | array | - | Target variable |
| `cv` | str/CVStrategy | 'stratified' | CV strategy |
| `n_splits` | int | 5 | Number of folds |
| `scoring` | str/list | None | Scoring metric(s) |
| `return_train_score` | bool | False | Return training scores |
| `return_estimator` | bool | False | Return fitted estimators |
| `n_jobs` | int | -1 | Parallel jobs |
| `groups` | array | None | Group labels for group CV |

**Returns:**
```python
{
    'test_score': [0.94, 0.93, 0.95, 0.92, 0.94],
    'fit_time': [0.5, 0.4, 0.5, 0.4, 0.5],
    'score_time': [0.1, 0.1, 0.1, 0.1, 0.1],
    'mean_test_score': 0.936,
    'std_test_score': 0.011,
    # If return_train_score=True:
    'train_score': [...],
    'mean_train_score': 0.98,
    # If return_estimator=True:
    'estimator': [model1, model2, ...]
}
```

**Example:**
```python
from mkyz import cross_validate, CVStrategy

# Basic usage
results = mkyz.cross_validate(model, X, y)
print(f"Accuracy: {results['mean_test_score']:.4f} ± {results['std_test_score']:.4f}")

# With strategy and multiple metrics
results = mkyz.cross_validate(
    model, X, y,
    cv=CVStrategy.STRATIFIED,
    n_splits=10,
    scoring=['accuracy', 'f1', 'roc_auc'],
    return_train_score=True
)

# Time series cross-validation
results = mkyz.cross_validate(
    model, X, y,
    cv=CVStrategy.TIME_SERIES,
    n_splits=5
)
```

---

### nested_cross_validation

```python
from mkyz.evaluation import nested_cross_validation

nested_cross_validation(
    model,
    X,
    y,
    param_grid,
    outer_cv=5,
    inner_cv=3,
    scoring='accuracy',
    n_jobs=-1
) -> dict
```

Perform nested cross-validation with hyperparameter tuning.

Gold standard for unbiased model evaluation.

**Returns:**
```python
{
    'outer_scores': [0.92, 0.94, 0.93, 0.91, 0.95],
    'mean_score': 0.93,
    'std_score': 0.014
}
```

---

### learning_curve_data

```python
from mkyz.evaluation import learning_curve_data

learning_curve_data(
    model, X, y,
    train_sizes=None,
    cv=5,
    scoring='accuracy'
) -> dict
```

Generate learning curve data.

**Returns:**
```python
{
    'train_sizes': [100, 250, 500, 750, 1000],
    'train_scores_mean': [...],
    'train_scores_std': [...],
    'test_scores_mean': [...],
    'test_scores_std': [...]
}
```

---

## Model Reporting

### ModelReport

```python
from mkyz import ModelReport
```

Generate comprehensive model evaluation reports.

#### Constructor

```python
ModelReport(
    model,
    X_test,
    y_test,
    task='classification',
    model_name=None,
    feature_names=None
)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | estimator | Trained model |
| `X_test` | array | Test features |
| `y_test` | array | Test labels |
| `task` | str | 'classification', 'regression', 'clustering' |
| `model_name` | str | Optional model name |
| `feature_names` | list | Optional feature names |

#### Methods

##### `generate(**options) -> ModelReport`

Generate the report data.

```python
report = ModelReport(model, X_test, y_test)
report.generate(
    include_feature_importance=True,
    include_confusion_matrix=True
)
```

##### `summary() -> str`

Get text summary of the report.

```python
print(report.summary())
```

Output:
```
============================================================
Model Report: RandomForestClassifier
============================================================
Task: Classification
Samples: 200
Features: 15

Metrics:
----------------------------------------
  accuracy: 0.9500
  precision: 0.9400
  recall: 0.9350
  f1_score: 0.9375

Top 10 Features:
----------------------------------------
  1. feature_a: 0.2341
  2. feature_b: 0.1892
  ...
============================================================
```

##### `export_html(path, include_plots=True) -> str`

Export report as HTML file.

```python
report.export_html('reports/model_report.html')
```

##### `to_dict() -> dict`

Convert report to dictionary.

#### Properties

##### `metrics`
Get calculated metrics dictionary.

##### `predictions`
Get model predictions.

**Complete Example:**
```python
from mkyz import ModelReport

# Create report
report = ModelReport(
    model=trained_rf,
    X_test=X_test,
    y_test=y_test,
    task='classification',
    model_name='Customer Churn Predictor'
)

# Generate and export
report.generate()
print(report.summary())
report.export_html('reports/churn_model_report.html')

# Access metrics
print(f"Accuracy: {report.metrics['accuracy']:.4f}")
```

---

## See Also

- [Model Evaluation Guide](../guides/model_evaluation.md)
- [Quick Start](../quickstart.md)
