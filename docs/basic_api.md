# Basic API Reference

Complete reference for the Basic API functions.

## Overview

The Basic API provides simple, high-level functions for common machine learning tasks. These functions automatically handle preprocessing, training, and evaluation with minimal code.

**Best for:** Quick prototyping, learning ML basics, simple projects

**Key functions:** `prepare_data()`, `train()`, `auto_train()`, `predict()`, `evaluate()`, `optimize_model()`, `visualize()`

---

## Quick Start

```python
import mkyz

# End-to-end workflow in 5 lines
data = mkyz.prepare_data('dataset.csv', target_column='target')
model = mkyz.train(data, task='classification', model='rf')
predictions = mkyz.predict(data, model, task='classification')
scores = mkyz.evaluate(data, predictions, task='classification')
print(scores)
```

---

## Functions

### `prepare_data()`

Automatically load, clean, and prepare your data for training.

```python
data = mkyz.prepare_data(
    path_or_df,
    target_column=None,
    test_size=0.2,
    random_state=42,
    drop_columns=None,
    outlier_strategy='remove',
    categorical_transform_method='onehot'
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path_or_df` | str or DataFrame | Required | Path to CSV file or pandas DataFrame |
| `target_column` | str | None | Name of target column (uses last column if None) |
| `test_size` | float | 0.2 | Fraction of data for test set (0.0 to 1.0) |
| `random_state` | int | 42 | Random seed for reproducibility |
| `drop_columns` | list | None | Column names to drop from dataset |
| `outlier_strategy` | str | 'remove' | How to handle outliers: 'remove', 'cap', or 'keep' |
| `categorical_transform_method` | str | 'onehot' | Categorical encoding: 'onehot' or 'frequency' |

**Returns:** Tuple containing:
- `X_train` - Training features
- `X_test` - Test features
- `y_train` - Training labels
- `y_test` - Test labels
- `df` - Full processed DataFrame
- `target` - Name of target column
- `num_cols` - List of numerical column names
- `cat_cols` - List of categorical column names

**Example:**

```python
data = mkyz.prepare_data(
    'data.csv',
    target_column='price',
    test_size=0.2,
    random_state=42
)
X_train, X_test, y_train, y_test, df, target, num_cols, cat_cols = data
```

**What it does automatically:**
- Detects numerical and categorical columns
- Handles missing values (imputation)
- Detects and handles outliers
- Encodes categorical variables
- Splits data into train/test sets

---

### `train()`

Train a single machine learning model.

```python
model = mkyz.train(
    data,
    task='classification',
    model='rf',
    **kwargs
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | tuple | Required | Data tuple from `prepare_data()` |
| `task` | str | 'classification' | Type of task: 'classification', 'regression', or 'clustering' |
| `model` | str | 'rf' | Model key (see Supported Models below) |
| `**kwargs` | dict | None | Model-specific hyperparameters |

**Returns:** Trained model object (scikit-learn compatible)

**Example:**

```python
# Classification with Random Forest
model = mkyz.train(
    data,
    task='classification',
    model='rf',
    n_estimators=100,
    max_depth=10
)

# Regression with Linear Regression
model = mkyz.train(
    data,
    task='regression',
    model='lr'
)

# Clustering with K-Means
model = mkyz.train(
    data,
    task='clustering',
    model='kmeans',
    n_clusters=5
)
```

---

### `auto_train()`

Automatically train multiple models and return the best one.

```python
best_model = mkyz.auto_train(
    data,
    task='classification',
    n_threads=4,
    optimize_models=False,
    optimization_method='grid_search'
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | tuple | Required | Data tuple from `prepare_data()` |
| `task` | str | 'classification' | Type of task: 'classification', 'regression', or 'clustering' |
| `n_threads` | int | 4 | Number of threads for parallel training |
| `optimize_models` | bool | False | Enable hyperparameter tuning |
| `optimization_method` | str | 'grid_search' | Optimization method: 'grid_search' or 'bayesian' |

**Returns:** Best performing model object

**Example:**

```python
# Find the best classification model
best_model = mkyz.auto_train(
    data,
    task='classification',
    n_threads=4
)

# With hyperparameter optimization
best_model = mkyz.auto_train(
    data,
    task='classification',
    optimize_models=True,
    optimization_method='bayesian'
)
```

---

### `predict()`

Make predictions on test data.

```python
predictions = mkyz.predict(
    data,
    model,
    task='classification'
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | tuple | Required | Data tuple from `prepare_data()` |
| `model` | object | Required | Trained model from `train()` or `auto_train()` |
| `task` | str | 'classification' | Type of task: 'classification', 'regression', or 'clustering' |

**Returns:** Array of predictions

**Example:**

```python
predictions = mkyz.predict(data, model, task='classification')
```

---

### `evaluate()`

Quick evaluation of model performance.

```python
scores = mkyz.evaluate(
    data,
    predictions,
    task='classification'
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | tuple | Required | Data tuple from `prepare_data()` |
| `predictions` | array | Required | Predictions from `predict()` |
| `task` | str | 'classification' | Type of task |

**Returns:** Dictionary of metrics

**Classification metrics:**
- `accuracy` - Overall correctness
- `f1_score` - F1 score (harmonic mean of precision and recall)
- `precision` - Precision score
- `recall` - Recall score
- `roc_auc` - ROC AUC score (binary classification)

**Regression metrics:**
- `r2` - R-squared score
- `rmse` - Root mean squared error
- `mae` - Mean absolute error

**Clustering metrics:**
- `silhouette_score` - Silhouette coefficient
- `inertia` - Sum of squared distances to centroids

**Example:**

```python
scores = mkyz.evaluate(data, predictions, task='classification')
print(f"Accuracy: {scores['accuracy']:.4f}")
print(f"F1 Score: {scores['f1_score']:.4f}")
```

---

### `optimize_model()`

Hyperparameter optimization for a single model.

```python
best_model, best_params, best_score = mkyz.optimize_model(
    X_train,
    y_train,
    model_class,
    param_grid,
    cv=5,
    method='grid_search',
    task='classification'
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `X_train` | array | Required | Training features |
| `y_train` | array | Required | Training labels |
| `model_class` | class | Required | Model class (e.g., RandomForestClassifier) |
| `param_grid` | dict | Required | Parameter grid for tuning |
| `cv` | int | 5 | Number of cross-validation folds |
| `method` | str | 'grid_search' | 'grid_search' or 'bayesian' |
| `task` | str | 'classification' | Task type |

**Returns:** Tuple of (best_model, best_params, best_score)

**Example:**

```python
from sklearn.ensemble import RandomForestClassifier

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15]
}

best_model, best_params, best_score = mkyz.optimize_model(
    X_train, y_train,
    model_class=RandomForestClassifier,
    param_grid=param_grid,
    cv=5,
    task='classification'
)
```

---

### `visualize()`

Generate visualizations for exploratory data analysis.

```python
mkyz.visualize(
    data,
    graphics='histogram',
    target_column=None,
    numerical_columns=None,
    categorical_columns=None
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | tuple or DataFrame | Required | Data tuple from `prepare_data()` or DataFrame |
| `graphics` | str | 'histogram' | Type of plot (see below) |
| `target_column` | str | None | Target column for color-coding |
| `numerical_columns` | list | None | Specific numerical columns to plot |
| `categorical_columns` | list | None | Specific categorical columns to plot |

**Available graphics types:**

| Type | Description |
|------|-------------|
| `histogram` | Distribution of numerical columns |
| `bar` | Bar charts for categorical columns |
| `box` | Box plots for outlier detection |
| `scatter` | Scatter plot for relationships |
| `corr` | Correlation heatmap |
| `kde` | Kernel density estimation |
| `violin` | Violin plots |
| `pair` | Pair plot for multiple variables |

**Example:**

```python
# Histogram of all numerical features
mkyz.visualize(data, graphics='histogram')

# Correlation heatmap
mkyz.visualize(data, graphics='corr')

# Box plots by target
mkyz.visualize(data, graphics='box', target_column='target')
```

---

## Supported Models

### Classification Models

| Key | Model | Description |
|-----|-------|-------------|
| `rf` | Random Forest | Ensemble of decision trees |
| `lr` | Logistic Regression | Linear classification |
| `svm` | Support Vector Machine | Margin-based classifier |
| `knn` | K-Nearest Neighbors | Instance-based learning |
| `dt` | Decision Tree | Single tree classifier |
| `nb` | Naive Bayes | Probabilistic classifier |
| `gb` | Gradient Boosting | Boosted trees |
| `xgb` | XGBoost | Extreme Gradient Boosting |
| `lgbm` | LightGBM | Light Gradient Boosting |
| `catboost` | CatBoost | Categorical Boosting |

### Regression Models

| Key | Model | Description |
|-----|-------|-------------|
| `rf` | Random Forest | Ensemble regressor |
| `lr` | Linear Regression | Ordinary least squares |
| `svm` | Support Vector Regression | SVR |
| `knn` | K-Nearest Neighbors | Instance-based regression |
| `dt` | Decision Tree | Single tree regressor |
| `gb` | Gradient Boosting | Boosted trees |
| `xgb` | XGBoost | Extreme Gradient Boosting |
| `lgbm` | LightGBM | Light Gradient Boosting |

### Clustering Models

| Key | Model | Description |
|-----|-------|-------------|
| `kmeans` | K-Means | Centroid-based clustering |
| `dbscan` | DBSCAN | Density-based clustering |
| `agglomerative` | Agglomerative | Hierarchical clustering |
| `gmm` | Gaussian Mixture | Probabilistic clustering |
| `mean_shift` | Mean Shift | Mode-seeking clustering |

---

## Complete Examples

### Binary Classification

```python
import mkyz

# Prepare data
data = mkyz.prepare_data('heart.csv', target_column='output')
X_train, X_test, y_train, y_test, df, target, num_cols, cat_cols = data

# Train model
model = mkyz.train(data, task='classification', model='rf')

# Predict
predictions = mkyz.predict(data, model, task='classification')

# Evaluate
scores = mkyz.evaluate(data, predictions, task='classification')
print(f"Accuracy: {scores['accuracy']:.4f}")
```

### Regression

```python
import mkyz

# Prepare data
data = mkyz.prepare_data('exam_scores.csv', target_column='score')

# Train model
model = mkyz.train(data, task='regression', model='lr')

# Predict and evaluate
predictions = mkyz.predict(data, model, task='regression')
scores = mkyz.evaluate(data, predictions, task='regression')
print(f"R²: {scores['r2']:.4f}")
```

### Clustering

```python
import mkyz

# Load data
df = mkyz.load_data('customers.csv')

# Select features for clustering
X = df[['Income', 'SpendingScore']].values

# Train clustering model
model = mkyz.train(
    (X, None, None, None, df, None, list(X.columns), []),
    task='clustering',
    model='kmeans',
    n_clusters=5
)

# Get cluster labels
labels = model.labels_
```

### AutoML

```python
import mkyz

# Prepare data
data = mkyz.prepare_data('dataset.csv', target_column='target')

# Automatically find the best model
best_model = mkyz.auto_train(
    data,
    task='classification',
    n_threads=4,
    optimize_models=True
)

# Use the best model
predictions = mkyz.predict(data, best_model, task='classification')
scores = mkyz.evaluate(data, predictions, task='classification')
```

---

## Example Notebooks

See the [examples/simple_api/](../examples/simple_api/) directory for complete notebooks:

| Notebook | Description |
|----------|-------------|
| [01_binary_classification_heart.ipynb](../examples/simple_api/01_binary_classification_heart.ipynb) | Binary classification for heart disease prediction |
| [02_regression_exam.ipynb](../examples/simple_api/02_regression_exam.ipynb) | Regression for exam score prediction |
| [03_clustering_customers.ipynb](../examples/simple_api/03_clustering_customers.ipynb) | Clustering for customer segmentation |

---

## When to Use Modular API Instead

Consider upgrading to the **Modular API** when you need:

- **Data validation** before processing
- **Feature engineering** (polynomial, datetime features)
- **Feature selection** (mutual info, recursive elimination)
- **Advanced cross-validation** strategies
- **Detailed metrics** and reporting
- **Model persistence** with metadata
- **Custom preprocessing** pipelines

See the [Modular API Reference](api/index.md) for more details.
