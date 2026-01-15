# Model Training Guide

Learn how to train, optimize, and compare machine learning models with MKYZ.

## Basic Training

### Single Model Training

```python
import mkyz

# Prepare data
data = mkyz.prepare_data('dataset.csv', target_column='target')

# Train a Random Forest classifier
model = mkyz.train(
    data,
    task='classification',
    model='rf',
    n_estimators=100,
    max_depth=10
)
```

### Available Models

#### Classification Models
| Key | Model | Best For |
|-----|-------|----------|
| `rf` | Random Forest | General purpose, robust |
| `lr` | Logistic Regression | Linear boundaries, interpretable |
| `svm` | Support Vector Machine | High-dimensional data |
| `knn` | K-Nearest Neighbors | Small datasets |
| `dt` | Decision Tree | Interpretability |
| `nb` | Naive Bayes | Text classification |
| `gb` | Gradient Boosting | High accuracy |
| `xgb` | XGBoost | Competitions, structured data |
| `lgbm` | LightGBM | Large datasets, fast |
| `catboost` | CatBoost | Categorical features |

#### Regression Models
| Key | Model |
|-----|-------|
| `rf` | Random Forest Regressor |
| `lr` | Linear Regression |
| `svm` | Support Vector Regression |
| `knn` | KNN Regressor |
| `dt` | Decision Tree Regressor |

#### Clustering Models
| Key | Model |
|-----|-------|
| `kmeans` | K-Means |
| `dbscan` | DBSCAN |
| `agglomerative` | Agglomerative Clustering |
| `gmm` | Gaussian Mixture |
| `mean_shift` | Mean Shift |

---

## AutoML - Find the Best Model

Let MKYZ automatically find the best model for your data.

```python
# Train all models and compare
best_model = mkyz.auto_train(
    data,
    task='classification',
    n_threads=4,              # Parallel training
    optimize_models=False     # Quick comparison
)
```

### With Hyperparameter Optimization

```python
# Full optimization (slower but better)
best_model = mkyz.auto_train(
    data,
    task='classification',
    n_threads=4,
    optimize_models=True,
    optimization_method='grid_search'  # or 'bayesian'
)
```

**Output:**
```
Starting training of 10 models...
Model 'rf' training started.
Model 'rf' training completed.
Model 'lr' training started.
...

┌──────────────────────────────────────────────┐
│         Model Training Results               │
├──────────┬───────────┬────────┬─────────────┤
│ Model    │   Score   │ Time   │ Best CV     │
├──────────┼───────────┼────────┼─────────────┤
│ xgb      │   0.9650  │  2.3s  │   0.9612    │
│ rf       │   0.9540  │  1.8s  │   0.9498    │
│ lgbm     │   0.9520  │  1.2s  │   0.9487    │
│ ...      │   ...     │  ...   │   ...       │
└──────────┴───────────┴────────┴─────────────┘
```

---

## Hyperparameter Optimization

### Grid Search

```python
best_model, best_params, best_score = mkyz.optimize_model(
    X_train, y_train,
    model_class=RandomForestClassifier,
    param_grid={
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10]
    },
    cv=5,
    method='grid_search',
    task='classification'
)

print(f"Best parameters: {best_params}")
print(f"Best CV score: {best_score:.4f}")
```

### Bayesian Optimization

```python
# Requires: pip install optuna

best_model, best_params, best_score = mkyz.optimize_model(
    X_train, y_train,
    model_class=RandomForestClassifier,
    param_grid={
        'n_estimators': (50, 500),  # Range for Bayesian
        'max_depth': [5, 10, 20, None],
        'min_samples_split': (2, 20)
    },
    method='bayesian',
    task='classification'
)
```

---

## Cross-Validation

Always evaluate with cross-validation for reliable estimates.

```python
from mkyz import cross_validate, CVStrategy

results = mkyz.cross_validate(
    model, X_train, y_train,
    cv=CVStrategy.STRATIFIED,
    n_splits=5,
    return_train_score=True
)

print(f"Train: {results['mean_train_score']:.4f}")
print(f"Test:  {results['mean_test_score']:.4f} ± {results['std_test_score']:.4f}")
```

### Choosing CV Strategy

| Strategy | When to Use |
|----------|-------------|
| `STRATIFIED` | Classification (default) |
| `KFOLD` | Regression |
| `TIME_SERIES` | Time-dependent data |
| `GROUP` | Grouped data (e.g., patients) |
| `REPEATED_STRATIFIED` | Small datasets, more reliable |

---

## Making Predictions

```python
# Predictions on test set
predictions = mkyz.predict(data, model, task='classification')

# On new data
new_predictions = model.predict(X_new)

# Probabilities (classification)
probabilities = model.predict_proba(X_new)
```

---

## Complete Training Pipeline

```python
import mkyz

# 1. Load and validate
df = mkyz.load_data('customer_churn.csv')
validation = mkyz.validate_dataset(df, target_column='churn')

if not validation['is_valid']:
    print(validation['issues'])

# 2. Check class balance
balance = mkyz.check_target_balance(df['churn'])
if balance['is_imbalanced']:
    print("Consider using class_weight='balanced'")

# 3. Prepare data
data = mkyz.prepare_data(
    'customer_churn.csv',
    target_column='churn',
    outlier_strategy='cap'
)

# 4. Quick model comparison
best_model = mkyz.auto_train(data, task='classification')

# 5. Cross-validate best model
X_train, X_test, y_train, y_test = data[:4]
results = mkyz.cross_validate(best_model, X_train, y_train)
print(f"CV Accuracy: {results['mean_test_score']:.4f}")

# 6. Final evaluation on test set
predictions = best_model.predict(X_test)
metrics = mkyz.classification_metrics(y_test, predictions)
print(metrics)

# 7. Save the model
mkyz.save_model(
    best_model,
    'models/churn_predictor',
    metadata={'accuracy': metrics['accuracy']}
)
```

---

## Tips

### Speed vs Accuracy

| Goal | Settings |
|------|----------|
| Quick comparison | `optimize_models=False`, `n_threads=4` |
| Best accuracy | `optimize_models=True`, `method='bayesian'` |
| Production | Use best model, save with metadata |

### Common Issues

| Issue | Solution |
|-------|----------|
| Slow training | Reduce `n_estimators`, use LightGBM |
| Overfitting | Add regularization, use cross-validation |
| Class imbalance | Use `class_weight='balanced'` |
| Memory error | Use `n_jobs=1`, batch processing |

---

## Next Steps

- [Model Evaluation Guide](model_evaluation.md)
- [Model Persistence Guide](model_persistence.md)
- [API Reference](../api/evaluation.md)
