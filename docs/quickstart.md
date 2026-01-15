# Quick Start Guide

Get up and running with MKYZ in 5 minutes.

## 1. Import MKYZ

```python
import mkyz
```

You should see:
```
mkyz package initialized. Version: 0.2.0
```

## 2. Load Your Data

### Option A: Using prepare_data (Original API)

```python
# Automatically handles everything
data = mkyz.prepare_data(
    'your_dataset.csv',
    target_column='target',
    test_size=0.2,
    random_state=42
)

# Returns: X_train, X_test, y_train, y_test, df, target, num_cols, cat_cols
X_train, X_test, y_train, y_test, df, target, num_cols, cat_cols = data
```

### Option B: Using load_data (New API)

```python
# More flexible loading
df = mkyz.load_data('data.csv')  # Also supports .xlsx, .json, .parquet

# Validate the dataset
validation = mkyz.validate_dataset(df, target_column='target')
print(validation)
```

## 3. Train a Model

### Single Model Training

```python
# Train a Random Forest classifier
model = mkyz.train(
    data,
    task='classification',
    model='rf',
    n_estimators=100
)
```

### Available Models

| Task | Models |
|------|--------|
| Classification | `rf`, `lr`, `svm`, `knn`, `dt`, `nb`, `gb`, `xgb`, `lgbm`, `catboost` |
| Regression | `rf`, `lr`, `svm`, `knn`, `dt`, `gb`, `xgb`, `lgbm` |
| Clustering | `kmeans`, `dbscan`, `agglomerative`, `gmm`, `mean_shift` |

### AutoML - Find the Best Model

```python
# Automatically train all models and find the best one
best_model = mkyz.auto_train(
    data,
    task='classification',
    n_threads=4,               # Parallel training
    optimize_models=True,      # Hyperparameter tuning
    optimization_method='grid_search'  # or 'bayesian'
)
```

## 4. Make Predictions

```python
predictions = mkyz.predict(data, model, task='classification')
```

## 5. Evaluate Performance

### Quick Evaluation

```python
scores = mkyz.evaluate(data, predictions, task='classification')
print(scores)
# {'accuracy': 0.95, 'f1_score': 0.94, 'precision': 0.93, 'recall': 0.94}
```

### Detailed Metrics

```python
from mkyz import classification_metrics

metrics = classification_metrics(y_test, predictions)
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")
```

### Cross-Validation

```python
from mkyz import cross_validate, CVStrategy

results = cross_validate(
    model, X_train, y_train,
    cv=CVStrategy.STRATIFIED,
    n_splits=5
)

print(f"Mean accuracy: {results['mean_test_score']:.4f}")
print(f"Std: {results['std_test_score']:.4f}")
```

## 6. Save Your Model

```python
# Save with metadata
mkyz.save_model(
    model,
    'models/my_model',
    format='joblib',
    metadata={'accuracy': 0.95, 'version': '1.0'}
)
```

## 7. Load and Use Later

```python
# Load the model
loaded_model, metadata = mkyz.load_model(
    'models/my_model.joblib',
    return_metadata=True
)

print(f"Model saved at: {metadata['saved_at']}")

# Make predictions with loaded model
new_predictions = loaded_model.predict(X_new)
```

## 8. Generate a Report

```python
from mkyz import ModelReport

# Create comprehensive report
report = ModelReport(
    model=model,
    X_test=X_test,
    y_test=y_test,
    task='classification',
    model_name='Random Forest Classifier'
)

# Generate report
report.generate()

# Print summary
print(report.summary())

# Export to HTML
report.export_html('reports/model_report.html')
```

## 9. Visualize Results

```python
# Various visualizations
mkyz.visualize(data, plot_type='histogram')
mkyz.visualize(data, plot_type='correlation')
mkyz.visualize(data, plot_type='boxplot')
```

---

## Complete Example

```python
import mkyz

# 1. Prepare data
data = mkyz.prepare_data('titanic.csv', target_column='Survived')
X_train, X_test, y_train, y_test, df, target, num_cols, cat_cols = data

# 2. Auto-train and find best model
best_model = mkyz.auto_train(data, task='classification')

# 3. Evaluate with cross-validation
results = mkyz.cross_validate(best_model, X_train, y_train, cv='stratified')
print(f"CV Accuracy: {results['mean_test_score']:.4f}")

# 4. Generate report
report = mkyz.ModelReport(best_model, X_test, y_test, task='classification')
report.generate()
report.export_html('titanic_report.html')

# 5. Save model
mkyz.save_model(best_model, 'titanic_model')

print("Done! Check titanic_report.html for results.")
```

---

## Next Steps

- [Data Preparation Guide](guides/data_preparation.md) - Learn about preprocessing
- [Feature Engineering Guide](guides/feature_engineering.md) - Create powerful features
- [Model Training Guide](guides/model_training.md) - Advanced training options
- [API Reference](api/index.md) - Complete function documentation
