# Quick Start Guide

Get up and running with MKYZ in 5 minutes.

MKYZ offers two APIs - choose the one that fits your needs:

- **Basic API**: Simple functions for quick tasks
- **Modular API**: Granular control for advanced workflows

---

## Basic API Quick Start

Perfect for beginners and quick prototyping.

### 1. Import

```python
import mkyz
```

### 2. Prepare Data

```python
data = mkyz.prepare_data(
    'your_dataset.csv',
    target_column='target',
    test_size=0.2,
    random_state=42
)
# Returns: X_train, X_test, y_train, y_test, df, target, num_cols, cat_cols
X_train, X_test, y_train, y_test, df, target, num_cols, cat_cols = data
```

**What it does automatically:**
- Detects numerical and categorical columns
- Handles missing values
- Handles outliers
- Encodes categorical variables
- Splits data into train/test sets

### 3. Train Model

```python
# Single model
model = mkyz.train(data, task='classification', model='rf')

# Or AutoML - find best model automatically
best_model = mkyz.auto_train(data, task='classification')
```

### 4. Predict and Evaluate

```python
predictions = mkyz.predict(data, model, task='classification')
scores = mkyz.evaluate(data, predictions, task='classification')
print(scores)
# {'accuracy': 0.95, 'f1_score': 0.94, 'precision': 0.93, 'recall': 0.94}
```

### 5. Visualize

```python
mkyz.visualize(data, graphics='histogram')
mkyz.visualize(data, graphics='corr')  # Correlation heatmap
```

### Complete Basic API Example

```python
import mkyz

# End-to-end workflow
data = mkyz.prepare_data('titanic.csv', target_column='Survived')
best_model = mkyz.auto_train(data, task='classification')
predictions = mkyz.predict(data, best_model, task='classification')
scores = mkyz.evaluate(data, predictions, task='classification')
print(f"Accuracy: {scores['accuracy']:.4f}")
```

---

## Modular API Quick Start

Perfect for production and advanced workflows.

### 1. Import and Configure

```python
import mkyz

# Configure globally
mkyz.set_config(random_state=42, n_jobs=-1, verbose=1)
```

### 2. Load and Validate

```python
# Flexible loading (CSV, Excel, JSON, Parquet)
df = mkyz.load_data('data.csv')

# Validate before processing
validation = mkyz.validate_dataset(df, target_column='target')
if not validation['is_valid']:
    print(validation['issues'])
```

### 3. Exploratory Data Analysis

```python
# Create data profile for EDA
profile = mkyz.DataProfile(df)

# Generate summary statistics
profile.generate_summary()
print(profile.summary)

# Show correlation matrix
profile.show_correlation_matrix()
```

### 4. Feature Engineering

```python
fe = mkyz.FeatureEngineer()

# Create datetime features
df = fe.create_datetime_features(df, 'date_column')

# Create polynomial features
df = fe.create_polynomial_features(df, ['age', 'income'], degree=2)

# Select best features
selected = mkyz.select_features(X, y, k=10, method='mutual_info')
```

### 5. Advanced Training

```python
# Prepare data
data = mkyz.prepare_data('data.csv', target_column='target')
X_train, X_test, y_train, y_test = data[:4]

# Train model
model = mkyz.train(data, task='classification', model='rf', n_estimators=100)
```

### 6. Cross-Validation

```python
from mkyz import cross_validate, CVStrategy

results = cross_validate(
    model, X_train, y_train,
    cv=CVStrategy.STRATIFIED,
    n_splits=5
)
print(f"Mean accuracy: {results['mean_test_score']:.4f}")
```

### 7. Save and Report

```python
# Save with metadata
mkyz.save_model(model, 'models/my_model', metadata={'version': '1.0'})

# Generate comprehensive report
report = mkyz.ModelReport(model, X_test, y_test, task='classification')
report.generate()
report.export_html('reports/model_report.html')
print(report.summary())
```

### Complete Modular API Example

```python
import mkyz

# Configure
mkyz.set_config(random_state=42, n_jobs=-1)

# Load and validate
df = mkyz.load_data('titanic.csv')
validation = mkyz.validate_dataset(df, target_column='Survived')

# Prepare and train
data = mkyz.prepare_data('titanic.csv', target_column='Survived')
X_train, X_test, y_train, y_test = data[:4]

model = mkyz.train(data, task='classification', model='rf')

# Cross-validate
results = mkyz.cross_validate(model, X_train, y_train, cv='stratified')
print(f"CV Accuracy: {results['mean_test_score']:.4f}")

# Generate report
report = mkyz.ModelReport(model, X_test, y_test, task='classification')
report.export_html('titanic_report.html')

# Save model
mkyz.save_model(model, 'titanic_model')
```

---

## Which API Should I Use?

| Use Case | Recommended API |
|----------|-----------------|
| Quick prototype | Basic API |
| Learning ML basics | Basic API |
| Simple projects | Basic API |
| Production system | Modular API |
| Custom preprocessing | Modular API |
| Need validation/feature engineering | Modular API |
| Model versioning | Modular API |

---

## Supported Models

### Classification
`rf`, `lr`, `svm`, `knn`, `dt`, `nb`, `gb`, `xgb`, `lgbm`, `catboost`

### Regression
`rf`, `lr`, `svm`, `knn`, `dt`, `gb`, `xgb`, `lgbm`

### Clustering
`kmeans`, `dbscan`, `agglomerative`, `gmm`, `mean_shift`

---

## Next Steps

### Basic API
- [Basic API Reference](basic_api.md) - Complete function documentation
- [examples/simple_api/](../examples/simple_api/) - Basic API notebooks

### Modular API
- [Data Preparation Guide](guides/data_preparation.md) - Learn about preprocessing
- [Feature Engineering Guide](guides/feature_engineering.md) - Create powerful features
- [Model Training Guide](guides/model_training.md) - Advanced training options
- [Metrics and Visualizations Guide](guides/metrics_and_visualizations.md) - Choose the right metrics and plots
- [API Reference](api/index.md) - Complete module documentation

### Examples
- [examples/modular_api/](../examples/modular_api/) - Modular API notebooks
