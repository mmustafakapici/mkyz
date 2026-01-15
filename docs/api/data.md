# API Reference - Data Module

The data module provides data loading, preprocessing, and feature engineering.

```python
from mkyz.data import (
    load_data, DataLoader,
    prepare_data, fill_missing_values, detect_outliers,
    FeatureEngineer, create_polynomial_features,
    validate_dataset, check_target_balance
)
```

---

## Data Loading

### load_data

```python
mkyz.load_data(path, format=None, **kwargs) -> pd.DataFrame
```

Load data from file.

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | str | Path to data file |
| `format` | str | File format (auto-detected if None) |
| `**kwargs` | dict | Arguments passed to pandas reader |

**Supported formats:** `csv`, `xlsx`, `xls`, `json`, `parquet`, `feather`, `pickle`

**Example:**
```python
df = mkyz.load_data('data.csv')
df = mkyz.load_data('data.xlsx', sheet_name='Sheet1')
df = mkyz.load_data('data.json')
df = mkyz.load_data('data.parquet')
```

---

### DataLoader

```python
from mkyz import DataLoader
```

Flexible data loader class.

#### Methods

##### `load(path, format=None, **kwargs) -> pd.DataFrame`
Load data from file.

##### `save(df, path, format=None, **kwargs) -> str`
Save DataFrame to file.

**Example:**
```python
loader = DataLoader()
df = loader.load('input.csv')
loader.save(df, 'output.parquet')
```

---

## Data Preparation

### prepare_data

```python
mkyz.prepare_data(
    filepath,
    target_column=None,
    numerical_columns=None,
    categorical_columns=None,
    test_size=0.2,
    random_state=42,
    drop_columns=None,
    outlier_strategy='remove',
    categorical_transform_method='onehot'
) -> tuple
```

Prepare and preprocess data for machine learning.

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `filepath` | str | - | Path to CSV file |
| `target_column` | str | None | Target column name |
| `numerical_columns` | list | None | Numerical feature columns |
| `categorical_columns` | list | None | Categorical feature columns |
| `test_size` | float | 0.2 | Test set proportion |
| `random_state` | int | 42 | Random seed |
| `drop_columns` | list | None | Columns to drop |
| `outlier_strategy` | str | 'remove' | 'remove' or 'cap' |
| `categorical_transform_method` | str | 'onehot' | Encoding method |

**Returns:** `(X_train, X_test, y_train, y_test, df, target, num_cols, cat_cols)`

**Example:**
```python
data = mkyz.prepare_data(
    'dataset.csv',
    target_column='price',
    drop_columns=['id'],
    outlier_strategy='cap'
)
X_train, X_test, y_train, y_test, df, target, num_cols, cat_cols = data
```

---

### fill_missing_values

```python
fill_missing_values(df, numerical_columns, categorical_columns) -> pd.DataFrame
```

Fill missing values in DataFrame.

- Numerical: Mean imputation
- Categorical: Mode imputation

---

### detect_outliers

```python
detect_outliers(df, numerical_columns, threshold=1.5) -> dict
```

Detect outliers using IQR method.

**Returns:** Dictionary mapping column names to outlier indices.

---

### handle_outliers

```python
handle_outliers(df, outliers, strategy='remove') -> pd.DataFrame
```

Handle detected outliers.

**Strategies:**
- `'remove'`: Drop outlier rows
- `'cap'`: Cap at IQR bounds

---

## Feature Engineering

### FeatureEngineer

```python
from mkyz import FeatureEngineer

fe = FeatureEngineer(random_state=42)
```

Comprehensive feature engineering class.

#### Methods

##### `create_polynomial_features(df, columns, degree=2)`

Create polynomial and interaction features.

```python
df = fe.create_polynomial_features(df, ['age', 'income'], degree=2)
# Adds: age^2, income^2, age*income
```

##### `create_interaction_features(df, columns, operations=None)`

Create interaction features between columns.

**Operations:** `'multiply'`, `'add'`, `'subtract'`, `'divide'`

```python
df = fe.create_interaction_features(
    df, 
    ['feature1', 'feature2'],
    operations=['multiply', 'add']
)
```

##### `create_datetime_features(df, datetime_column, features=None)`

Extract features from datetime column.

**Available features:**
- `'year'`, `'month'`, `'day'`, `'dayofweek'`, `'hour'`, `'minute'`
- `'quarter'`, `'week'`, `'is_weekend'`
- `'is_month_start'`, `'is_month_end'`, `'days_since_epoch'`

```python
df = fe.create_datetime_features(df, 'signup_date')
# Adds: signup_date_year, signup_date_month, signup_date_day, etc.
```

##### `create_lag_features(df, columns, lags=None)`

Create lag features for time series.

```python
df = fe.create_lag_features(df, ['sales'], lags=[1, 7, 30])
# Adds: sales_lag_1, sales_lag_7, sales_lag_30
```

##### `create_rolling_features(df, columns, windows=None, functions=None)`

Create rolling window features.

```python
df = fe.create_rolling_features(
    df, ['sales'], 
    windows=[7, 30],
    functions=['mean', 'std']
)
# Adds: sales_rolling_7_mean, sales_rolling_7_std, etc.
```

##### `create_aggregation_features(df, group_column, agg_columns, agg_functions=None)`

Create group aggregation features.

```python
df = fe.create_aggregation_features(
    df,
    group_column='category',
    agg_columns=['price'],
    agg_functions=['mean', 'std']
)
```

##### `select_features(X, y, k=10, method='mutual_info', task='classification')`

Select top k features.

```python
selected = fe.select_features(X, y, k=10, method='mutual_info')
X_selected = X[selected]
```

#### Properties

##### `created_features`
List of all created feature names.

```python
print(fe.created_features)
```

---

### Convenience Functions

#### create_polynomial_features

```python
mkyz.create_polynomial_features(df, columns, degree=2) -> pd.DataFrame
```

#### create_datetime_features

```python
mkyz.create_datetime_features(df, datetime_column, features=None) -> pd.DataFrame
```

#### select_features

```python
mkyz.select_features(X, y, k=10, method='mutual_info', task='classification') -> List[str]
```

**Methods:** `'mutual_info'`, `'f_score'`

---

## Data Validation

### validate_dataset

```python
mkyz.validate_dataset(
    df,
    target_column=None,
    required_columns=None,
    check_missing=True,
    check_duplicates=True,
    check_infinity=True
) -> dict
```

Comprehensive dataset validation.

**Returns:**
```python
{
    'is_valid': True/False,
    'issues': [],
    'warnings': [],
    'statistics': {
        'n_rows': 1000,
        'n_columns': 15,
        'missing_values': {...},
        'n_duplicates': 5
    }
}
```

**Example:**
```python
result = mkyz.validate_dataset(df, target_column='price')
if not result['is_valid']:
    print("Issues:", result['issues'])
```

---

### check_target_balance

```python
mkyz.check_target_balance(y, imbalance_threshold=0.1) -> dict
```

Check class balance for classification.

**Returns:**
```python
{
    'n_classes': 3,
    'class_distribution': {...},
    'is_imbalanced': True/False,
    'recommendation': 'Consider class weights or SMOTE'
}
```

---

## See Also

- [Data Preparation Guide](../guides/data_preparation.md)
- [Feature Engineering Guide](../guides/feature_engineering.md)
