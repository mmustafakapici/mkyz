# Data Preparation Guide

Learn how to load, validate, and preprocess your data with MKYZ.

## Loading Data

### Quick Load

```python
import mkyz

# CSV file
df = mkyz.load_data('data.csv')

# Excel file
df = mkyz.load_data('data.xlsx', sheet_name='Sheet1')

# JSON file
df = mkyz.load_data('data.json')

# Parquet (fast for large files)
df = mkyz.load_data('data.parquet')
```

### Using DataLoader

```python
from mkyz import DataLoader

loader = DataLoader(default_encoding='utf-8')
df = loader.load('data.csv')

# Save in different format
loader.save(df, 'output.parquet')
```

---

## Validating Data

Always validate your data before training.

```python
validation = mkyz.validate_dataset(
    df,
    target_column='price',
    check_missing=True,
    check_duplicates=True
)

print(f"Valid: {validation['is_valid']}")
print(f"Issues: {validation['issues']}")
print(f"Warnings: {validation['warnings']}")
print(f"Stats: {validation['statistics']}")
```

### Check Class Balance (Classification)

```python
balance = mkyz.check_target_balance(df['target'])

print(f"Classes: {balance['n_classes']}")
print(f"Distribution: {balance['class_distribution']}")
print(f"Imbalanced: {balance['is_imbalanced']}")
```

---

## Using prepare_data

The `prepare_data` function handles everything automatically.

```python
data = mkyz.prepare_data(
    'dataset.csv',
    target_column='price',
    test_size=0.2,
    random_state=42,
    drop_columns=['id', 'name'],
    outlier_strategy='cap',
    categorical_transform_method='onehot'
)

X_train, X_test, y_train, y_test, df, target, num_cols, cat_cols = data
```

### Parameters

| Parameter | Default | Options |
|-----------|---------|---------|
| `target_column` | Last column | Any column name |
| `test_size` | 0.2 | 0.0 - 1.0 |
| `outlier_strategy` | 'remove' | 'remove', 'cap' |
| `categorical_transform_method` | 'onehot' | 'onehot', 'frequency' |

---

## Manual Preprocessing

### Fill Missing Values

```python
from mkyz.data import fill_missing_values

df = fill_missing_values(
    df,
    numerical_columns=['age', 'income'],
    categorical_columns=['gender', 'city']
)
# Numerical: filled with mean
# Categorical: filled with mode
```

### Handle Outliers

```python
from mkyz.data import detect_outliers, handle_outliers

# Detect outliers using IQR
outliers = detect_outliers(df, numerical_columns=['price', 'quantity'])
print(f"Outliers found: {outliers}")

# Remove or cap outliers
df = handle_outliers(df, outliers, strategy='cap')
```

### Scale Features

```python
from mkyz.data import scale_features

df = scale_features(
    df,
    columns=['price', 'quantity'],
    method='standard'  # 'standard', 'minmax', 'robust'
)
```

---

## Best Practices

1. **Always validate first** - Check for issues before training
2. **Handle missing values** - Don't let NaN break your model
3. **Check for leakage** - Ensure no future data in training
4. **Document transformations** - Track what you changed

---

## Next Steps

- [Feature Engineering Guide](feature_engineering.md)
- [Model Training Guide](model_training.md)
