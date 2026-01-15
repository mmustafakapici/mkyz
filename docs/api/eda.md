# API Reference - EDA Module

The EDA (Exploratory Data Analysis) module provides comprehensive data profiling and insights.

```python
from mkyz import DataProfile, data_info, quick_eda
from mkyz.data import describe_column, get_summary_stats
```

---

## DataProfile

```python
from mkyz import DataProfile
```

Comprehensive data profiling class.

### Constructor

```python
DataProfile(df, target_column=None)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `df` | DataFrame | Data to analyze |
| `target_column` | str | Optional target column |

### Methods

#### `generate() -> DataProfile`

Generate the complete data profile.

```python
profile = DataProfile(df, target_column='price')
profile.generate()
```

#### `summary() -> str`

Get text summary of the data.

```python
print(profile.summary())
```

**Output:**
```
============================================================
DATA PROFILE SUMMARY
============================================================

📊 Overview
----------------------------------------
  Rows: 10,000
  Columns: 15
  Memory: 1.23 MB

📈 Column Types
----------------------------------------
  Numerical: 10
  Categorical: 4
  Datetime: 1

❓ Missing Values
----------------------------------------
  Total: 250 (1.67%)
  Columns affected: 3
  Complete rows: 9,750

🔄 Duplicates
----------------------------------------
  Duplicate rows: 15
============================================================
```

#### `get_column_info(column) -> Dict`

Get detailed info for a specific column.

```python
info = profile.get_column_info('price')
print(f"Mean: {info['mean']:.2f}")
print(f"Outliers: {info['n_outliers']}")
```

#### `get_recommendations() -> List[str]`

Get preprocessing recommendations.

```python
for rec in profile.get_recommendations():
    print(rec)
```

**Output:**
```
⚠️ High missing rate (12.5%). Consider imputation.
⚠️ Column 'category' has high cardinality (150 unique).
ℹ️ High correlation (0.95) between 'height' and 'weight'.
```

#### `export_report(path) -> str`

Export as HTML report.

```python
profile.export_report('reports/data_profile.html')
```

---

## data_info

```python
mkyz.data_info(df, target_column=None, detailed=False) -> Dict
```

Quick data information.

**Returns:**
```python
{
    'n_rows': 10000,
    'n_columns': 15,
    'n_numerical': 10,
    'n_categorical': 4,
    'memory_mb': 1.23,
    'missing_pct': 1.67,
    'n_duplicates': 15,
    'columns': ['col1', 'col2', ...]
}
```

**Example:**
```python
info = mkyz.data_info(df)
print(f"Dataset: {info['n_rows']} rows, {info['n_columns']} columns")
print(f"Missing: {info['missing_pct']:.1f}%")
```

---

## quick_eda

```python
mkyz.quick_eda(df, target_column=None, show_plots=False)
```

Print quick EDA report to console.

**Example:**
```python
mkyz.quick_eda(df, target_column='price')

# With plots (requires matplotlib)
mkyz.quick_eda(df, show_plots=True)
```

---

## describe_column

```python
from mkyz.data import describe_column

describe_column(df, column) -> Dict
```

Get detailed statistics for a single column.

**Example:**
```python
stats = describe_column(df, 'price')
print(f"Mean: {stats['mean']:.2f}")
print(f"Std: {stats['std']:.2f}")
print(f"Outliers: {stats['n_outliers']}")
```

---

## get_summary_stats

```python
from mkyz.data import get_summary_stats

get_summary_stats(df) -> pd.DataFrame
```

Enhanced describe() with more statistics.

**Example:**
```python
summary = get_summary_stats(df)
print(summary)
```

**Output:**
```
       column      dtype  n_unique  n_missing missing_pct     mean      std
0       price    float64       450          0         0.0   150.50    45.23
1    quantity      int64        25          5        0.5%    12.30     8.12
2    category     object        15          0         0.0        -        -
```

---

## Complete Example

```python
import mkyz

# Load data
df = mkyz.load_data('customers.csv')

# Quick overview
print(mkyz.data_info(df))

# Full EDA
mkyz.quick_eda(df, target_column='churn')

# Detailed profiling
profile = mkyz.DataProfile(df, target_column='churn')
profile.generate()

# Get recommendations
for rec in profile.get_recommendations():
    print(rec)

# Export HTML report
profile.export_report('reports/customer_profile.html')
```

---

## See Also

- [Data Preparation Guide](../guides/data_preparation.md)
- [Data Module API](data.md)
