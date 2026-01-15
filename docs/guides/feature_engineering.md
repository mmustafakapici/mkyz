# Feature Engineering Guide

Learn how to create powerful features that improve model performance.

## Why Feature Engineering?

Good features can:
- ✅ Significantly improve model accuracy
- ✅ Reduce training time by simplifying patterns
- ✅ Make models more interpretable
- ✅ Extract hidden information from raw data

---

## Getting Started

```python
from mkyz import FeatureEngineer, load_data

# Load your data
df = load_data('sales_data.csv')

# Initialize feature engineer
fe = FeatureEngineer(random_state=42)
```

---

## Polynomial Features

Create non-linear relationships between features.

```python
# Create polynomial features
df = fe.create_polynomial_features(
    df,
    columns=['price', 'quantity'],
    degree=2,
    interaction_only=False  # Include squares
)

# New columns created:
# price^2, quantity^2, price*quantity
```

**When to use:**
- Non-linear relationships exist
- Model is linear (e.g., Linear Regression)
- Few numerical features

---

## Interaction Features

Capture relationships between multiple features.

```python
df = fe.create_interaction_features(
    df,
    columns=['height', 'width', 'depth'],
    operations=['multiply', 'add', 'divide']
)

# Creates:
# height_x_width, height_x_depth, width_x_depth
# height_+_width, height_+_depth, width_+_depth
# height_/_width, etc.
```

**Example: BMI Calculation**
```python
df = fe.create_interaction_features(
    df,
    columns=['weight', 'height_squared'],
    operations=['divide']
)
# Creates weight/height² = BMI
```

---

## DateTime Features

Extract time-based patterns from datetime columns.

```python
df = fe.create_datetime_features(
    df,
    datetime_column='order_date',
    features=[
        'year', 'month', 'day', 
        'dayofweek', 'hour',
        'is_weekend', 'quarter'
    ]
)
```

**All Available Features:**

| Feature | Description | Example |
|---------|-------------|---------|
| `year` | Year | 2024 |
| `month` | Month (1-12) | 3 |
| `day` | Day of month | 15 |
| `dayofweek` | Day of week (0=Mon) | 2 |
| `hour` | Hour (0-23) | 14 |
| `minute` | Minute | 30 |
| `quarter` | Quarter (1-4) | 1 |
| `week` | Week of year | 12 |
| `is_weekend` | Saturday/Sunday | 1/0 |
| `is_month_start` | First day of month | 1/0 |
| `is_month_end` | Last day of month | 1/0 |
| `days_since_epoch` | Days since 1970-01-01 | 19732 |

**Example: E-commerce Analysis**
```python
# Identify shopping patterns
df = fe.create_datetime_features(df, 'purchase_time', features=[
    'hour', 'dayofweek', 'is_weekend', 'month'
])

# Now you can analyze:
# - Peak shopping hours
# - Weekday vs weekend behavior
# - Seasonal trends
```

---

## Lag Features (Time Series)

Create features from past values.

```python
# Create lag features for sales prediction
df = fe.create_lag_features(
    df,
    columns=['sales'],
    lags=[1, 7, 30],  # Yesterday, last week, last month
    sort_column='date'
)

# Creates:
# sales_lag_1 (yesterday's sales)
# sales_lag_7 (sales 7 days ago)
# sales_lag_30 (sales 30 days ago)
```

---

## Rolling Features

Capture trends with moving averages.

```python
df = fe.create_rolling_features(
    df,
    columns=['sales', 'revenue'],
    windows=[7, 30],
    functions=['mean', 'std', 'min', 'max']
)

# Creates:
# sales_rolling_7_mean (7-day moving average)
# sales_rolling_7_std (7-day volatility)
# sales_rolling_30_mean (monthly average)
# etc.
```

---

## Aggregation Features

Create group-based statistics.

```python
df = fe.create_aggregation_features(
    df,
    group_column='category',
    agg_columns=['price', 'quantity'],
    agg_functions=['mean', 'std', 'median']
)

# Creates:
# price_mean_by_category
# price_std_by_category
# quantity_mean_by_category
# etc.
```

**Example: Customer Features**
```python
df = fe.create_aggregation_features(
    df,
    group_column='customer_id',
    agg_columns=['order_value'],
    agg_functions=['mean', 'sum', 'count']
)
# Average order value per customer
# Total spending per customer
# Number of orders per customer
```

---

## Feature Selection

Remove irrelevant features to improve performance.

```python
# Select top 10 features using mutual information
selected_features = fe.select_features(
    X, y,
    k=10,
    method='mutual_info',  # or 'f_score'
    task='classification'
)

# Use only selected features
X_selected = X[selected_features]
print(f"Selected: {selected_features}")
```

---

## Complete Example

```python
import mkyz
from mkyz import FeatureEngineer

# Load data
df = mkyz.load_data('ecommerce_data.csv')

# Initialize
fe = FeatureEngineer(random_state=42)

# 1. DateTime features
df = fe.create_datetime_features(df, 'order_date', features=[
    'hour', 'dayofweek', 'is_weekend', 'month', 'quarter'
])

# 2. Polynomial features for pricing
df = fe.create_polynomial_features(
    df, ['unit_price', 'quantity'], degree=2
)

# 3. Customer aggregations
df = fe.create_aggregation_features(
    df,
    group_column='customer_id',
    agg_columns=['total_amount'],
    agg_functions=['mean', 'sum', 'count']
)

# 4. Time series features (if applicable)
df = fe.create_lag_features(df, ['daily_sales'], lags=[1, 7, 30])
df = fe.create_rolling_features(df, ['daily_sales'], windows=[7, 30])

# See all created features
print(f"Created {len(fe.created_features)} new features:")
for feat in fe.created_features[:10]:
    print(f"  - {feat}")

# 5. Feature selection
X = df.drop('target', axis=1)
y = df['target']
selected = fe.select_features(X, y, k=20)

print(f"\nTop 20 features: {selected}")
```

---

## Tips and Best Practices

### ✅ Do's

1. **Start simple** - Add features incrementally
2. **Domain knowledge** - Use business understanding
3. **Cross-validate** - Check if features actually help
4. **Handle missing values** - Lag/rolling features create NaNs

### ❌ Don'ts

1. **Don't overfit** - Too many features = overfitting
2. **Avoid data leakage** - Future data in training
3. **Watch cardinality** - One-hot on high cardinality = explosion

---

## Next Steps

- [Data Preparation Guide](data_preparation.md)
- [Model Training Guide](model_training.md)
- [API Reference](../api/data.md)
