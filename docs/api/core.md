# API Reference - Core Module

The core module provides configuration, exceptions, and base classes.

```python
from mkyz.core import Config, DEFAULT_CONFIG, get_config, set_config
from mkyz.core.exceptions import MKYZError, DataValidationError
```

---

## Configuration

### Config Class

```python
from mkyz import Config, DEFAULT_CONFIG
```

A dataclass containing all global configuration options.

#### Attributes

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `random_state` | int | 42 | Random seed for reproducibility |
| `n_jobs` | int | -1 | Number of parallel jobs (-1 = all CPUs) |
| `verbose` | int | 1 | Verbosity level (0=silent, 1=progress, 2=debug) |
| `cv_folds` | int | 5 | Default cross-validation folds |
| `test_size` | float | 0.2 | Default test set proportion |
| `optimization_method` | str | 'grid_search' | 'grid_search' or 'bayesian' |
| `n_trials` | int | 50 | Trials for Bayesian optimization |
| `log_level` | str | 'INFO' | Logging level |
| `suppress_warnings` | bool | True | Suppress sklearn warnings |
| `default_classification_model` | str | 'rf' | Default classifier |
| `default_regression_model` | str | 'rf' | Default regressor |
| `dark_mode` | bool | True | Dark visualization theme |
| `missing_value_strategy` | str | 'mean' | 'mean', 'median', 'mode', 'drop' |
| `outlier_strategy` | str | 'remove' | 'remove', 'cap', 'keep' |
| `categorical_encoding` | str | 'onehot' | 'onehot', 'label', 'frequency' |

#### Methods

##### `update(**kwargs) -> Config`
Update configuration with new values.

```python
config = Config()
config.update(random_state=123, n_jobs=4)
```

##### `to_dict() -> Dict[str, Any]`
Convert configuration to dictionary.

```python
config_dict = config.to_dict()
```

##### `from_dict(config_dict) -> Config`
Create Config from dictionary.

```python
config = Config.from_dict({'random_state': 123})
```

---

### get_config

```python
mkyz.get_config() -> Config
```

Get the current global configuration.

**Returns:** Current Config instance

**Example:**
```python
config = mkyz.get_config()
print(config.random_state)  # 42
```

---

### set_config

```python
mkyz.set_config(**kwargs) -> Config
```

Update global configuration.

**Parameters:**
- `**kwargs`: Configuration values to update

**Returns:** Updated Config instance

**Example:**
```python
mkyz.set_config(
    random_state=123,
    n_jobs=4,
    verbose=2
)
```

---

## Exceptions

All MKYZ exceptions inherit from `MKYZError`.

### MKYZError

```python
from mkyz import MKYZError
```

Base exception class for all MKYZ errors.

```python
try:
    # MKYZ operation
    pass
except MKYZError as e:
    print(f"MKYZ error: {e}")
```

---

### DataValidationError

```python
from mkyz import DataValidationError
```

Raised when data validation fails.

**Common causes:**
- Missing required columns
- Invalid data types
- Empty datasets
- Incompatible data shapes

```python
try:
    mkyz.validate_dataset(df)
except DataValidationError as e:
    print(f"Data issue: {e}")
```

---

### ModelNotTrainedError

```python
from mkyz import ModelNotTrainedError
```

Raised when using a model that hasn't been trained.

**Common causes:**
- Calling predict() before fit()
- Accessing model parameters before training

---

### UnsupportedTaskError

```python
from mkyz import UnsupportedTaskError
```

Raised when an unsupported task type is specified.

**Supported tasks:** `classification`, `regression`, `clustering`, `dimensionality_reduction`

---

### UnsupportedModelError

```python
from mkyz import UnsupportedModelError
```

Raised when an unsupported model type is specified.

---

### OptimizationError

```python
from mkyz import OptimizationError
```

Raised when hyperparameter optimization fails.

---

### PersistenceError

```python
from mkyz import PersistenceError
```

Raised when model saving or loading fails.

**Common causes:**
- File not found
- Corrupted model file
- Incompatible model version

---

## Base Classes

### BaseEstimator

```python
from mkyz.core.base import BaseEstimator
```

Abstract base class for all MKYZ estimators.

#### Methods

| Method | Description |
|--------|-------------|
| `fit(X, y=None)` | Fit the model to training data |
| `predict(X)` | Make predictions on new data |
| `get_params()` | Get model parameters |
| `set_params(**params)` | Set model parameters |
| `is_fitted` | Property: True if model is fitted |

---

### DataMixin

```python
from mkyz.core.base import DataMixin
```

Mixin class providing data validation utilities.

#### Static Methods

##### `validate_dataframe(df, required_columns=None)`
Validate DataFrame input.

##### `validate_array(arr, expected_dim=None)`
Validate numpy array input.

##### `get_column_types(df) -> Dict`
Get numerical and categorical column names.

```python
types = DataMixin.get_column_types(df)
print(types['numerical'])
print(types['categorical'])
```

---

### ModelMixin

```python
from mkyz.core.base import ModelMixin
```

Mixin class providing model utilities.

#### Static Methods

##### `get_scorer(task, metric=None)`
Get appropriate scorer for the task.

##### `format_scores(scores, precision=4) -> str`
Format scores dictionary as readable string.

---

## See Also

- [Data Module](data.md)
- [Evaluation Module](evaluation.md)
- [Persistence Module](persistence.md)
