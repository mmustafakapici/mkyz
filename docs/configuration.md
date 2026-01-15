# Configuration Guide

Learn how to configure MKYZ globally.

## Quick Configuration

```python
import mkyz

# Update settings
mkyz.set_config(
    random_state=42,
    n_jobs=-1,
    verbose=1
)
```

---

## All Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `random_state` | 42 | Random seed |
| `n_jobs` | -1 | Parallel jobs (-1 = all CPUs) |
| `cv_folds` | 5 | Cross-validation folds |
| `test_size` | 0.2 | Test split ratio |
| `verbose` | 1 | Verbosity (0-2) |
| `optimization_method` | 'grid_search' | Hyperparameter method |
| `n_trials` | 50 | Bayesian optimization trials |
| `log_level` | 'INFO' | Logging level |
| `suppress_warnings` | True | Hide sklearn warnings |
| `default_classification_model` | 'rf' | Default classifier |
| `default_regression_model` | 'rf' | Default regressor |
| `dark_mode` | True | Dark plot theme |
| `missing_value_strategy` | 'mean' | Missing value handling |
| `outlier_strategy` | 'remove' | Outlier handling |
| `categorical_encoding` | 'onehot' | Encoding method |

---

## View Current Config

```python
config = mkyz.get_config()
print(config.to_dict())
```

---

## Environment-Based Config

```python
import os

if os.getenv('MKYZ_ENV') == 'production':
    mkyz.set_config(verbose=0, n_jobs=-1)
else:
    mkyz.set_config(verbose=2, n_jobs=1)
```

---

## See Also

- [API Reference - Core](api/core.md)
