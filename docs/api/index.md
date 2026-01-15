# API Reference

Complete reference documentation for all MKYZ modules and functions.

## Modules

| Module | Description |
|--------|-------------|
| [Core](core.md) | Configuration, exceptions, base classes |
| [Data](data.md) | Data loading, preprocessing, feature engineering |
| [EDA](eda.md) | Exploratory data analysis, profiling, insights |
| [Evaluation](evaluation.md) | Metrics, cross-validation, reporting |
| [Persistence](persistence.md) | Model saving and loading |
| [Utils](utils.md) | Logging and parallel processing |

---

## Quick Reference

### Most Used Functions


```python
import mkyz

# Data
mkyz.load_data(path)                    # Load data
mkyz.prepare_data(path, target)         # Full preprocessing
mkyz.validate_dataset(df)               # Validate data

# Training
mkyz.train(data, task, model)           # Train single model
mkyz.auto_train(data, task)             # AutoML
mkyz.predict(data, model)               # Make predictions

# Evaluation
mkyz.evaluate(data, predictions)        # Quick evaluation
mkyz.cross_validate(model, X, y)        # Cross-validation
mkyz.classification_metrics(y, pred)    # Detailed metrics

# Persistence
mkyz.save_model(model, path)            # Save model
mkyz.load_model(path)                   # Load model

# Reporting
mkyz.ModelReport(model, X, y).export_html('report.html')
```

---

## Import Paths

```python
# Main imports (recommended)
import mkyz

# Or specific modules
from mkyz.core import Config, MKYZError
from mkyz.data import FeatureEngineer, DataLoader
from mkyz.evaluation import cross_validate, CVStrategy, ModelReport
from mkyz.persistence import save_model, load_model
from mkyz.utils import setup_logging, parallel_map
```
