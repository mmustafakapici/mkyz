# API Reference - Persistence Module

The persistence module provides model saving and loading.

```python
from mkyz import save_model, load_model
from mkyz.persistence import export_pipeline, import_pipeline
```

---

## save_model

```python
mkyz.save_model(
    model,
    path,
    format='joblib',
    metadata=None,
    overwrite=False
) -> str
```

Save a trained model to disk.

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | object | - | Trained model to save |
| `path` | str | - | Output file path |
| `format` | str | 'joblib' | 'joblib' or 'pickle' |
| `metadata` | dict | None | Optional metadata |
| `overwrite` | bool | False | Overwrite existing file |

**Returns:** Absolute path to saved file

**Example:**
```python
# Basic save
mkyz.save_model(model, 'models/my_model')
# Saves to: models/my_model.joblib

# With metadata
mkyz.save_model(
    model,
    'models/classifier_v1',
    metadata={
        'accuracy': 0.95,
        'version': '1.0',
        'trained_on': 'dataset_v2',
        'features': feature_names
    }
)

# Force overwrite
mkyz.save_model(model, 'models/my_model', overwrite=True)
```

**Saved Data Structure:**
```python
{
    'model': <trained_model>,
    'metadata': {...},
    'saved_at': '2024-01-15T10:30:00',
    'mkyz_version': '0.2.0'
}
```

---

## load_model

```python
mkyz.load_model(path, return_metadata=False) -> model | tuple
```

Load a trained model from disk.

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | str | - | Path to saved model |
| `return_metadata` | bool | False | Return (model, metadata) tuple |

**Returns:** 
- If `return_metadata=False`: Loaded model
- If `return_metadata=True`: `(model, metadata)` tuple

**Example:**
```python
# Load model only
model = mkyz.load_model('models/my_model.joblib')

# Load with metadata
model, metadata = mkyz.load_model(
    'models/classifier_v1.joblib',
    return_metadata=True
)

print(f"Model saved at: {metadata['saved_at']}")
print(f"Accuracy: {metadata['accuracy']}")
print(f"MKYZ version: {metadata['mkyz_version']}")

# Use the model
predictions = model.predict(X_new)
```

**Metadata Contents:**
```python
{
    'accuracy': 0.95,        # User-provided metadata
    'version': '1.0',
    'saved_at': '2024-01-15T10:30:00',
    'loaded_at': '2024-01-16T14:20:00',
    'mkyz_version': '0.2.0'
}
```

---

## export_pipeline

```python
from mkyz.persistence import export_pipeline

export_pipeline(pipeline, path, overwrite=False) -> str
```

Export ML pipeline configuration as JSON.

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `pipeline` | dict | Pipeline configuration |
| `path` | str | Output JSON path |
| `overwrite` | bool | Overwrite existing file |

**Example:**
```python
pipeline_config = {
    'preprocessing': {
        'missing_values': 'mean',
        'outlier_strategy': 'cap',
        'encoding': 'onehot',
        'scaling': 'standard'
    },
    'feature_engineering': {
        'polynomial_degree': 2,
        'datetime_features': ['year', 'month', 'dayofweek']
    },
    'model': {
        'type': 'rf',
        'params': {
            'n_estimators': 100,
            'max_depth': 10
        }
    },
    'target': 'price'
}

export_pipeline(pipeline_config, 'pipelines/price_predictor.json')
```

---

## import_pipeline

```python
from mkyz.persistence import import_pipeline

import_pipeline(path) -> dict
```

Import pipeline configuration from JSON.

**Example:**
```python
config = import_pipeline('pipelines/price_predictor.json')
print(config['model']['type'])  # 'rf'
```

---

## Error Handling

```python
from mkyz import PersistenceError

try:
    model = mkyz.load_model('nonexistent_model.joblib')
except FileNotFoundError:
    print("Model file not found")
except PersistenceError as e:
    print(f"Failed to load: {e}")
```

---

## Best Practices

### 1. Always Include Metadata

```python
mkyz.save_model(
    model,
    'models/classifier',
    metadata={
        'accuracy': test_accuracy,
        'features': list(X.columns),
        'training_samples': len(X_train),
        'created_by': 'your_name'
    }
)
```

### 2. Version Your Models

```python
import datetime

version = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
mkyz.save_model(model, f'models/classifier_{version}')
```

### 3. Verify After Loading

```python
model = mkyz.load_model('models/classifier.joblib')

# Quick sanity check
test_pred = model.predict(X_test[:5])
print(f"Test predictions: {test_pred}")
```

---

## See Also

- [Model Persistence Guide](../guides/model_persistence.md)
- [Quick Start](../quickstart.md)
