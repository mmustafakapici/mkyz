# Model Persistence Guide

Learn how to save, load, and manage your trained models.

## Saving Models

### Basic Save

```python
import mkyz

# Train your model
model = mkyz.train(data, task='classification', model='rf')

# Save it
mkyz.save_model(model, 'models/my_model')
# Saved to: models/my_model.joblib
```

### With Metadata

```python
mkyz.save_model(
    model,
    'models/classifier_v1',
    metadata={
        'accuracy': 0.95,
        'features': list(X.columns),
        'version': '1.0',
        'trained_by': 'your_name',
        'dataset': 'customers_2024'
    }
)
```

### Choose Format

```python
# Joblib (default, faster for numpy arrays)
mkyz.save_model(model, 'model', format='joblib')

# Pickle (more compatible)
mkyz.save_model(model, 'model', format='pickle')
```

---

## Loading Models

### Basic Load

```python
model = mkyz.load_model('models/my_model.joblib')
predictions = model.predict(X_new)
```

### With Metadata

```python
model, metadata = mkyz.load_model(
    'models/classifier_v1.joblib',
    return_metadata=True
)

print(f"Saved at: {metadata['saved_at']}")
print(f"Accuracy: {metadata['accuracy']}")
print(f"Features: {metadata['features']}")
```

---

## Version Control

```python
import datetime

# Version with timestamp
version = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
mkyz.save_model(model, f'models/classifier_{version}')
# models/classifier_20240115_143000.joblib
```

---

## Best Practices

1. **Always include metadata** - Track accuracy, features, version
2. **Version your models** - Use timestamps or version numbers
3. **Store feature names** - Ensure correct input order
4. **Verify after loading** - Quick sanity check

```python
# Verification after loading
model = mkyz.load_model('model.joblib')
test_pred = model.predict(X_test[:5])
print(f"Model works: {test_pred}")
```

---

## Next Steps

- [API Reference](../api/persistence.md)
- [Quick Start](../quickstart.md)
