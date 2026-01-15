# Model Evaluation Guide

Learn how to properly evaluate your machine learning models.

## Quick Evaluation

```python
import mkyz

# After training
predictions = mkyz.predict(data, model)
scores = mkyz.evaluate(data, predictions, task='classification')
print(scores)
```

---

## Detailed Metrics

### Classification

```python
metrics = mkyz.classification_metrics(y_test, predictions, y_proba)

print(f"Accuracy:  {metrics['accuracy']:.4f}")
print(f"Precision: {metrics['precision']:.4f}")
print(f"Recall:    {metrics['recall']:.4f}")
print(f"F1 Score:  {metrics['f1_score']:.4f}")
print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
```

### Regression

```python
metrics = mkyz.regression_metrics(y_test, predictions)

print(f"RMSE:      {metrics['rmse']:.4f}")
print(f"MAE:       {metrics['mae']:.4f}")
print(f"R² Score:  {metrics['r2_score']:.4f}")
```

---

## Cross-Validation

Never evaluate on training data alone!

```python
from mkyz import cross_validate, CVStrategy

results = mkyz.cross_validate(
    model, X, y,
    cv=CVStrategy.STRATIFIED,
    n_splits=5
)

print(f"CV Score: {results['mean_test_score']:.4f} ± {results['std_test_score']:.4f}")
```

### Strategy Selection

| Data Type | Strategy |
|-----------|----------|
| Classification | `STRATIFIED` |
| Regression | `KFOLD` |
| Time Series | `TIME_SERIES` |
| Grouped Data | `GROUP` |

---

## Generating Reports

```python
from mkyz import ModelReport

report = ModelReport(
    model=model,
    X_test=X_test,
    y_test=y_test,
    task='classification'
)

# Generate
report.generate()

# View summary
print(report.summary())

# Export HTML
report.export_html('reports/model_report.html')
```

---

## Comparing Models

```python
from mkyz import cross_validate

models = {
    'rf': RandomForestClassifier(),
    'xgb': XGBClassifier(),
    'lgbm': LGBMClassifier()
}

for name, model in models.items():
    results = mkyz.cross_validate(model, X, y)
    print(f"{name}: {results['mean_test_score']:.4f}")
```

---

## Next Steps

- [Model Persistence Guide](model_persistence.md)
- [API Reference](../api/evaluation.md)
