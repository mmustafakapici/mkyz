# MKYZ Documentation

Welcome to the MKYZ Machine Learning Library documentation.

## 📚 Contents

### Getting Started
- [Installation](installation.md) - How to install MKYZ
- [Quick Start](quickstart.md) - Get up and running in 5 minutes
- [Configuration](configuration.md) - Customize MKYZ behavior

### User Guides
- [Data Preparation](guides/data_preparation.md) - Load and preprocess data
- [Feature Engineering](guides/feature_engineering.md) - Create powerful features
- [Model Training](guides/model_training.md) - Train and optimize models
- [Model Evaluation](guides/model_evaluation.md) - Assess model performance
- [Model Persistence](guides/model_persistence.md) - Save and load models

### API Reference
- [Core Module](api/core.md) - Configuration and exceptions
- [Data Module](api/data.md) - Data processing functions
- [Evaluation Module](api/evaluation.md) - Metrics and cross-validation
- [Persistence Module](api/persistence.md) - Model serialization
- [Utils Module](api/utils.md) - Logging and parallel processing

### Additional Resources
- [Examples](examples/index.md) - Code examples and notebooks
- [Changelog](changelog.md) - Version history
- [Contributing](contributing.md) - How to contribute
- [FAQ](faq.md) - Frequently asked questions

---

## Quick Links

| Task | Function | Example |
|------|----------|---------|
| Load data | `mkyz.load_data()` | `df = mkyz.load_data('data.csv')` |
| Prepare data | `mkyz.prepare_data()` | `data = mkyz.prepare_data('data.csv', target='y')` |
| Train model | `mkyz.train()` | `model = mkyz.train(data, model='rf')` |
| Auto-train | `mkyz.auto_train()` | `best = mkyz.auto_train(data)` |
| Evaluate | `mkyz.evaluate()` | `scores = mkyz.evaluate(data, predictions)` |
| Cross-validate | `mkyz.cross_validate()` | `results = mkyz.cross_validate(model, X, y)` |
| Save model | `mkyz.save_model()` | `mkyz.save_model(model, 'model.joblib')` |
| Load model | `mkyz.load_model()` | `model = mkyz.load_model('model.joblib')` |
| Generate report | `mkyz.ModelReport()` | `report = mkyz.ModelReport(model, X, y)` |

---

## Version Information

- **Current Version:** 0.2.0
- **Python Support:** 3.6+
- **License:** MIT

## Support

- **GitHub Issues:** [Report bugs](https://github.com/mmustafakapici/mkyz/issues)
- **Email:** m.mustafakapici@gmail.com
