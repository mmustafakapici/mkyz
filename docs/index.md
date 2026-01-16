# MKYZ Documentation

Welcome to the MKYZ Machine Learning Library documentation.

## Choose Your API

MKYZ provides two APIs for different use cases:

### Basic API
Simple functions for quick ML tasks. See [Basic API Reference](basic_api.md)

**Quick start:** `import mkyz; data = mkyz.prepare_data('file.csv')`

**Best for:** Quick prototyping, learning, simple projects

### Modular API
Granular control for advanced workflows. See [Modular API Reference](api/index.md)

**Quick start:** `import mkyz; df = mkyz.load_data('file.csv')`

**Best for:** Production systems, custom workflows, advanced features

---

## 📚 Contents

### Getting Started
- [Installation](installation.md) - How to install MKYZ
- [Quick Start](quickstart.md) - Get up and running in 5 minutes
- [Basic API Reference](basic_api.md) - Complete Basic API documentation
- [Configuration](configuration.md) - Customize MKYZ behavior

### User Guides
- [Data Preparation](guides/data_preparation.md) - Load and preprocess data
- [Feature Engineering](guides/feature_engineering.md) - Create powerful features
- [Model Training](guides/model_training.md) - Train and optimize models
- [Model Evaluation](guides/model_evaluation.md) - Assess model performance
- [Metrics and Visualizations Guide](guides/metrics_and_visualizations.md) - **NEW!** Choose the right metrics and plots for your task
- [Model Persistence](guides/model_persistence.md) - Save and load models

### API Reference

#### Basic API
- [Basic API Reference](basic_api.md) - All Basic API functions

#### Modular API Modules
- [Core Module](api/core.md) - Configuration and exceptions
- [Data Module](api/data.md) - Data processing functions
- [Evaluation Module](api/evaluation.md) - Metrics and cross-validation
- [EDA Module](api/eda.md) - Exploratory data analysis
- [Persistence Module](api/persistence.md) - Model serialization
- [Utils Module](api/utils.md) - Logging and parallel processing

### Additional Resources
- [Examples](../examples/) - Code examples and notebooks
- [Changelog](changelog.md) - Version history
- [Contributing](../CONTRIBUTING.md) - How to contribute

---

## Quick Links

### Basic API

| Task | Function | Example |
|------|----------|---------|
| Prepare data | `mkyz.prepare_data()` | `data = mkyz.prepare_data('data.csv', target='y')` |
| Train model | `mkyz.train()` | `model = mkyz.train(data, model='rf')` |
| Auto-train | `mkyz.auto_train()` | `best = mkyz.auto_train(data)` |
| Predict | `mkyz.predict()` | `preds = mkyz.predict(data, model)` |
| Evaluate | `mkyz.evaluate()` | `scores = mkyz.evaluate(data, preds)` |
| Visualize | `mkyz.visualize()` | `mkyz.visualize(data, graphics='histogram')` |

### Modular API

| Task | Function | Example |
|------|----------|---------|
| Load data | `mkyz.load_data()` | `df = mkyz.load_data('data.csv')` |
| Validate | `mkyz.validate_dataset()` | `validation = mkyz.validate_dataset(df)` |
| Configure | `mkyz.set_config()` | `mkyz.set_config(random_state=42)` |
| Cross-validate | `mkyz.cross_validate()` | `results = mkyz.cross_validate(model, X, y)` |
| Save model | `mkyz.save_model()` | `mkyz.save_model(model, 'model.joblib')` |
| Load model | `mkyz.load_model()` | `model = mkyz.load_model('model.joblib')` |
| Generate report | `mkyz.ModelReport()` | `report = mkyz.ModelReport(model, X, y)` |

---

## Version Information

- **Current Version:** 0.2.3
- **Python Support:** 3.6+
- **License:** MIT

## Support

- **GitHub Issues:** [Report bugs](https://github.com/mmustafakapici/mkyz/issues)
- **Email:** m.mustafakapici@gmail.com
