# Installation Guide

## Requirements

- Python 3.6 or higher
- pip package manager

## Quick Install

The simplest way to install MKYZ is via pip:

```bash
pip install mkyz
```

## Install from Source

For the latest development version:

```bash
# Clone the repository
git clone https://github.com/mmustafakapici/mkyz.git
cd mkyz

# Install in development mode
pip install -e .
```

## Dependencies

MKYZ will automatically install the following dependencies:

### Core Dependencies
| Package | Purpose |
|---------|---------|
| `pandas` | Data manipulation |
| `numpy` | Numerical operations |
| `scikit-learn` | Machine learning models |

### Visualization
| Package | Purpose |
|---------|---------|
| `matplotlib` | Static plots |
| `seaborn` | Statistical visualization |
| `plotly` | Interactive plots |

### Advanced Models
| Package | Purpose |
|---------|---------|
| `xgboost` | Extreme Gradient Boosting |
| `lightgbm` | Light Gradient Boosting |
| `catboost` | Categorical Boosting |

### Utilities
| Package | Purpose |
|---------|---------|
| `rich` | Beautiful console output |
| `mlxtend` | Association rule learning |
| `statsmodels` | Statistical models |
| `joblib` | Model serialization |

## Optional Dependencies

For hyperparameter optimization with Bayesian search:

```bash
pip install optuna
```

For Jupyter notebook support:

```bash
pip install jupyter ipywidgets
```

## Verify Installation

```python
import mkyz
print(f"MKYZ version: {mkyz.__version__}")
# Expected output: mkyz package initialized. Version: 0.2.0
```

## Troubleshooting

### Common Issues

**Issue: ImportError for optional dependencies**
```
Solution: Install the missing package
pip install <package_name>
```

**Issue: Conflicting sklearn versions**
```
Solution: Upgrade scikit-learn
pip install --upgrade scikit-learn
```

**Issue: CatBoost installation fails on Windows**
```
Solution: Install Visual C++ Build Tools first
```

## Upgrading

To upgrade to the latest version:

```bash
pip install --upgrade mkyz
```

## Uninstalling

```bash
pip uninstall mkyz
```

---

Next: [Quick Start Guide](quickstart.md)
