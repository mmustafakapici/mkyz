# Visualization Quick Reference

**Which plot type to use for your data and problem type**

---

## 📊 For Understanding Your Data (EDA)

| Plot Type | When to Use | What It Shows |
|-----------|-------------|---------------|
| **Histogram** | Distribution of numerical features | Shape, spread, outliers |
| **Box Plot** | Outlier detection, comparing groups | Min, max, median, quartiles |
| **Bar Chart** | Categorical variable distribution | Count/frequency per category |
| **Scatter Plot** | Relationship between 2 numerical variables | Correlation, patterns, clusters |
| **Heatmap** | Correlation matrix | Which features are related |
| **Pair Plot** | Multiple numerical variables | All pairwise relationships |
| **Violin Plot** | Distribution comparison (like box + density) | Shape + summary stats |
| **KDE Plot** | Smooth distribution curve | Probability density |
| **Count Plot** | Categorical counts | Frequency of each category |

---

## 🎯 By Problem Type

### Classification Problems

| Visualization | Purpose | When to Use |
|---------------|---------|-------------|
| **Confusion Matrix** | See prediction errors | Always - understand which classes get confused |
| **ROC Curve** | True positive vs false positive trade-off | Binary classification, threshold selection |
| **Precision-Recall Curve** | Performance at different thresholds | Imbalanced datasets |
| **Class Distribution** | Check if classes are balanced | Before training - decide if stratification needed |
| **Feature Importance** | Which features matter most | Model interpretation, feature selection |
| **Decision Boundary** | How model separates classes | 2D feature space visualization |

### Regression Problems

| Visualization | Purpose | When to Use |
|---------------|---------|-------------|
| **Actual vs Predicted** | Overall model quality | Always - first plot to check |
| **Residual Plot** | Check for patterns in errors | Always - validate assumptions |
| **Q-Q Plot** | Check if residuals are normal | Linear regression diagnostics |
| **Residual Histogram** | Error distribution | Understanding error spread |
| **Prediction Interval Plot** | Uncertainty in predictions | When confidence intervals matter |
| **Time Series Plot** | Trend over time | Sequential/forecasting data |

### Clustering Problems

| Visualization | Purpose | When to Use |
|---------------|---------|-------------|
| **Elbow Plot** | Find optimal number of clusters | Before final clustering |
| **Silhouette Plot** | Cluster quality assessment | Evaluating clustering results |
| **Cluster Scatter** | Visualize clusters in 2D/3D | Presenting results |
| **Dendrogram** | Hierarchical cluster structure | Hierarchical clustering |
| **PCA Plot** | Visualize high-dimensional clusters | More than 3 features |

---

## 🎨 MKYZ Visualization Options

```python
import mkyz

# Available graphics options:
mkyz.visualize(data, graphics='<type>')

# EDA Plots
'histogram'      # Distribution of numerical columns
'bar'            # Bar charts for categorical columns
'box'            # Box plots for outlier detection
'scatter'        # Scatter plot for relationships
'corr'           # Correlation heatmap
'kde'            # Kernel density estimation
'violin'         # Violin plots
'pair'           # Pair plot for multiple variables
'heatmap'        # Generic heatmaps

# Model Evaluation Plots
'confusion_matrix'   # Classification errors
'roc'                # ROC curve
'pr_curve'           # Precision-Recall curve
'qq'                # Q-Q plot for residuals
'residual'          # Residual plot
'feature_importance' # Feature importance
```

---

## 🚀 Quick Decision Guide

```
What do you want to see?

┌─────────────────────────────────────────────────────┐
│  DATA DISTRIBUTION?                                 │
│  → Numerical:  Histogram, KDE, Box Plot             │
│  → Categorical: Bar Chart, Count Plot               │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│  RELATIONSHIPS?                                     │
│  → 2 numerical: Scatter Plot                        │
│  → Many numerical: Correlation Heatmap, Pair Plot   │
│  → Num vs Cat: Box Plot, Violin Plot               │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│  CLASSIFICATION RESULTS?                            │
│  → Confusion Matrix (always)                        │
│  → ROC Curve (binary)                               │
│  → PR Curve (imbalanced)                            │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│  REGRESSION RESULTS?                                │
│  → Actual vs Predicted (always)                     │
│  → Residual Plot (always)                           │
│  → Q-Q Plot (check assumptions)                     │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│  CLUSTERING RESULTS?                                │
│  → Elbow Plot (find k)                              │
│  → Silhouette Plot (quality)                        │
│  → Scatter Plot (visualize)                         │
└─────────────────────────────────────────────────────┘
```

---

## 📋 Example Workflows

### Classification Workflow
```python
import mkyz

# 1. Check class balance
mkyz.visualize(data, graphics='bar', target_column='class')

# 2. Train model
model = mkyz.train(data, task='classification', model='rf')

# 3. Evaluate with confusion matrix
mkyz.visualize(data, graphics='confusion_matrix', model=model)

# 4. ROC curve for binary
mkyz.visualize(data, graphics='roc', model=model)
```

### Regression Workflow
```python
import mkyz

# 1. Check correlations
mkyz.visualize(data, graphics='corr')

# 2. Train model
model = mkyz.train(data, task='regression', model='lr')

# 3. Actual vs Predicted
mkyz.visualize(data, graphics='scatter', model=model)

# 4. Check residuals
mkyz.visualize(data, graphics='residual', model=model)

# 5. Q-Q plot for normality
mkyz.visualize(data, graphics='qq', model=model)
```

### Clustering Workflow
```python
import mkyz

# 1. Find optimal k
for k in range(2, 11):
    model = mkyz.train(data, task='clustering', model='kmeans', n_clusters=k)
    # Track inertia for elbow plot

# 2. Visualize clusters
mkyz.visualize(data, graphics='scatter')

# 3. Check silhouette score
# Use sklearn.metrics.silhouette_score
```

---

## 🔍 One-Page Summary

| Goal | Use This Plot |
|------|---------------|
| See distribution | **Histogram** |
| Find outliers | **Box Plot** |
| Compare categories | **Bar Chart** |
| Check correlation | **Scatter Plot** |
| See all correlations | **Heatmap** |
| Classification errors | **Confusion Matrix** |
| Binary class performance | **ROC Curve** |
| Imbalanced class performance | **PR Curve** |
| Regression quality | **Actual vs Predicted** |
| Check regression errors | **Residual Plot** |
| Find cluster count | **Elbow Plot** |
| Cluster quality | **Silhouette Plot** |
| Feature importance | **Feature Importance** |
| Feature relationships | **Pair Plot** |
| Probability distribution | **KDE Plot** |
| Distribution + stats | **Violin Plot** |
| Normality check | **Q-Q Plot** |
| Time patterns | **Line Plot** |
