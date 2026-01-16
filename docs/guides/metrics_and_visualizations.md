# Metrics and Visualizations Guide

A comprehensive guide to choosing the right metrics and visualizations for your machine learning tasks.

## Table of Contents

- [Classification Metrics](#classification-metrics)
- [Classification Visualizations](#classification-visualizations)
- [Regression Metrics](#regression-metrics)
- [Regression Visualizations](#regression-visualizations)
- [Clustering Metrics](#clustering-metrics)
- [Clustering Visualizations](#clustering-visualizations)
- [When to Use What](#when-to-use-what)

---

## Classification Metrics

### Primary Metrics

| Metric | Range | Best For | When to Use |
|--------|-------|----------|-------------|
| **Accuracy** | 0-1 | Balanced datasets | Classes are roughly equal |
| **F1 Score** | 0-1 | Imbalanced datasets | When precision and recall both matter |
| **Precision** | 0-1 | When false positives are costly | Spam detection, fraud detection |
| **Recall** | 0-1 | When false negatives are costly | Medical diagnosis, disease screening |
| **ROC AUC** | 0-1 | Model comparison | When you need threshold-independent evaluation |

### Secondary Metrics

| Metric | Description | Use Case |
|--------|-------------|----------|
| **Specificity** | True negative rate | When true negatives are important |
| **PR AUC** | Area under precision-recall curve | Highly imbalanced datasets |
| **Log Loss** | Probabilistic prediction quality | When predicted probabilities matter |
| **Cohen's Kappa** | Agreement beyond chance | Multi-class problems |

### Classification Metric Guide

```
┌─────────────────────────────────────────────────────────────────┐
│                    CLASSIFICATION METRICS                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  BALANCED DATASET  ────────────────►  USE ACCURACY              │
│                                                                  │
│  IMBALANCED DATASET  ──────────────►  USE F1-SCORE              │
│                                                                  │
│  FALSE POSITIVES COSTLY  ───────────►  USE PRECISION            │
│  (e.g., Spam Detection, False Alarms)                           │
│                                                                  │
│  FALSE NEGATIVES COSTLY  ──────────►  USE RECALL                │
│  (e.g., Medical Diagnosis, Fraud)                               │
│                                                                  │
│  THRESHOLD-FREE EVALUATION  ───────►  USE ROC-AUC               │
│  (e.g., Model Comparison)                                       │
│                                                                  │
│  HIGHLY IMBALANCED  ───────────────►  USE PR-AUC                │
│  (e.g., Rare Disease Detection)                                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Classification Visualizations

### Confusion Matrix

**Best for:** Understanding model errors, class-wise performance

```python
import mkyz
mkyz.visualize(data, graphics='confusion_matrix')
```

**When to use:**
- Debugging classification errors
- Multi-class problems
- Presenting results to stakeholders

### ROC Curve

**Best for:** Comparing models, understanding true positive vs false positive trade-off

```python
mkyz.visualize(data, graphics='roc')
```

**When to use:**
- Binary classification
- Selecting optimal threshold
- Model comparison

### Precision-Recall Curve

**Best for:** Imbalanced datasets

```python
mkyz.visualize(data, graphics='pr_curve')
```

**When to use:**
- Highly imbalanced classes
- When positive class is rare
- More informative than ROC for imbalanced data

### Class Distribution Plot

**Best for:** Checking class balance before training

```python
mkyz.visualize(data, graphics='bar', target_column='class')
```

**When to use:**
- Initial EDA
- Deciding whether to use class weights
- Understanding dataset composition

### Feature Importance Plot

**Best for:** Model interpretability

```python
mkyz.visualize(data, graphics='feature_importance', model=model)
```

**When to use:**
- Feature selection
- Explaining model to stakeholders
- Identifying key predictors

### Classification Visualization Guide

```
┌─────────────────────────────────────────────────────────────────┐
│               CLASSIFICATION VISUALIZATIONS                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  UNDERSTANDING ERRORS    ─────────►  CONFUSION MATRIX           │
│  "Which classes are confused?"                                   │
│                                                                  │
│  BINARY CLASSIFICATION      ────►  ROC CURVE                     │
│  "What's the trade-off between TPR and FPR?"                     │
│                                                                  │
│  IMBALANCED DATA          ─────────►  PR CURVE                   │
│  "How does precision change with recall?"                        │
│                                                                  │
│  CHECKING CLASS BALANCE   ─────────►  BAR PLOT (target)          │
│  "Are my classes balanced?"                                      │
│                                                                  │
│  MODEL INTERPRETABILITY  ─────────►  FEATURE IMPORTANCE          │
│  "Which features matter most?"                                   │
│                                                                  │
│  PROBABILITY CALIBRATION   ───────►  CALIBRATION PLOT            │
│  "Are predicted probabilities reliable?"                         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Regression Metrics

### Primary Metrics

| Metric | Range | Best For | When to Use |
|--------|-------|----------|-------------|
| **R² (R-Squared)** | -∞ to 1 | Overall model fit | Explaining variance, model comparison |
| **RMSE** | 0 to +∞ | Large errors matter | When big errors are disproportionately bad |
| **MAE** | 0 to +∞ | Interpretability | When all errors scale linearly |
| **MAPE** | 0% to +∞ | Business context | When you need percentage error |

### Secondary Metrics

| Metric | Description | Use Case |
|--------|-------------|----------|
| **Adjusted R²** | R² adjusted for predictors | Comparing models with different feature counts |
| **Max Error** | Largest single error | Worst-case scenario analysis |
| **Median Absolute Error** | Median of absolute errors | Robust to outliers |
| **Explained Variance** | Variance explained by model | Alternative to R² |

### Regression Metric Guide

```
┌─────────────────────────────────────────────────────────────────┐
│                      REGRESSION METRICS                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  OVERALL MODEL FIT        ────────────►  R² (R-SQUARED)         │
│  "How much variance is explained?"                                │
│                                                                  │
│  LARGE ERRORS MATTER       ────────────►  RMSE                   │
│  "Big errors are much worse" (penalizes outliers)                │
│                                                                  │
│  INTERPRETABILITY NEEDED   ────────────►  MAE                    │
│  "Average error in same units as target"                         │
│                                                                  │
│  BUSINESS CONTEXT          ────────────►  MAPE                   │
│  "Error as percentage for stakeholders"                          │
│                                                                  │
│  OUTLIER-ROBUST            ────────────►  MEDIAN ABSOLUTE ERROR  │
│  "Typical error ignoring extremes"                                │
│                                                                  │
│  WORST CASE ANALYSIS       ────────────►  MAX ERROR              │
│  "What's the worst prediction?"                                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Metric Interpretation

| Metric | Good | Excellent | Notes |
|--------|------|-----------|-------|
| **R²** | > 0.7 | > 0.9 | Domain dependent |
| **RMSE** | Lower is better | Context dependent | Compare to target mean |
| **MAE** | Lower is better | Context dependent | More interpretable than RMSE |
| **MAPE** | < 10% | < 5% | Expressed as percentage |

---

## Regression Visualizations

### Actual vs Predicted Plot

**Best for:** Overall model assessment, identifying systematic errors

```python
import matplotlib.pyplot as plt
plt.scatter(y_test, predictions)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted')
```

**When to use:**
- Always use this first
- Identifies bias (systematic over/under prediction)
- Shows variance heterogeneity

### Residual Plot

**Best for:** Checking model assumptions, detecting patterns

```python
plt.scatter(predictions, y_test - predictions)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
```

**When to use:**
- Checking for non-linearity
- Detecting heteroscedasticity
- Model diagnostic

### Q-Q Plot

**Best for:** Checking residual normality

```python
mkyz.visualize(data, graphics='qq')
```

**When to use:**
- Validating normality assumption
- Required for some statistical tests
- Linear regression diagnostics

### Prediction Error Distribution

**Best for:** Understanding error spread

```python
mkyz.visualize(data, graphics='histogram')
```

**When to use:**
- Understanding error distribution
- Identifying outliers in predictions
- Setting confidence intervals

### Feature vs Target Plots

**Best for:** Understanding relationships

```python
mkyz.visualize(data, graphics='scatter')
```

**When to use:**
- Feature selection
- Detecting non-linear relationships
- Initial EDA

### Regression Visualization Guide

```
┌─────────────────────────────────────────────────────────────────┐
│                REGRESSION VISUALIZATIONS                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  MODEL ASSESSMENT         ───────────►  ACTUAL VS PREDICTED     │
│  "How close are predictions to reality?"                         │
│                                                                  │
│  MODEL DIAGNOSTICS        ───────────►  RESIDUAL PLOT            │
│  "Are there patterns in errors?"                                 │
│                                                                  │
│  CHECKING ASSUMPTIONS      ───────────►  Q-Q PLOT                │
│  "Are residuals normally distributed?"                            │
│                                                                  │
│  ERROR ANALYSIS           ───────────►  ERROR DISTRIBUTION       │
│  "What's the spread of errors?"                                  │
│                                                                  │
│  FEATURE RELATIONSHIPS    ───────────►  SCATTER PLOT MATRIX      │
│  "How do features relate to target?"                             │
│                                                                  │
│  FEATURE IMPORTANCE       ───────────►  FEATURE IMPORTANCE       │
│  "Which features drive predictions?"                             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Clustering Metrics

### Primary Metrics

| Metric | Range | Best For | When to Use |
|--------|-------|----------|-------------|
| **Silhouette Score** | -1 to 1 | General cluster quality | Evaluating cluster separation and cohesion |
| **Inertia (WCSS)** | 0 to +∞ | Elbow method | Finding optimal number of clusters |
| **Davies-Bouldin Index** | 0 to +∞ | Cluster separation | Lower is better, similar to silhouette |
| **Calinski-Harabasz** | 0 to +∞ | Cluster dispersion | Higher is better |

### Clustering Metric Guide

```
┌─────────────────────────────────────────────────────────────────┐
│                      CLUSTERING METRICS                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  GENERAL QUALITY          ───────────►  SILHOUETTE SCORE         │
│  > 0.5: Good structure                                             │
│  > 0.7: Strong structure                                          │
│  < 0.25: No meaningful structure                                  │
│                                                                  │
│  FINDING K (OPTIMAL)      ───────────►  ELBOW METHOD (Inertia)   │
│  Look for "elbow" in the curve                                    │
│                                                                  │
│  CLUSTER SEPARATION       ───────────►  DAVIES-BOULDIN INDEX     │
│  Lower = Better separation                                         │
│                                                                  │
│  CLUSTER DISPERSION      ───────────►  CALINSKI-HARABASZ         │
│  Higher = Better defined clusters                                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Clustering Visualizations

### Cluster Scatter Plot

**Best for:** 2D cluster visualization

```python
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200)
```

**When to use:**
- 2D or 3D data
- Presenting results to non-technical audience
- Quick cluster assessment

### Elbow Plot

**Best for:** Finding optimal number of clusters

```python
# Plot inertia vs k
inertias = []
for k in range(1, 11):
    kmeans = mkyz.train(data, task='clustering', model='kmeans', n_clusters=k)
    inertias.append(kmeans.inertia_)
plt.plot(range(1, 11), inertias, 'bo-')
```

**When to use:**
- Determining k before final clustering
- K-means clustering
- Always use for unsupervised cluster count selection

### Silhouette Plot

**Best for:** Detailed cluster quality assessment

```python
from sklearn.metrics import silhouette_samples
import matplotlib.pyplot as plt

silhouette_values = silhouette_samples(X, labels)
# Plot silhouette for each sample
```

**When to use:**
- Understanding individual cluster quality
- Identifying poorly clustered points
- Refining cluster assignments

### Dendrogram

**Best for:** Hierarchical clustering

```python
from scipy.cluster.hierarchy import dendrogram, linkage
linkage_matrix = linkage(X, method='ward')
dendrogram(linkage_matrix)
```

**When to use:**
- Hierarchical clustering
- Understanding cluster hierarchy
- Determining cluster count

### PCA Cluster Plot

**Best for:** High-dimensional data visualization

```python
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels)
```

**When to use:**
- Data with more than 3 dimensions
- Visualizing high-dimensional clusters
- EDA before clustering

### Clustering Visualization Guide

```
┌─────────────────────────────────────────────────────────────────┐
│                CLUSTERING VISUALIZATIONS                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  2D VISUALIZATION          ───────────►  SCATTER PLOT            │
│  "What do the clusters look like?"                               │
│                                                                  │
│  FINDING OPTIMAL K         ───────────►  ELBOW PLOT              │
│  "How many clusters should I use?"                               │
│                                                                  │
│  CLUSTER QUALITY          ───────────►  SILHOUETTE PLOT          │
│  "How well-defined are each cluster?"                            │
│                                                                  │
│  HIERARCHICAL STRUCTURE    ───────────►  DENDROGRAM              │
│  "What's the cluster hierarchy?"                                 │
│                                                                  │
│  HIGH-DIMENSIONAL DATA     ───────────►  PCA PLOT                │
│  "Visualize clusters in 2D"                                       │
│                                                                  │
│  CLUSTER PROFILES         ───────────►  PARALLEL COORDINATES     │
│  "What are the characteristics of each cluster?"                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## When to Use What

### Decision Tree: Choosing Metrics by Task

```
                    ┌─────────────────┐
                    │   START HERE    │
                    └────────┬────────┘
                             │
                 ┌───────────┴───────────┐
                 │   What is your task?  │
                 └───┬───────────────┬───┘
                     │               │
              ┌──────▼──────┐   ┌───▼──────┐
              │SUPERVISED   │   │UNSUPERVISED│
              └──────┬──────┘   └───┬──────┘
                     │               │
        ┌────────────┴────┬──────────┘
        │                 │
    ┌───▼─────┐     ┌────▼───┐
    │Predict  │     │Grouping│
    │Category │     │Patterns│
    └───┬─────┘     └────┬───┘
        │                │
   CLASSIFICATION    CLUSTERING
   - Accuracy        - Silhouette
   - F1-Score        - Elbow Method
   - ROC-AUC         - Inertia
   - Confusion Matrix
        │
    ┌───▼─────┐
    │Predict  │
    │Number  │
    └───┬─────┘
        │
   REGRESSION
   - R²
   - RMSE
   - MAE
   - Residual Plot
```

### Quick Reference Card

| Task | Primary Metrics | Key Visualizations | Best For |
|------|-----------------|---------------------|----------|
| **Binary Classification** | Accuracy, F1, ROC-AUC | Confusion Matrix, ROC Curve | Spam detection, fraud, churn |
| **Multi-class Classification** | Accuracy, F1-macro | Confusion Matrix | Image classification, sentiment |
| **Imbalanced Classification** | F1, PR-AUC, Recall | PR Curve, Confusion Matrix | Rare disease, defect detection |
| **Regression** | R², RMSE, MAE | Actual vs Predicted, Residuals | Price prediction, demand forecasting |
| **Time Series** | MAPE, RMSE | Time series plot, ACF | Sales forecasting, stock prediction |
| **Clustering** | Silhouette, Inertia | Elbow plot, Scatter plot | Customer segmentation, anomaly detection |

---

## Industry-Specific Recommendations

### Healthcare / Medical

**Metrics:**
- Classification: **Recall** (sensitivity) - don't miss diagnoses
- Binary: **ROC-AUC** for model comparison
- Regression: **MAE** for interpretable error ranges

**Visualizations:**
- Confusion Matrix (show false negatives clearly)
- Calibration Curve (probability reliability)
- ROC Curve (threshold selection)

### Finance / Fraud Detection

**Metrics:**
- Classification: **Precision** (minimize false alarms)
- Cost-sensitive metrics: Custom loss function
- Regression: **MAPE** for percentage errors

**Visualizations:**
- Precision-Recall Curve (imbalanced data)
- Confusion Matrix with costs
- Feature Importance (regulatory requirements)

### Marketing / Customer Analytics

**Metrics:**
- Classification: **F1-Score** (balance precision/recall)
- Clustering: **Silhouette Score** (segment quality)
- Regression: **MAE** (interpretable for business)

**Visualizations:**
- Cluster profiles (parallel coordinates)
- Customer journey plots
- Feature importance for explainability

### E-Commerce / Retail

**Metrics:**
- Classification: **Accuracy** (balanced cart abandonment)
- Regression: **RMSE** (large errors in revenue prediction)
- Recommendation: NDCG, MAP

**Visualizations:**
- Actual vs Predicted (sales forecasting)
- Feature importance (product affinity)
- Cluster scatter plots (customer segments)

---

## Common Mistakes to Avoid

| Mistake | Why It's Wrong | What to Do Instead |
|---------|----------------|-------------------|
| Using Accuracy for Imbalanced Data | 99% accuracy = useless if minority class never predicted | Use F1-Score or PR-AUC |
| Ignoring Residual Plots | Can miss systematic errors | Always check residuals for regression |
| Over-relying on R² | Can be misleading with non-linear data | Use RMSE/MAE alongside R² |
| Not Visualizing Before Metrics | Numbers don't tell the whole story | Always plot your data first |
| Using Silhouette Alone | Doesn't work well on all cluster shapes | Combine with domain knowledge |
| ROC Curve for Imbalanced Data | Can be overly optimistic | Use Precision-Recall Curve instead |

---

## Summary Checklist

### Classification
- [ ] Is data balanced? → Use Accuracy if yes, F1 if no
- [ ] Are false positives costly? → Check Precision
- [ ] Are false negatives costly? → Check Recall
- [ ] Need threshold-independent metric? → Use ROC-AUC
- [ ] Visualize with Confusion Matrix

### Regression
- [ ] Need overall fit? → Use R²
- [ ] Large errors especially bad? → Use RMSE
- [ ] Need interpretability? → Use MAE
- [ ] Business context? → Use MAPE
- [ ] Always plot Actual vs Predicted
- [ ] Check Residual Plot for patterns

### Clustering
- [ ] Finding optimal k? → Use Elbow Method
- [ ] Cluster quality? → Use Silhouette Score
- [ ] Visualize clusters in 2D/3D
- [ ] Check cluster profiles
- [ ] Validate with domain knowledge

---

## Additional Resources

- [Model Evaluation Guide](model_evaluation.md) - Detailed evaluation techniques
- [Data Visualization Guide](../api/visualization.md) - Complete visualization reference
- [Model Training Guide](model_training.md) - Training best practices
