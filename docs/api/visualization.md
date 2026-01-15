# API Reference - Visualization Module

The Visualization module provides a unified interface for creating various types of plots for data analysis.

```python
from mkyz.visualization import visualize
```

---

## visualize

```python
visualize(
    data,
    target_column=None,
    numerical_columns=None,
    categorical_columns=None,
    graphics='kde',
    cols=3,
    figsize=(15, 15),
    palette=DEFAULT_PALETTE,
    max_plots_per_fig=20
)
```

Visualizes data using the specified type of graphics.

### Arguments

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `pd.DataFrame` or `tuple` | Required | The data to visualize. Can be a pandas DataFrame or a pre-processed tuple. |
| `target_column` | `str` | `None` | The name of the target column (required if `data` is a DataFrame). |
| `numerical_columns` | `list` | `None` | List of numerical column names. If None, inferred from DataFrame. |
| `categorical_columns` | `list` | `None` | List of categorical column names. If None, inferred from DataFrame. |
| `graphics` | `str` | `'kde'` | Type of graph to create. See [Supported Graphics](#supported-graphics). |
| `cols` | `int` | `3` | Number of columns in each subplot grid. |
| `figsize` | `tuple` | `(15, 15)` | Size of each figure. |
| `palette` | `list` | `DEFAULT_PALETTE` | Color palette for the plots. |
| `max_plots_per_fig` | `int` | `20` | Maximum number of subplots per figure. |

### Supported Graphics

**Continuous Visualization:**
- `histogram`: Histograms
- `box`: Box plots
- `scatter`: Scatter plots
- `line`: Line plots
- `kde`: Kernel Density Estimation plots
- `pair`: Pair plots
- `violin`: Violin plots
- `ridge`: Ridge plots
- `area`: Area plots
- `step`: Step plots
- `density`: Density plots
- `bubble`: Bubble plots
- `3dscatter`: 3D Scatter plots (Plotly)
- `parallel`: Parallel coordinates
- `hexbin`: Hexbin plots
- `boxen`: Boxen plots
- `3dsurface`: 3D Surface plots (Plotly)
- `pca`: PCA visualization
- `tsne`: t-SNE visualization
- `regression`: Regression plots
- `joint`: Joint plots

**Categorical Visualization:**
- `bar`: Bar plots
- `pie`: Pie charts
- `swarm`: Swarm plots
- `strip`: Strip plots
- `trellis`: Trellis plots
- `lollipop`: Lollipop charts
- `mosaic`: Mosaic plots
- `donut`: Donut charts
- `sunburst`: Sunburst chart (Plotly)
- `radar`: Radar chart
- `waterfall`: Waterfall chart
- `funnel`: Funnel chart (Plotly)
- `stackedbar`: Stacked bar plots
- `dendrogram`: Dendrogram
- `facetgrid`: Facet grid
- `corr`: Correlation matrix

### Examples

#### Using a DataFrame directly

```python
import pandas as pd
from mkyz.visualization import visualize

# Load data
df = pd.read_csv('data.csv')

# Visualize using correlation matrix
visualize(
    data=df,
    target_column='target',
    graphics='corr'
)

# Visualize distributions
visualize(
    data=df,
    target_column='target',
    graphics='histogram'
)
```

#### Using pre-processed data tuple

```python
from mkyz.data_processing import prepare_data
from mkyz.visualization import visualize

# Prepare data returns a tuple
data_tuple = prepare_data('data.csv', target_column='target')

# Visualize
visualize(data_tuple, graphics='scatter')
```
