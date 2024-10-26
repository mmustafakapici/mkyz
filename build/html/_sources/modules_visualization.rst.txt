Visualization
=============

This module provides tools for visualizing data and model results.

### Functions

.. autofunction:: mkyz.visualization.visualize

Generates different types of plots to help visualize your data or model results.

**Usage:**

.. code-block:: python

   from mkyz import visualization as viz
   viz.visualize(data=data, plot_type='corr', title='Correlation Matrix')

Arguments:
- `data`: The dataset to visualize.
- `plot_type`: The type of plot (e.g., 'scatter', 'corr').
- `**plot_params`: Additional parameters for the plot such as title, labels, etc.

Returns:
- Displays the generated plot.
