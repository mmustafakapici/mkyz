Data Processing
===============

This module provides tools for preparing data for machine learning tasks.

### Functions

.. autofunction:: mkyz.data_processing.prepare_data

Prepares the dataset for training by handling missing values, encoding categorical variables, and splitting the data into features and target labels.

**Usage:**

.. code-block:: python

   from mkyz import data_processing as dp
   data = dp.prepare_data('winequality-red.csv', target_column='quality', task='classification')

Arguments:
- `data`: A path to the dataset (CSV file) or a pandas DataFrame.
- `target_column`: The target variable (classification or regression task).
- `task`: The type of task ('classification' or 'regression').

Returns:
- A tuple of `(X, y)` where `X` is the feature matrix and `y` is the target variable.
