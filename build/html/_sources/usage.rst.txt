Usage
=====

Here are examples of how to use the **mkyz** library.

### Data Preparation

Load and prepare your dataset:

.. code-block:: python

   from mkyz import data_processing as dp
   data = dp.prepare_data('winequality-red.csv', target_column='quality')

### Visualization

Visualize your dataset, such as generating a correlation matrix:

.. code-block:: python

   from mkyz import visualization as viz
   viz.visualize(data=data, plot_type='corr')

### Model Training

Train a classification model:

.. code-block:: python

   from mkyz import training as tr
   model = tr.train(data=data, task='classification', model='rf')

### Prediction

Make predictions using your trained model:

.. code-block:: python

   predictions = tr.predict(data=data, fitted_model=model, task='classification')

### Evaluation

Evaluate your model's performance:

.. code-block:: python

   results = tr.evaluate(data=data, predictions=predictions, task='classification')
   print(results)
