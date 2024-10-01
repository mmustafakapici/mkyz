Quickstart
==========

This quickstart guide will help you get up and running with **mkyz** quickly.

### 1. Prepare Your Data

You can load and prepare your data using the `prepare_data` function:

.. code-block:: python

   from mkyz import data_processing as dp
   data = dp.prepare_data('winequality-red.csv')

### 2. Visualize the Data

Next, visualize your data with the built-in `visualize` function:

.. code-block:: python

   from mkyz import visualization as viz
   viz.visualize(data=data, plot_type='corr')

### 3. Train a Model

Train a machine learning model using the `train` function:

.. code-block:: python

   from mkyz import training as tr
   model = tr.train(data=data, task='classification', model='rf')

### 4. Make Predictions

Once trained, make predictions with your model:

.. code-block:: python

   predictions = tr.predict(data=data, fitted_model=model, task='classification')

### 5. Evaluate the Model

Finally, evaluate your model's performance:

.. code-block:: python

   results = tr.evaluate(data=data, predictions=predictions, task='classification')
   print(results)
