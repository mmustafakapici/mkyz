Training
========

This module provides functionality to train machine learning models, make predictions, and evaluate model performance.

### Functions

#### Training

.. autofunction:: mkyz.training.train

#### Prediction

.. autofunction:: mkyz.training.predict

#### Evaluation

.. autofunction:: mkyz.training.evaluate

**Usage:**

.. code-block:: python

   from mkyz import training as tr
   model = tr.train(data=data, task='classification', model='rf')

   predictions = tr.predict(data=data, fitted_model=model, task='classification')

   results = tr.evaluate(data=data, predictions=predictions, task='classification')
