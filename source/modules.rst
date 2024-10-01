Modules
=======

The **mkyz** library contains several key modules that work together to process data, train models, and generate visualizations.

.. toctree::
   :maxdepth: 2
   :caption: Modules:

   modules/data_processing
   modules/training
   modules/visualization

Each module has specific functionalities:

### Data Processing

- **`data_processing.prepare_data`**: Prepares your dataset for training by handling missing values and encoding categorical variables.

### Training

- **`training.train`**: Trains a machine learning model based on the specified task and model type.

- **`training.predict`**: Makes predictions using a trained model.

- **`training.evaluate`**: Evaluates the performance of a trained model using various metrics.

### Visualization

- **`visualization.visualize`**: Visualizes your data using different plot types such as scatter plots, bar charts, and correlation matrices.
