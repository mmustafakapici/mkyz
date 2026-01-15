# mkyz/core/exceptions.py
"""Custom exception classes for MKYZ library."""


class MKYZError(Exception):
    """Base exception class for MKYZ library.
    
    All custom exceptions in MKYZ inherit from this class,
    making it easy to catch all MKYZ-specific errors.
    """
    pass


class DataValidationError(MKYZError):
    """Raised when data validation fails.
    
    Examples:
        - Missing required columns
        - Invalid data types
        - Empty datasets
        - Incompatible data shapes
    """
    pass


class ModelNotTrainedError(MKYZError):
    """Raised when attempting to use a model that hasn't been trained.
    
    Examples:
        - Calling predict() before fit()
        - Accessing model parameters before training
    """
    pass


class UnsupportedTaskError(MKYZError):
    """Raised when an unsupported task type is specified.
    
    Supported tasks: classification, regression, clustering, dimensionality_reduction
    """
    pass


class UnsupportedModelError(MKYZError):
    """Raised when an unsupported model type is specified.
    
    Each task has a specific set of supported models defined in the
    models module.
    """
    pass


class OptimizationError(MKYZError):
    """Raised when hyperparameter optimization fails.
    
    Examples:
        - GridSearchCV fails to converge
        - Optuna study fails
        - Invalid parameter grid
    """
    pass


class FeatureEngineeringError(MKYZError):
    """Raised when feature engineering operations fail.
    
    Examples:
        - Cannot create polynomial features
        - DateTime parsing fails
        - Invalid column specifications
    """
    pass


class PersistenceError(MKYZError):
    """Raised when model saving or loading fails.
    
    Examples:
        - File not found
        - Corrupted model file
        - Incompatible model version
    """
    pass
