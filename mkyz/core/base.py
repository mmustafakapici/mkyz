# mkyz/core/base.py
"""Base classes and mixins for MKYZ library."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union, List
import pandas as pd
import numpy as np


class BaseEstimator(ABC):
    """Abstract base class for all MKYZ estimators.
    
    Provides a consistent interface for all models in the library.
    Subclasses must implement fit(), predict(), and get_params() methods.
    """
    
    def __init__(self, **params):
        """Initialize the estimator with given parameters.
        
        Args:
            **params: Model-specific parameters
        """
        self._params = params
        self._is_fitted = False
        self._model = None
    
    @abstractmethod
    def fit(self, X: Union[pd.DataFrame, np.ndarray], 
            y: Optional[Union[pd.Series, np.ndarray]] = None) -> 'BaseEstimator':
        """Fit the model to the training data.
        
        Args:
            X: Training features
            y: Target variable (optional for unsupervised learning)
            
        Returns:
            self: The fitted estimator
        """
        pass
    
    @abstractmethod
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Make predictions on new data.
        
        Args:
            X: Features to predict on
            
        Returns:
            Predictions array
        """
        pass
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters.
        
        Returns:
            Dictionary of model parameters
        """
        return self._params.copy()
    
    def set_params(self, **params) -> 'BaseEstimator':
        """Set model parameters.
        
        Args:
            **params: Parameters to set
            
        Returns:
            self: The estimator with updated parameters
        """
        self._params.update(params)
        return self
    
    @property
    def is_fitted(self) -> bool:
        """Check if the model has been fitted.
        
        Returns:
            True if fitted, False otherwise
        """
        return self._is_fitted
    
    def _check_is_fitted(self) -> None:
        """Raise error if model is not fitted.
        
        Raises:
            ModelNotTrainedError: If model hasn't been fitted
        """
        from .exceptions import ModelNotTrainedError
        if not self._is_fitted:
            raise ModelNotTrainedError(
                f"{self.__class__.__name__} has not been fitted. "
                "Call fit() before using predict()."
            )


class DataMixin:
    """Mixin class providing data validation and preprocessing utilities."""
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame, 
                          required_columns: Optional[List[str]] = None) -> None:
        """Validate that input is a proper DataFrame.
        
        Args:
            df: DataFrame to validate
            required_columns: List of required column names
            
        Raises:
            DataValidationError: If validation fails
        """
        from .exceptions import DataValidationError
        
        if not isinstance(df, pd.DataFrame):
            raise DataValidationError(
                f"Expected pandas DataFrame, got {type(df).__name__}"
            )
        
        if df.empty:
            raise DataValidationError("DataFrame is empty")
        
        if required_columns:
            missing_cols = set(required_columns) - set(df.columns)
            if missing_cols:
                raise DataValidationError(
                    f"Missing required columns: {missing_cols}"
                )
    
    @staticmethod
    def validate_array(arr: np.ndarray, 
                      expected_dim: Optional[int] = None) -> None:
        """Validate that input is a proper numpy array.
        
        Args:
            arr: Array to validate
            expected_dim: Expected number of dimensions
            
        Raises:
            DataValidationError: If validation fails
        """
        from .exceptions import DataValidationError
        
        if not isinstance(arr, np.ndarray):
            raise DataValidationError(
                f"Expected numpy array, got {type(arr).__name__}"
            )
        
        if arr.size == 0:
            raise DataValidationError("Array is empty")
        
        if expected_dim and arr.ndim != expected_dim:
            raise DataValidationError(
                f"Expected {expected_dim}D array, got {arr.ndim}D"
            )
    
    @staticmethod
    def get_column_types(df: pd.DataFrame) -> Dict[str, List[str]]:
        """Get numerical and categorical column names.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with 'numerical' and 'categorical' column lists
        """
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(
            include=['object', 'category']
        ).columns.tolist()
        
        return {
            'numerical': numerical_cols,
            'categorical': categorical_cols
        }


class ModelMixin:
    """Mixin class providing model evaluation and scoring utilities."""
    
    @staticmethod
    def get_scorer(task: str, metric: Optional[str] = None):
        """Get appropriate scorer for the task.
        
        Args:
            task: Task type ('classification', 'regression', 'clustering')
            metric: Specific metric to use (optional)
            
        Returns:
            Scorer function or string
        """
        default_metrics = {
            'classification': 'accuracy',
            'regression': 'neg_mean_squared_error',
            'clustering': 'silhouette'
        }
        
        return metric or default_metrics.get(task, 'accuracy')
    
    @staticmethod
    def format_scores(scores: Dict[str, float], precision: int = 4) -> str:
        """Format scores dictionary as a readable string.
        
        Args:
            scores: Dictionary of metric names and values
            precision: Number of decimal places
            
        Returns:
            Formatted string of scores
        """
        lines = []
        for metric, value in scores.items():
            formatted_value = f"{value:.{precision}f}" if isinstance(value, float) else str(value)
            lines.append(f"  {metric}: {formatted_value}")
        return "\n".join(lines)
