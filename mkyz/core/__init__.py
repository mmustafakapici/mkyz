# mkyz/core/__init__.py
"""Core module containing base classes, exceptions, and configuration."""

from .exceptions import (
    MKYZError,
    DataValidationError,
    ModelNotTrainedError,
    UnsupportedTaskError,
    UnsupportedModelError,
    OptimizationError
)
from .config import Config, DEFAULT_CONFIG
from .base import BaseEstimator, DataMixin, ModelMixin

__all__ = [
    'MKYZError',
    'DataValidationError', 
    'ModelNotTrainedError',
    'UnsupportedTaskError',
    'UnsupportedModelError',
    'OptimizationError',
    'Config',
    'DEFAULT_CONFIG',
    'BaseEstimator',
    'DataMixin',
    'ModelMixin'
]
