# mkyz/data/__init__.py
"""Data processing, feature engineering, and EDA module."""

from .preprocessing import (
    fill_missing_values,
    detect_outliers,
    handle_outliers,
    transform_categorical,
    scale_features,
    prepare_data
)
from .feature_engineering import (
    FeatureEngineer,
    create_polynomial_features,
    create_interaction_features,
    create_datetime_features,
    select_features
)
from .validation import (
    validate_dataset,
    check_target_balance,
    detect_data_leakage
)
from .loading import (
    load_data,
    DataLoader
)
from .eda import (
    DataProfile,
    data_info,
    describe_column,
    get_summary_stats,
    quick_eda
)

__all__ = [
    # Preprocessing
    'fill_missing_values',
    'detect_outliers',
    'handle_outliers',
    'transform_categorical',
    'scale_features',
    'prepare_data',
    # Feature Engineering
    'FeatureEngineer',
    'create_polynomial_features',
    'create_interaction_features',
    'create_datetime_features',
    'select_features',
    # Validation
    'validate_dataset',
    'check_target_balance',
    'detect_data_leakage',
    # Loading
    'load_data',
    'DataLoader',
    # EDA
    'DataProfile',
    'data_info',
    'describe_column',
    'get_summary_stats',
    'quick_eda'
]
