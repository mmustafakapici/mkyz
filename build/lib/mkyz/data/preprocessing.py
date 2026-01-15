# mkyz/data/preprocessing.py
"""Data preprocessing module - wraps and extends existing functionality."""

# Re-export functions from the original data_processing module for backward compatibility
from mkyz.data_processing import (
    fill_missing_values,
    detect_outliers,
    handle_outliers,
    transform_categorical,
    prepare_data
)

# Additional preprocessing utilities
from typing import List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    LabelEncoder, OrdinalEncoder
)


def scale_features(df: pd.DataFrame,
                  columns: List[str],
                  method: str = 'standard',
                  fit_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Scale numerical features.
    
    Args:
        df: Input DataFrame
        columns: Columns to scale
        method: Scaling method ('standard', 'minmax', 'robust')
        fit_data: Data to fit scaler on (if different from df)
        
    Returns:
        DataFrame with scaled features
    """
    scalers = {
        'standard': StandardScaler(),
        'minmax': MinMaxScaler(),
        'robust': RobustScaler()
    }
    
    if method not in scalers:
        raise ValueError(f"Unknown scaling method: {method}")
    
    scaler = scalers[method]
    result = df.copy()
    
    fit_df = fit_data if fit_data is not None else df
    scaler.fit(fit_df[columns])
    
    result[columns] = scaler.transform(df[columns])
    
    return result


def encode_labels(df: pd.DataFrame,
                 columns: List[str],
                 method: str = 'label') -> pd.DataFrame:
    """Encode categorical labels.
    
    Args:
        df: Input DataFrame
        columns: Columns to encode
        method: Encoding method ('label', 'ordinal')
        
    Returns:
        DataFrame with encoded labels
    """
    result = df.copy()
    
    for col in columns:
        if method == 'label':
            le = LabelEncoder()
            result[col] = le.fit_transform(result[col].astype(str))
        elif method == 'ordinal':
            oe = OrdinalEncoder()
            result[[col]] = oe.fit_transform(result[[col]])
    
    return result


def remove_duplicates(df: pd.DataFrame,
                     subset: Optional[List[str]] = None,
                     keep: str = 'first') -> pd.DataFrame:
    """Remove duplicate rows from DataFrame.
    
    Args:
        df: Input DataFrame
        subset: Columns to consider for duplicates (all if None)
        keep: Which duplicates to keep ('first', 'last', False)
        
    Returns:
        DataFrame with duplicates removed
    """
    return df.drop_duplicates(subset=subset, keep=keep).reset_index(drop=True)


def clip_values(df: pd.DataFrame,
               columns: List[str],
               lower: Optional[float] = None,
               upper: Optional[float] = None,
               percentile_lower: Optional[float] = None,
               percentile_upper: Optional[float] = None) -> pd.DataFrame:
    """Clip values in columns to specified range.
    
    Args:
        df: Input DataFrame
        columns: Columns to clip
        lower: Lower bound value
        upper: Upper bound value
        percentile_lower: Lower bound percentile (0-100)
        percentile_upper: Upper bound percentile (0-100)
        
    Returns:
        DataFrame with clipped values
    """
    result = df.copy()
    
    for col in columns:
        lower_bound = lower
        upper_bound = upper
        
        if percentile_lower is not None:
            lower_bound = np.percentile(df[col].dropna(), percentile_lower)
        if percentile_upper is not None:
            upper_bound = np.percentile(df[col].dropna(), percentile_upper)
        
        result[col] = result[col].clip(lower=lower_bound, upper=upper_bound)
    
    return result


__all__ = [
    # From original data_processing
    'fill_missing_values',
    'detect_outliers', 
    'handle_outliers',
    'transform_categorical',
    'prepare_data',
    # New functions
    'scale_features',
    'encode_labels',
    'remove_duplicates',
    'clip_values'
]
