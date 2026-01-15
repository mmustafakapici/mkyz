# mkyz/data/validation.py
"""Data validation utilities for MKYZ library."""

from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd
from collections import Counter


def validate_dataset(df: pd.DataFrame,
                    target_column: Optional[str] = None,
                    required_columns: Optional[List[str]] = None,
                    check_missing: bool = True,
                    check_duplicates: bool = True,
                    check_infinity: bool = True) -> Dict[str, Any]:
    """Comprehensive dataset validation.
    
    Args:
        df: DataFrame to validate
        target_column: Name of target column (if applicable)
        required_columns: List of required columns
        check_missing: Check for missing values
        check_duplicates: Check for duplicate rows
        check_infinity: Check for infinite values
        
    Returns:
        Dictionary with validation results
    
    Examples:
        >>> results = validate_dataset(df, target_column='price')
        >>> if not results['is_valid']:
        ...     print(results['issues'])
    """
    from mkyz.core.exceptions import DataValidationError
    
    results = {
        'is_valid': True,
        'issues': [],
        'warnings': [],
        'statistics': {}
    }
    
    # Check if DataFrame is empty
    if df.empty:
        results['is_valid'] = False
        results['issues'].append("DataFrame is empty")
        return results
    
    # Basic statistics
    results['statistics'] = {
        'n_rows': len(df),
        'n_columns': len(df.columns),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2
    }
    
    # Check required columns
    if required_columns:
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            results['is_valid'] = False
            results['issues'].append(f"Missing required columns: {missing_cols}")
    
    # Check target column
    if target_column:
        if target_column not in df.columns:
            results['is_valid'] = False
            results['issues'].append(f"Target column '{target_column}' not found")
        else:
            target_missing = df[target_column].isna().sum()
            if target_missing > 0:
                results['is_valid'] = False
                results['issues'].append(
                    f"Target column has {target_missing} missing values"
                )
    
    # Check for missing values
    if check_missing:
        missing_counts = df.isna().sum()
        cols_with_missing = missing_counts[missing_counts > 0]
        if len(cols_with_missing) > 0:
            results['warnings'].append(
                f"Columns with missing values: {dict(cols_with_missing)}"
            )
            results['statistics']['missing_values'] = dict(cols_with_missing)
            results['statistics']['total_missing'] = cols_with_missing.sum()
    
    # Check for duplicate rows
    if check_duplicates:
        n_duplicates = df.duplicated().sum()
        if n_duplicates > 0:
            results['warnings'].append(f"Found {n_duplicates} duplicate rows")
            results['statistics']['n_duplicates'] = n_duplicates
    
    # Check for infinite values in numeric columns
    if check_infinity:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            inf_count = np.isinf(df[col]).sum()
            if inf_count > 0:
                results['warnings'].append(
                    f"Column '{col}' has {inf_count} infinite values"
                )
    
    return results


def check_target_balance(y: Union[pd.Series, np.ndarray],
                        imbalance_threshold: float = 0.1) -> Dict[str, Any]:
    """Check class balance for classification targets.
    
    Args:
        y: Target variable
        imbalance_threshold: Threshold for considering classes imbalanced
        
    Returns:
        Dictionary with balance information
    
    Examples:
        >>> balance_info = check_target_balance(y_train)
        >>> if balance_info['is_imbalanced']:
        ...     print("Consider using class weights or resampling")
    """
    if isinstance(y, pd.Series):
        y = y.values
    
    counter = Counter(y)
    total = len(y)
    
    distribution = {
        str(k): {'count': v, 'percentage': v / total * 100}
        for k, v in counter.items()
    }
    
    min_ratio = min(counter.values()) / max(counter.values())
    
    is_imbalanced = min_ratio < imbalance_threshold
    
    return {
        'n_classes': len(counter),
        'class_distribution': distribution,
        'min_class_ratio': min_ratio,
        'is_imbalanced': is_imbalanced,
        'recommendation': 'Consider class weights or SMOTE' if is_imbalanced else None
    }


def detect_data_leakage(X_train: pd.DataFrame,
                       X_test: pd.DataFrame,
                       threshold: float = 0.9) -> Dict[str, Any]:
    """Detect potential data leakage between train and test sets.
    
    Args:
        X_train: Training features
        X_test: Test features
        threshold: Overlap threshold for flagging issues
        
    Returns:
        Dictionary with leakage detection results
    
    Examples:
        >>> leakage = detect_data_leakage(X_train, X_test)
        >>> if leakage['has_leakage']:
        ...     print(f"Warning: {leakage['overlap_percentage']:.1f}% overlap")
    """
    # Check for exact row matches
    train_tuples = set(map(tuple, X_train.values))
    test_tuples = set(map(tuple, X_test.values))
    
    overlap = train_tuples.intersection(test_tuples)
    overlap_count = len(overlap)
    overlap_pct = overlap_count / len(test_tuples) * 100 if test_tuples else 0
    
    has_leakage = overlap_pct > (threshold * 100)
    
    # Check for columns with suspiciously high correlation with future info
    suspicious_columns = []
    
    return {
        'has_leakage': has_leakage,
        'overlap_count': overlap_count,
        'overlap_percentage': overlap_pct,
        'suspicious_columns': suspicious_columns,
        'recommendation': 'Review data splitting methodology' if has_leakage else None
    }


def check_feature_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Identify feature types in DataFrame.
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        Dictionary mapping type names to column lists
    """
    result = {
        'numeric': [],
        'categorical': [],
        'datetime': [],
        'boolean': [],
        'text': [],
        'other': []
    }
    
    for col in df.columns:
        dtype = df[col].dtype
        
        if pd.api.types.is_numeric_dtype(dtype):
            result['numeric'].append(col)
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            result['datetime'].append(col)
        elif pd.api.types.is_bool_dtype(dtype):
            result['boolean'].append(col)
        elif pd.api.types.is_categorical_dtype(dtype) or dtype == object:
            # Check if it's text (long strings) or categorical
            sample = df[col].dropna().head(100)
            if len(sample) > 0:
                avg_len = sample.astype(str).str.len().mean()
                if avg_len > 50:
                    result['text'].append(col)
                else:
                    result['categorical'].append(col)
            else:
                result['categorical'].append(col)
        else:
            result['other'].append(col)
    
    return result


def suggest_preprocessing(df: pd.DataFrame,
                         target_column: Optional[str] = None) -> List[str]:
    """Suggest preprocessing steps based on data characteristics.
    
    Args:
        df: DataFrame to analyze
        target_column: Name of target column
        
    Returns:
        List of preprocessing suggestions
    """
    suggestions = []
    
    # Check for missing values
    missing_pct = df.isna().sum() / len(df) * 100
    high_missing = missing_pct[missing_pct > 20].index.tolist()
    
    if high_missing:
        suggestions.append(
            f"Consider dropping or carefully imputing columns with >20% missing: {high_missing}"
        )
    
    # Check for categorical variables
    feature_types = check_feature_types(df)
    
    if feature_types['categorical']:
        suggestions.append(
            f"Encode categorical columns: {feature_types['categorical'][:5]}..."
        )
    
    if feature_types['datetime']:
        suggestions.append(
            f"Extract features from datetime columns: {feature_types['datetime']}"
        )
    
    if feature_types['text']:
        suggestions.append(
            f"Apply text vectorization to: {feature_types['text']}"
        )
    
    # Check for scaling needs
    if feature_types['numeric']:
        numeric_df = df[feature_types['numeric']]
        range_ratio = numeric_df.max() / (numeric_df.min().replace(0, np.nan))
        high_range = range_ratio[range_ratio > 100].index.tolist()
        
        if high_range:
            suggestions.append(
                f"Consider scaling numeric features with high range: {high_range[:5]}"
            )
    
    return suggestions
