# mkyz/data/feature_engineering.py
"""Feature engineering utilities for MKYZ library."""

from typing import List, Optional, Union, Dict, Any
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import (
    SelectKBest, mutual_info_classif, mutual_info_regression,
    f_classif, f_regression, RFE
)


class FeatureEngineer:
    """Comprehensive feature engineering class.
    
    Provides methods for creating new features, selecting important features,
    and transforming existing features.
    
    Examples:
        >>> fe = FeatureEngineer()
        >>> df_enhanced = fe.create_all_features(df, 
        ...     numerical_cols=['age', 'income'],
        ...     datetime_col='signup_date',
        ...     polynomial_degree=2
        ... )
    """
    
    def __init__(self, random_state: int = 42):
        """Initialize FeatureEngineer.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self._created_features = []
    
    def create_polynomial_features(self,
                                   df: pd.DataFrame,
                                   columns: List[str],
                                   degree: int = 2,
                                   include_bias: bool = False,
                                   interaction_only: bool = False) -> pd.DataFrame:
        """Create polynomial features from numerical columns.
        
        Args:
            df: Input DataFrame
            columns: Columns to transform
            degree: Polynomial degree
            include_bias: Include bias column
            interaction_only: Only include interaction terms
            
        Returns:
            DataFrame with new polynomial features added
        """
        poly = PolynomialFeatures(
            degree=degree,
            include_bias=include_bias,
            interaction_only=interaction_only
        )
        
        X_poly = poly.fit_transform(df[columns])
        feature_names = poly.get_feature_names_out(columns)
        
        # Create DataFrame with new features
        poly_df = pd.DataFrame(X_poly, columns=feature_names, index=df.index)
        
        # Remove original columns (they're included in poly features)
        new_cols = [c for c in poly_df.columns if c not in columns]
        
        result = df.copy()
        for col in new_cols:
            result[col] = poly_df[col]
            self._created_features.append(col)
        
        return result
    
    def create_interaction_features(self,
                                   df: pd.DataFrame,
                                   columns: List[str],
                                   operations: List[str] = None) -> pd.DataFrame:
        """Create interaction features between columns.
        
        Args:
            df: Input DataFrame
            columns: Columns to create interactions from
            operations: Operations to apply ('multiply', 'divide', 'add', 'subtract')
            
        Returns:
            DataFrame with interaction features added
        """
        if operations is None:
            operations = ['multiply', 'add']
        
        result = df.copy()
        
        for i, col1 in enumerate(columns):
            for col2 in columns[i+1:]:
                if 'multiply' in operations:
                    name = f"{col1}_x_{col2}"
                    result[name] = df[col1] * df[col2]
                    self._created_features.append(name)
                
                if 'add' in operations:
                    name = f"{col1}_+_{col2}"
                    result[name] = df[col1] + df[col2]
                    self._created_features.append(name)
                
                if 'subtract' in operations:
                    name = f"{col1}_-_{col2}"
                    result[name] = df[col1] - df[col2]
                    self._created_features.append(name)
                
                if 'divide' in operations:
                    # Avoid division by zero
                    name = f"{col1}_/__{col2}"
                    result[name] = df[col1] / (df[col2].replace(0, np.nan))
                    self._created_features.append(name)
        
        return result
    
    def create_datetime_features(self,
                                df: pd.DataFrame,
                                datetime_column: str,
                                features: List[str] = None) -> pd.DataFrame:
        """Extract features from datetime column.
        
        Args:
            df: Input DataFrame
            datetime_column: Name of datetime column
            features: Features to extract. Options:
                'year', 'month', 'day', 'dayofweek', 'hour', 'minute',
                'quarter', 'is_weekend', 'is_month_start', 'is_month_end',
                'days_since_epoch'
                
        Returns:
            DataFrame with datetime features added
        """
        if features is None:
            features = ['year', 'month', 'day', 'dayofweek', 'hour', 'is_weekend']
        
        result = df.copy()
        
        # Convert to datetime if not already
        dt = pd.to_datetime(result[datetime_column])
        
        feature_extractors = {
            'year': lambda x: x.dt.year,
            'month': lambda x: x.dt.month,
            'day': lambda x: x.dt.day,
            'dayofweek': lambda x: x.dt.dayofweek,
            'hour': lambda x: x.dt.hour,
            'minute': lambda x: x.dt.minute,
            'second': lambda x: x.dt.second,
            'quarter': lambda x: x.dt.quarter,
            'week': lambda x: x.dt.isocalendar().week,
            'is_weekend': lambda x: x.dt.dayofweek.isin([5, 6]).astype(int),
            'is_month_start': lambda x: x.dt.is_month_start.astype(int),
            'is_month_end': lambda x: x.dt.is_month_end.astype(int),
            'is_year_start': lambda x: x.dt.is_year_start.astype(int),
            'is_year_end': lambda x: x.dt.is_year_end.astype(int),
            'days_since_epoch': lambda x: (x - pd.Timestamp('1970-01-01')).dt.days
        }
        
        for feature in features:
            if feature in feature_extractors:
                col_name = f"{datetime_column}_{feature}"
                result[col_name] = feature_extractors[feature](dt)
                self._created_features.append(col_name)
        
        return result
    
    def create_aggregation_features(self,
                                   df: pd.DataFrame,
                                   group_column: str,
                                   agg_columns: List[str],
                                   agg_functions: List[str] = None) -> pd.DataFrame:
        """Create aggregation features based on groups.
        
        Args:
            df: Input DataFrame
            group_column: Column to group by
            agg_columns: Columns to aggregate
            agg_functions: Aggregation functions ('mean', 'std', 'min', 'max', 'sum')
            
        Returns:
            DataFrame with aggregation features added
        """
        if agg_functions is None:
            agg_functions = ['mean', 'std']
        
        result = df.copy()
        
        for agg_col in agg_columns:
            for func in agg_functions:
                col_name = f"{agg_col}_{func}_by_{group_column}"
                agg_values = df.groupby(group_column)[agg_col].transform(func)
                result[col_name] = agg_values
                self._created_features.append(col_name)
        
        return result
    
    def create_lag_features(self,
                           df: pd.DataFrame,
                           columns: List[str],
                           lags: List[int] = None,
                           sort_column: str = None) -> pd.DataFrame:
        """Create lag features for time series data.
        
        Args:
            df: Input DataFrame
            columns: Columns to create lags for
            lags: List of lag values (e.g., [1, 2, 3] for 1, 2, 3 periods back)
            sort_column: Column to sort by before creating lags
            
        Returns:
            DataFrame with lag features added
        """
        if lags is None:
            lags = [1, 2, 3]
        
        result = df.copy()
        
        if sort_column:
            result = result.sort_values(sort_column)
        
        for col in columns:
            for lag in lags:
                col_name = f"{col}_lag_{lag}"
                result[col_name] = result[col].shift(lag)
                self._created_features.append(col_name)
        
        return result
    
    def create_rolling_features(self,
                               df: pd.DataFrame,
                               columns: List[str],
                               windows: List[int] = None,
                               functions: List[str] = None) -> pd.DataFrame:
        """Create rolling window features.
        
        Args:
            df: Input DataFrame
            columns: Columns to create rolling features for
            windows: Window sizes
            functions: Aggregation functions
            
        Returns:
            DataFrame with rolling features added
        """
        if windows is None:
            windows = [3, 7]
        if functions is None:
            functions = ['mean', 'std']
        
        result = df.copy()
        
        for col in columns:
            for window in windows:
                for func in functions:
                    col_name = f"{col}_rolling_{window}_{func}"
                    rolling = result[col].rolling(window=window, min_periods=1)
                    result[col_name] = getattr(rolling, func)()
                    self._created_features.append(col_name)
        
        return result
    
    def select_features(self,
                       X: pd.DataFrame,
                       y: pd.Series,
                       k: int = 10,
                       method: str = 'mutual_info',
                       task: str = 'classification') -> List[str]:
        """Select top k features.
        
        Args:
            X: Feature matrix
            y: Target variable
            k: Number of features to select
            method: Selection method ('mutual_info', 'f_score', 'rfe')
            task: Task type ('classification', 'regression')
            
        Returns:
            List of selected feature names
        """
        return select_features(X, y, k, method, task)
    
    @property
    def created_features(self) -> List[str]:
        """Get list of all created feature names."""
        return self._created_features.copy()
    
    def reset(self):
        """Reset the created features list."""
        self._created_features = []


def create_polynomial_features(df: pd.DataFrame,
                              columns: List[str],
                              degree: int = 2) -> pd.DataFrame:
    """Create polynomial features (convenience function).
    
    Args:
        df: Input DataFrame
        columns: Columns to transform
        degree: Polynomial degree
        
    Returns:
        DataFrame with polynomial features
    """
    fe = FeatureEngineer()
    return fe.create_polynomial_features(df, columns, degree)


def create_interaction_features(df: pd.DataFrame,
                               columns: List[str]) -> pd.DataFrame:
    """Create interaction features (convenience function).
    
    Args:
        df: Input DataFrame
        columns: Columns to create interactions from
        
    Returns:
        DataFrame with interaction features
    """
    fe = FeatureEngineer()
    return fe.create_interaction_features(df, columns)


def create_datetime_features(df: pd.DataFrame,
                            datetime_column: str,
                            features: List[str] = None) -> pd.DataFrame:
    """Extract datetime features (convenience function).
    
    Args:
        df: Input DataFrame
        datetime_column: Datetime column name
        features: Features to extract
        
    Returns:
        DataFrame with datetime features
    """
    fe = FeatureEngineer()
    return fe.create_datetime_features(df, datetime_column, features)


def select_features(X: pd.DataFrame,
                   y: pd.Series,
                   k: int = 10,
                   method: str = 'mutual_info',
                   task: str = 'classification') -> List[str]:
    """Select top k features.
    
    Args:
        X: Feature matrix
        y: Target variable
        k: Number of features to select
        method: Selection method ('mutual_info', 'f_score')
        task: Task type ('classification', 'regression')
        
    Returns:
        List of selected feature names
    """
    # Ensure k doesn't exceed number of features
    k = min(k, X.shape[1])
    
    # Select score function
    if method == 'mutual_info':
        if task == 'classification':
            score_func = mutual_info_classif
        else:
            score_func = mutual_info_regression
    elif method == 'f_score':
        if task == 'classification':
            score_func = f_classif
        else:
            score_func = f_regression
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Fit selector
    selector = SelectKBest(score_func=score_func, k=k)
    selector.fit(X, y)
    
    # Get selected feature names
    mask = selector.get_support()
    
    if hasattr(X, 'columns'):
        return list(X.columns[mask])
    else:
        return [f'feature_{i}' for i in range(X.shape[1]) if mask[i]]
