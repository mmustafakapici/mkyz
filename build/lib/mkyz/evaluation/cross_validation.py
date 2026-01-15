# mkyz/evaluation/cross_validation.py
"""Cross-validation strategies and utilities."""

from enum import Enum
from typing import Any, Dict, Optional, Union, List, Callable
import numpy as np
import pandas as pd
from sklearn.model_selection import (
    cross_val_score, cross_validate as sklearn_cross_validate,
    KFold, StratifiedKFold, TimeSeriesSplit, GroupKFold,
    RepeatedKFold, RepeatedStratifiedKFold, LeaveOneOut, LeaveOneGroupOut,
    ShuffleSplit, StratifiedShuffleSplit
)


class CVStrategy(Enum):
    """Cross-validation strategy enumeration."""
    KFOLD = 'kfold'
    STRATIFIED = 'stratified'
    TIME_SERIES = 'time_series'
    GROUP = 'group'
    REPEATED = 'repeated'
    REPEATED_STRATIFIED = 'repeated_stratified'
    LEAVE_ONE_OUT = 'loo'
    LEAVE_ONE_GROUP_OUT = 'logo'
    SHUFFLE = 'shuffle'
    STRATIFIED_SHUFFLE = 'stratified_shuffle'


def get_cv_strategy(strategy: Union[str, CVStrategy],
                   n_splits: int = 5,
                   n_repeats: int = 3,
                   test_size: float = 0.2,
                   random_state: int = 42,
                   shuffle: bool = True,
                   groups: Optional[np.ndarray] = None) -> Any:
    """Get cross-validation splitter object.
    
    Args:
        strategy: CV strategy name or CVStrategy enum
        n_splits: Number of folds/splits
        n_repeats: Number of repeats (for repeated strategies)
        test_size: Test set size (for shuffle strategies)
        random_state: Random seed for reproducibility
        shuffle: Whether to shuffle data before splitting
        groups: Group labels for group-based CV
        
    Returns:
        Sklearn CV splitter object
    
    Examples:
        >>> from mkyz.evaluation import get_cv_strategy, CVStrategy
        >>> cv = get_cv_strategy(CVStrategy.STRATIFIED, n_splits=5)
        >>> cv = get_cv_strategy('time_series', n_splits=10)
    """
    if isinstance(strategy, CVStrategy):
        strategy = strategy.value
    
    strategy = strategy.lower()
    
    strategies = {
        'kfold': KFold(
            n_splits=n_splits, shuffle=shuffle, random_state=random_state
        ),
        'stratified': StratifiedKFold(
            n_splits=n_splits, shuffle=shuffle, random_state=random_state
        ),
        'time_series': TimeSeriesSplit(n_splits=n_splits),
        'group': GroupKFold(n_splits=n_splits),
        'repeated': RepeatedKFold(
            n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
        ),
        'repeated_stratified': RepeatedStratifiedKFold(
            n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
        ),
        'loo': LeaveOneOut(),
        'logo': LeaveOneGroupOut(),
        'shuffle': ShuffleSplit(
            n_splits=n_splits, test_size=test_size, random_state=random_state
        ),
        'stratified_shuffle': StratifiedShuffleSplit(
            n_splits=n_splits, test_size=test_size, random_state=random_state
        )
    }
    
    if strategy not in strategies:
        valid = list(strategies.keys())
        raise ValueError(f"Unknown CV strategy: {strategy}. Valid: {valid}")
    
    return strategies[strategy]


def cross_validate(model: Any,
                  X: Union[pd.DataFrame, np.ndarray],
                  y: Union[pd.Series, np.ndarray],
                  cv: Union[str, CVStrategy, Any] = 'stratified',
                  n_splits: int = 5,
                  scoring: Optional[Union[str, List[str]]] = None,
                  return_train_score: bool = False,
                  return_estimator: bool = False,
                  n_jobs: int = -1,
                  groups: Optional[np.ndarray] = None,
                  verbose: int = 0) -> Dict[str, Any]:
    """Perform cross-validation with multiple metrics.
    
    Args:
        model: Sklearn-compatible estimator
        X: Feature matrix
        y: Target variable
        cv: Cross-validation strategy or splitter
        n_splits: Number of folds (if cv is a string/enum)
        scoring: Scoring metric(s) to use
        return_train_score: Whether to return training scores
        return_estimator: Whether to return fitted estimators
        n_jobs: Number of parallel jobs (-1 for all CPUs)
        groups: Group labels for group-based CV
        verbose: Verbosity level
        
    Returns:
        Dictionary containing:
            - test_score: Test scores for each fold
            - train_score: Training scores (if return_train_score=True)
            - fit_time: Time to fit each fold
            - score_time: Time to score each fold
            - estimator: Fitted estimators (if return_estimator=True)
            - mean_test_score: Mean test score
            - std_test_score: Standard deviation of test scores
    
    Examples:
        >>> from mkyz.evaluation import cross_validate
        >>> results = cross_validate(rf_model, X_train, y_train, cv='stratified')
        >>> print(f"Mean accuracy: {results['mean_test_score']:.4f}")
    """
    # Get CV splitter if string/enum provided
    if isinstance(cv, (str, CVStrategy)):
        cv = get_cv_strategy(cv, n_splits=n_splits, groups=groups)
    
    # Default scoring based on common tasks
    if scoring is None:
        scoring = 'accuracy'
    
    # Perform cross-validation
    cv_results = sklearn_cross_validate(
        model, X, y,
        cv=cv,
        scoring=scoring,
        return_train_score=return_train_score,
        return_estimator=return_estimator,
        n_jobs=n_jobs,
        groups=groups,
        verbose=verbose
    )
    
    # Add summary statistics
    test_key = 'test_score' if isinstance(scoring, str) else f'test_{scoring[0]}'
    if test_key in cv_results:
        cv_results['mean_test_score'] = np.mean(cv_results[test_key])
        cv_results['std_test_score'] = np.std(cv_results[test_key])
    
    if return_train_score:
        train_key = 'train_score' if isinstance(scoring, str) else f'train_{scoring[0]}'
        if train_key in cv_results:
            cv_results['mean_train_score'] = np.mean(cv_results[train_key])
            cv_results['std_train_score'] = np.std(cv_results[train_key])
    
    return cv_results


def cross_val_predict_proba(model: Any,
                           X: Union[pd.DataFrame, np.ndarray],
                           y: Union[pd.Series, np.ndarray],
                           cv: Union[str, CVStrategy, Any] = 'stratified',
                           n_splits: int = 5,
                           method: str = 'predict_proba') -> np.ndarray:
    """Get cross-validated probability predictions.
    
    Args:
        model: Sklearn-compatible estimator with predict_proba method
        X: Feature matrix
        y: Target variable
        cv: Cross-validation strategy or splitter
        n_splits: Number of folds
        method: Prediction method ('predict_proba', 'predict_log_proba', 'decision_function')
        
    Returns:
        Array of probability predictions
    """
    from sklearn.model_selection import cross_val_predict
    
    if isinstance(cv, (str, CVStrategy)):
        cv = get_cv_strategy(cv, n_splits=n_splits)
    
    return cross_val_predict(model, X, y, cv=cv, method=method)


def nested_cross_validation(model: Any,
                           X: Union[pd.DataFrame, np.ndarray],
                           y: Union[pd.Series, np.ndarray],
                           param_grid: Dict[str, List],
                           outer_cv: int = 5,
                           inner_cv: int = 3,
                           scoring: str = 'accuracy',
                           n_jobs: int = -1) -> Dict[str, Any]:
    """Perform nested cross-validation with hyperparameter tuning.
    
    This is the gold standard for model evaluation as it provides
    an unbiased estimate of generalization performance.
    
    Args:
        model: Sklearn-compatible estimator
        X: Feature matrix
        y: Target variable
        param_grid: Hyperparameter grid for inner CV
        outer_cv: Number of outer folds
        inner_cv: Number of inner folds
        scoring: Scoring metric
        n_jobs: Number of parallel jobs
        
    Returns:
        Dictionary with outer scores and best parameters per fold
    """
    from sklearn.model_selection import GridSearchCV, cross_val_score
    
    # Create inner CV with GridSearchCV
    grid_search = GridSearchCV(
        model, param_grid, cv=inner_cv, scoring=scoring, n_jobs=n_jobs
    )
    
    # Perform outer cross-validation
    outer_scores = cross_val_score(
        grid_search, X, y, cv=outer_cv, scoring=scoring, n_jobs=n_jobs
    )
    
    return {
        'outer_scores': outer_scores,
        'mean_score': np.mean(outer_scores),
        'std_score': np.std(outer_scores),
        'outer_cv': outer_cv,
        'inner_cv': inner_cv
    }


def learning_curve_data(model: Any,
                       X: Union[pd.DataFrame, np.ndarray],
                       y: Union[pd.Series, np.ndarray],
                       train_sizes: Optional[List[float]] = None,
                       cv: int = 5,
                       scoring: str = 'accuracy',
                       n_jobs: int = -1) -> Dict[str, np.ndarray]:
    """Generate learning curve data.
    
    Args:
        model: Sklearn-compatible estimator
        X: Feature matrix
        y: Target variable
        train_sizes: List of training set sizes (fractions or absolute)
        cv: Number of CV folds
        scoring: Scoring metric
        n_jobs: Number of parallel jobs
        
    Returns:
        Dictionary with train_sizes, train_scores, test_scores
    """
    from sklearn.model_selection import learning_curve
    
    if train_sizes is None:
        train_sizes = [0.1, 0.25, 0.5, 0.75, 1.0]
    
    train_sizes_abs, train_scores, test_scores = learning_curve(
        model, X, y,
        train_sizes=train_sizes,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs
    )
    
    return {
        'train_sizes': train_sizes_abs,
        'train_scores_mean': np.mean(train_scores, axis=1),
        'train_scores_std': np.std(train_scores, axis=1),
        'test_scores_mean': np.mean(test_scores, axis=1),
        'test_scores_std': np.std(test_scores, axis=1)
    }
