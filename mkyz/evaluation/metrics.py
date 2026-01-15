# mkyz/evaluation/metrics.py
"""Model evaluation metrics for different task types."""

from typing import Dict, Any, Optional, List, Union
import numpy as np
from sklearn.metrics import (
    # Classification metrics
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    log_loss, matthews_corrcoef, cohen_kappa_score,
    # Regression metrics
    mean_squared_error, mean_absolute_error, r2_score,
    mean_squared_log_error, explained_variance_score, max_error,
    mean_absolute_percentage_error,
    # Clustering metrics
    silhouette_score, calinski_harabasz_score, davies_bouldin_score
)


def classification_metrics(y_true: np.ndarray,
                          y_pred: np.ndarray,
                          y_proba: Optional[np.ndarray] = None,
                          average: str = 'weighted') -> Dict[str, float]:
    """Calculate comprehensive classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (optional, for ROC-AUC)
        average: Averaging method for multi-class ('weighted', 'macro', 'micro')
        
    Returns:
        Dictionary of metric names and values
    
    Examples:
        >>> from mkyz.evaluation import classification_metrics
        >>> metrics = classification_metrics(y_test, predictions)
        >>> print(f"Accuracy: {metrics['accuracy']:.4f}")
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average=average, zero_division=0),
        'mcc': matthews_corrcoef(y_true, y_pred),
        'cohen_kappa': cohen_kappa_score(y_true, y_pred)
    }
    
    # Add ROC-AUC if probabilities are provided
    if y_proba is not None:
        try:
            n_classes = len(np.unique(y_true))
            if n_classes == 2:
                # Binary classification
                if y_proba.ndim == 2:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
                else:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
            else:
                # Multi-class
                metrics['roc_auc'] = roc_auc_score(
                    y_true, y_proba, average=average, multi_class='ovr'
                )
        except ValueError:
            # ROC-AUC not applicable
            metrics['roc_auc'] = None
    
    return metrics


def regression_metrics(y_true: np.ndarray,
                      y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate comprehensive regression metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary of metric names and values
    
    Examples:
        >>> from mkyz.evaluation import regression_metrics
        >>> metrics = regression_metrics(y_test, predictions)
        >>> print(f"R2 Score: {metrics['r2_score']:.4f}")
    """
    metrics = {
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2_score': r2_score(y_true, y_pred),
        'explained_variance': explained_variance_score(y_true, y_pred),
        'max_error': max_error(y_true, y_pred)
    }
    
    # MAPE (handle zero values)
    try:
        metrics['mape'] = mean_absolute_percentage_error(y_true, y_pred)
    except ValueError:
        metrics['mape'] = None
    
    # MSLE (only for positive values)
    try:
        if np.all(y_true >= 0) and np.all(y_pred >= 0):
            metrics['msle'] = mean_squared_log_error(y_true, y_pred)
        else:
            metrics['msle'] = None
    except ValueError:
        metrics['msle'] = None
    
    return metrics


def clustering_metrics(X: np.ndarray,
                      labels: np.ndarray) -> Dict[str, float]:
    """Calculate clustering quality metrics.
    
    Args:
        X: Feature matrix used for clustering
        labels: Cluster labels
        
    Returns:
        Dictionary of metric names and values
    
    Examples:
        >>> from mkyz.evaluation import clustering_metrics
        >>> metrics = clustering_metrics(X, cluster_labels)
        >>> print(f"Silhouette: {metrics['silhouette_score']:.4f}")
    """
    n_labels = len(np.unique(labels))
    
    # Need at least 2 clusters for meaningful metrics
    if n_labels < 2:
        return {
            'silhouette_score': -1,
            'calinski_harabasz': None,
            'davies_bouldin': None,
            'n_clusters': n_labels
        }
    
    metrics = {
        'silhouette_score': silhouette_score(X, labels),
        'calinski_harabasz': calinski_harabasz_score(X, labels),
        'davies_bouldin': davies_bouldin_score(X, labels),
        'n_clusters': n_labels
    }
    
    return metrics


def dimensionality_reduction_metrics(X_original: np.ndarray,
                                    X_reduced: np.ndarray,
                                    model: Any = None) -> Dict[str, float]:
    """Calculate dimensionality reduction quality metrics.
    
    Args:
        X_original: Original feature matrix
        X_reduced: Reduced feature matrix
        model: The fitted reduction model (for explained variance)
        
    Returns:
        Dictionary of metric names and values
    """
    metrics = {
        'original_dimensions': X_original.shape[1],
        'reduced_dimensions': X_reduced.shape[1],
        'compression_ratio': X_original.shape[1] / X_reduced.shape[1]
    }
    
    # Get explained variance if available (e.g., PCA)
    if model is not None and hasattr(model, 'explained_variance_ratio_'):
        metrics['total_explained_variance'] = sum(model.explained_variance_ratio_)
        metrics['explained_variance_per_component'] = list(model.explained_variance_ratio_)
    
    return metrics


def get_all_metrics(task: str,
                   y_true: np.ndarray = None,
                   y_pred: np.ndarray = None,
                   y_proba: np.ndarray = None,
                   X: np.ndarray = None,
                   labels: np.ndarray = None,
                   **kwargs) -> Dict[str, float]:
    """Get all metrics for a given task type.
    
    Args:
        task: Task type ('classification', 'regression', 'clustering')
        y_true: True values/labels
        y_pred: Predicted values/labels
        y_proba: Predicted probabilities (optional)
        X: Feature matrix (for clustering)
        labels: Cluster labels (for clustering)
        **kwargs: Additional arguments passed to metric functions
        
    Returns:
        Dictionary of metric names and values
    
    Examples:
        >>> metrics = get_all_metrics('classification', y_true, y_pred, y_proba)
        >>> metrics = get_all_metrics('regression', y_true, y_pred)
        >>> metrics = get_all_metrics('clustering', X=X, labels=labels)
    """
    if task == 'classification':
        return classification_metrics(y_true, y_pred, y_proba, **kwargs)
    elif task == 'regression':
        return regression_metrics(y_true, y_pred)
    elif task == 'clustering':
        return clustering_metrics(X, labels)
    else:
        raise ValueError(f"Unsupported task type: {task}")


def get_confusion_matrix(y_true: np.ndarray,
                        y_pred: np.ndarray,
                        labels: Optional[List] = None) -> np.ndarray:
    """Get confusion matrix for classification.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Optional list of label values
        
    Returns:
        Confusion matrix as numpy array
    """
    return confusion_matrix(y_true, y_pred, labels=labels)


def get_classification_report(y_true: np.ndarray,
                             y_pred: np.ndarray,
                             target_names: Optional[List[str]] = None,
                             as_dict: bool = False) -> Union[str, Dict]:
    """Get detailed classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        target_names: Optional names for each class
        as_dict: If True, return as dictionary instead of string
        
    Returns:
        Classification report as string or dictionary
    """
    return classification_report(
        y_true, y_pred,
        target_names=target_names,
        output_dict=as_dict,
        zero_division=0
    )
