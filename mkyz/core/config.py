# mkyz/core/config.py
"""Global configuration for MKYZ library."""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List


@dataclass
class Config:
    """Global configuration class for MKYZ library.
    
    Attributes:
        random_state: Random seed for reproducibility
        n_jobs: Number of parallel jobs (-1 for all CPUs)
        verbose: Verbosity level (0=silent, 1=progress, 2=debug)
        cv_folds: Default number of cross-validation folds
        test_size: Default test set proportion
        optimization_method: Default optimization method ('grid_search' or 'bayesian')
        n_trials: Number of trials for Bayesian optimization
        log_level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        suppress_warnings: Whether to suppress sklearn warnings
    """
    random_state: int = 42
    n_jobs: int = -1
    verbose: int = 1
    cv_folds: int = 5
    test_size: float = 0.2
    optimization_method: str = 'grid_search'
    n_trials: int = 50
    log_level: str = 'INFO'
    suppress_warnings: bool = True
    
    # Model defaults
    default_classification_model: str = 'rf'
    default_regression_model: str = 'rf'
    default_clustering_model: str = 'kmeans'
    default_reduction_model: str = 'pca'
    
    # Visualization defaults
    figure_size: tuple = (15, 10)
    color_palette: str = 'tab20'
    dark_mode: bool = True
    
    # Feature engineering defaults
    polynomial_degree: int = 2
    feature_selection_k: int = 10
    
    # Data processing defaults
    missing_value_strategy: str = 'mean'  # 'mean', 'median', 'mode', 'drop'
    outlier_strategy: str = 'remove'  # 'remove', 'cap', 'keep'
    outlier_threshold: float = 1.5
    categorical_encoding: str = 'onehot'  # 'onehot', 'label', 'frequency', 'target'
    
    def update(self, **kwargs) -> 'Config':
        """Update config with new values.
        
        Args:
            **kwargs: Configuration values to update
            
        Returns:
            Updated Config instance
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown config key: {key}")
        return self
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary.
        
        Returns:
            Dictionary of configuration values
        """
        return {
            'random_state': self.random_state,
            'n_jobs': self.n_jobs,
            'verbose': self.verbose,
            'cv_folds': self.cv_folds,
            'test_size': self.test_size,
            'optimization_method': self.optimization_method,
            'n_trials': self.n_trials,
            'log_level': self.log_level,
            'suppress_warnings': self.suppress_warnings,
            'default_classification_model': self.default_classification_model,
            'default_regression_model': self.default_regression_model,
            'default_clustering_model': self.default_clustering_model,
            'default_reduction_model': self.default_reduction_model,
            'figure_size': self.figure_size,
            'color_palette': self.color_palette,
            'dark_mode': self.dark_mode,
            'polynomial_degree': self.polynomial_degree,
            'feature_selection_k': self.feature_selection_k,
            'missing_value_strategy': self.missing_value_strategy,
            'outlier_strategy': self.outlier_strategy,
            'outlier_threshold': self.outlier_threshold,
            'categorical_encoding': self.categorical_encoding,
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create Config from dictionary.
        
        Args:
            config_dict: Dictionary of configuration values
            
        Returns:
            New Config instance
        """
        return cls(**config_dict)


# Default global configuration instance
DEFAULT_CONFIG = Config()


def get_config() -> Config:
    """Get the current global configuration.
    
    Returns:
        Current Config instance
    """
    return DEFAULT_CONFIG


def set_config(**kwargs) -> Config:
    """Update global configuration.
    
    Args:
        **kwargs: Configuration values to update
        
    Returns:
        Updated Config instance
    """
    return DEFAULT_CONFIG.update(**kwargs)
