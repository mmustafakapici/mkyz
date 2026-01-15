# mkyz - Machine Learning Library
# Simplifies data processing, model training, evaluation, and visualization

__version__ = "0.2.0"

# ============================================================
# Constants
# ============================================================
DEFAULT_SEED = 42

# ============================================================
# Initialization
# ============================================================
def init(verbose: bool = True):
    """Initialize the mkyz package.
    
    Args:
        verbose: Whether to print initialization message
    """
    if verbose:
        print(f"mkyz package initialized. Version: {__version__}")


# ============================================================
# Backward-compatible imports (original API)
# These may fail if optional dependencies are missing
# ============================================================
try:
    from .data_processing import prepare_data
except ImportError as e:
    prepare_data = None

try:
    from .training import train, predict, evaluate, auto_train, optimize_model
except ImportError as e:
    # Optional dependencies like optuna may be missing
    train = predict = evaluate = auto_train = optimize_model = None

try:
    from .visualization import visualize
except ImportError as e:
    visualize = None

# ============================================================
# New modular imports (Phase 1 & 2 enhancements)
# These have minimal dependencies
# ============================================================

# Core module - always available
from .core.config import Config, DEFAULT_CONFIG, get_config, set_config
from .core.exceptions import (
    MKYZError,
    DataValidationError,
    ModelNotTrainedError,
    UnsupportedTaskError,
    UnsupportedModelError,
    OptimizationError,
    PersistenceError
)

# Persistence module
from .persistence import save_model, load_model

# Evaluation module
from .evaluation import (
    classification_metrics,
    regression_metrics,
    clustering_metrics,
    cross_validate,
    CVStrategy,
    ModelReport
)

# Data module
from .data import (
    FeatureEngineer,
    create_polynomial_features,
    create_datetime_features,
    select_features,
    validate_dataset,
    check_target_balance,
    load_data,
    DataLoader,
    # EDA
    DataProfile,
    data_info,
    quick_eda
)

# Utils module
from .utils import setup_logging, get_logger

# Initialize on import
init(verbose=True)

# ============================================================
# Public API
# ============================================================
__all__ = [
    # Version
    '__version__',
    # Original API (may be None if deps missing)
    'prepare_data',
    'train',
    'predict',
    'evaluate',
    'auto_train',
    'optimize_model',
    'visualize',
    # Config
    'Config',
    'DEFAULT_CONFIG',
    'get_config',
    'set_config',
    # Exceptions
    'MKYZError',
    'DataValidationError',
    'ModelNotTrainedError',
    'UnsupportedTaskError',
    'UnsupportedModelError',
    'OptimizationError',
    'PersistenceError',
    # Persistence
    'save_model',
    'load_model',
    # Evaluation
    'classification_metrics',
    'regression_metrics',
    'clustering_metrics',
    'cross_validate',
    'CVStrategy',
    'ModelReport',
    # Data
    'FeatureEngineer',
    'create_polynomial_features',
    'create_datetime_features',
    'select_features',
    'validate_dataset',
    'check_target_balance',
    'load_data',
    'DataLoader',
    # EDA
    'DataProfile',
    'data_info',
    'quick_eda',
    # Utils
    'setup_logging',
    'get_logger',
    # Constants
    'DEFAULT_SEED',
    'init'
]
