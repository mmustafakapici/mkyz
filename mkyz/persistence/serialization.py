# mkyz/persistence/serialization.py
"""Model serialization and persistence utilities."""

import os
import json
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Union
from datetime import datetime

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False


def save_model(model: Any, 
               path: Union[str, Path], 
               format: str = 'joblib',
               metadata: Optional[Dict[str, Any]] = None,
               overwrite: bool = False) -> str:
    """Save a trained model to disk.
    
    Args:
        model: Trained model object to save
        path: File path to save the model (extension will be added if missing)
        format: Serialization format ('joblib', 'pickle')
        metadata: Optional metadata to save alongside the model
        overwrite: Whether to overwrite existing file
        
    Returns:
        Absolute path where the model was saved
        
    Raises:
        PersistenceError: If saving fails
        FileExistsError: If file exists and overwrite=False
    
    Examples:
        >>> from mkyz.persistence import save_model
        >>> save_model(trained_rf, 'models/my_model')
        'models/my_model.joblib'
        
        >>> save_model(trained_rf, 'models/my_model', format='pickle')
        'models/my_model.pkl'
    """
    from mkyz.core.exceptions import PersistenceError
    
    path = Path(path)
    
    # Add extension if not present
    extensions = {'joblib': '.joblib', 'pickle': '.pkl'}
    if path.suffix not in extensions.values():
        path = path.with_suffix(extensions.get(format, '.joblib'))
    
    # Check if file exists
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"File {path} already exists. Set overwrite=True to replace it."
        )
    
    # Create directory if needed
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare save data
    save_data = {
        'model': model,
        'metadata': metadata or {},
        'saved_at': datetime.now().isoformat(),
        'format': format,
        'mkyz_version': _get_version()
    }
    
    try:
        if format == 'joblib':
            if not JOBLIB_AVAILABLE:
                raise PersistenceError(
                    "joblib is not installed. Install it with: pip install joblib"
                )
            joblib.dump(save_data, path)
        elif format == 'pickle':
            with open(path, 'wb') as f:
                pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            raise PersistenceError(f"Unsupported format: {format}")
            
    except Exception as e:
        raise PersistenceError(f"Failed to save model: {e}")
    
    return str(path.absolute())


def load_model(path: Union[str, Path], 
               return_metadata: bool = False) -> Union[Any, tuple]:
    """Load a trained model from disk.
    
    Args:
        path: Path to the saved model file
        return_metadata: If True, return (model, metadata) tuple
        
    Returns:
        Loaded model object, or (model, metadata) tuple if return_metadata=True
        
    Raises:
        PersistenceError: If loading fails
        FileNotFoundError: If model file doesn't exist
    
    Examples:
        >>> from mkyz.persistence import load_model
        >>> model = load_model('models/my_model.joblib')
        
        >>> model, metadata = load_model('models/my_model.joblib', return_metadata=True)
        >>> print(metadata['saved_at'])
    """
    from mkyz.core.exceptions import PersistenceError
    
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    
    try:
        if path.suffix == '.joblib':
            if not JOBLIB_AVAILABLE:
                raise PersistenceError(
                    "joblib is not installed. Install it with: pip install joblib"
                )
            save_data = joblib.load(path)
        elif path.suffix == '.pkl':
            with open(path, 'rb') as f:
                save_data = pickle.load(f)
        else:
            raise PersistenceError(f"Unsupported file extension: {path.suffix}")
            
    except Exception as e:
        raise PersistenceError(f"Failed to load model: {e}")
    
    model = save_data.get('model')
    metadata = save_data.get('metadata', {})
    
    # Add load info to metadata
    metadata['loaded_at'] = datetime.now().isoformat()
    metadata['saved_at'] = save_data.get('saved_at')
    metadata['mkyz_version'] = save_data.get('mkyz_version')
    
    if return_metadata:
        return model, metadata
    return model


def export_pipeline(pipeline: Dict[str, Any],
                   path: Union[str, Path],
                   overwrite: bool = False) -> str:
    """Export a complete ML pipeline configuration.
    
    Exports preprocessing steps, model configuration, and metadata
    as a JSON file that can be shared and reproduced.
    
    Args:
        pipeline: Dictionary containing pipeline configuration
        path: Path to save the pipeline JSON
        overwrite: Whether to overwrite existing file
        
    Returns:
        Absolute path where the pipeline was saved
        
    Examples:
        >>> pipeline = {
        ...     'preprocessing': {'missing': 'mean', 'encoding': 'onehot'},
        ...     'model': {'type': 'rf', 'params': {'n_estimators': 100}},
        ...     'target': 'price'
        ... }
        >>> export_pipeline(pipeline, 'pipelines/my_pipeline.json')
    """
    from mkyz.core.exceptions import PersistenceError
    
    path = Path(path)
    if path.suffix != '.json':
        path = path.with_suffix('.json')
    
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"File {path} already exists. Set overwrite=True to replace it."
        )
    
    path.parent.mkdir(parents=True, exist_ok=True)
    
    export_data = {
        'pipeline': pipeline,
        'exported_at': datetime.now().isoformat(),
        'mkyz_version': _get_version()
    }
    
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, default=str)
    except Exception as e:
        raise PersistenceError(f"Failed to export pipeline: {e}")
    
    return str(path.absolute())


def import_pipeline(path: Union[str, Path]) -> Dict[str, Any]:
    """Import a pipeline configuration from JSON.
    
    Args:
        path: Path to the pipeline JSON file
        
    Returns:
        Pipeline configuration dictionary
        
    Raises:
        PersistenceError: If import fails
        FileNotFoundError: If file doesn't exist
    """
    from mkyz.core.exceptions import PersistenceError
    
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Pipeline file not found: {path}")
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        raise PersistenceError(f"Failed to import pipeline: {e}")
    
    return data.get('pipeline', data)


def _get_version() -> str:
    """Get current MKYZ version."""
    try:
        from mkyz import __version__
        return __version__
    except ImportError:
        return 'unknown'
