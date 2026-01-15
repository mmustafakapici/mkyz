# mkyz/data/loading.py
"""Data loading utilities for MKYZ library."""

from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import pandas as pd


class DataLoader:
    """Flexible data loader supporting multiple formats.
    
    Examples:
        >>> loader = DataLoader()
        >>> df = loader.load('data.csv')
        >>> df = loader.load('data.xlsx', sheet_name='Sheet1')
        >>> df = loader.load('data.json')
    """
    
    SUPPORTED_FORMATS = ['csv', 'xlsx', 'xls', 'json', 'parquet', 'feather', 'pickle', 'pkl']
    
    def __init__(self, default_encoding: str = 'utf-8'):
        """Initialize DataLoader.
        
        Args:
            default_encoding: Default file encoding
        """
        self.default_encoding = default_encoding
        self._last_loaded = None
    
    def load(self, 
             path: Union[str, Path],
             format: Optional[str] = None,
             **kwargs) -> pd.DataFrame:
        """Load data from file.
        
        Args:
            path: Path to data file
            format: File format (auto-detected if not specified)
            **kwargs: Additional arguments passed to pandas reader
            
        Returns:
            Loaded DataFrame
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        # Auto-detect format from extension
        if format is None:
            format = path.suffix.lstrip('.').lower()
        
        if format not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported format: {format}. Supported: {self.SUPPORTED_FORMATS}"
            )
        
        # Load based on format
        loaders = {
            'csv': self._load_csv,
            'xlsx': self._load_excel,
            'xls': self._load_excel,
            'json': self._load_json,
            'parquet': self._load_parquet,
            'feather': self._load_feather,
            'pickle': self._load_pickle,
            'pkl': self._load_pickle
        }
        
        df = loaders[format](path, **kwargs)
        self._last_loaded = path
        
        return df
    
    def _load_csv(self, path: Path, **kwargs) -> pd.DataFrame:
        """Load CSV file."""
        if 'encoding' not in kwargs:
            kwargs['encoding'] = self.default_encoding
        
        # Try to auto-detect separator
        if 'sep' not in kwargs:
            with open(path, 'r', encoding=kwargs['encoding']) as f:
                first_line = f.readline()
                if ';' in first_line and ',' not in first_line:
                    kwargs['sep'] = ';'
                elif '\t' in first_line:
                    kwargs['sep'] = '\t'
        
        return pd.read_csv(path, **kwargs)
    
    def _load_excel(self, path: Path, **kwargs) -> pd.DataFrame:
        """Load Excel file."""
        return pd.read_excel(path, **kwargs)
    
    def _load_json(self, path: Path, **kwargs) -> pd.DataFrame:
        """Load JSON file."""
        return pd.read_json(path, **kwargs)
    
    def _load_parquet(self, path: Path, **kwargs) -> pd.DataFrame:
        """Load Parquet file."""
        return pd.read_parquet(path, **kwargs)
    
    def _load_feather(self, path: Path, **kwargs) -> pd.DataFrame:
        """Load Feather file."""
        return pd.read_feather(path, **kwargs)
    
    def _load_pickle(self, path: Path, **kwargs) -> pd.DataFrame:
        """Load Pickle file."""
        return pd.read_pickle(path, **kwargs)
    
    def save(self,
             df: pd.DataFrame,
             path: Union[str, Path],
             format: Optional[str] = None,
             **kwargs) -> str:
        """Save DataFrame to file.
        
        Args:
            df: DataFrame to save
            path: Output path
            format: File format (auto-detected if not specified)
            **kwargs: Additional arguments passed to pandas writer
            
        Returns:
            Absolute path to saved file
        """
        path = Path(path)
        
        # Auto-detect format from extension
        if format is None:
            format = path.suffix.lstrip('.').lower()
        
        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save based on format
        savers = {
            'csv': lambda: df.to_csv(path, index=False, **kwargs),
            'xlsx': lambda: df.to_excel(path, index=False, **kwargs),
            'json': lambda: df.to_json(path, **kwargs),
            'parquet': lambda: df.to_parquet(path, **kwargs),
            'feather': lambda: df.to_feather(path, **kwargs),
            'pickle': lambda: df.to_pickle(path, **kwargs),
            'pkl': lambda: df.to_pickle(path, **kwargs)
        }
        
        if format not in savers:
            raise ValueError(f"Unsupported format: {format}")
        
        savers[format]()
        
        return str(path.absolute())


def load_data(path: Union[str, Path],
             format: Optional[str] = None,
             **kwargs) -> pd.DataFrame:
    """Load data from file (convenience function).
    
    Args:
        path: Path to data file
        format: File format (auto-detected if not specified)
        **kwargs: Additional arguments passed to pandas reader
        
    Returns:
        Loaded DataFrame
    
    Examples:
        >>> from mkyz.data import load_data
        >>> df = load_data('data.csv')
        >>> df = load_data('data.xlsx', sheet_name='Sheet1')
    """
    loader = DataLoader()
    return loader.load(path, format, **kwargs)


def load_sample_dataset(name: str = 'iris') -> pd.DataFrame:
    """Load a sample dataset for testing.
    
    Args:
        name: Dataset name ('iris', 'boston', 'titanic', 'wine')
        
    Returns:
        Sample DataFrame
    """
    from sklearn.datasets import load_iris, load_wine
    
    datasets = {
        'iris': lambda: _sklearn_to_df(load_iris()),
        'wine': lambda: _sklearn_to_df(load_wine()),
    }
    
    if name not in datasets:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(datasets.keys())}")
    
    return datasets[name]()


def _sklearn_to_df(sklearn_dataset) -> pd.DataFrame:
    """Convert sklearn dataset to DataFrame."""
    df = pd.DataFrame(
        sklearn_dataset.data,
        columns=sklearn_dataset.feature_names
    )
    df['target'] = sklearn_dataset.target
    return df
