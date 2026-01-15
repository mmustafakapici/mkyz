# mkyz/data/eda.py
"""Exploratory Data Analysis (EDA) module for MKYZ library.

Provides comprehensive data profiling, summary statistics, and insights.
"""

from typing import Dict, List, Optional, Any, Union, Tuple
import numpy as np
import pandas as pd
from collections import Counter


class DataProfile:
    """Comprehensive data profiling and EDA class.
    
    Generates detailed statistics and insights about a dataset.
    
    Examples:
        >>> from mkyz.data import DataProfile
        >>> profile = DataProfile(df)
        >>> print(profile.summary())
        >>> profile.export_report('data_profile.html')
    """
    
    def __init__(self, df: pd.DataFrame, target_column: Optional[str] = None):
        """Initialize DataProfile.
        
        Args:
            df: DataFrame to analyze
            target_column: Optional target column for supervised analysis
        """
        self.df = df.copy()
        self.target_column = target_column
        self._profile = None
    
    def generate(self) -> 'DataProfile':
        """Generate the complete data profile.
        
        Returns:
            self for method chaining
        """
        self._profile = {
            'overview': self._get_overview(),
            'columns': self._analyze_columns(),
            'missing': self._analyze_missing(),
            'correlations': self._analyze_correlations(),
            'duplicates': self._analyze_duplicates(),
            'samples': self._get_samples()
        }
        
        if self.target_column and self.target_column in self.df.columns:
            self._profile['target'] = self._analyze_target()
        
        return self
    
    def _get_overview(self) -> Dict[str, Any]:
        """Get dataset overview statistics."""
        return {
            'n_rows': len(self.df),
            'n_columns': len(self.df.columns),
            'memory_usage_mb': self.df.memory_usage(deep=True).sum() / 1024**2,
            'n_numerical': len(self.df.select_dtypes(include=[np.number]).columns),
            'n_categorical': len(self.df.select_dtypes(include=['object', 'category']).columns),
            'n_datetime': len(self.df.select_dtypes(include=['datetime64']).columns),
            'n_boolean': len(self.df.select_dtypes(include=['bool']).columns),
            'total_missing': self.df.isna().sum().sum(),
            'missing_percentage': (self.df.isna().sum().sum() / self.df.size) * 100,
            'n_duplicates': self.df.duplicated().sum()
        }
    
    def _analyze_columns(self) -> Dict[str, Dict]:
        """Analyze each column in detail."""
        columns = {}
        
        for col in self.df.columns:
            col_data = self.df[col]
            col_info = {
                'dtype': str(col_data.dtype),
                'n_unique': col_data.nunique(),
                'n_missing': col_data.isna().sum(),
                'missing_pct': (col_data.isna().sum() / len(col_data)) * 100
            }
            
            # Numerical column statistics
            if pd.api.types.is_numeric_dtype(col_data):
                col_info.update(self._numerical_stats(col_data))
            
            # Categorical column statistics
            elif pd.api.types.is_object_dtype(col_data) or pd.api.types.is_categorical_dtype(col_data):
                col_info.update(self._categorical_stats(col_data))
            
            # Datetime statistics
            elif pd.api.types.is_datetime64_any_dtype(col_data):
                col_info.update(self._datetime_stats(col_data))
            
            columns[col] = col_info
        
        return columns
    
    def _numerical_stats(self, series: pd.Series) -> Dict[str, Any]:
        """Calculate numerical column statistics."""
        clean = series.dropna()
        if len(clean) == 0:
            return {'stats': 'No valid data'}
        
        q1, q3 = clean.quantile([0.25, 0.75])
        iqr = q3 - q1
        
        return {
            'type': 'numerical',
            'mean': clean.mean(),
            'std': clean.std(),
            'min': clean.min(),
            'max': clean.max(),
            'median': clean.median(),
            'q1': q1,
            'q3': q3,
            'iqr': iqr,
            'skewness': clean.skew(),
            'kurtosis': clean.kurtosis(),
            'n_zeros': (clean == 0).sum(),
            'n_negative': (clean < 0).sum(),
            'n_outliers': ((clean < (q1 - 1.5 * iqr)) | (clean > (q3 + 1.5 * iqr))).sum()
        }
    
    def _categorical_stats(self, series: pd.Series) -> Dict[str, Any]:
        """Calculate categorical column statistics."""
        clean = series.dropna()
        value_counts = clean.value_counts()
        
        return {
            'type': 'categorical',
            'n_unique': len(value_counts),
            'top_value': value_counts.index[0] if len(value_counts) > 0 else None,
            'top_freq': value_counts.iloc[0] if len(value_counts) > 0 else 0,
            'top_5': dict(value_counts.head(5)),
            'avg_length': clean.astype(str).str.len().mean()
        }
    
    def _datetime_stats(self, series: pd.Series) -> Dict[str, Any]:
        """Calculate datetime column statistics."""
        clean = series.dropna()
        if len(clean) == 0:
            return {'stats': 'No valid data'}
        
        return {
            'type': 'datetime',
            'min_date': clean.min(),
            'max_date': clean.max(),
            'range_days': (clean.max() - clean.min()).days,
            'n_unique_dates': clean.dt.date.nunique()
        }
    
    def _analyze_missing(self) -> Dict[str, Any]:
        """Analyze missing value patterns."""
        missing_counts = self.df.isna().sum()
        missing_cols = missing_counts[missing_counts > 0]
        
        return {
            'total_missing': missing_counts.sum(),
            'columns_with_missing': len(missing_cols),
            'missing_by_column': dict(missing_cols),
            'missing_pct_by_column': dict((missing_cols / len(self.df)) * 100),
            'rows_with_any_missing': self.df.isna().any(axis=1).sum(),
            'complete_rows': (~self.df.isna().any(axis=1)).sum()
        }
    
    def _analyze_correlations(self) -> Dict[str, Any]:
        """Analyze correlations between numerical columns."""
        num_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if len(num_cols) < 2:
            return {'message': 'Not enough numerical columns for correlation'}
        
        corr_matrix = self.df[num_cols].corr()
        
        # Find highly correlated pairs
        high_corr = []
        for i, col1 in enumerate(num_cols):
            for col2 in num_cols[i+1:]:
                corr = corr_matrix.loc[col1, col2]
                if abs(corr) > 0.7:
                    high_corr.append({
                        'column1': col1,
                        'column2': col2,
                        'correlation': corr
                    })
        
        return {
            'correlation_matrix': corr_matrix.to_dict(),
            'high_correlations': sorted(high_corr, key=lambda x: abs(x['correlation']), reverse=True)
        }
    
    def _analyze_duplicates(self) -> Dict[str, Any]:
        """Analyze duplicate rows."""
        n_duplicates = self.df.duplicated().sum()
        
        return {
            'n_duplicates': n_duplicates,
            'pct_duplicates': (n_duplicates / len(self.df)) * 100,
            'duplicate_indices': list(self.df[self.df.duplicated()].index[:10])
        }
    
    def _get_samples(self) -> Dict[str, Any]:
        """Get sample rows."""
        return {
            'head': self.df.head(5).to_dict('records'),
            'tail': self.df.tail(5).to_dict('records'),
            'random': self.df.sample(min(5, len(self.df))).to_dict('records')
        }
    
    def _analyze_target(self) -> Dict[str, Any]:
        """Analyze target column for supervised learning."""
        target = self.df[self.target_column]
        
        if pd.api.types.is_numeric_dtype(target) and target.nunique() > 10:
            # Regression target
            return {
                'task_type': 'regression',
                **self._numerical_stats(target)
            }
        else:
            # Classification target
            value_counts = target.value_counts()
            return {
                'task_type': 'classification',
                'n_classes': len(value_counts),
                'class_distribution': dict(value_counts),
                'class_balance': dict((value_counts / len(target)) * 100),
                'is_imbalanced': value_counts.min() / value_counts.max() < 0.1
            }
    
    def summary(self) -> str:
        """Generate text summary of the profile.
        
        Returns:
            Formatted summary string
        """
        if self._profile is None:
            self.generate()
        
        overview = self._profile['overview']
        missing = self._profile['missing']
        
        lines = [
            "=" * 60,
            "DATA PROFILE SUMMARY",
            "=" * 60,
            "",
            "📊 Overview",
            "-" * 40,
            f"  Rows: {overview['n_rows']:,}",
            f"  Columns: {overview['n_columns']}",
            f"  Memory: {overview['memory_usage_mb']:.2f} MB",
            "",
            "📈 Column Types",
            "-" * 40,
            f"  Numerical: {overview['n_numerical']}",
            f"  Categorical: {overview['n_categorical']}",
            f"  Datetime: {overview['n_datetime']}",
            f"  Boolean: {overview['n_boolean']}",
            "",
            "❓ Missing Values",
            "-" * 40,
            f"  Total: {missing['total_missing']:,} ({overview['missing_percentage']:.2f}%)",
            f"  Columns affected: {missing['columns_with_missing']}",
            f"  Complete rows: {missing['complete_rows']:,}",
            "",
            "🔄 Duplicates",
            "-" * 40,
            f"  Duplicate rows: {overview['n_duplicates']}",
        ]
        
        # Add target info if available
        if 'target' in self._profile:
            target_info = self._profile['target']
            lines.extend([
                "",
                "🎯 Target Analysis",
                "-" * 40,
                f"  Task type: {target_info['task_type']}",
            ])
            if target_info['task_type'] == 'classification':
                lines.append(f"  Classes: {target_info['n_classes']}")
                lines.append(f"  Imbalanced: {'Yes' if target_info['is_imbalanced'] else 'No'}")
        
        lines.extend(["", "=" * 60])
        
        return "\n".join(lines)
    
    def get_column_info(self, column: str) -> Dict[str, Any]:
        """Get detailed info for a specific column.
        
        Args:
            column: Column name
            
        Returns:
            Column statistics and info
        """
        if self._profile is None:
            self.generate()
        
        if column not in self._profile['columns']:
            raise KeyError(f"Column '{column}' not found")
        
        return self._profile['columns'][column]
    
    def get_recommendations(self) -> List[str]:
        """Get preprocessing recommendations based on analysis.
        
        Returns:
            List of recommendation strings
        """
        if self._profile is None:
            self.generate()
        
        recommendations = []
        overview = self._profile['overview']
        columns = self._profile['columns']
        missing = self._profile['missing']
        
        # Missing value recommendations
        if overview['missing_percentage'] > 5:
            recommendations.append(
                f"⚠️ High missing rate ({overview['missing_percentage']:.1f}%). "
                "Consider imputation or dropping columns with >50% missing."
            )
        
        # High cardinality categorical
        for col, info in columns.items():
            if info.get('type') == 'categorical' and info.get('n_unique', 0) > 50:
                recommendations.append(
                    f"⚠️ Column '{col}' has high cardinality ({info['n_unique']} unique). "
                    "Consider target encoding or grouping rare categories."
                )
        
        # Duplicates
        if overview['n_duplicates'] > 0:
            recommendations.append(
                f"ℹ️ Found {overview['n_duplicates']} duplicate rows. Review and decide to keep/remove."
            )
        
        # Outliers
        for col, info in columns.items():
            if info.get('n_outliers', 0) > 0:
                outlier_pct = (info['n_outliers'] / overview['n_rows']) * 100
                if outlier_pct > 5:
                    recommendations.append(
                        f"⚠️ Column '{col}' has {outlier_pct:.1f}% outliers. Consider capping or investigation."
                    )
        
        # High correlations
        high_corr = self._profile['correlations'].get('high_correlations', [])
        if high_corr:
            for pair in high_corr[:3]:
                recommendations.append(
                    f"ℹ️ High correlation ({pair['correlation']:.2f}) between "
                    f"'{pair['column1']}' and '{pair['column2']}'. Consider removing one."
                )
        
        # Class imbalance
        if 'target' in self._profile and self._profile['target'].get('is_imbalanced'):
            recommendations.append(
                "⚠️ Target is imbalanced. Consider SMOTE, class weights, or stratified sampling."
            )
        
        return recommendations if recommendations else ["✅ Dataset looks clean!"]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert profile to dictionary.
        
        Returns:
            Complete profile as dictionary
        """
        if self._profile is None:
            self.generate()
        return self._profile
    
    def export_report(self, path: str) -> str:
        """Export profile as HTML report.
        
        Args:
            path: Output file path
            
        Returns:
            Absolute path to saved file
        """
        from pathlib import Path
        
        if self._profile is None:
            self.generate()
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        html = self._generate_html()
        
        with open(path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        return str(path.absolute())
    
    def _generate_html(self) -> str:
        """Generate HTML report content."""
        overview = self._profile['overview']
        
        # Build columns table
        columns_rows = ""
        for col, info in self._profile['columns'].items():
            col_type = info.get('type', info['dtype'])
            missing_pct = f"{info['missing_pct']:.1f}%"
            unique = info['n_unique']
            columns_rows += f"<tr><td>{col}</td><td>{col_type}</td><td>{unique}</td><td>{missing_pct}</td></tr>\n"
        
        # Build recommendations
        recommendations_html = "<ul>" + "".join(f"<li>{r}</li>" for r in self.get_recommendations()) + "</ul>"
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Data Profile Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 40px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 3px solid #007bff; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; }}
        .stat-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin: 20px 0; }}
        .stat-card {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px; text-align: center; }}
        .stat-card .value {{ font-size: 2em; font-weight: bold; }}
        .stat-card .label {{ opacity: 0.9; margin-top: 5px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #007bff; color: white; }}
        tr:hover {{ background: #f5f5f5; }}
        .recommendations {{ background: #fff3cd; padding: 20px; border-radius: 8px; margin: 20px 0; }}
        .recommendations ul {{ margin: 10px 0; padding-left: 20px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Data Profile Report</h1>
        
        <div class="stat-grid">
            <div class="stat-card">
                <div class="value">{overview['n_rows']:,}</div>
                <div class="label">Rows</div>
            </div>
            <div class="stat-card">
                <div class="value">{overview['n_columns']}</div>
                <div class="label">Columns</div>
            </div>
            <div class="stat-card">
                <div class="value">{overview['missing_percentage']:.1f}%</div>
                <div class="label">Missing</div>
            </div>
            <div class="stat-card">
                <div class="value">{overview['n_duplicates']}</div>
                <div class="label">Duplicates</div>
            </div>
        </div>
        
        <h2>📈 Column Overview</h2>
        <table>
            <tr><th>Column</th><th>Type</th><th>Unique</th><th>Missing</th></tr>
            {columns_rows}
        </table>
        
        <h2>💡 Recommendations</h2>
        <div class="recommendations">
            {recommendations_html}
        </div>
        
        <footer style="margin-top: 40px; color: #888; font-size: 12px;">
            Generated by MKYZ Library - Data Profile
        </footer>
    </div>
</body>
</html>
        """
        
        return html


def data_info(df: pd.DataFrame, 
             target_column: Optional[str] = None,
             detailed: bool = False) -> Dict[str, Any]:
    """Quick data information and statistics.
    
    Args:
        df: DataFrame to analyze
        target_column: Optional target column
        detailed: If True, return full profile
        
    Returns:
        Dictionary with data information
    
    Examples:
        >>> from mkyz.data import data_info
        >>> info = data_info(df)
        >>> print(f"Rows: {info['n_rows']}, Missing: {info['missing_pct']:.1f}%")
    """
    profile = DataProfile(df, target_column)
    profile.generate()
    
    if detailed:
        return profile.to_dict()
    
    overview = profile._profile['overview']
    return {
        'n_rows': overview['n_rows'],
        'n_columns': overview['n_columns'],
        'n_numerical': overview['n_numerical'],
        'n_categorical': overview['n_categorical'],
        'memory_mb': round(overview['memory_usage_mb'], 2),
        'missing_pct': round(overview['missing_percentage'], 2),
        'n_duplicates': overview['n_duplicates'],
        'columns': list(df.columns)
    }


def describe_column(df: pd.DataFrame, column: str) -> Dict[str, Any]:
    """Get detailed statistics for a single column.
    
    Args:
        df: DataFrame
        column: Column name
        
    Returns:
        Column statistics
    """
    profile = DataProfile(df)
    profile.generate()
    return profile.get_column_info(column)


def get_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Get summary statistics similar to pandas describe() but enhanced.
    
    Args:
        df: DataFrame
        
    Returns:
        Summary statistics DataFrame
    """
    stats = []
    
    for col in df.columns:
        row = {
            'column': col,
            'dtype': str(df[col].dtype),
            'n_unique': df[col].nunique(),
            'n_missing': df[col].isna().sum(),
            'missing_pct': f"{(df[col].isna().sum() / len(df)) * 100:.1f}%"
        }
        
        if pd.api.types.is_numeric_dtype(df[col]):
            row['mean'] = df[col].mean()
            row['std'] = df[col].std()
            row['min'] = df[col].min()
            row['max'] = df[col].max()
        else:
            row['mean'] = '-'
            row['std'] = '-'
            row['min'] = '-'
            row['max'] = df[col].dropna().iloc[0] if len(df[col].dropna()) > 0 else '-'
        
        stats.append(row)
    
    return pd.DataFrame(stats)


def quick_eda(df: pd.DataFrame, 
             target_column: Optional[str] = None,
             show_plots: bool = False) -> None:
    """Print a quick EDA report to console.
    
    Args:
        df: DataFrame to analyze
        target_column: Optional target column
        show_plots: If True, display matplotlib plots
    """
    profile = DataProfile(df, target_column)
    profile.generate()
    
    print(profile.summary())
    print("\n💡 RECOMMENDATIONS")
    print("-" * 40)
    for rec in profile.get_recommendations():
        print(f"  {rec}")
    
    if show_plots:
        try:
            import matplotlib.pyplot as plt
            
            # Missing values plot
            missing = df.isna().sum()
            if missing.sum() > 0:
                plt.figure(figsize=(10, 4))
                missing[missing > 0].plot(kind='bar')
                plt.title('Missing Values by Column')
                plt.ylabel('Count')
                plt.tight_layout()
                plt.show()
            
            # Numerical distributions
            num_cols = df.select_dtypes(include=[np.number]).columns[:6]
            if len(num_cols) > 0:
                fig, axes = plt.subplots(2, 3, figsize=(12, 6))
                for i, col in enumerate(num_cols):
                    ax = axes.flatten()[i]
                    df[col].hist(ax=ax, bins=30)
                    ax.set_title(col)
                plt.tight_layout()
                plt.show()
                
        except ImportError:
            print("Install matplotlib for plots: pip install matplotlib")
