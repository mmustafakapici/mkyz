# mkyz/evaluation/reports.py
"""Automatic model reporting and export utilities."""

from typing import Any, Dict, Optional, List, Union
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd


class ModelReport:
    """Generate comprehensive model evaluation reports.
    
    Creates detailed reports with metrics, visualizations, and insights
    that can be exported to HTML or PDF formats.
    
    Examples:
        >>> from mkyz.evaluation import ModelReport
        >>> report = ModelReport(model, X_test, y_test, task='classification')
        >>> report.generate()
        >>> report.export_html('reports/model_report.html')
    """
    
    def __init__(self,
                 model: Any,
                 X_test: Union[pd.DataFrame, np.ndarray],
                 y_test: Union[pd.Series, np.ndarray],
                 task: str = 'classification',
                 model_name: Optional[str] = None,
                 feature_names: Optional[List[str]] = None):
        """Initialize ModelReport.
        
        Args:
            model: Trained model to evaluate
            X_test: Test features
            y_test: True test labels
            task: Task type ('classification', 'regression', 'clustering')
            model_name: Optional name for the model
            feature_names: Optional list of feature names
        """
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.task = task
        self.model_name = model_name or type(model).__name__
        self.feature_names = feature_names
        
        self._metrics = None
        self._predictions = None
        self._report_data = {}
        self._generated = False
    
    def generate(self,
                include_feature_importance: bool = True,
                include_confusion_matrix: bool = True,
                include_roc_curve: bool = True) -> 'ModelReport':
        """Generate the report data.
        
        Args:
            include_feature_importance: Include feature importance analysis
            include_confusion_matrix: Include confusion matrix (classification)
            include_roc_curve: Include ROC curve (classification)
            
        Returns:
            self for method chaining
        """
        from .metrics import get_all_metrics, get_confusion_matrix, get_classification_report
        
        # Get predictions
        self._predictions = self.model.predict(self.X_test)
        
        # Get probabilities if available
        y_proba = None
        if hasattr(self.model, 'predict_proba'):
            try:
                y_proba = self.model.predict_proba(self.X_test)
            except:
                pass
        
        # Calculate metrics
        if self.task == 'classification':
            self._metrics = get_all_metrics(
                self.task, self.y_test, self._predictions, y_proba
            )
            
            if include_confusion_matrix:
                self._report_data['confusion_matrix'] = get_confusion_matrix(
                    self.y_test, self._predictions
                )
            
            self._report_data['classification_report'] = get_classification_report(
                self.y_test, self._predictions, as_dict=True
            )
            
        elif self.task == 'regression':
            self._metrics = get_all_metrics(
                self.task, self.y_test, self._predictions
            )
            
            # Add residuals
            self._report_data['residuals'] = self.y_test - self._predictions
            
        elif self.task == 'clustering':
            self._metrics = get_all_metrics(
                self.task, X=self.X_test, labels=self._predictions
            )
        
        # Feature importance
        if include_feature_importance:
            self._report_data['feature_importance'] = self._get_feature_importance()
        
        # Add metadata
        self._report_data['metadata'] = {
            'model_name': self.model_name,
            'task': self.task,
            'n_samples': len(self.y_test),
            'n_features': self.X_test.shape[1],
            'generated_at': datetime.now().isoformat()
        }
        
        self._generated = True
        return self
    
    def _get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Extract feature importance from model if available."""
        importance = None
        
        # Try different attribute names
        for attr in ['feature_importances_', 'coef_', 'feature_importance']:
            if hasattr(self.model, attr):
                importance = getattr(self.model, attr)
                break
        
        if importance is None:
            return None
        
        # Flatten if needed (e.g., for linear models with coef_)
        if importance.ndim > 1:
            importance = np.abs(importance).mean(axis=0)
        else:
            importance = np.abs(importance)
        
        # Create dictionary with feature names
        if self.feature_names:
            names = self.feature_names
        elif hasattr(self.X_test, 'columns'):
            names = list(self.X_test.columns)
        else:
            names = [f'feature_{i}' for i in range(len(importance))]
        
        return dict(sorted(
            zip(names, importance),
            key=lambda x: x[1],
            reverse=True
        ))
    
    @property
    def metrics(self) -> Dict[str, float]:
        """Get calculated metrics."""
        if not self._generated:
            self.generate()
        return self._metrics
    
    @property
    def predictions(self) -> np.ndarray:
        """Get model predictions."""
        if not self._generated:
            self.generate()
        return self._predictions
    
    def summary(self) -> str:
        """Get text summary of the report.
        
        Returns:
            Formatted string summary
        """
        if not self._generated:
            self.generate()
        
        lines = [
            f"=" * 60,
            f"Model Report: {self.model_name}",
            f"=" * 60,
            f"Task: {self.task.title()}",
            f"Samples: {len(self.y_test)}",
            f"Features: {self.X_test.shape[1]}",
            "",
            "Metrics:",
            "-" * 40
        ]
        
        for metric, value in self._metrics.items():
            if value is not None:
                if isinstance(value, float):
                    lines.append(f"  {metric}: {value:.4f}")
                else:
                    lines.append(f"  {metric}: {value}")
        
        if 'feature_importance' in self._report_data and self._report_data['feature_importance']:
            lines.extend([
                "",
                "Top 10 Features:",
                "-" * 40
            ])
            for i, (feat, imp) in enumerate(list(self._report_data['feature_importance'].items())[:10]):
                lines.append(f"  {i+1}. {feat}: {imp:.4f}")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary.
        
        Returns:
            Complete report as dictionary
        """
        if not self._generated:
            self.generate()
        
        return {
            'metrics': self._metrics,
            'predictions': self._predictions.tolist(),
            **self._report_data
        }
    
    def export_html(self, path: Union[str, Path], 
                   include_plots: bool = True) -> str:
        """Export report as HTML file.
        
        Args:
            path: Output file path
            include_plots: Whether to embed plots
            
        Returns:
            Absolute path to saved file
        """
        if not self._generated:
            self.generate()
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        html_content = self._generate_html(include_plots)
        
        with open(path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return str(path.absolute())
    
    def _generate_html(self, include_plots: bool = True) -> str:
        """Generate HTML content for the report."""
        metadata = self._report_data.get('metadata', {})
        
        # Build metrics table
        metrics_rows = ""
        for metric, value in self._metrics.items():
            if value is not None:
                formatted = f"{value:.4f}" if isinstance(value, float) else str(value)
                metrics_rows += f"<tr><td>{metric}</td><td>{formatted}</td></tr>\n"
        
        # Build feature importance table
        fi_rows = ""
        if 'feature_importance' in self._report_data and self._report_data['feature_importance']:
            for feat, imp in list(self._report_data['feature_importance'].items())[:15]:
                fi_rows += f"<tr><td>{feat}</td><td>{imp:.4f}</td></tr>\n"
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Model Report: {self.model_name}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 40px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 3px solid #007bff; padding-bottom: 10px; }}
        h2 {{ color: #666; margin-top: 30px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #007bff; color: white; }}
        tr:hover {{ background: #f5f5f5; }}
        .metadata {{ color: #888; font-size: 14px; }}
        .metric-value {{ font-weight: bold; color: #007bff; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Model Report: {self.model_name}</h1>
        <p class="metadata">
            Task: {self.task.title()} | 
            Samples: {metadata.get('n_samples', 'N/A')} | 
            Features: {metadata.get('n_features', 'N/A')} |
            Generated: {metadata.get('generated_at', 'N/A')}
        </p>
        
        <h2>Performance Metrics</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            {metrics_rows}
        </table>
        
        {f'''
        <h2>Feature Importance (Top 15)</h2>
        <table>
            <tr><th>Feature</th><th>Importance</th></tr>
            {fi_rows}
        </table>
        ''' if fi_rows else ''}
        
        <footer style="margin-top: 40px; color: #888; font-size: 12px;">
            Generated by MKYZ Library
        </footer>
    </div>
</body>
</html>
        """
        
        return html
    
    def __repr__(self) -> str:
        status = "generated" if self._generated else "not generated"
        return f"ModelReport(model={self.model_name}, task={self.task}, status={status})"
