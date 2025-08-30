"""
ResultsVisualizer class for creating charts and graphs of model results.

This module provides visualization functionality for DARTS model evaluation results
including prediction vs actual plots, performance comparison charts, and feature displays.
"""

from typing import Dict, Any, List, Optional, Tuple
import warnings
import numpy as np
import pandas as pd
from dataclasses import dataclass

# Try to import plotting libraries, handle gracefully if not available
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Matplotlib not available: {e}")
    plt = None
    Figure = Any
    MATPLOTLIB_AVAILABLE = False

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    sns = None
    SEABORN_AVAILABLE = False

# Try to import DARTS components for type hints
try:
    from darts import TimeSeries
    DARTS_AVAILABLE = True
except ImportError:
    TimeSeries = Any
    DARTS_AVAILABLE = False

# Import evaluation results
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from model_evaluator import EvaluationResults


@dataclass
class VisualizationResults:
    """Data class to store visualization results and metadata."""
    model_name: str
    prediction_plot: Optional[Figure]
    feature_summary_plot: Optional[Figure]
    comparison_plot: Optional[Figure]
    plot_paths: List[str]
    visualization_time: float


class VisualizationError(Exception):
    """Custom exception for visualization errors."""
    pass


class ResultsVisualizer:
    """
    Visualizer class for DARTS model evaluation results.
    
    Provides comprehensive visualization functionality including prediction vs actual plots,
    performance comparison charts, and feature column displays.
    """
    
    def __init__(self, 
                 figure_size: Tuple[int, int] = (12, 8),
                 style: str = 'seaborn-v0_8',
                 save_plots: bool = True,
                 output_dir: str = 'plots',
                 verbose: bool = True):
        """
        Initialize ResultsVisualizer with configuration.
        
        Args:
            figure_size (Tuple[int, int]): Default figure size for plots
            style (str): Matplotlib style to use
            save_plots (bool): Whether to save plots to files
            output_dir (str): Directory to save plots
            verbose (bool): Whether to print visualization progress
        """
        self.figure_size = figure_size
        self.style = style
        self.save_plots = save_plots
        self.output_dir = output_dir
        self.verbose = verbose
        
        # Visualization history
        self.visualization_history = {}
        
        # Set up plotting style
        self._setup_plotting_style()
        
        # Create output directory if needed
        if self.save_plots:
            self._create_output_directory()
    
    def visualize_results(self, 
                         results: Dict[str, EvaluationResults],
                         feature_data: Optional[pd.DataFrame] = None,
                         time_index: Optional[pd.DatetimeIndex] = None) -> Dict[str, VisualizationResults]:
        """
        Create comprehensive visualizations for model evaluation results.
        
        Args:
            results: Dictionary of evaluation results for each model
            feature_data: Optional DataFrame with feature columns for display
            time_index: Optional datetime index for time series plots
            
        Returns:
            Dict[str, VisualizationResults]: Visualization results for each model
            
        Raises:
            VisualizationError: If visualization fails
        """
        if not MATPLOTLIB_AVAILABLE:
            raise VisualizationError("Matplotlib is not available. Please install matplotlib.")
        
        if not results:
            raise VisualizationError("No evaluation results provided for visualization.")
        
        if self.verbose:
            print(f"\n📊 Creating visualizations for {len(results)} models...")
        
        try:
            import time
            start_time = time.time()
            
            visualization_results = {}
            
            # Create individual model visualizations
            for model_name, eval_result in results.items():
                viz_result = self._create_model_visualization(
                    model_name, eval_result, time_index
                )
                visualization_results[model_name] = viz_result
            
            # Create comparison visualizations
            comparison_plots = self._create_comparison_plots(results)
            
            # Create feature summary if provided
            if feature_data is not None:
                feature_plots = self._create_feature_summary_plots(feature_data)
            
            total_time = time.time() - start_time
            
            if self.verbose:
                print(f"   ✓ Visualizations created in {total_time:.2f}s")
                if self.save_plots:
                    print(f"   ✓ Plots saved to {self.output_dir}/")
            
            return visualization_results
            
        except Exception as e:
            error_msg = f"Failed to create visualizations: {str(e)}"
            if self.verbose:
                print(f"   ❌ {error_msg}")
            raise VisualizationError(error_msg) from e
    
    def _create_model_visualization(self, 
                                  model_name: str, 
                                  eval_result: EvaluationResults,
                                  time_index: Optional[pd.DatetimeIndex] = None) -> VisualizationResults:
        """
        Create visualization for a single model's results.
        
        Args:
            model_name: Name of the model
            eval_result: Evaluation results for the model
            time_index: Optional datetime index for plots
            
        Returns:
            VisualizationResults: Visualization results
        """
        import time
        start_time = time.time()
        
        plot_paths = []
        
        # Create prediction vs actual plot
        prediction_plot = self._create_prediction_plot(
            model_name, eval_result.predictions, eval_result.actuals, time_index
        )
        
        if self.save_plots and prediction_plot:
            plot_path = os.path.join(self.output_dir, f"{model_name}_predictions.png")
            prediction_plot.savefig(plot_path, dpi=300, bbox_inches='tight')
            plot_paths.append(plot_path)
        
        visualization_time = time.time() - start_time
        
        return VisualizationResults(
            model_name=model_name,
            prediction_plot=prediction_plot,
            feature_summary_plot=None,
            comparison_plot=None,
            plot_paths=plot_paths,
            visualization_time=visualization_time
        )
    
    def _create_prediction_plot(self, 
                              model_name: str,
                              predictions: np.ndarray,
                              actuals: np.ndarray,
                              time_index: Optional[pd.DatetimeIndex] = None) -> Optional[Figure]:
        """
        Create prediction vs actual plot for a model.
        
        Args:
            model_name: Name of the model
            predictions: Model predictions
            actuals: Actual values
            time_index: Optional datetime index
            
        Returns:
            Optional[Figure]: Matplotlib figure or None if creation fails
        """
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figure_size, 
                                          gridspec_kw={'height_ratios': [3, 1]})
            
            # Create time index if not provided
            if time_index is None:
                time_index = pd.date_range('2023-01-01', periods=len(predictions), freq='D')
            else:
                # Ensure we have the right length
                time_index = time_index[:len(predictions)]
            
            # Main prediction plot
            ax1.plot(time_index, actuals, label='Actual', color='blue', linewidth=2, marker='o')
            ax1.plot(time_index, predictions, label='Predicted', color='red', linewidth=2, marker='s')
            ax1.set_title(f'{model_name} - Predictions vs Actual Values', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Value', fontsize=12)
            ax1.legend(fontsize=11)
            ax1.grid(True, alpha=0.3)
            
            # Format x-axis for dates
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax1.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(predictions)//5)))
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
            
            # Residuals plot
            residuals = actuals - predictions
            ax2.plot(time_index, residuals, color='green', linewidth=1, marker='x')
            ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax2.set_title('Residuals (Actual - Predicted)', fontsize=12)
            ax2.set_xlabel('Date', fontsize=12)
            ax2.set_ylabel('Residual', fontsize=12)
            ax2.grid(True, alpha=0.3)
            
            # Format x-axis for dates
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax2.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(predictions)//5)))
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
            
            # Add metrics text box
            mae = np.mean(np.abs(residuals))
            rmse = np.sqrt(np.mean(residuals**2))
            mape = np.mean(np.abs(residuals / actuals)) * 100 if np.all(actuals != 0) else float('inf')
            
            metrics_text = f'MAE: {mae:.4f}\nRMSE: {rmse:.4f}\nMAPE: {mape:.2f}%'
            ax1.text(0.02, 0.98, metrics_text, transform=ax1.transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            warnings.warn(f"Failed to create prediction plot for {model_name}: {e}")
            return None
    
    def _create_comparison_plots(self, results: Dict[str, EvaluationResults]) -> List[Figure]:
        """
        Create comparison plots across multiple models.
        
        Args:
            results: Dictionary of evaluation results
            
        Returns:
            List[Figure]: List of comparison figures
        """
        comparison_plots = []
        
        try:
            # Metrics comparison bar chart
            metrics_fig = self._create_metrics_comparison_plot(results)
            if metrics_fig:
                comparison_plots.append(metrics_fig)
                
                if self.save_plots:
                    plot_path = os.path.join(self.output_dir, "model_comparison_metrics.png")
                    metrics_fig.savefig(plot_path, dpi=300, bbox_inches='tight')
            
            # Performance scatter plot
            scatter_fig = self._create_performance_scatter_plot(results)
            if scatter_fig:
                comparison_plots.append(scatter_fig)
                
                if self.save_plots:
                    plot_path = os.path.join(self.output_dir, "model_performance_scatter.png")
                    scatter_fig.savefig(plot_path, dpi=300, bbox_inches='tight')
            
        except Exception as e:
            warnings.warn(f"Failed to create comparison plots: {e}")
        
        return comparison_plots
    
    def _create_metrics_comparison_plot(self, results: Dict[str, EvaluationResults]) -> Optional[Figure]:
        """
        Create bar chart comparing metrics across models.
        
        Args:
            results: Dictionary of evaluation results
            
        Returns:
            Optional[Figure]: Matplotlib figure or None if creation fails
        """
        try:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            model_names = list(results.keys())
            mae_values = [results[name].mae for name in model_names]
            rmse_values = [results[name].rmse for name in model_names]
            mape_values = [results[name].mape for name in model_names]
            
            # MAE comparison
            bars1 = axes[0].bar(model_names, mae_values, color='skyblue', alpha=0.7)
            axes[0].set_title('Mean Absolute Error (MAE)', fontweight='bold')
            axes[0].set_ylabel('MAE')
            axes[0].tick_params(axis='x', rotation=45)
            self._add_value_labels(axes[0], bars1)
            
            # RMSE comparison
            bars2 = axes[1].bar(model_names, rmse_values, color='lightcoral', alpha=0.7)
            axes[1].set_title('Root Mean Square Error (RMSE)', fontweight='bold')
            axes[1].set_ylabel('RMSE')
            axes[1].tick_params(axis='x', rotation=45)
            self._add_value_labels(axes[1], bars2)
            
            # MAPE comparison
            bars3 = axes[2].bar(model_names, mape_values, color='lightgreen', alpha=0.7)
            axes[2].set_title('Mean Absolute Percentage Error (MAPE)', fontweight='bold')
            axes[2].set_ylabel('MAPE (%)')
            axes[2].tick_params(axis='x', rotation=45)
            self._add_value_labels(axes[2], bars3)
            
            plt.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
            plt.tight_layout()
            return fig
            
        except Exception as e:
            warnings.warn(f"Failed to create metrics comparison plot: {e}")
            return None
    
    def _create_performance_scatter_plot(self, results: Dict[str, EvaluationResults]) -> Optional[Figure]:
        """
        Create scatter plot of model performance metrics.
        
        Args:
            results: Dictionary of evaluation results
            
        Returns:
            Optional[Figure]: Matplotlib figure or None if creation fails
        """
        try:
            fig, ax = plt.subplots(figsize=self.figure_size)
            
            mae_values = [results[name].mae for name in results.keys()]
            rmse_values = [results[name].rmse for name in results.keys()]
            model_names = list(results.keys())
            
            # Create scatter plot
            scatter = ax.scatter(mae_values, rmse_values, s=100, alpha=0.7, c=range(len(model_names)), cmap='viridis')
            
            # Add model name labels
            for i, name in enumerate(model_names):
                ax.annotate(name, (mae_values[i], rmse_values[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=10)
            
            ax.set_xlabel('Mean Absolute Error (MAE)', fontsize=12)
            ax.set_ylabel('Root Mean Square Error (RMSE)', fontsize=12)
            ax.set_title('Model Performance: MAE vs RMSE', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Add diagonal reference line
            min_val = min(min(mae_values), min(rmse_values))
            max_val = max(max(mae_values), max(rmse_values))
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='MAE = RMSE')
            ax.legend()
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            warnings.warn(f"Failed to create performance scatter plot: {e}")
            return None
    
    def _create_feature_summary_plots(self, feature_data: pd.DataFrame) -> List[Figure]:
        """
        Create plots showing head and tail of feature columns.
        
        Args:
            feature_data: DataFrame with feature columns
            
        Returns:
            List[Figure]: List of feature summary figures
        """
        feature_plots = []
        
        try:
            # Display head and tail of features
            head_tail_fig = self._create_head_tail_plot(feature_data)
            if head_tail_fig:
                feature_plots.append(head_tail_fig)
                
                if self.save_plots:
                    plot_path = os.path.join(self.output_dir, "feature_head_tail.png")
                    head_tail_fig.savefig(plot_path, dpi=300, bbox_inches='tight')
            
            # Feature correlation heatmap if seaborn is available
            if SEABORN_AVAILABLE and len(feature_data.columns) > 1:
                corr_fig = self._create_correlation_heatmap(feature_data)
                if corr_fig:
                    feature_plots.append(corr_fig)
                    
                    if self.save_plots:
                        plot_path = os.path.join(self.output_dir, "feature_correlation.png")
                        corr_fig.savefig(plot_path, dpi=300, bbox_inches='tight')
            
        except Exception as e:
            warnings.warn(f"Failed to create feature summary plots: {e}")
        
        return feature_plots
    
    def _create_head_tail_plot(self, feature_data: pd.DataFrame) -> Optional[Figure]:
        """
        Create plot showing head and tail of feature data.
        
        Args:
            feature_data: DataFrame with feature columns
            
        Returns:
            Optional[Figure]: Matplotlib figure or None if creation fails
        """
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figure_size)
            
            # Show first 5 rows
            head_data = feature_data.head()
            head_data.plot(kind='line', ax=ax1, marker='o')
            ax1.set_title('Feature Data - First 5 Rows', fontweight='bold')
            ax1.set_ylabel('Value')
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax1.grid(True, alpha=0.3)
            
            # Show last 5 rows
            tail_data = feature_data.tail()
            tail_data.plot(kind='line', ax=ax2, marker='s')
            ax2.set_title('Feature Data - Last 5 Rows', fontweight='bold')
            ax2.set_xlabel('Index')
            ax2.set_ylabel('Value')
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            warnings.warn(f"Failed to create head/tail plot: {e}")
            return None
    
    def _create_correlation_heatmap(self, feature_data: pd.DataFrame) -> Optional[Figure]:
        """
        Create correlation heatmap of features.
        
        Args:
            feature_data: DataFrame with feature columns
            
        Returns:
            Optional[Figure]: Matplotlib figure or None if creation fails
        """
        try:
            fig, ax = plt.subplots(figsize=self.figure_size)
            
            # Calculate correlation matrix
            corr_matrix = feature_data.corr()
            
            # Create heatmap
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                       square=True, ax=ax, cbar_kws={'shrink': 0.8})
            
            ax.set_title('Feature Correlation Matrix', fontweight='bold')
            plt.tight_layout()
            return fig
            
        except Exception as e:
            warnings.warn(f"Failed to create correlation heatmap: {e}")
            return None
    
    def _add_value_labels(self, ax, bars):
        """Add value labels on top of bars."""
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}', ha='center', va='bottom', fontsize=9)
    
    def _setup_plotting_style(self):
        """Set up matplotlib plotting style."""
        if MATPLOTLIB_AVAILABLE:
            try:
                plt.style.use(self.style)
            except Exception:
                # Fallback to default style
                try:
                    plt.style.use('default')
                except Exception:
                    pass  # Use whatever style is available
    
    def _create_output_directory(self):
        """Create output directory for saving plots."""
        try:
            os.makedirs(self.output_dir, exist_ok=True)
        except Exception as e:
            warnings.warn(f"Failed to create output directory {self.output_dir}: {e}")
            self.save_plots = False
    
    def show_plots(self):
        """Display all created plots."""
        if MATPLOTLIB_AVAILABLE:
            plt.show()
    
    def close_all_plots(self):
        """Close all matplotlib figures to free memory."""
        if MATPLOTLIB_AVAILABLE:
            plt.close('all')
    
    def get_visualization_summary(self) -> Dict[str, Any]:
        """
        Get summary of visualization results.
        
        Returns:
            Dict[str, Any]: Summary of visualizations created
        """
        if not self.visualization_history:
            return {"message": "No visualization history available"}
        
        summary = {
            "total_visualizations": len(self.visualization_history),
            "models_visualized": list(self.visualization_history.keys()),
            "plots_saved": self.save_plots,
            "output_directory": self.output_dir if self.save_plots else None,
            "total_plots_created": sum(
                len(viz.plot_paths) for viz in self.visualization_history.values()
            )
        }
        
        return summary