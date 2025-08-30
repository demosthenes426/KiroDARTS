"""
Unit tests for ResultsVisualizer class.

Tests the visualization functionality including prediction plots, comparison charts,
and feature displays.
"""

import unittest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
import warnings
import sys
import os
import tempfile
import shutil

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from results_visualizer import ResultsVisualizer, VisualizationResults, VisualizationError
from model_evaluator import EvaluationResults


class TestResultsVisualizer(unittest.TestCase):
    """Test cases for ResultsVisualizer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for test plots
        self.temp_dir = tempfile.mkdtemp()
        
        self.visualizer = ResultsVisualizer(
            save_plots=True,
            output_dir=self.temp_dir,
            verbose=False
        )
        
        # Create test evaluation results
        self.test_results = {
            'Model1': EvaluationResults(
                model_name='Model1',
                predictions=np.array([100.1, 101.2, 102.1, 103.3, 104.2]),
                actuals=np.array([100.0, 101.0, 102.0, 103.0, 104.0]),
                mae=0.2,
                rmse=0.25,
                mape=0.2,
                prediction_length=5,
                evaluation_time=1.0,
                performance_degraded=False
            ),
            'Model2': EvaluationResults(
                model_name='Model2',
                predictions=np.array([100.5, 101.8, 102.5, 103.8, 104.5]),
                actuals=np.array([100.0, 101.0, 102.0, 103.0, 104.0]),
                mae=0.5,
                rmse=0.6,
                mape=0.5,
                prediction_length=5,
                evaluation_time=1.2,
                performance_degraded=False
            )
        }
        
        # Create test feature data
        self.test_features = pd.DataFrame({
            'feature1': np.random.randn(10),
            'feature2': np.random.randn(10),
            'feature3': np.random.randn(10)
        })
        
        # Suppress warnings during tests
        warnings.filterwarnings("ignore")
    
    def tearDown(self):
        """Clean up after tests."""
        warnings.resetwarnings()
        # Clean up temporary directory
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_visualizer_initialization(self):
        """Test ResultsVisualizer initialization."""
        visualizer = ResultsVisualizer(
            figure_size=(10, 6),
            style='default',
            save_plots=False,
            output_dir='test_plots',
            verbose=True
        )
        
        self.assertEqual(visualizer.figure_size, (10, 6))
        self.assertEqual(visualizer.style, 'default')
        self.assertFalse(visualizer.save_plots)
        self.assertEqual(visualizer.output_dir, 'test_plots')
        self.assertTrue(visualizer.verbose)
    
    @patch('results_visualizer.MATPLOTLIB_AVAILABLE', False)
    def test_visualize_results_matplotlib_unavailable(self):
        """Test visualization when matplotlib is not available."""
        with self.assertRaises(VisualizationError) as context:
            self.visualizer.visualize_results(self.test_results)
        
        self.assertIn("Matplotlib is not available", str(context.exception))
    
    def test_visualize_results_empty_results(self):
        """Test visualization with empty results."""
        with self.assertRaises(VisualizationError) as context:
            self.visualizer.visualize_results({})
        
        self.assertIn("No evaluation results provided", str(context.exception))
    
    @patch('results_visualizer.MATPLOTLIB_AVAILABLE', True)
    @patch('results_visualizer.plt')
    def test_visualize_results_success(self, mock_plt):
        """Test successful visualization creation."""
        # Mock matplotlib components
        mock_fig = MagicMock()
        mock_axes = [MagicMock(), MagicMock()]
        mock_plt.subplots.return_value = (mock_fig, mock_axes)
        
        results = self.visualizer.visualize_results(self.test_results)
        
        self.assertEqual(len(results), 2)
        self.assertIn('Model1', results)
        self.assertIn('Model2', results)
        
        # Verify all results are VisualizationResults objects
        for result in results.values():
            self.assertIsInstance(result, VisualizationResults)
    
    @patch('results_visualizer.MATPLOTLIB_AVAILABLE', True)
    @patch('results_visualizer.plt')
    def test_create_prediction_plot(self, mock_plt):
        """Test prediction plot creation."""
        # Mock matplotlib components
        mock_fig = MagicMock()
        mock_axes = [MagicMock(), MagicMock()]
        mock_plt.subplots.return_value = (mock_fig, mock_axes)
        
        predictions = np.array([100.1, 101.2, 102.1])
        actuals = np.array([100.0, 101.0, 102.0])
        time_index = pd.date_range('2023-01-01', periods=3, freq='D')
        
        fig = self.visualizer._create_prediction_plot(
            'TestModel', predictions, actuals, time_index
        )
        
        # Should return the mocked figure
        self.assertEqual(fig, mock_fig)
        
        # Verify subplots was called
        mock_plt.subplots.assert_called()
    
    @patch('results_visualizer.MATPLOTLIB_AVAILABLE', True)
    @patch('results_visualizer.plt')
    def test_create_metrics_comparison_plot(self, mock_plt):
        """Test metrics comparison plot creation."""
        # Mock matplotlib components
        mock_fig = MagicMock()
        mock_axes = [MagicMock(), MagicMock(), MagicMock()]
        mock_plt.subplots.return_value = (mock_fig, mock_axes)
        
        fig = self.visualizer._create_metrics_comparison_plot(self.test_results)
        
        # Should return the mocked figure
        self.assertEqual(fig, mock_fig)
        
        # Verify subplots was called with correct parameters
        mock_plt.subplots.assert_called_with(1, 3, figsize=(15, 5))
    
    @patch('results_visualizer.MATPLOTLIB_AVAILABLE', True)
    @patch('results_visualizer.plt')
    def test_create_performance_scatter_plot(self, mock_plt):
        """Test performance scatter plot creation."""
        # Mock matplotlib components
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        
        fig = self.visualizer._create_performance_scatter_plot(self.test_results)
        
        # Should return the mocked figure
        self.assertEqual(fig, mock_fig)
        
        # Verify scatter plot was called
        mock_ax.scatter.assert_called()
    
    @patch('results_visualizer.MATPLOTLIB_AVAILABLE', True)
    @patch('results_visualizer.plt')
    def test_create_head_tail_plot(self, mock_plt):
        """Test head/tail feature plot creation."""
        # Mock matplotlib components
        mock_fig = MagicMock()
        mock_axes = [MagicMock(), MagicMock()]
        mock_plt.subplots.return_value = (mock_fig, mock_axes)
        mock_plt.tight_layout = MagicMock()
        
        # Mock the DataFrame plot method
        with patch.object(pd.DataFrame, 'plot') as mock_plot:
            fig = self.visualizer._create_head_tail_plot(self.test_features)
            
            # Should return the mocked figure
            self.assertEqual(fig, mock_fig)
            
            # Verify subplots was called
            mock_plt.subplots.assert_called()
            
            # Verify plot was called on head and tail data
            self.assertEqual(mock_plot.call_count, 2)
    
    @patch('results_visualizer.SEABORN_AVAILABLE', True)
    @patch('results_visualizer.MATPLOTLIB_AVAILABLE', True)
    @patch('results_visualizer.sns')
    @patch('results_visualizer.plt')
    def test_create_correlation_heatmap(self, mock_plt, mock_sns):
        """Test correlation heatmap creation."""
        # Mock matplotlib components
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        
        fig = self.visualizer._create_correlation_heatmap(self.test_features)
        
        # Should return the mocked figure
        self.assertEqual(fig, mock_fig)
        
        # Verify heatmap was called
        mock_sns.heatmap.assert_called()
    
    def test_create_output_directory(self):
        """Test output directory creation."""
        test_dir = os.path.join(self.temp_dir, 'test_output')
        visualizer = ResultsVisualizer(output_dir=test_dir, save_plots=True, verbose=False)
        
        # Directory should be created
        self.assertTrue(os.path.exists(test_dir))
    
    def test_create_output_directory_failure(self):
        """Test handling of output directory creation failure."""
        # Mock os.makedirs to raise an exception
        with patch('results_visualizer.os.makedirs') as mock_makedirs:
            mock_makedirs.side_effect = Exception("Permission denied")
            
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                visualizer = ResultsVisualizer(
                    output_dir='invalid_dir', 
                    save_plots=True, 
                    verbose=False
                )
                
                # Should disable save_plots and warn
                self.assertFalse(visualizer.save_plots)
                self.assertTrue(len(w) > 0)
    
    @patch('results_visualizer.MATPLOTLIB_AVAILABLE', True)
    @patch('results_visualizer.plt')
    def test_plot_creation_failure_handling(self, mock_plt):
        """Test handling of plot creation failures."""
        # Make subplots raise an exception
        mock_plt.subplots.side_effect = Exception("Plot creation failed")
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fig = self.visualizer._create_prediction_plot(
                'TestModel', 
                np.array([1, 2, 3]), 
                np.array([1.1, 2.1, 3.1])
            )
            
            # Should return None and warn
            self.assertIsNone(fig)
            self.assertTrue(len(w) > 0)
    
    def test_visualization_summary_empty(self):
        """Test visualization summary with no history."""
        summary = self.visualizer.get_visualization_summary()
        self.assertIn("No visualization history available", summary["message"])
    
    def test_visualization_summary_with_history(self):
        """Test visualization summary with history."""
        # Add mock visualization history
        self.visualizer.visualization_history = {
            'Model1': VisualizationResults(
                model_name='Model1',
                prediction_plot=None,
                feature_summary_plot=None,
                comparison_plot=None,
                plot_paths=['plot1.png', 'plot2.png'],
                visualization_time=1.0
            )
        }
        
        summary = self.visualizer.get_visualization_summary()
        
        self.assertEqual(summary['total_visualizations'], 1)
        self.assertEqual(summary['models_visualized'], ['Model1'])
        self.assertTrue(summary['plots_saved'])
        self.assertEqual(summary['total_plots_created'], 2)
    
    @patch('results_visualizer.MATPLOTLIB_AVAILABLE', True)
    @patch('results_visualizer.plt')
    def test_show_plots(self, mock_plt):
        """Test show plots functionality."""
        self.visualizer.show_plots()
        mock_plt.show.assert_called_once()
    
    @patch('results_visualizer.MATPLOTLIB_AVAILABLE', True)
    @patch('results_visualizer.plt')
    def test_close_all_plots(self, mock_plt):
        """Test close all plots functionality."""
        self.visualizer.close_all_plots()
        mock_plt.close.assert_called_once_with('all')
    
    def test_add_value_labels(self):
        """Test adding value labels to bars."""
        # Create mock axes and bars
        mock_ax = MagicMock()
        mock_bar = MagicMock()
        mock_bar.get_height.return_value = 1.5
        mock_bar.get_x.return_value = 0.0
        mock_bar.get_width.return_value = 0.8
        
        bars = [mock_bar]
        
        self.visualizer._add_value_labels(mock_ax, bars)
        
        # Verify text was added
        mock_ax.text.assert_called()
    
    @patch('results_visualizer.MATPLOTLIB_AVAILABLE', True)
    @patch('results_visualizer.plt')
    def test_setup_plotting_style(self, mock_plt):
        """Test plotting style setup."""
        # Reset mock call count
        mock_plt.style.use.reset_mock()
        
        # Test with valid style
        visualizer = ResultsVisualizer(style='default', verbose=False)
        mock_plt.style.use.assert_called_with('default')
        
        # Reset and test with invalid style (should fallback)
        mock_plt.style.use.reset_mock()
        mock_plt.style.use.side_effect = [Exception("Invalid style"), None]
        visualizer = ResultsVisualizer(style='invalid_style', verbose=False)
        
        # Should try the invalid style, then fallback to default
        self.assertEqual(mock_plt.style.use.call_count, 2)
    
    def test_edge_cases(self):
        """Test various edge cases."""
        # Test with single model
        single_result = {'Model1': self.test_results['Model1']}
        
        with patch('results_visualizer.MATPLOTLIB_AVAILABLE', True), \
             patch('results_visualizer.plt') as mock_plt:
            mock_fig = MagicMock()
            mock_axes = [MagicMock(), MagicMock()]
            mock_plt.subplots.return_value = (mock_fig, mock_axes)
            
            results = self.visualizer.visualize_results(single_result)
            self.assertEqual(len(results), 1)
        
        # Test with empty feature data
        empty_features = pd.DataFrame()
        plots = self.visualizer._create_feature_summary_plots(empty_features)
        # Should handle gracefully (may return empty list or plots with warnings)
        self.assertIsInstance(plots, list)
    
    def test_time_index_handling(self):
        """Test handling of different time index scenarios."""
        predictions = np.array([100, 101, 102])
        actuals = np.array([100.1, 101.1, 102.1])
        
        with patch('results_visualizer.MATPLOTLIB_AVAILABLE', True), \
             patch('results_visualizer.plt') as mock_plt:
            mock_fig = MagicMock()
            mock_axes = [MagicMock(), MagicMock()]
            mock_plt.subplots.return_value = (mock_fig, mock_axes)
            
            # Test with None time index (should create default)
            fig1 = self.visualizer._create_prediction_plot(
                'TestModel', predictions, actuals, None
            )
            self.assertEqual(fig1, mock_fig)
            
            # Test with provided time index
            time_index = pd.date_range('2023-01-01', periods=5, freq='D')  # Longer than predictions
            fig2 = self.visualizer._create_prediction_plot(
                'TestModel', predictions, actuals, time_index
            )
            self.assertEqual(fig2, mock_fig)
    
    def test_metrics_calculation_in_plots(self):
        """Test that metrics are correctly calculated in plots."""
        predictions = np.array([100, 101, 102])
        actuals = np.array([100.1, 101.1, 102.1])
        
        with patch('results_visualizer.MATPLOTLIB_AVAILABLE', True), \
             patch('results_visualizer.plt') as mock_plt:
            mock_fig = MagicMock()
            mock_axes = [MagicMock(), MagicMock()]
            mock_plt.subplots.return_value = (mock_fig, mock_axes)
            
            fig = self.visualizer._create_prediction_plot(
                'TestModel', predictions, actuals
            )
            
            # Verify that text was added to the plot (metrics box)
            mock_axes[0].text.assert_called()


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)