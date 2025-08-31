"""
Unit tests for ModelEvaluator class.

This module tests the ModelEvaluator class functionality including:
- Model evaluation with accuracy metrics
- Performance degradation detection
- Baseline comparison
- Multiple model evaluation
"""

import unittest
import numpy as np
import pandas as pd
import sys
import os
from unittest.mock import Mock, patch
import warnings

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Try to import dependencies
try:
    from darts import TimeSeries
    from model_evaluator import ModelEvaluator, EvaluationResults, ModelEvaluationError
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    DEPENDENCIES_AVAILABLE = False
    print(f"Warning: Dependencies not available: {e}")


@unittest.skipUnless(DEPENDENCIES_AVAILABLE, "Dependencies not available")
class TestModelEvaluator(unittest.TestCase):
    """Test cases for ModelEvaluator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create sample data
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        values = np.random.normal(100, 10, 100)
        self.test_ts = TimeSeries.from_pandas(pd.Series(values, index=dates))
        
        # Create mock model
        self.mock_model = Mock()
        self.mock_model.predict.return_value = TimeSeries.from_pandas(
            pd.Series(np.random.normal(100, 10, 5), 
                     index=pd.date_range('2023-04-11', periods=5, freq='D'))
        )
        
        # Initialize evaluator
        self.evaluator = ModelEvaluator(verbose=False)
    
    def test_init(self):
        """Test ModelEvaluator initialization."""
        evaluator = ModelEvaluator(
            performance_threshold=0.3,
            baseline_metrics={'test_model': {'mae': 5.0}},
            verbose=True
        )
        
        self.assertEqual(evaluator.performance_threshold, 0.3)
        self.assertEqual(evaluator.baseline_metrics['test_model']['mae'], 5.0)
        self.assertTrue(evaluator.verbose)
        self.assertEqual(len(evaluator.evaluation_history), 0)
    
    def test_evaluate_model(self):
        """Test single model evaluation."""
        result = self.evaluator.evaluate_model(
            self.mock_model, 
            self.test_ts, 
            "TestModel", 
            prediction_length=5
        )
        
        # Check result structure
        self.assertIsInstance(result, EvaluationResults)
        self.assertEqual(result.model_name, "TestModel")
        self.assertEqual(result.prediction_length, 5)
        self.assertEqual(len(result.predictions), 5)
        self.assertEqual(len(result.actuals), 5)
        
        # Check metrics are calculated
        self.assertIsInstance(result.mae, float)
        self.assertIsInstance(result.rmse, float)
        self.assertIsInstance(result.mape, float)
        self.assertGreaterEqual(result.mae, 0)
        self.assertGreaterEqual(result.rmse, 0)
        
        # Check evaluation history is updated
        self.assertIn("TestModel", self.evaluator.evaluation_history)
    
    def test_evaluate_multiple_models(self):
        """Test multiple model evaluation."""
        models = {
            'Model1': self.mock_model,
            'Model2': self.mock_model
        }
        
        results = self.evaluator.evaluate_multiple_models(
            models, 
            self.test_ts, 
            prediction_length=5
        )
        
        # Check results structure
        self.assertEqual(len(results), 2)
        self.assertIn('Model1', results)
        self.assertIn('Model2', results)
        
        for model_name, result in results.items():
            self.assertIsInstance(result, EvaluationResults)
            self.assertEqual(result.model_name, model_name)
    
    def test_calculate_mae(self):
        """Test MAE calculation."""
        predictions = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        actuals = np.array([1.1, 2.1, 2.9, 4.2, 4.8])
        
        mae = self.evaluator._calculate_mae(predictions, actuals)
        expected_mae = np.mean(np.abs(predictions - actuals))
        
        self.assertAlmostEqual(mae, expected_mae, places=6)
    
    def test_calculate_rmse(self):
        """Test RMSE calculation."""
        predictions = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        actuals = np.array([1.1, 2.1, 2.9, 4.2, 4.8])
        
        rmse = self.evaluator._calculate_rmse(predictions, actuals)
        expected_rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
        
        self.assertAlmostEqual(rmse, expected_rmse, places=6)
    
    def test_calculate_mape(self):
        """Test MAPE calculation."""
        predictions = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        actuals = np.array([1.1, 2.1, 2.9, 4.2, 4.8])
        
        mape = self.evaluator._calculate_mape(predictions, actuals)
        expected_mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
        
        self.assertAlmostEqual(mape, expected_mape, places=6)
    
    def test_calculate_mape_with_zeros(self):
        """Test MAPE calculation with zero values."""
        predictions = np.array([1.0, 2.0, 3.0])
        actuals = np.array([0.0, 2.1, 2.9])  # Contains zero
        
        mape = self.evaluator._calculate_mape(predictions, actuals)
        
        # Should handle zeros gracefully
        self.assertIsInstance(mape, float)
        self.assertGreaterEqual(mape, 0)
    
    def test_performance_degradation_detection(self):
        """Test performance degradation detection."""
        # Set baseline metrics
        baseline_metrics = {
            'TestModel': {
                'mae': 1.0,
                'rmse': 1.5,
                'mape': 5.0
            }
        }
        self.evaluator.set_baseline_metrics(baseline_metrics)
        
        # Test no degradation
        degraded = self.evaluator._check_performance_degradation(
            'TestModel', mae=1.1, rmse=1.6, mape=5.5
        )
        self.assertFalse(degraded)
        
        # Test degradation (25% increase, threshold is 20%)
        degraded = self.evaluator._check_performance_degradation(
            'TestModel', mae=1.3, rmse=1.9, mape=6.5
        )
        self.assertTrue(degraded)
    
    def test_baseline_comparison(self):
        """Test baseline comparison."""
        baseline_metrics = {
            'TestModel': {
                'mae': 1.0,
                'rmse': 1.5,
                'mape': 5.0
            }
        }
        self.evaluator.set_baseline_metrics(baseline_metrics)
        
        comparison = self.evaluator._compare_to_baseline(
            'TestModel', mae=1.1, rmse=1.6, mape=5.5
        )
        
        self.assertIn('mae_change', comparison)
        self.assertIn('rmse_change', comparison)
        self.assertIn('mape_change', comparison)
        
        # Check percentage changes
        self.assertAlmostEqual(comparison['mae_change'], 10.0, places=1)
        self.assertAlmostEqual(comparison['rmse_change'], 6.67, places=1)
        self.assertAlmostEqual(comparison['mape_change'], 10.0, places=1)
    
    def test_validate_evaluation_requirements(self):
        """Test evaluation requirements validation."""
        # Test valid requirements
        valid = self.evaluator.validate_evaluation_requirements(
            self.test_ts, prediction_length=5
        )
        self.assertTrue(valid)
        
        # Test insufficient data
        short_ts = self.test_ts[:3]  # Only 3 points
        with self.assertRaises(ModelEvaluationError):
            self.evaluator.validate_evaluation_requirements(
                short_ts, prediction_length=5
            )
    
    def test_get_evaluation_summary(self):
        """Test evaluation summary generation."""
        # Add some evaluation history
        result1 = EvaluationResults(
            model_name="Model1",
            predictions=np.array([1, 2, 3]),
            actuals=np.array([1.1, 2.1, 2.9]),
            mae=0.1,
            rmse=0.15,
            mape=5.0,
            prediction_length=3,
            evaluation_time=1.0,
            performance_degraded=False
        )
        
        result2 = EvaluationResults(
            model_name="Model2",
            predictions=np.array([1, 2, 3]),
            actuals=np.array([1.2, 2.2, 2.8]),
            mae=0.2,
            rmse=0.25,
            mape=8.0,
            prediction_length=3,
            evaluation_time=1.5,
            performance_degraded=True
        )
        
        self.evaluator.evaluation_history = {
            "Model1": result1,
            "Model2": result2
        }
        
        summary = self.evaluator.get_evaluation_summary()
        
        self.assertEqual(summary["total_models_evaluated"], 2)
        self.assertEqual(summary["best_model_mae"], "Model1")
        self.assertEqual(summary["best_model_rmse"], "Model1")
        self.assertEqual(summary["best_model_mape"], "Model1")
        self.assertIn("Model2", summary["degraded_models"])
        self.assertAlmostEqual(summary["average_mae"], 0.15, places=2)
    
    def test_compare_models(self):
        """Test model comparison functionality."""
        results = {
            "Model1": EvaluationResults(
                model_name="Model1",
                predictions=np.array([1, 2, 3]),
                actuals=np.array([1.1, 2.1, 2.9]),
                mae=0.1,
                rmse=0.15,
                mape=5.0,
                prediction_length=3,
                evaluation_time=1.0,
                performance_degraded=False
            ),
            "Model2": EvaluationResults(
                model_name="Model2",
                predictions=np.array([1, 2, 3]),
                actuals=np.array([1.2, 2.2, 2.8]),
                mae=0.2,
                rmse=0.25,
                mape=8.0,
                prediction_length=3,
                evaluation_time=1.5,
                performance_degraded=True
            )
        }
        
        comparison = self.evaluator.compare_models(results)
        
        self.assertEqual(comparison["model_count"], 2)
        self.assertIn("metrics_comparison", comparison)
        self.assertIn("rankings", comparison)
        self.assertIn("performance_analysis", comparison)
        
        # Check rankings (Model1 should be better)
        self.assertEqual(comparison["rankings"]["mae"][0], "Model1")
        self.assertEqual(comparison["rankings"]["rmse"][0], "Model1")
        self.assertEqual(comparison["rankings"]["mape"][0], "Model1")
        
        # Check performance analysis
        self.assertEqual(comparison["performance_analysis"]["degraded_models"], 1)
        self.assertEqual(comparison["performance_analysis"]["degradation_rate"], 50.0)
    
    def test_error_handling(self):
        """Test error handling in evaluation."""
        # Test with failing model
        failing_model = Mock()
        failing_model.predict.side_effect = Exception("Model prediction failed")
        
        with self.assertRaises(ModelEvaluationError):
            self.evaluator.evaluate_model(failing_model, self.test_ts, "FailingModel")
    
    def test_empty_results_handling(self):
        """Test handling of empty results."""
        # Test empty evaluation summary
        summary = self.evaluator.get_evaluation_summary()
        self.assertIn("message", summary)
        self.assertEqual(summary["message"], "No evaluation history available")
        
        # Test empty model comparison
        comparison = self.evaluator.compare_models({})
        self.assertIn("message", comparison)
        self.assertEqual(comparison["message"], "No results to compare")


if __name__ == '__main__':
    unittest.main()