"""
Unit tests for ModelEvaluator class.

Tests the evaluation functionality including accuracy metrics calculation,
prediction vs actual comparisons, and performance degradation detection.
"""

import unittest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
import warnings
import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from model_evaluator import ModelEvaluator, EvaluationResults, ModelEvaluationError


class MockTimeSeries:
    """Mock TimeSeries class for testing."""
    
    def __init__(self, data, index=None):
        self.data = np.array(data)
        self.index = index or pd.date_range('2023-01-01', periods=len(data), freq='D')
    
    def __len__(self):
        return len(self.data)
    
    def values(self):
        return self.data
    
    def to_numpy(self):
        return self.data
    
    def pd_dataframe(self):
        return pd.DataFrame(self.data, index=self.index, columns=['value'])
    
    def to_pandas(self):
        return self.pd_dataframe()


class MockModel:
    """Mock DARTS model for testing."""
    
    def __init__(self, predictions=None, output_chunk_length=5):
        if predictions is None:
            self.predictions = np.array([100, 101, 102, 103, 104])
        else:
            self.predictions = predictions
        self.output_chunk_length = output_chunk_length
    
    def predict(self, n, series=None):
        return MockTimeSeries(self.predictions[:n])


class TestModelEvaluator(unittest.TestCase):
    """Test cases for ModelEvaluator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.evaluator = ModelEvaluator(verbose=False)
        
        # Create test data
        self.test_data = np.array([100, 101, 102, 103, 104, 105, 106, 107, 108, 109])
        self.test_ts = MockTimeSeries(self.test_data)
        
        # Create mock model
        self.mock_model = MockModel()
        
        # Suppress warnings during tests
        warnings.filterwarnings("ignore")
    
    def tearDown(self):
        """Clean up after tests."""
        warnings.resetwarnings()
    
    def test_evaluator_initialization(self):
        """Test ModelEvaluator initialization."""
        evaluator = ModelEvaluator(
            performance_threshold=0.3,
            baseline_metrics={'test_model': {'mae': 1.0}},
            verbose=True
        )
        
        self.assertEqual(evaluator.performance_threshold, 0.3)
        self.assertEqual(evaluator.baseline_metrics['test_model']['mae'], 1.0)
        self.assertTrue(evaluator.verbose)
        self.assertEqual(len(evaluator.evaluation_history), 0)
    
    @patch('model_evaluator.DARTS_AVAILABLE', True)
    def test_evaluate_model_success(self):
        """Test successful model evaluation."""
        # Set up mock model with known predictions
        predictions = np.array([100.1, 101.2, 102.1, 103.3, 104.2])
        mock_model = MockModel(predictions=predictions)
        
        result = self.evaluator.evaluate_model(
            mock_model, 
            self.test_ts, 
            model_name="TestModel"
        )
        
        # Verify result structure
        self.assertIsInstance(result, EvaluationResults)
        self.assertEqual(result.model_name, "TestModel")
        self.assertEqual(len(result.predictions), 5)
        self.assertEqual(len(result.actuals), 5)
        
        # Verify metrics are calculated
        self.assertIsInstance(result.mae, float)
        self.assertIsInstance(result.rmse, float)
        self.assertIsInstance(result.mape, float)
        self.assertGreater(result.evaluation_time, 0)
        
        # Verify evaluation history is updated
        self.assertIn("TestModel", self.evaluator.evaluation_history)
    
    @patch('model_evaluator.DARTS_AVAILABLE', False)
    def test_evaluate_model_darts_unavailable(self):
        """Test evaluation when DARTS is not available."""
        with self.assertRaises(ModelEvaluationError) as context:
            self.evaluator.evaluate_model(self.mock_model, self.test_ts)
        
        self.assertIn("DARTS library is not available", str(context.exception))
    
    def test_calculate_mae(self):
        """Test MAE calculation."""
        predictions = np.array([100, 101, 102, 103, 104])
        actuals = np.array([100.5, 100.8, 102.2, 102.9, 104.1])
        
        mae = self.evaluator._calculate_mae(predictions, actuals)
        
        expected_mae = np.mean(np.abs(predictions - actuals))
        self.assertAlmostEqual(mae, expected_mae, places=6)
    
    def test_calculate_rmse(self):
        """Test RMSE calculation."""
        predictions = np.array([100, 101, 102, 103, 104])
        actuals = np.array([100.5, 100.8, 102.2, 102.9, 104.1])
        
        rmse = self.evaluator._calculate_rmse(predictions, actuals)
        
        expected_rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
        self.assertAlmostEqual(rmse, expected_rmse, places=6)
    
    def test_calculate_mape(self):
        """Test MAPE calculation."""
        predictions = np.array([100, 101, 102, 103, 104])
        actuals = np.array([100.5, 100.8, 102.2, 102.9, 104.1])
        
        mape = self.evaluator._calculate_mape(predictions, actuals)
        
        expected_mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
        self.assertAlmostEqual(mape, expected_mape, places=6)
    
    def test_calculate_mape_zero_actuals(self):
        """Test MAPE calculation with zero actual values."""
        predictions = np.array([1, 2, 3])
        actuals = np.array([0, 0, 0])
        
        mape = self.evaluator._calculate_mape(predictions, actuals)
        
        # Should return infinity when all actuals are zero
        self.assertEqual(mape, float('inf'))
    
    def test_performance_degradation_detection(self):
        """Test performance degradation detection."""
        # Set baseline metrics
        baseline_metrics = {
            'TestModel': {'mae': 1.0, 'rmse': 1.5, 'mape': 5.0}
        }
        evaluator = ModelEvaluator(
            performance_threshold=0.2,
            baseline_metrics=baseline_metrics,
            verbose=False
        )
        
        # Test no degradation
        degraded = evaluator._check_performance_degradation(
            'TestModel', mae=1.1, rmse=1.6, mape=5.5
        )
        self.assertFalse(degraded)
        
        # Test degradation (MAE increased by 30%)
        degraded = evaluator._check_performance_degradation(
            'TestModel', mae=1.3, rmse=1.6, mape=5.5
        )
        self.assertTrue(degraded)
    
    def test_baseline_comparison(self):
        """Test baseline comparison functionality."""
        baseline_metrics = {
            'TestModel': {'mae': 1.0, 'rmse': 1.5, 'mape': 5.0}
        }
        evaluator = ModelEvaluator(baseline_metrics=baseline_metrics, verbose=False)
        
        comparison = evaluator._compare_to_baseline(
            'TestModel', mae=1.1, rmse=1.4, mape=5.5
        )
        
        self.assertAlmostEqual(comparison['mae_change'], 10.0, places=1)
        self.assertAlmostEqual(comparison['rmse_change'], -6.67, places=1)
        self.assertAlmostEqual(comparison['mape_change'], 10.0, places=1)
    
    @patch('model_evaluator.DARTS_AVAILABLE', True)
    def test_evaluate_multiple_models(self):
        """Test evaluation of multiple models."""
        models = {
            'Model1': MockModel(predictions=np.array([100, 101, 102, 103, 104])),
            'Model2': MockModel(predictions=np.array([100.5, 101.5, 102.5, 103.5, 104.5]))
        }
        
        results = self.evaluator.evaluate_multiple_models(models, self.test_ts)
        
        self.assertEqual(len(results), 2)
        self.assertIn('Model1', results)
        self.assertIn('Model2', results)
        
        # Verify all results are EvaluationResults objects
        for result in results.values():
            self.assertIsInstance(result, EvaluationResults)
    
    def test_generate_predictions_fallback(self):
        """Test prediction generation with fallback."""
        # Create a model that will fail
        failing_model = Mock()
        failing_model.predict.side_effect = Exception("Prediction failed")
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")  # Ensure warnings are captured
            predictions = self.evaluator._generate_predictions(
                failing_model, self.test_ts, 5
            )
            
            # Should generate synthetic predictions and warn
            self.assertEqual(len(predictions), 5)
            self.assertTrue(len(w) > 0)
            self.assertIn("synthetic data", str(w[0].message))
    
    def test_extract_actuals_fallback(self):
        """Test actual value extraction with fallback."""
        # Create a TimeSeries that will fail
        failing_ts = Mock()
        failing_ts.values.side_effect = Exception("Values failed")
        failing_ts.to_numpy.side_effect = Exception("to_numpy failed")
        failing_ts.pd_dataframe.side_effect = Exception("pd_dataframe failed")
        failing_ts.to_pandas.side_effect = Exception("to_pandas failed")
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")  # Ensure warnings are captured
            actuals = self.evaluator._extract_actuals(failing_ts, 5)
            
            # Should generate synthetic actuals and warn
            self.assertEqual(len(actuals), 5)
            self.assertTrue(len(w) > 0)
            self.assertIn("synthetic data", str(w[0].message))
    
    def test_evaluation_summary(self):
        """Test evaluation summary generation."""
        # Add some mock results to history
        self.evaluator.evaluation_history = {
            'Model1': EvaluationResults(
                model_name='Model1',
                predictions=np.array([1, 2, 3]),
                actuals=np.array([1.1, 2.1, 3.1]),
                mae=0.1,
                rmse=0.15,
                mape=5.0,
                prediction_length=3,
                evaluation_time=1.0,
                performance_degraded=False
            ),
            'Model2': EvaluationResults(
                model_name='Model2',
                predictions=np.array([1, 2, 3]),
                actuals=np.array([1.2, 2.2, 3.2]),
                mae=0.2,
                rmse=0.25,
                mape=10.0,
                prediction_length=3,
                evaluation_time=1.5,
                performance_degraded=True
            )
        }
        
        summary = self.evaluator.get_evaluation_summary()
        
        self.assertEqual(summary['total_models_evaluated'], 2)
        self.assertEqual(summary['best_model_mae'], 'Model1')
        self.assertEqual(summary['best_model_rmse'], 'Model1')
        self.assertEqual(summary['best_model_mape'], 'Model1')
        self.assertAlmostEqual(summary['average_mae'], 0.15, places=2)
        self.assertEqual(len(summary['degraded_models']), 1)
        self.assertIn('Model2', summary['degraded_models'])
    
    def test_validation_requirements(self):
        """Test validation of evaluation requirements."""
        # Test with sufficient data
        large_ts = MockTimeSeries(np.random.randn(20))
        self.assertTrue(self.evaluator.validate_evaluation_requirements(large_ts, 5))
        
        # Test with insufficient data
        small_ts = MockTimeSeries(np.random.randn(3))
        with self.assertRaises(ModelEvaluationError) as context:
            self.evaluator.validate_evaluation_requirements(small_ts, 5)
        
        self.assertIn("Test data too small", str(context.exception))
    
    def test_compare_models(self):
        """Test model comparison functionality."""
        results = {
            'Model1': EvaluationResults(
                model_name='Model1',
                predictions=np.array([1, 2, 3]),
                actuals=np.array([1.1, 2.1, 3.1]),
                mae=0.1,
                rmse=0.15,
                mape=5.0,
                prediction_length=3,
                evaluation_time=1.0,
                performance_degraded=False
            ),
            'Model2': EvaluationResults(
                model_name='Model2',
                predictions=np.array([1, 2, 3]),
                actuals=np.array([1.2, 2.2, 3.2]),
                mae=0.2,
                rmse=0.25,
                mape=10.0,
                prediction_length=3,
                evaluation_time=1.5,
                performance_degraded=True
            )
        }
        
        comparison = self.evaluator.compare_models(results)
        
        self.assertEqual(comparison['model_count'], 2)
        self.assertEqual(comparison['rankings']['mae'], ['Model1', 'Model2'])
        self.assertEqual(comparison['rankings']['rmse'], ['Model1', 'Model2'])
        self.assertEqual(comparison['rankings']['mape'], ['Model1', 'Model2'])
        
        # Check best/worst for each metric
        self.assertEqual(comparison['metrics_comparison']['mae']['best'][0], 'Model1')
        self.assertEqual(comparison['metrics_comparison']['mae']['worst'][0], 'Model2')
        
        # Check performance analysis
        self.assertEqual(comparison['performance_analysis']['degraded_models'], 1)
        self.assertEqual(comparison['performance_analysis']['degradation_rate'], 50.0)
    
    def test_set_baseline_metrics(self):
        """Test setting baseline metrics."""
        baseline = {
            'Model1': {'mae': 1.0, 'rmse': 1.5},
            'Model2': {'mae': 2.0, 'rmse': 2.5}
        }
        
        self.evaluator.set_baseline_metrics(baseline)
        
        self.assertEqual(self.evaluator.baseline_metrics, baseline)
    
    def test_multivariate_data_handling(self):
        """Test handling of multivariate data."""
        # Create multivariate test data
        multivariate_data = np.random.randn(10, 3)  # 10 timesteps, 3 features
        
        # Test prediction extraction (should take first column)
        predictions = self.evaluator._generate_predictions(
            MockModel(predictions=multivariate_data[:5]), 
            self.test_ts, 
            5
        )
        
        self.assertEqual(len(predictions), 5)
        self.assertEqual(predictions.ndim, 1)  # Should be flattened
    
    def test_edge_cases(self):
        """Test various edge cases."""
        # Test with empty evaluation history
        summary = ModelEvaluator(verbose=False).get_evaluation_summary()
        self.assertIn("No evaluation history available", summary["message"])
        
        # Test comparison with empty results
        comparison = self.evaluator.compare_models({})
        self.assertIn("No results to compare", comparison["message"])
        
        # Test performance degradation with no baseline
        degraded = self.evaluator._check_performance_degradation(
            'UnknownModel', 1.0, 1.5, 5.0
        )
        self.assertFalse(degraded)


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)