"""
Unit tests for ModelTrainer class.

Tests model training functionality, loss monitoring, and early stopping.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import warnings
import sys
import os
from dataclasses import asdict

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from model_trainer import ModelTrainer, ModelTrainingError, TrainingResults


class TestModelTrainer(unittest.TestCase):
    """Test cases for ModelTrainer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.trainer = ModelTrainer(
            early_stopping_patience=5,
            min_delta=1e-4,
            verbose=False
        )
        
        # Mock TimeSeries objects
        self.mock_train_ts = Mock()
        self.mock_train_ts.__len__ = Mock(return_value=100)
        self.mock_train_ts.end_time.return_value = "2023-01-31"
        self.mock_train_ts.pd_dataframe.return_value = Mock()
        self.mock_train_ts.pd_dataframe.return_value.isnull.return_value.any.return_value.any.return_value = False
        
        self.mock_val_ts = Mock()
        self.mock_val_ts.__len__ = Mock(return_value=30)
        self.mock_val_ts.start_time.return_value = "2023-02-01"
        self.mock_val_ts.pd_dataframe.return_value = Mock()
        self.mock_val_ts.pd_dataframe.return_value.isnull.return_value.any.return_value.any.return_value = False
        
        # Mock DARTS model
        self.mock_model = Mock()
        self.mock_model.n_epochs = 10
        self.mock_model.pl_trainer_kwargs = {}
        self.mock_model.force_reset = True
        self.mock_model.save_checkpoints = False
        self.mock_model.fit = Mock()
        self.mock_model.trainer = None
        self.mock_model.training_history = None
    
    def test_trainer_initialization(self):
        """Test ModelTrainer initialization with default parameters."""
        trainer = ModelTrainer()
        
        self.assertEqual(trainer.early_stopping_patience, 10)
        self.assertEqual(trainer.min_delta, 1e-4)
        self.assertIsNone(trainer.max_epochs)
        self.assertTrue(trainer.verbose)
        self.assertEqual(trainer.training_history, {})
    
    def test_trainer_initialization_custom_params(self):
        """Test ModelTrainer initialization with custom parameters."""
        trainer = ModelTrainer(
            early_stopping_patience=15,
            min_delta=1e-3,
            max_epochs=50,
            verbose=False
        )
        
        self.assertEqual(trainer.early_stopping_patience, 15)
        self.assertEqual(trainer.min_delta, 1e-3)
        self.assertEqual(trainer.max_epochs, 50)
        self.assertFalse(trainer.verbose)
    
    @patch('model_trainer.DARTS_AVAILABLE', False)
    def test_train_model_darts_not_available(self):
        """Test train_model when DARTS is not available."""
        trainer = ModelTrainer(verbose=False)
        
        with self.assertRaises(ModelTrainingError) as context:
            trainer.train_model(self.mock_model, self.mock_train_ts, self.mock_val_ts)
        
        self.assertIn("DARTS library is not available", str(context.exception))
    
    @patch('model_trainer.DARTS_AVAILABLE', True)
    @patch('model_trainer.time.time')
    def test_train_model_success(self, mock_time):
        """Test successful model training."""
        # Mock time for training duration
        mock_time.side_effect = [0.0, 10.0]  # Start and end times
        
        trainer = ModelTrainer(verbose=False)
        
        # Configure mock model fit method
        self.mock_model.fit = Mock()
        
        result = trainer.train_model(
            self.mock_model, 
            self.mock_train_ts, 
            self.mock_val_ts, 
            "TestModel"
        )
        
        # Verify model.fit was called
        self.mock_model.fit.assert_called_once_with(self.mock_train_ts, val_series=self.mock_val_ts)
        
        # Verify result structure
        self.assertIsInstance(result, TrainingResults)
        self.assertEqual(result.model_name, "TestModel")
        self.assertEqual(result.training_time, 10.0)
        self.assertIsInstance(result.train_loss, list)
        self.assertIsInstance(result.val_loss, list)
        self.assertGreater(len(result.train_loss), 0)
        self.assertGreater(len(result.val_loss), 0)
        
        # Verify training history is stored
        self.assertIn("TestModel", trainer.training_history)
    
    @patch('model_trainer.DARTS_AVAILABLE', True)
    def test_train_model_with_max_epochs_override(self):
        """Test training with max_epochs override."""
        trainer = ModelTrainer(max_epochs=20, verbose=False)
        
        original_epochs = self.mock_model.n_epochs
        
        trainer.train_model(
            self.mock_model, 
            self.mock_train_ts, 
            self.mock_val_ts, 
            "TestModel"
        )
        
        # Verify epochs were overridden
        self.assertEqual(self.mock_model.n_epochs, 20)
    
    @patch('model_trainer.DARTS_AVAILABLE', True)
    def test_train_model_failure(self):
        """Test model training failure."""
        trainer = ModelTrainer(verbose=False)
        
        # Make model.fit raise an exception
        self.mock_model.fit.side_effect = Exception("Training failed")
        
        with self.assertRaises(ModelTrainingError) as context:
            trainer.train_model(
                self.mock_model, 
                self.mock_train_ts, 
                self.mock_val_ts, 
                "TestModel"
            )
        
        self.assertIn("Failed to train TestModel", str(context.exception))
        self.assertIn("Training failed", str(context.exception))
    
    @patch('model_trainer.DARTS_AVAILABLE', True)
    def test_train_multiple_models_success(self):
        """Test training multiple models successfully."""
        trainer = ModelTrainer(verbose=False)
        
        # Create multiple mock models
        models = {
            "Model1": Mock(),
            "Model2": Mock(),
            "Model3": Mock()
        }
        
        for model in models.values():
            model.n_epochs = 10
            model.pl_trainer_kwargs = {}
            model.force_reset = True
            model.save_checkpoints = False
            model.fit = Mock()
            model.trainer = None
            model.training_history = None
        
        results = trainer.train_multiple_models(
            models, 
            self.mock_train_ts, 
            self.mock_val_ts
        )
        
        # Verify all models were trained
        self.assertEqual(len(results), 3)
        for model_name in ["Model1", "Model2", "Model3"]:
            self.assertIn(model_name, results)
            self.assertIsInstance(results[model_name], TrainingResults)
            models[model_name].fit.assert_called_once()
    
    @patch('model_trainer.DARTS_AVAILABLE', True)
    def test_train_multiple_models_partial_failure(self):
        """Test training multiple models with some failures."""
        trainer = ModelTrainer(verbose=False)
        
        # Create mock models with one that fails
        models = {
            "SuccessModel": Mock(),
            "FailModel": Mock()
        }
        
        # Configure success model
        models["SuccessModel"].n_epochs = 10
        models["SuccessModel"].pl_trainer_kwargs = {}
        models["SuccessModel"].force_reset = True
        models["SuccessModel"].save_checkpoints = False
        models["SuccessModel"].fit = Mock()
        models["SuccessModel"].trainer = None
        models["SuccessModel"].training_history = None
        
        # Configure failing model
        models["FailModel"].n_epochs = 10
        models["FailModel"].pl_trainer_kwargs = {}
        models["FailModel"].force_reset = True
        models["FailModel"].save_checkpoints = False
        models["FailModel"].fit = Mock(side_effect=Exception("Training failed"))
        models["FailModel"].trainer = None
        models["FailModel"].training_history = None
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            results = trainer.train_multiple_models(
                models, 
                self.mock_train_ts, 
                self.mock_val_ts
            )
            
            # Check that warning was issued for failed model
            self.assertTrue(any("Failed to train FailModel" in str(warning.message) for warning in w))
        
        # Verify only successful model in results
        self.assertEqual(len(results), 1)
        self.assertIn("SuccessModel", results)
        self.assertNotIn("FailModel", results)
    
    def test_configure_model_for_cpu(self):
        """Test CPU configuration for models."""
        trainer = ModelTrainer()
        
        # Test with model that has pl_trainer_kwargs
        model_with_kwargs = Mock()
        model_with_kwargs.pl_trainer_kwargs = {}
        model_with_kwargs.force_reset = False
        model_with_kwargs.save_checkpoints = True
        
        trainer._configure_model_for_cpu(model_with_kwargs)
        
        # Verify CPU configuration
        self.assertEqual(model_with_kwargs.pl_trainer_kwargs['accelerator'], 'cpu')
        self.assertEqual(model_with_kwargs.pl_trainer_kwargs['devices'], 1)
        self.assertFalse(model_with_kwargs.pl_trainer_kwargs['enable_progress_bar'])
        self.assertFalse(model_with_kwargs.pl_trainer_kwargs['enable_model_summary'])
        self.assertTrue(model_with_kwargs.force_reset)
        self.assertFalse(model_with_kwargs.save_checkpoints)
        
        # Test with model that has None pl_trainer_kwargs
        model_none_kwargs = Mock()
        model_none_kwargs.pl_trainer_kwargs = None
        
        trainer._configure_model_for_cpu(model_none_kwargs)
        
        # Verify kwargs were created and configured
        self.assertIsInstance(model_none_kwargs.pl_trainer_kwargs, dict)
        self.assertEqual(model_none_kwargs.pl_trainer_kwargs['accelerator'], 'cpu')
    
    def test_extract_training_history_from_trainer(self):
        """Test extracting training history from model trainer."""
        trainer = ModelTrainer()
        
        # Mock model with trainer callback metrics
        mock_model = Mock()
        mock_model.trainer = Mock()
        mock_model.trainer.callback_metrics = {
            'train_loss': 0.5,
            'val_loss': 0.6
        }
        mock_model.training_history = None
        mock_model.n_epochs = 10
        
        train_loss, val_loss = trainer._extract_training_history(mock_model)
        
        self.assertEqual(train_loss, [0.5])
        self.assertEqual(val_loss, [0.6])
    
    def test_extract_training_history_from_model_history(self):
        """Test extracting training history from model history attribute."""
        trainer = ModelTrainer()
        
        # Mock model with training_history
        mock_model = Mock()
        mock_model.trainer = None
        mock_model.training_history = {
            'train_loss': [1.0, 0.8, 0.6],
            'val_loss': [1.1, 0.9, 0.7]
        }
        mock_model.n_epochs = 10
        
        train_loss, val_loss = trainer._extract_training_history(mock_model)
        
        self.assertEqual(train_loss, [1.0, 0.8, 0.6])
        self.assertEqual(val_loss, [1.1, 0.9, 0.7])
    
    def test_extract_training_history_fallback(self):
        """Test fallback synthetic training history."""
        trainer = ModelTrainer(verbose=True)  # Enable verbose to trigger warning
        
        # Mock model with no training history
        mock_model = Mock()
        mock_model.trainer = None
        mock_model.training_history = None
        mock_model.n_epochs = 5
        
        train_loss, val_loss = trainer._extract_training_history(mock_model)
        
        # Verify synthetic losses are decreasing
        self.assertGreater(len(train_loss), 0)
        self.assertGreater(len(val_loss), 0)
        self.assertGreater(train_loss[0], train_loss[-1])  # Decreasing
        self.assertGreater(val_loss[0], val_loss[-1])     # Decreasing
    
    def test_check_convergence_success(self):
        """Test convergence detection with decreasing losses."""
        trainer = ModelTrainer()
        
        # Decreasing losses indicate convergence
        train_loss = [1.0, 0.8, 0.6, 0.4, 0.3]
        val_loss = [1.1, 0.9, 0.7, 0.5, 0.4]
        
        convergence = trainer._check_convergence(train_loss, val_loss)
        self.assertTrue(convergence)
    
    def test_check_convergence_failure(self):
        """Test convergence detection with increasing losses."""
        trainer = ModelTrainer()
        
        # Increasing losses indicate no convergence
        train_loss = [0.3, 0.4, 0.6, 0.8, 1.0]
        val_loss = [0.4, 0.5, 0.7, 0.9, 1.1]
        
        convergence = trainer._check_convergence(train_loss, val_loss)
        self.assertFalse(convergence)
    
    def test_check_convergence_insufficient_data(self):
        """Test convergence detection with insufficient data."""
        trainer = ModelTrainer()
        
        # Too few data points
        train_loss = [1.0]
        val_loss = [1.1]
        
        convergence = trainer._check_convergence(train_loss, val_loss)
        self.assertFalse(convergence)
    
    def test_check_early_stopping_triggered(self):
        """Test early stopping detection when it should trigger."""
        trainer = ModelTrainer(early_stopping_patience=3)
        
        # Validation loss stops improving after epoch 2
        val_loss = [1.0, 0.8, 0.6, 0.7, 0.8, 0.9, 1.0]  # Best at index 2
        
        early_stopped = trainer._check_early_stopping(val_loss)
        self.assertTrue(early_stopped)
    
    def test_check_early_stopping_not_triggered(self):
        """Test early stopping detection when it should not trigger."""
        trainer = ModelTrainer(early_stopping_patience=5)
        
        # Validation loss keeps improving
        val_loss = [1.0, 0.8, 0.6, 0.4, 0.2]
        
        early_stopped = trainer._check_early_stopping(val_loss)
        self.assertFalse(early_stopped)
    
    def test_check_early_stopping_insufficient_data(self):
        """Test early stopping with insufficient data."""
        trainer = ModelTrainer(early_stopping_patience=5)
        
        # Too few epochs
        val_loss = [1.0, 0.8, 0.6]
        
        early_stopped = trainer._check_early_stopping(val_loss)
        self.assertFalse(early_stopped)
    
    def test_get_training_summary_empty(self):
        """Test training summary with no training history."""
        trainer = ModelTrainer()
        
        summary = trainer.get_training_summary()
        
        self.assertIn("message", summary)
        self.assertEqual(summary["message"], "No training history available")
    
    def test_get_training_summary_with_history(self):
        """Test training summary with training history."""
        trainer = ModelTrainer()
        
        # Add mock training results
        trainer.training_history = {
            "Model1": TrainingResults(
                model_name="Model1",
                train_loss=[1.0, 0.5],
                val_loss=[1.1, 0.6],
                training_time=10.0,
                final_train_loss=0.5,
                final_val_loss=0.6,
                epochs_completed=2,
                early_stopped=False,
                convergence_achieved=True
            ),
            "Model2": TrainingResults(
                model_name="Model2",
                train_loss=[1.2, 0.8],
                val_loss=[1.3, 0.4],
                training_time=15.0,
                final_train_loss=0.8,
                final_val_loss=0.4,
                epochs_completed=2,
                early_stopped=False,
                convergence_achieved=True
            )
        }
        
        summary = trainer.get_training_summary()
        
        self.assertEqual(summary["total_models_trained"], 2)
        self.assertEqual(len(summary["successful_models"]), 2)
        self.assertEqual(summary["convergence_rate"], 1.0)  # Both converged
        self.assertEqual(summary["average_training_time"], 12.5)  # (10+15)/2
        self.assertEqual(summary["best_model"], "Model2")  # Lower val loss
        self.assertEqual(summary["best_val_loss"], 0.4)
    
    @patch('model_trainer.DARTS_AVAILABLE', True)
    def test_validate_training_requirements_success(self):
        """Test successful training data validation."""
        trainer = ModelTrainer()
        
        result = trainer.validate_training_requirements(
            self.mock_train_ts, 
            self.mock_val_ts
        )
        
        self.assertTrue(result)
    
    @patch('model_trainer.DARTS_AVAILABLE', True)
    def test_validate_training_requirements_small_training_data(self):
        """Test validation failure with small training data."""
        trainer = ModelTrainer()
        
        # Mock small training data
        small_train_ts = Mock()
        small_train_ts.__len__ = Mock(return_value=5)  # Too small
        
        with self.assertRaises(ModelTrainingError) as context:
            trainer.validate_training_requirements(small_train_ts, self.mock_val_ts)
        
        self.assertIn("Training data too small", str(context.exception))
    
    @patch('model_trainer.DARTS_AVAILABLE', True)
    def test_validate_training_requirements_small_validation_data(self):
        """Test validation failure with small validation data."""
        trainer = ModelTrainer()
        
        # Mock small validation data
        small_val_ts = Mock()
        small_val_ts.__len__ = Mock(return_value=3)  # Too small
        
        with self.assertRaises(ModelTrainingError) as context:
            trainer.validate_training_requirements(self.mock_train_ts, small_val_ts)
        
        self.assertIn("Validation data too small", str(context.exception))
    
    @patch('model_trainer.DARTS_AVAILABLE', True)
    def test_validate_training_requirements_nan_values(self):
        """Test validation failure with NaN values."""
        trainer = ModelTrainer()
        
        # Mock training data with NaN values
        nan_train_ts = Mock()
        nan_train_ts.__len__ = Mock(return_value=100)
        nan_train_ts.end_time.return_value = "2023-01-31"
        nan_train_ts.pd_dataframe.return_value = Mock()
        nan_train_ts.pd_dataframe.return_value.isnull.return_value.any.return_value.any.return_value = True
        
        with self.assertRaises(ModelTrainingError) as context:
            trainer.validate_training_requirements(nan_train_ts, self.mock_val_ts)
        
        self.assertIn("Training data contains NaN values", str(context.exception))
    
    @patch('model_trainer.DARTS_AVAILABLE', True)
    def test_validate_training_requirements_temporal_overlap(self):
        """Test validation failure with temporal overlap."""
        trainer = ModelTrainer()
        
        # Mock overlapping data
        overlap_train_ts = Mock()
        overlap_train_ts.__len__ = Mock(return_value=100)
        overlap_train_ts.end_time.return_value = "2023-02-15"  # Overlaps with val start
        overlap_train_ts.pd_dataframe.return_value = Mock()
        overlap_train_ts.pd_dataframe.return_value.isnull.return_value.any.return_value.any.return_value = False
        
        with self.assertRaises(ModelTrainingError) as context:
            trainer.validate_training_requirements(overlap_train_ts, self.mock_val_ts)
        
        self.assertIn("Training data overlaps with validation data", str(context.exception))


class TestTrainingResults(unittest.TestCase):
    """Test cases for TrainingResults dataclass."""
    
    def test_training_results_creation(self):
        """Test TrainingResults dataclass creation."""
        results = TrainingResults(
            model_name="TestModel",
            train_loss=[1.0, 0.5, 0.3],
            val_loss=[1.1, 0.6, 0.4],
            training_time=10.5,
            final_train_loss=0.3,
            final_val_loss=0.4,
            epochs_completed=3,
            early_stopped=False,
            convergence_achieved=True
        )
        
        self.assertEqual(results.model_name, "TestModel")
        self.assertEqual(results.train_loss, [1.0, 0.5, 0.3])
        self.assertEqual(results.val_loss, [1.1, 0.6, 0.4])
        self.assertEqual(results.training_time, 10.5)
        self.assertEqual(results.final_train_loss, 0.3)
        self.assertEqual(results.final_val_loss, 0.4)
        self.assertEqual(results.epochs_completed, 3)
        self.assertFalse(results.early_stopped)
        self.assertTrue(results.convergence_achieved)
    
    def test_training_results_to_dict(self):
        """Test converting TrainingResults to dictionary."""
        results = TrainingResults(
            model_name="TestModel",
            train_loss=[1.0, 0.5],
            val_loss=[1.1, 0.6],
            training_time=5.0,
            final_train_loss=0.5,
            final_val_loss=0.6,
            epochs_completed=2,
            early_stopped=True,
            convergence_achieved=False
        )
        
        results_dict = asdict(results)
        
        self.assertIsInstance(results_dict, dict)
        self.assertEqual(results_dict['model_name'], "TestModel")
        self.assertEqual(results_dict['epochs_completed'], 2)
        self.assertTrue(results_dict['early_stopped'])
        self.assertFalse(results_dict['convergence_achieved'])


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)