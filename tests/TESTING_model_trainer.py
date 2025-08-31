"""
Unit tests for ModelTrainer class.

This module tests the ModelTrainer class functionality including:
- Model training with CPU configuration
- Training loss monitoring
- Early stopping detection
- Convergence analysis
"""

import unittest
import numpy as np
import pandas as pd
import sys
import os
from unittest.mock import Mock, patch, MagicMock
import tempfile
import warnings

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Try to import dependencies
try:
    from darts import TimeSeries
    from model_trainer import ModelTrainer, TrainingResults, ModelTrainingError
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    DEPENDENCIES_AVAILABLE = False
    print(f"Warning: Dependencies not available: {e}")


@unittest.skipUnless(DEPENDENCIES_AVAILABLE, "Dependencies not available")
class TestModelTrainer(unittest.TestCase):
    """Test cases for ModelTrainer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create sample TimeSeries data
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        train_values = np.random.normal(100, 10, 70)
        val_values = np.random.normal(100, 10, 30)
        
        self.train_ts = TimeSeries.from_pandas(pd.Series(train_values, index=dates[:70]))
        self.val_ts = TimeSeries.from_pandas(pd.Series(val_values, index=dates[70:]))
        
        # Create mock model
        self.mock_model = Mock()
        self.mock_model.fit = Mock()
        self.mock_model.trainer = Mock()
        self.mock_model.trainer.current_epoch = 10
        self.mock_model.n_epochs = 50
        
        # Initialize trainer
        self.trainer = ModelTrainer(verbose=False)
    
    def test_init(self):
        """Test ModelTrainer initialization."""
        trainer = ModelTrainer(
            early_stopping_patience=15,
            min_delta=1e-3,
            max_epochs=100,
            verbose=True
        )
        
        self.assertEqual(trainer.early_stopping_patience, 15)
        self.assertEqual(trainer.min_delta, 1e-3)
        self.assertEqual(trainer.max_epochs, 100)
        self.assertTrue(trainer.verbose)
        self.assertEqual(len(trainer.training_history), 0)
    
    @patch('model_trainer.CSVLogger')
    @patch('model_trainer.pd.read_csv')
    def test_train_model_success(self, mock_read_csv, mock_csv_logger):
        """Test successful model training."""
        # Mock CSV logger and metrics
        mock_logger_instance = Mock()
        mock_logger_instance.log_dir = "temp_logs/test_model"
        mock_csv_logger.return_value = mock_logger_instance
        
        # Mock metrics CSV data
        mock_metrics_df = pd.DataFrame({
            'train_loss': [1.0, 0.8, 0.6, 0.5, 0.4],
            'val_loss': [1.1, 0.9, 0.7, 0.6, 0.5]
        })
        mock_read_csv.return_value = mock_metrics_df
        
        # Configure mock model
        self.mock_model.pl_trainer_kwargs = {}
        
        result = self.trainer.train_model(
            self.mock_model,
            self.train_ts,
            self.val_ts,
            "TestModel"
        )
        
        # Check result structure
        self.assertIsInstance(result, TrainingResults)
        self.assertEqual(result.model_name, "TestModel")
        self.assertEqual(len(result.train_loss), 5)
        self.assertEqual(len(result.val_loss), 5)
        self.assertEqual(result.final_train_loss, 0.4)
        self.assertEqual(result.final_val_loss, 0.5)
        self.assertGreater(result.training_time, 0)
        
        # Check that model.fit was called
        self.mock_model.fit.assert_called_once_with(self.train_ts, val_series=self.val_ts)
        
        # Check training history is updated
        self.assertIn("TestModel", self.trainer.training_history)
    
    @patch('model_trainer.CSVLogger')
    def test_train_model_with_max_epochs_override(self, mock_csv_logger):
        """Test training with max_epochs override."""
        # Mock CSV logger
        mock_logger_instance = Mock()
        mock_logger_instance.log_dir = "temp_logs/test_model"
        mock_csv_logger.return_value = mock_logger_instance
        
        # Set max_epochs override
        trainer = ModelTrainer(max_epochs=25, verbose=False)
        
        # Configure mock model
        self.mock_model.pl_trainer_kwargs = {}
        original_epochs = self.mock_model.n_epochs
        
        with patch('model_trainer.pd.read_csv') as mock_read_csv:
            mock_read_csv.side_effect = FileNotFoundError("No metrics file")
            
            trainer.train_model(
                self.mock_model,
                self.train_ts,
                self.val_ts,
                "TestModel"
            )
        
        # Check that epochs were overridden and restored
        self.assertEqual(self.mock_model.n_epochs, original_epochs)
    
    def test_configure_trainer(self):
        """Test trainer configuration for CPU."""
        mock_logger = self.trainer._configure_trainer(self.mock_model, "TestModel")
        
        # Check that pl_trainer_kwargs were set correctly
        self.assertIn('accelerator', self.mock_model.pl_trainer_kwargs)
        self.assertEqual(self.mock_model.pl_trainer_kwargs['accelerator'], 'cpu')
        self.assertEqual(self.mock_model.pl_trainer_kwargs['devices'], 1)
        self.assertIn('logger', self.mock_model.pl_trainer_kwargs)
        
        # Check logger configuration
        self.assertIsNotNone(mock_logger)
    
    def test_check_convergence(self):
        """Test convergence checking."""
        # Test convergence achieved
        train_loss = [1.0, 0.8, 0.6, 0.5, 0.4]
        val_loss = [1.1, 0.9, 0.7, 0.6, 0.5]
        
        converged = self.trainer._check_convergence(train_loss, val_loss)
        self.assertTrue(converged)
        
        # Test no convergence (increasing loss)
        train_loss = [1.0, 0.8, 0.9, 1.0, 1.1]
        val_loss = [1.1, 0.9, 1.0, 1.1, 1.2]
        
        converged = self.trainer._check_convergence(train_loss, val_loss)
        self.assertFalse(converged)
        
        # Test insufficient data
        train_loss = [1.0, 0.8]
        val_loss = [1.1, 0.9]
        
        converged = self.trainer._check_convergence(train_loss, val_loss)
        self.assertFalse(converged)
    
    def test_check_early_stopping(self):
        """Test early stopping detection."""
        # Set patience to 3
        trainer = ModelTrainer(early_stopping_patience=3, min_delta=0.01, verbose=False)
        
        # Test early stopping triggered (no improvement for 3 epochs)
        val_loss = [1.0, 0.9, 0.8, 0.79, 0.785, 0.784, 0.783]  # No significant improvement
        
        early_stopped = trainer._check_early_stopping(val_loss)
        self.assertTrue(early_stopped)
        
        # Test no early stopping (improvement within patience)
        val_loss = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]  # Continuous improvement
        
        early_stopped = trainer._check_early_stopping(val_loss)
        self.assertFalse(early_stopped)
        
        # Test insufficient data
        val_loss = [1.0, 0.9]
        
        early_stopped = trainer._check_early_stopping(val_loss)
        self.assertFalse(early_stopped)
    
    def test_training_results_dataclass(self):
        """Test TrainingResults dataclass."""
        result = TrainingResults(
            model_name="TestModel",
            train_loss=[1.0, 0.8, 0.6],
            val_loss=[1.1, 0.9, 0.7],
            training_time=10.5,
            final_train_loss=0.6,
            final_val_loss=0.7,
            epochs_completed=3,
            early_stopped=False,
            convergence_achieved=True
        )
        
        self.assertEqual(result.model_name, "TestModel")
        self.assertEqual(len(result.train_loss), 3)
        self.assertEqual(len(result.val_loss), 3)
        self.assertEqual(result.training_time, 10.5)
        self.assertEqual(result.final_train_loss, 0.6)
        self.assertEqual(result.final_val_loss, 0.7)
        self.assertEqual(result.epochs_completed, 3)
        self.assertFalse(result.early_stopped)
        self.assertTrue(result.convergence_achieved)
    
    def test_error_handling(self):
        """Test error handling during training."""
        # Test with failing model
        failing_model = Mock()
        failing_model.fit.side_effect = Exception("Training failed")
        failing_model.pl_trainer_kwargs = {}
        
        with self.assertRaises(ModelTrainingError):
            self.trainer.train_model(
                failing_model,
                self.train_ts,
                self.val_ts,
                "FailingModel"
            )
    
    @patch('model_trainer.CSVLogger')
    @patch('model_trainer.pd.read_csv')
    def test_missing_metrics_file(self, mock_read_csv, mock_csv_logger):
        """Test handling of missing metrics file."""
        # Mock CSV logger
        mock_logger_instance = Mock()
        mock_logger_instance.log_dir = "temp_logs/test_model"
        mock_csv_logger.return_value = mock_logger_instance
        
        # Mock missing metrics file
        mock_read_csv.side_effect = FileNotFoundError("Metrics file not found")
        
        # Configure mock model
        self.mock_model.pl_trainer_kwargs = {}
        
        result = self.trainer.train_model(
            self.mock_model,
            self.train_ts,
            self.val_ts,
            "TestModel"
        )
        
        # Should handle missing metrics gracefully
        self.assertIsInstance(result, TrainingResults)
        self.assertEqual(len(result.train_loss), 0)
        self.assertEqual(len(result.val_loss), 0)
        self.assertEqual(result.final_train_loss, float('inf'))
        self.assertEqual(result.final_val_loss, float('inf'))
    
    def test_model_without_trainer_attribute(self):
        """Test handling of model without trainer attribute."""
        # Create model without trainer attribute
        model_without_trainer = Mock()
        model_without_trainer.fit = Mock()
        model_without_trainer.pl_trainer_kwargs = {}
        del model_without_trainer.trainer  # Remove trainer attribute
        
        with patch('model_trainer.CSVLogger') as mock_csv_logger, \
             patch('model_trainer.pd.read_csv') as mock_read_csv:
            
            mock_logger_instance = Mock()
            mock_logger_instance.log_dir = "temp_logs/test_model"
            mock_csv_logger.return_value = mock_logger_instance
            mock_read_csv.side_effect = FileNotFoundError("No metrics")
            
            result = self.trainer.train_model(
                model_without_trainer,
                self.train_ts,
                self.val_ts,
                "TestModel"
            )
            
            # Should handle missing trainer gracefully
            self.assertIsInstance(result, TrainingResults)
            self.assertEqual(result.epochs_completed, 0)
    
    def test_model_configuration_preservation(self):
        """Test that model configuration is preserved after training."""
        # Configure mock model with existing pl_trainer_kwargs
        original_kwargs = {'some_param': 'original_value'}
        self.mock_model.pl_trainer_kwargs = original_kwargs.copy()
        
        with patch('model_trainer.CSVLogger') as mock_csv_logger, \
             patch('model_trainer.pd.read_csv') as mock_read_csv:
            
            mock_logger_instance = Mock()
            mock_logger_instance.log_dir = "temp_logs/test_model"
            mock_csv_logger.return_value = mock_logger_instance
            mock_read_csv.side_effect = FileNotFoundError("No metrics")
            
            self.trainer.train_model(
                self.mock_model,
                self.train_ts,
                self.val_ts,
                "TestModel"
            )
            
            # Check that original parameters are preserved
            self.assertIn('some_param', self.mock_model.pl_trainer_kwargs)
            self.assertEqual(
                self.mock_model.pl_trainer_kwargs['some_param'], 
                'original_value'
            )
            
            # Check that CPU configuration was added
            self.assertEqual(self.mock_model.pl_trainer_kwargs['accelerator'], 'cpu')
            self.assertEqual(self.mock_model.pl_trainer_kwargs['devices'], 1)


if __name__ == '__main__':
    unittest.main()