"""
Unit tests for ModelFactory class.

Tests model instantiation, configuration, and validation for all DARTS neural network models.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import warnings
import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from model_factory import ModelFactory, ModelInstantiationError


class TestModelFactory(unittest.TestCase):
    """Test cases for ModelFactory class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.factory = ModelFactory()
        
        # Mock model classes for testing
        self.mock_models = {}
        for model_name in ['RNNModel', 'TCNModel', 'TransformerModel', 'NBEATSModel', 
                          'TFTModel', 'NHiTSModel', 'DLinearModel', 'NLinearModel']:
            mock_model = Mock()
            mock_model.__name__ = model_name
            mock_model.input_chunk_length = 30
            mock_model.output_chunk_length = 5
            mock_model.n_epochs = 50
            mock_model.batch_size = 32
            mock_model.supports_multivariate = True
            self.mock_models[model_name] = mock_model
    
    def test_factory_initialization(self):
        """Test ModelFactory initialization with default parameters."""
        factory = ModelFactory()
        
        # Check default parameters
        self.assertEqual(factory.params['input_chunk_length'], 30)
        self.assertEqual(factory.params['output_chunk_length'], 5)
        self.assertEqual(factory.params['n_epochs'], 50)
        self.assertEqual(factory.params['batch_size'], 32)
        self.assertEqual(factory.params['random_state'], 42)
        
        # Check CPU configuration
        self.assertIn('pl_trainer_kwargs', factory.params)
        self.assertEqual(factory.params['pl_trainer_kwargs']['accelerator'], 'cpu')
        self.assertEqual(factory.params['pl_trainer_kwargs']['devices'], 1)
    
    def test_factory_initialization_custom_params(self):
        """Test ModelFactory initialization with custom parameters."""
        factory = ModelFactory(
            input_chunk_length=60,
            output_chunk_length=10,
            n_epochs=100,
            batch_size=64,
            random_state=123
        )
        
        self.assertEqual(factory.params['input_chunk_length'], 60)
        self.assertEqual(factory.params['output_chunk_length'], 10)
        self.assertEqual(factory.params['n_epochs'], 100)
        self.assertEqual(factory.params['batch_size'], 64)
        self.assertEqual(factory.params['random_state'], 123)
    
    @patch('model_factory.DARTS_AVAILABLE', False)
    def test_create_models_darts_not_available(self):
        """Test create_models when DARTS is not available."""
        factory = ModelFactory()
        
        with self.assertRaises(ModelInstantiationError) as context:
            factory.create_models()
        
        self.assertIn("DARTS library is not available", str(context.exception))
    
    @patch('model_factory.DARTS_AVAILABLE', True)
    @patch('model_factory.RNNModel')
    @patch('model_factory.TCNModel')
    @patch('model_factory.TransformerModel')
    @patch('model_factory.NBEATSModel')
    @patch('model_factory.TFTModel')
    @patch('model_factory.NHiTSModel')
    @patch('model_factory.DLinearModel')
    @patch('model_factory.NLinearModel')
    def test_create_models_success(self, mock_nlinear, mock_dlinear, mock_nhits, 
                                  mock_tft, mock_nbeats, mock_transformer, 
                                  mock_tcn, mock_rnn):
        """Test successful model creation."""
        # Set up mock model classes
        mock_classes = {
            'RNNModel': mock_rnn,
            'TCNModel': mock_tcn,
            'TransformerModel': mock_transformer,
            'NBEATSModel': mock_nbeats,
            'TFTModel': mock_tft,
            'NHiTSModel': mock_nhits,
            'DLinearModel': mock_dlinear,
            'NLinearModel': mock_nlinear,
        }
        
        # Configure mocks to return model instances
        for name, mock_class in mock_classes.items():
            mock_instance = Mock()
            mock_instance.__class__.__name__ = name
            mock_class.return_value = mock_instance
        
        factory = ModelFactory()
        models = factory.create_models()
        
        # Verify all models were created
        self.assertEqual(len(models), 8)
        expected_models = ['RNNModel', 'TCNModel', 'TransformerModel', 'NBEATSModel',
                          'TFTModel', 'NHiTSModel', 'DLinearModel', 'NLinearModel']
        
        for model_name in expected_models:
            self.assertIn(model_name, models)
            self.assertIsNotNone(models[model_name])
        
        # Verify each mock was called
        for mock_class in mock_classes.values():
            mock_class.assert_called_once()
    
    @patch('model_factory.DARTS_AVAILABLE', True)
    @patch('model_factory.RNNModel')
    @patch('model_factory.TCNModel')
    def test_create_models_partial_failure(self, mock_tcn, mock_rnn):
        """Test model creation with some models failing."""
        # Set up one successful and one failing model
        mock_rnn_instance = Mock()
        mock_rnn_instance.__class__.__name__ = 'RNNModel'
        mock_rnn.return_value = mock_rnn_instance
        
        mock_tcn.side_effect = Exception("TCN creation failed")
        
        # Patch other models to None to simulate unavailability
        with patch('model_factory.TransformerModel', None), \
             patch('model_factory.NBEATSModel', None), \
             patch('model_factory.TFTModel', None), \
             patch('model_factory.NHiTSModel', None), \
             patch('model_factory.DLinearModel', None), \
             patch('model_factory.NLinearModel', None):
            
            factory = ModelFactory()
            
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                models = factory.create_models()
                
                # Check that warning was issued for failed model
                self.assertTrue(any("Failed to create TCNModel" in str(warning.message) for warning in w))
        
        # Should have one successful model
        self.assertEqual(len(models), 1)
        self.assertIn('RNNModel', models)
        self.assertNotIn('TCNModel', models)
    
    @patch('model_factory.DARTS_AVAILABLE', True)
    def test_create_models_all_fail(self):
        """Test create_models when all models fail to instantiate."""
        # Patch all model classes to raise exceptions
        with patch('model_factory.RNNModel', side_effect=Exception("RNN failed")), \
             patch('model_factory.TCNModel', side_effect=Exception("TCN failed")), \
             patch('model_factory.TransformerModel', side_effect=Exception("Transformer failed")), \
             patch('model_factory.NBEATSModel', side_effect=Exception("NBEATS failed")), \
             patch('model_factory.TFTModel', side_effect=Exception("TFT failed")), \
             patch('model_factory.NHiTSModel', side_effect=Exception("NHiTS failed")), \
             patch('model_factory.DLinearModel', side_effect=Exception("DLinear failed")), \
             patch('model_factory.NLinearModel', side_effect=Exception("NLinear failed")):
            
            factory = ModelFactory()
            
            with self.assertRaises(ModelInstantiationError) as context:
                factory.create_models()
            
            self.assertIn("Failed to create any models", str(context.exception))
    
    def test_filter_supported_params(self):
        """Test parameter filtering for different models."""
        factory = ModelFactory()
        
        # Test parameters
        all_params = {
            'input_chunk_length': 30,
            'output_chunk_length': 5,
            'n_epochs': 50,
            'batch_size': 32,
            'random_state': 42,
            'force_reset': True,
            'save_checkpoints': False,
            'pl_trainer_kwargs': {'accelerator': 'cpu'},
            'hidden_dim': 64,  # RNN-specific
            'kernel_size': 3,  # TCN/DLinear-specific
            'unsupported_param': 'should_be_filtered'
        }
        
        # Test RNNModel parameter filtering
        rnn_params = factory._filter_supported_params('RNNModel', all_params)
        self.assertIn('hidden_dim', rnn_params)
        self.assertNotIn('kernel_size', rnn_params)
        self.assertNotIn('unsupported_param', rnn_params)
        
        # Test DLinearModel parameter filtering
        dlinear_params = factory._filter_supported_params('DLinearModel', all_params)
        self.assertIn('kernel_size', dlinear_params)
        self.assertNotIn('hidden_dim', dlinear_params)
        self.assertNotIn('unsupported_param', dlinear_params)
        
        # Test unknown model (should get common parameters only)
        unknown_params = factory._filter_supported_params('UnknownModel', all_params)
        self.assertNotIn('hidden_dim', unknown_params)
        self.assertNotIn('kernel_size', unknown_params)
        self.assertNotIn('unsupported_param', unknown_params)
        self.assertIn('input_chunk_length', unknown_params)
    
    def test_get_model_info(self):
        """Test getting model information."""
        factory = ModelFactory()
        
        # Create mock models
        mock_models = {}
        for name in ['RNNModel', 'TCNModel']:
            mock_model = Mock()
            mock_model.__class__.__name__ = name
            mock_model.input_chunk_length = 30
            mock_model.output_chunk_length = 5
            mock_model.n_epochs = 50
            mock_model.batch_size = 32
            mock_models[name] = mock_model
        
        model_info = factory.get_model_info(mock_models)
        
        self.assertEqual(len(model_info), 2)
        
        for name in ['RNNModel', 'TCNModel']:
            self.assertIn(name, model_info)
            info = model_info[name]
            self.assertEqual(info['model_class'], name)
            self.assertEqual(info['input_chunk_length'], 30)
            self.assertEqual(info['output_chunk_length'], 5)
            self.assertEqual(info['n_epochs'], 50)
            self.assertEqual(info['batch_size'], 32)
            self.assertTrue(info['supports_multivariate'])
    
    def test_get_model_info_with_error(self):
        """Test getting model information when model raises error."""
        factory = ModelFactory()
        
        # Create mock model that raises error during getattr() call
        mock_model = Mock()
        mock_model.__class__.__name__ = 'ErrorModel'
        # Make getattr raise an exception for specific attributes
        def side_effect_getattr(obj, name, default=None):
            if name == 'input_chunk_length':
                raise Exception("Access error")
            return default
        
        # Patch getattr to raise exception
        with patch('builtins.getattr', side_effect=side_effect_getattr):
            model_info = factory.get_model_info({'ErrorModel': mock_model})
        
        self.assertIn('ErrorModel', model_info)
        self.assertIn('error', model_info['ErrorModel'])
        self.assertIn('Failed to get info', model_info['ErrorModel']['error'])
    
    def test_validate_models_for_multivariate(self):
        """Test multivariate validation for models."""
        factory = ModelFactory()
        
        # Create mock models with different multivariate support
        mock_models = {
            'SupportedModel': Mock(supports_multivariate=True),
            'UnsupportedModel': Mock(supports_multivariate=False),
            'NoAttributeModel': Mock(spec=[])  # No supports_multivariate attribute
        }
        
        validation_results = factory.validate_models_for_multivariate(mock_models)
        
        self.assertEqual(len(validation_results), 3)
        self.assertTrue(validation_results['SupportedModel'])
        self.assertFalse(validation_results['UnsupportedModel'])
        self.assertTrue(validation_results['NoAttributeModel'])  # Default to True for neural networks
    
    def test_validate_models_for_multivariate_with_error(self):
        """Test multivariate validation when model raises error."""
        factory = ModelFactory()
        
        # Create mock model that raises error during hasattr check
        mock_model = Mock()
        # Make hasattr raise an exception
        def side_effect_hasattr(obj, name):
            if name == 'supports_multivariate':
                raise Exception("Validation error")
            return False
        
        mock_models = {'ErrorModel': mock_model}
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with patch('builtins.hasattr', side_effect=side_effect_hasattr):
                validation_results = factory.validate_models_for_multivariate(mock_models)
            
            # Check that warning was issued
            self.assertTrue(any("Failed to validate ErrorModel" in str(warning.message) for warning in w))
        
        self.assertFalse(validation_results['ErrorModel'])
    
    def test_update_model_params(self):
        """Test updating model parameters."""
        factory = ModelFactory()
        
        original_epochs = factory.params['n_epochs']
        original_batch_size = factory.params['batch_size']
        
        # Update parameters
        factory.update_model_params(
            n_epochs=100,
            batch_size=64,
            new_param='new_value'
        )
        
        # Check updates
        self.assertEqual(factory.params['n_epochs'], 100)
        self.assertEqual(factory.params['batch_size'], 64)
        self.assertEqual(factory.params['new_param'], 'new_value')
        
        # Ensure CPU configuration is maintained
        self.assertEqual(factory.params['pl_trainer_kwargs']['accelerator'], 'cpu')
        self.assertEqual(factory.params['pl_trainer_kwargs']['devices'], 1)
    
    def test_model_specific_parameters(self):
        """Test that model-specific parameters are correctly defined."""
        factory = ModelFactory()
        
        # Check that all expected models have specific parameters
        expected_models = ['RNNModel', 'TCNModel', 'TransformerModel', 'NBEATSModel',
                          'TFTModel', 'NHiTSModel', 'DLinearModel', 'NLinearModel']
        
        for model_name in expected_models:
            self.assertIn(model_name, factory.MODEL_SPECIFIC_PARAMS)
            params = factory.MODEL_SPECIFIC_PARAMS[model_name]
            self.assertIsInstance(params, dict)
            self.assertGreater(len(params), 0)
    
    def test_cpu_configuration_enforcement(self):
        """Test that CPU configuration is enforced."""
        factory = ModelFactory()
        
        # Try to update with GPU configuration
        factory.update_model_params(
            pl_trainer_kwargs={'accelerator': 'gpu', 'devices': 2}
        )
        
        # Should still enforce CPU
        self.assertEqual(factory.params['pl_trainer_kwargs']['accelerator'], 'cpu')
        self.assertEqual(factory.params['pl_trainer_kwargs']['devices'], 1)
    
    def test_create_single_model_success(self):
        """Test successful single model creation."""
        factory = ModelFactory()
        
        # Mock model class
        mock_model_class = Mock()
        mock_instance = Mock()
        mock_model_class.return_value = mock_instance
        
        result = factory._create_single_model('TestModel', mock_model_class)
        
        self.assertEqual(result, mock_instance)
        mock_model_class.assert_called_once()
    
    def test_create_single_model_failure(self):
        """Test single model creation failure."""
        factory = ModelFactory()
        
        # Mock model class that raises exception
        mock_model_class = Mock(side_effect=Exception("Model creation failed"))
        
        with self.assertRaises(ModelInstantiationError) as context:
            factory._create_single_model('TestModel', mock_model_class)
        
        self.assertIn("Failed to create TestModel", str(context.exception))
        self.assertIn("Model creation failed", str(context.exception))


class TestModelFactoryIntegration(unittest.TestCase):
    """Integration tests for ModelFactory with real DARTS models (if available)."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.factory = ModelFactory(
            input_chunk_length=10,  # Smaller for faster testing
            output_chunk_length=3,
            n_epochs=1,  # Minimal epochs for testing
            batch_size=16
        )
    
    @patch('model_factory.DARTS_AVAILABLE', True)
    def test_integration_model_creation_parameters(self):
        """Test that models are created with correct parameters."""
        # This test verifies parameter passing without actually creating models
        factory = ModelFactory(
            input_chunk_length=15,
            output_chunk_length=7,
            n_epochs=25,
            batch_size=48,
            random_state=999
        )
        
        # Check that parameters are correctly set
        self.assertEqual(factory.params['input_chunk_length'], 15)
        self.assertEqual(factory.params['output_chunk_length'], 7)
        self.assertEqual(factory.params['n_epochs'], 25)
        self.assertEqual(factory.params['batch_size'], 48)
        self.assertEqual(factory.params['random_state'], 999)
        
        # Check CPU configuration
        trainer_kwargs = factory.params['pl_trainer_kwargs']
        self.assertEqual(trainer_kwargs['accelerator'], 'cpu')
        self.assertEqual(trainer_kwargs['devices'], 1)
        self.assertFalse(trainer_kwargs['enable_progress_bar'])
        self.assertFalse(trainer_kwargs['enable_model_summary'])


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)