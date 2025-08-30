"""
ModelFactory class for creating DARTS neural network models.

This module provides a factory for instantiating all available DARTS neural network models
configured for multi-variate input and CPU execution.
"""

from typing import Dict, Any, Optional
import warnings

# Try to import DARTS models, handle gracefully if not available
try:
    from darts.models import (
        RNNModel,
        TCNModel,
        TransformerModel,
        NBEATSModel,
        TFTModel,
        NHiTSModel,
        DLinearModel,
        NLinearModel
    )
    DARTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: DARTS models not available: {e}")
    # Create dummy classes for testing
    RNNModel = TCNModel = TransformerModel = NBEATSModel = None
    TFTModel = NHiTSModel = DLinearModel = NLinearModel = None
    DARTS_AVAILABLE = False


class ModelInstantiationError(Exception):
    """Custom exception for model instantiation errors."""
    pass


class ModelFactory:
    """
    Factory class for creating DARTS neural network models.
    
    Creates and configures all available DARTS neural network models for multi-variate
    time series forecasting with CPU execution and consistent parameters.
    """
    
    # Default parameters for all models
    DEFAULT_PARAMS = {
        'input_chunk_length': 30,
        'output_chunk_length': 5,
        'n_epochs': 50,
        'batch_size': 32,
        'random_state': 42,
        'force_reset': True,
        'save_checkpoints': False,
    }
    
    # Model-specific parameters
    MODEL_SPECIFIC_PARAMS = {
        'RNNModel': {
            'model': 'LSTM',
            'hidden_dim': 64,
            'n_rnn_layers': 2,
            'dropout': 0.1,
        },
        'TCNModel': {
            'kernel_size': 3,
            'num_filters': 64,
            'num_layers': 3,
            'dilation_base': 2,
            'dropout': 0.1,
        },
        'TransformerModel': {
            'd_model': 64,
            'nhead': 4,
            'num_encoder_layers': 3,
            'num_decoder_layers': 3,
            'dim_feedforward': 256,
            'dropout': 0.1,
        },
        'NBEATSModel': {
            'num_blocks': 3,
            'num_layers': 4,
            'layer_widths': 256,
            'expansion_coefficient_dim': 5,
        },
        'TFTModel': {
            'hidden_size': 64,
            'lstm_layers': 2,
            'num_attention_heads': 4,
            'dropout': 0.1,
        },
        'NHiTSModel': {
            'num_blocks': 3,
            'num_layers': 2,
            'layer_widths': 256,
            'pooling_kernel_sizes': [2, 2, 2],
            'n_freq_downsample': [2, 2, 2],
        },
        'DLinearModel': {
            'kernel_size': 25,
            'shared_weights': False,
        },
        'NLinearModel': {
            'shared_weights': False,
        }
    }
    
    def __init__(self, 
                 input_chunk_length: int = 30,
                 output_chunk_length: int = 5,
                 n_epochs: int = 50,
                 batch_size: int = 32,
                 random_state: int = 42):
        """
        Initialize ModelFactory with configuration parameters.
        
        Args:
            input_chunk_length (int): Number of input time steps
            output_chunk_length (int): Number of output time steps to predict
            n_epochs (int): Number of training epochs
            batch_size (int): Training batch size
            random_state (int): Random seed for reproducibility
        """
        self.params = self.DEFAULT_PARAMS.copy()
        self.params.update({
            'input_chunk_length': input_chunk_length,
            'output_chunk_length': output_chunk_length,
            'n_epochs': n_epochs,
            'batch_size': batch_size,
            'random_state': random_state,
        })
        
        # Force CPU execution
        self.params['pl_trainer_kwargs'] = {
            'accelerator': 'cpu',
            'devices': 1,
            'enable_progress_bar': False,
            'enable_model_summary': False,
        }
    
    def create_models(self) -> Dict[str, Any]:
        """
        Create all available DARTS neural network models.
        
        Returns:
            Dict[str, Any]: Dictionary mapping model names to instantiated model objects
            
        Raises:
            ModelInstantiationError: If model creation fails
        """
        if not DARTS_AVAILABLE:
            raise ModelInstantiationError("DARTS library is not available. Please install darts.")
        
        models = {}
        failed_models = []
        
        # List of model classes and their names
        model_classes = [
            ('RNNModel', RNNModel),
            ('TCNModel', TCNModel),
            ('TransformerModel', TransformerModel),
            ('NBEATSModel', NBEATSModel),
            ('TFTModel', TFTModel),
            ('NHiTSModel', NHiTSModel),
            ('DLinearModel', DLinearModel),
            ('NLinearModel', NLinearModel),
        ]
        
        # Filter out None classes (in case some models are not available)
        model_classes = [(name, cls) for name, cls in model_classes if cls is not None]
        
        for model_name, model_class in model_classes:
            try:
                model = self._create_single_model(model_name, model_class)
                models[model_name] = model
                print(f"✓ Successfully created {model_name}")
                
            except Exception as e:
                failed_models.append((model_name, str(e)))
                warnings.warn(f"Failed to create {model_name}: {e}")
                continue
        
        if not models:
            raise ModelInstantiationError(
                f"Failed to create any models. Errors: {failed_models}"
            )
        
        if failed_models:
            print(f"\nWarning: {len(failed_models)} models failed to instantiate:")
            for model_name, error in failed_models:
                print(f"  - {model_name}: {error}")
        
        print(f"\nSuccessfully created {len(models)} models: {list(models.keys())}")
        return models
    
    def _create_single_model(self, model_name: str, model_class: type) -> Any:
        """
        Create a single model instance with appropriate parameters.
        
        Args:
            model_name (str): Name of the model
            model_class (type): Model class to instantiate
            
        Returns:
            Any: Instantiated model object
            
        Raises:
            ModelInstantiationError: If model creation fails
        """
        try:
            # Combine default params with model-specific params
            model_params = self.params.copy()
            
            if model_name in self.MODEL_SPECIFIC_PARAMS:
                model_params.update(self.MODEL_SPECIFIC_PARAMS[model_name])
            
            # Remove parameters that might not be supported by all models
            safe_params = self._filter_supported_params(model_name, model_params)
            
            # Create model instance
            model = model_class(**safe_params)
            
            return model
            
        except Exception as e:
            raise ModelInstantiationError(f"Failed to create {model_name}: {e}")
    
    def _filter_supported_params(self, model_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Filter parameters to only include those supported by the specific model.
        
        Args:
            model_name (str): Name of the model
            params (Dict[str, Any]): All parameters
            
        Returns:
            Dict[str, Any]: Filtered parameters
        """
        # Common parameters supported by most models
        common_params = [
            'input_chunk_length',
            'output_chunk_length',
            'n_epochs',
            'batch_size',
            'random_state',
            'force_reset',
            'save_checkpoints',
            'pl_trainer_kwargs',
        ]
        
        # Model-specific parameter filtering
        model_specific_filters = {
            'DLinearModel': common_params + ['kernel_size', 'shared_weights'],
            'NLinearModel': common_params + ['shared_weights'],
            'RNNModel': common_params + ['model', 'hidden_dim', 'n_rnn_layers', 'dropout'],
            'TCNModel': common_params + ['kernel_size', 'num_filters', 'num_layers', 'dilation_base', 'dropout'],
            'TransformerModel': common_params + ['d_model', 'nhead', 'num_encoder_layers', 'num_decoder_layers', 'dim_feedforward', 'dropout'],
            'NBEATSModel': common_params + ['num_blocks', 'num_layers', 'layer_widths', 'expansion_coefficient_dim'],
            'TFTModel': common_params + ['hidden_size', 'lstm_layers', 'num_attention_heads', 'dropout'],
            'NHiTSModel': common_params + ['num_blocks', 'num_layers', 'layer_widths', 'pooling_kernel_sizes', 'n_freq_downsample'],
        }
        
        if model_name in model_specific_filters:
            allowed_params = model_specific_filters[model_name]
            filtered_params = {k: v for k, v in params.items() if k in allowed_params}
        else:
            # For unknown models, use common parameters only
            filtered_params = {k: v for k, v in params.items() if k in common_params}
        
        return filtered_params
    
    def get_model_info(self, models: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Get information about created models.
        
        Args:
            models (Dict[str, Any]): Dictionary of created models
            
        Returns:
            Dict[str, Dict[str, Any]]: Model information
        """
        model_info = {}
        
        for model_name, model in models.items():
            try:
                info = {
                    'model_class': type(model).__name__,
                    'input_chunk_length': getattr(model, 'input_chunk_length', 'Unknown'),
                    'output_chunk_length': getattr(model, 'output_chunk_length', 'Unknown'),
                    'n_epochs': getattr(model, 'n_epochs', 'Unknown'),
                    'batch_size': getattr(model, 'batch_size', 'Unknown'),
                    'supports_multivariate': True,
                }
                
                model_info[model_name] = info
                
            except Exception as e:
                model_info[model_name] = {'error': f"Failed to get info: {e}"}
        
        return model_info
    
    def validate_models_for_multivariate(self, models: Dict[str, Any]) -> Dict[str, bool]:
        """
        Validate that models support multivariate input.
        
        Args:
            models (Dict[str, Any]): Dictionary of created models
            
        Returns:
            Dict[str, bool]: Validation results for each model
        """
        validation_results = {}
        
        for model_name, model in models.items():
            try:
                # All DARTS neural network models should support multivariate input
                supports_multivariate = True
                
                # Additional checks could be added here based on model capabilities
                if hasattr(model, 'supports_multivariate'):
                    supports_multivariate = model.supports_multivariate
                
                validation_results[model_name] = supports_multivariate
                
            except Exception as e:
                validation_results[model_name] = False
                warnings.warn(f"Failed to validate {model_name}: {e}")
        
        return validation_results
    
    def create_single_model(self, model_name: str) -> Any:
        """
        Create a single model by name.
        
        Args:
            model_name (str): Name of the model to create
            
        Returns:
            Any: Instantiated model object
            
        Raises:
            ModelInstantiationError: If model creation fails
        """
        if not DARTS_AVAILABLE:
            raise ModelInstantiationError("DARTS library is not available. Please install darts.")
        
        # Map model names to classes
        model_classes = {
            'RNNModel': RNNModel,
            'TCNModel': TCNModel,
            'TransformerModel': TransformerModel,
            'NBEATSModel': NBEATSModel,
            'TFTModel': TFTModel,
            'NHiTSModel': NHiTSModel,
            'DLinearModel': DLinearModel,
            'NLinearModel': NLinearModel,
        }
        
        if model_name not in model_classes:
            raise ModelInstantiationError(f"Unknown model: {model_name}")
        
        model_class = model_classes[model_name]
        if model_class is None:
            raise ModelInstantiationError(f"Model {model_name} is not available")
        
        return self._create_single_model(model_name, model_class)
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """
        Get information about a specific model or all models.
        
        Args:
            model_name (str): Name of the model, or pass a dict of models for backward compatibility
            
        Returns:
            Dict[str, Any]: Model information
        """
        # Handle backward compatibility - if a dict is passed, process all models
        if isinstance(model_name, dict):
            models = model_name
            model_info = {}
            
            for name, model in models.items():
                try:
                    info = {
                        'model_class': type(model).__name__,
                        'input_chunk_length': getattr(model, 'input_chunk_length', 'Unknown'),
                        'output_chunk_length': getattr(model, 'output_chunk_length', 'Unknown'),
                        'n_epochs': getattr(model, 'n_epochs', 'Unknown'),
                        'batch_size': getattr(model, 'batch_size', 'Unknown'),
                        'supports_multivariate': True,
                    }
                    
                    model_info[name] = info
                    
                except Exception as e:
                    model_info[name] = {'error': f"Failed to get info: {e}"}
            
            return model_info
        
        # Handle single model info
        try:
            model = self.create_single_model(model_name)
            return {
                'model_class': type(model).__name__,
                'input_chunk_length': getattr(model, 'input_chunk_length', 'Unknown'),
                'output_chunk_length': getattr(model, 'output_chunk_length', 'Unknown'),
                'n_epochs': getattr(model, 'n_epochs', 'Unknown'),
                'batch_size': getattr(model, 'batch_size', 'Unknown'),
                'supports_multivariate': True,
            }
        except Exception as e:
            return {'error': f"Failed to get info: {e}"}
    
    def update_model_params(self, **kwargs) -> None:
        """
        Update default parameters for model creation.
        
        Args:
            **kwargs: Parameters to update
        """
        self.params.update(kwargs)
        
        # Ensure CPU execution is maintained
        if 'pl_trainer_kwargs' not in self.params:
            self.params['pl_trainer_kwargs'] = {}
        
        self.params['pl_trainer_kwargs'].update({
            'accelerator': 'cpu',
            'devices': 1,
        })