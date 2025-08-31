"""
ModelTrainer class for training DARTS neural network models.

This module provides a trainer for DARTS models with CPU-only configuration,
training/validation loss monitoring, and early stopping capabilities.
"""

from typing import Dict, Any, List, Optional, Tuple
import warnings
import time
from dataclasses import dataclass

# Try to import DARTS components, handle gracefully if not available
try:
    from darts import TimeSeries
    from darts.models.forecasting.forecasting_model import ForecastingModel
    from pytorch_lightning.callbacks import Callback
    DARTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: DARTS not available: {e}")
    TimeSeries = Any
    ForecastingModel = Any
    Callback = object  # Dummy class
    DARTS_AVAILABLE = False


@dataclass
class TrainingResults:
    """Data class to store training results and metrics."""
    model_name: str
    train_loss: List[float]
    val_loss: List[float]
    training_time: float
    final_train_loss: float
    final_val_loss: float
    epochs_completed: int
    early_stopped: bool
    convergence_achieved: bool


class ModelTrainingError(Exception):
    """Custom exception for model training errors."""
    pass


class LossLogger(Callback):
    """
    PyTorch Lightning callback to log training and validation losses.
    """
    def __init__(self):
        super().__init__()
        self.train_loss = []
        self.val_loss = []

    def on_train_epoch_end(self, trainer, pl_module):
        """Log training loss at the end of each training epoch."""
        if 'train_loss' in trainer.callback_metrics:
            self.train_loss.append(trainer.callback_metrics['train_loss'].item())

    def on_validation_epoch_end(self, trainer, pl_module):
        """Log validation loss at the end of each validation epoch."""
        if not trainer.sanity_checking:  # Exclude sanity check run
            if 'val_loss' in trainer.callback_metrics:
                self.val_loss.append(trainer.callback_metrics['val_loss'].item())


class ModelTrainer:
    """
    Trainer class for DARTS neural network models.
    
    Provides training functionality with CPU-only configuration, loss monitoring,
    and early stopping capabilities for time series forecasting models.
    """
    
    def __init__(self, 
                 early_stopping_patience: int = 10,
                 min_delta: float = 1e-4,
                 max_epochs: Optional[int] = None,
                 verbose: bool = True):
        """
        Initialize ModelTrainer with training configuration. 
        
        Args:
            early_stopping_patience (int): Number of epochs to wait for improvement
            min_delta (float): Minimum change to qualify as improvement
            max_epochs (Optional[int]): Maximum epochs to override model default
            verbose (bool): Whether to print training progress
        """
        self.early_stopping_patience = early_stopping_patience
        self.min_delta = min_delta
        self.max_epochs = max_epochs
        self.verbose = verbose
        
        # Training state
        self.training_history = {}
        
    def train_model(self, 
                   model: ForecastingModel,
                   train_ts: TimeSeries,
                   val_ts: TimeSeries,
                   model_name: str = "Unknown") -> TrainingResults:
        """
        Train a DARTS model with monitoring and early stopping.
        
        Args:
            model: DARTS forecasting model to train
            train_ts: Training TimeSeries data
            val_ts: Validation TimeSeries data
            model_name: Name of the model for logging
            
        Returns:
            TrainingResults: Training results and metrics
            
        Raises:
            ModelTrainingError: If training fails
        """
        if not DARTS_AVAILABLE:
            raise ModelTrainingError("DARTS library is not available. Please install darts.")
        
        if self.verbose:
            print(f"\n🚀 Starting training for {model_name}")
            print(f"   Training data: {len(train_ts)} points")
            print(f"   Validation data: {len(val_ts)} points")
        
        try:
            # Configure model for CPU training and add loss logger
            loss_logger = self._configure_trainer(model)
            
            # Override epochs if specified
            if self.max_epochs is not None and hasattr(model, 'n_epochs'):
                original_epochs = model.n_epochs
                model.n_epochs = self.max_epochs
                if self.verbose:
                    print(f"   Overriding epochs: {original_epochs} → {self.max_epochs}")
            
            # Start training
            start_time = time.time()
            model.fit(train_ts, val_series=val_ts)
            training_time = time.time() - start_time
            
            # Extract training history from the logger
            train_loss = loss_logger.train_loss
            val_loss = loss_logger.val_loss
            
            # Analyze training results
            final_train_loss = train_loss[-1] if train_loss else float('inf')
            final_val_loss = val_loss[-1] if val_loss else float('inf')
            epochs_completed = len(train_loss)
            
            convergence_achieved = self._check_convergence(train_loss, val_loss)
            early_stopped = self._check_early_stopping(val_loss)
            
            if self.verbose:
                print(f"   ✓ Training completed in {training_time:.2f}s")
                print(f"   ✓ Epochs: {epochs_completed}")
                print(f"   ✓ Final train loss: {final_train_loss:.6f}")
                print(f"   ✓ Final val loss: {final_val_loss:.6f}")
                print(f"   ✓ Convergence: {'Yes' if convergence_achieved else 'No'}")
                print(f"   ✓ Early stopped: {'Yes' if early_stopped else 'No'}")
            
            results = TrainingResults(
                model_name=model_name,
                train_loss=train_loss,
                val_loss=val_loss,
                training_time=training_time,
                final_train_loss=final_train_loss,
                final_val_loss=final_val_loss,
                epochs_completed=epochs_completed,
                early_stopped=early_stopped,
                convergence_achieved=convergence_achieved
            )
            
            self.training_history[model_name] = results
            return results
            
        except Exception as e:
            error_msg = f"Failed to train {model_name}: {str(e)}"
            if self.verbose:
                print(f"   ❌ {error_msg}")
            raise ModelTrainingError(error_msg) from e

    def _configure_trainer(self, model: ForecastingModel) -> LossLogger:
        """
        Configure model's pl_trainer_kwargs for CPU training and add loss logger.
        
        Args:
            model: DARTS forecasting model to configure.
            
        Returns:
            LossLogger: The instance of the loss logger callback.
        """
        loss_logger = LossLogger()
        
        if not hasattr(model, 'pl_trainer_kwargs') or model.pl_trainer_kwargs is None:
            model.pl_trainer_kwargs = {}
            
        # Ensure CPU configuration
        model.pl_trainer_kwargs.update({
            'accelerator': 'cpu',
            'devices': 1,
            'enable_progress_bar': self.verbose,
            'enable_model_summary': False,
        })
        
        # Add our loss logger to the callbacks
        callbacks = model.pl_trainer_kwargs.get('callbacks', [])
        if not any(isinstance(c, LossLogger) for c in callbacks):
            callbacks.append(loss_logger)
        model.pl_trainer_kwargs['callbacks'] = callbacks

        if hasattr(model, 'force_reset'):
            model.force_reset = True
        if hasattr(model, 'save_checkpoints'):
            model.save_checkpoints = False
            
        return loss_logger

    def _check_convergence(self, train_loss: List[float], val_loss: List[float]) -> bool:
        """
        Check if training has converged (losses are generally decreasing).
        
        Args:
            train_loss: Training loss history.
            val_loss: Validation loss history.
            
        Returns:
            bool: True if convergence is achieved.
        """
        if len(train_loss) < 3 or len(val_loss) < 3:
            return False
        
        return val_loss[-1] < val_loss[-3]

    def _check_early_stopping(self, val_loss: List[float]) -> bool:
        """
        Check if early stopping would have been triggered based on validation loss.
        
        Args:
            val_loss: Validation loss history.
            
        Returns:
            bool: True if early stopping condition is met.
        """
        if len(val_loss) < self.early_stopping_patience + 1:
            return False
        
        # Get the last `patience` number of validation losses
        recent_losses = val_loss[-self.early_stopping_patience:]
        
        # Check if the loss has not improved by min_delta
        for i in range(len(recent_losses) - 1):
            if recent_losses[i] - recent_losses[i+1] > self.min_delta:
                return False # There was an improvement
        
        return True
