"""
ModelTrainer class for training DARTS neural network models.

This module provides a trainer for DARTS models with CPU-only configuration,
training/validation loss monitoring, and early stopping capabilities.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import warnings
import time
from dataclasses import dataclass

# Try to import DARTS components, handle gracefully if not available
try:
    from darts import TimeSeries
    from darts.models.forecasting.forecasting_model import ForecastingModel
    from pytorch_lightning.loggers import CSVLogger
    DARTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: DARTS not available: {e}")
    TimeSeries = Any
    ForecastingModel = Any
    CSVLogger = object # Dummy class
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
            print(f"\nStarting training for {model_name}")
            print(f"   Training data: {len(train_ts)} points")
            print(f"   Validation data: {len(val_ts)} points")
        
        original_epochs = None
        try:
            # Configure model for CPU training and add loss logger
            logger = self._configure_trainer(model, model_name)
            
            # Override epochs if specified
            if self.max_epochs is not None and hasattr(model, 'n_epochs'):
                original_epochs = model.n_epochs
                model.n_epochs = self.max_epochs
                if self.verbose:
                    print(f"   Overriding epochs: {original_epochs} -> {self.max_epochs}")
            
            # Start training
            start_time = time.time()
            model.fit(train_ts, val_series=val_ts)
            training_time = time.time() - start_time
            
            # --- Read metrics from CSV logger ---
            log_dir = logger.log_dir
            metrics_path = f"{log_dir}/metrics.csv"
            try:
                metrics_df = pd.read_csv(metrics_path)
                train_loss = metrics_df['train_loss'].dropna().tolist()
                val_loss = metrics_df['val_loss'].dropna().tolist()
            except (FileNotFoundError, KeyError):
                train_loss = []
                val_loss = []

            # More reliable way to get epochs completed
            epochs_completed = 0
            if hasattr(model, 'trainer'):
                epochs_completed = model.trainer.current_epoch
            elif train_loss:
                epochs_completed = len(train_loss)

            # Analyze training results
            final_train_loss = train_loss[-1] if train_loss else float('inf')
            final_val_loss = val_loss[-1] if val_loss else float('inf')
            
            convergence_achieved = self._check_convergence(train_loss, val_loss)
            early_stopped = self._check_early_stopping(val_loss)
            
            if self.verbose:
                print(f"   V Training completed in {training_time:.2f}s")
                print(f"   V Epochs: {epochs_completed}")
                print(f"   V Final train loss: {final_train_loss:.6f}")
                print(f"   V Final val loss: {final_val_loss:.6f}")
                print(f"   V Convergence: {'Yes' if convergence_achieved else 'No'}")
                print(f"   V Early stopped: {'Yes' if early_stopped else 'No'}")
            
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
                print(f"   X {error_msg}")
            raise ModelTrainingError(error_msg) from e
        finally:
            if original_epochs is not None:
                model.n_epochs = original_epochs

    def _configure_trainer(self, model: ForecastingModel, model_name: str) -> CSVLogger:
        """
        Configure model's pl_trainer_kwargs for CPU training and add a CSV logger.
        
        Args:
            model: DARTS forecasting model to configure.
            model_name: The name of the model, used for the logger.
            
        Returns:
            CSVLogger: The instance of the CSV logger.
        """
        logger = CSVLogger(save_dir="temp_logs", name=model_name)
        
        if not hasattr(model, 'pl_trainer_kwargs') or model.pl_trainer_kwargs is None:
            model.pl_trainer_kwargs = {}
            
        # Ensure CPU configuration
        model.pl_trainer_kwargs.update({
            'accelerator': 'cpu',
            'devices': 1,
            'enable_progress_bar': self.verbose,
            'enable_model_summary': False,
            'logger': logger
        })
        
        if hasattr(model, 'force_reset'):
            model.force_reset = True
        if hasattr(model, 'save_checkpoints'):
            model.save_checkpoints = False
            
        return logger

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
    
    def train_multiple_models(self, 
                            models: Dict[str, ForecastingModel], 
                            train_ts: TimeSeries, 
                            val_ts: TimeSeries) -> Dict[str, TrainingResults]:
        """
        Train multiple models and return results.
        
        Args:
            models: Dictionary of model name to model instance
            train_ts: Training TimeSeries data
            val_ts: Validation TimeSeries data
            
        Returns:
            Dict[str, TrainingResults]: Training results for each model
        """
        results = {}
        failed_models = []
        
        if self.verbose:
            print(f"\n🚀 Training {len(models)} models...")
        
        for model_name, model in models.items():
            try:
                result = self.train_model(model, train_ts, val_ts, model_name)
                results[model_name] = result
                
            except Exception as e:
                failed_models.append((model_name, str(e)))
                warnings.warn(f"Failed to train {model_name}: {e}")
                continue
        
        if self.verbose:
            print(f"\n📊 Training Summary:")
            print(f"   ✓ Successful: {len(results)}")
            print(f"   ❌ Failed: {len(failed_models)}")
            
            if failed_models:
                print(f"\n   Failed models:")
                for model_name, error in failed_models:
                    print(f"     - {model_name}: {error}")
        
        return results
    
    def get_training_summary(self) -> Dict[str, Any]:
        """
        Get summary of all training results.
        
        Returns:
            Dict[str, Any]: Summary statistics
        """
        if not self.training_history:
            return {"message": "No training history available"}
        
        summary = {
            "total_models_trained": len(self.training_history),
            "successful_models": [],
            "converged_models": [],
            "early_stopped_models": [],
            "average_training_time": 0,
            "average_final_val_loss": 0,
            "best_model_by_val_loss": None
        }
        
        training_times = []
        val_losses = []
        best_val_loss = float('inf')
        
        for model_name, results in self.training_history.items():
            summary["successful_models"].append(model_name)
            
            if results.convergence_achieved:
                summary["converged_models"].append(model_name)
            
            if results.early_stopped:
                summary["early_stopped_models"].append(model_name)
            
            training_times.append(results.training_time)
            val_losses.append(results.final_val_loss)
            
            if results.final_val_loss < best_val_loss:
                best_val_loss = results.final_val_loss
                summary["best_model_by_val_loss"] = model_name
        
        if training_times:
            summary["average_training_time"] = np.mean(training_times)
            summary["average_final_val_loss"] = np.mean(val_losses)
        
        return summary
    
    def validate_training_requirements(self, train_ts: TimeSeries, val_ts: TimeSeries) -> bool:
        """
        Validate that training data meets requirements.
        
        Args:
            train_ts: Training TimeSeries
            val_ts: Validation TimeSeries
            
        Returns:
            bool: True if requirements are met
            
        Raises:
            ModelTrainingError: If validation fails
        """
        try:
            # Check minimum data requirements
            min_train_length = 30  # Minimum for meaningful training
            min_val_length = 10    # Minimum for validation
            
            if len(train_ts) < min_train_length:
                raise ModelTrainingError(
                    f"Training data too small: {len(train_ts)} points (minimum {min_train_length})"
                )
            
            if len(val_ts) < min_val_length:
                raise ModelTrainingError(
                    f"Validation data too small: {len(val_ts)} points (minimum {min_val_length})"
                )
            
            # Check for NaN values
            try:
                train_df = train_ts.pd_dataframe() if hasattr(train_ts, 'pd_dataframe') else train_ts.to_pandas()
                val_df = val_ts.pd_dataframe() if hasattr(val_ts, 'pd_dataframe') else val_ts.to_pandas()
                
                if train_df.isnull().any().any():
                    raise ModelTrainingError("Training data contains NaN values")
                
                if val_df.isnull().any().any():
                    raise ModelTrainingError("Validation data contains NaN values")
            except Exception:
                # If we can't check for NaN values, skip this validation
                pass
            
            # Check feature consistency
            if len(train_ts.columns) != len(val_ts.columns):
                raise ModelTrainingError(
                    f"Feature count mismatch: train={len(train_ts.columns)}, val={len(val_ts.columns)}"
                )
            
            return True
            
        except Exception as e:
            if isinstance(e, ModelTrainingError):
                raise
            raise ModelTrainingError(f"Training validation failed: {e}") from e
