#!/usr/bin/env python3
"""
DARTS Stock Forecasting System - Main Execution Script

This script implements a comprehensive time series forecasting pipeline using DARTS
neural network models to predict stock prices 5 business days into the future.

The script is structured like a Jupyter notebook with standardized cell division
comments for easy experimentation and development.
"""

# === CELL: markdown ===
# # DARTS Stock Forecasting System
# 
# This notebook implements a comprehensive time series forecasting pipeline that:
# - Processes multi-variate stock data with custom business day handling
# - Tests multiple DARTS neural network models
# - Provides comprehensive evaluation and visualization
# - Saves model artifacts for future predictions

# === CELL: code ===
# ## Import Dependencies

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import logging
import argparse
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timedelta

# DARTS imports
try:
    from darts import TimeSeries
    DARTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: DARTS not available: {e}")
    DARTS_AVAILABLE = False

# Custom component imports
from src.data_loader import DataLoader
from src.data_preprocessor import DataPreprocessor
from src.custom_holiday_calendar import CustomHolidayCalendar
from src.darts_timeseries_creator import DartsTimeSeriesCreator
from src.data_splitter import DataSplitter
from src.data_scaler import DataScaler
from src.target_creator import TargetCreator
from src.model_factory import ModelFactory
from src.model_trainer import ModelTrainer
from src.model_evaluator import ModelEvaluator
from src.results_visualizer import ResultsVisualizer
from src.model_artifact_saver import ModelArtifactSaver
from src.data_integrity_validator import DataIntegrityValidator

warnings.filterwarnings('ignore')

# === CELL: markdown ===
# # Configuration and Constants

# === CELL: code ===
# ## Global Configuration

def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('darts_forecasting.log')
        ]
    )
    return logging.getLogger(__name__)

def setup_directories() -> Dict[str, Path]:
    """Setup output directories."""
    paths = {
        'data': Path("Data/covariaterawdata1.csv"),
        'output': Path("output"),
        'models': Path("output/models"),
        'plots': Path("output/plots"),
        'artifacts': Path("model_artifacts")
    }
    
    # Create output directories
    for key, path in paths.items():
        if key != 'data':  # Don't create data file
            path.mkdir(exist_ok=True)
    
    return paths

# Global configuration constants
CONFIG = {
    'prediction_horizon': 5,  # 5-day future prediction
    'train_ratio': 0.7,
    'val_ratio': 0.15,
    'test_ratio': 0.15,
    'epochs': 100,
    'batch_size': 32,
    'learning_rate': 0.001,
    'early_stopping_patience': 10,
    'input_chunk_length': 30,
    'output_chunk_length': 5
}

# === CELL: markdown ===
# # Main Pipeline Functions

# === CELL: code ===
# ## Pipeline Implementation

def load_and_preprocess_data(data_path: Path, logger: logging.Logger) -> pd.DataFrame:
    """Load and preprocess raw stock data."""
    logger.info("=== Data Loading and Preprocessing Phase ===")
    
    try:
        # Load raw data
        logger.info(f"Loading data from {data_path}")
        data_loader = DataLoader()
        raw_df = data_loader.load_data(str(data_path))
        logger.info(f"Loaded {len(raw_df)} rows of data")
        
        # Preprocess data
        logger.info("Preprocessing data...")
        preprocessor = DataPreprocessor()
        processed_df = preprocessor.preprocess_data(raw_df)
        logger.info(f"Preprocessed data shape: {processed_df.shape}")
        
        return processed_df
        
    except Exception as e:
        logger.error(f"Error in data loading/preprocessing: {e}")
        raise

def create_timeseries_with_calendar(df: pd.DataFrame, logger: logging.Logger) -> Tuple[TimeSeries, Any]:
    """Create DARTS TimeSeries with custom holiday calendar."""
    logger.info("=== Custom Holiday Calendar and TimeSeries Creation Phase ===")
    
    try:
        # Create custom holiday calendar (for reference)
        logger.info("Creating custom holiday calendar...")
        calendar_creator = CustomHolidayCalendar()
        custom_freq, holidays = calendar_creator.create_custom_calendar(df)
        logger.info(f"Created custom calendar with {len(holidays)} holidays")
        
        # Create DARTS TimeSeries (uses data-driven holiday discovery internally)
        logger.info("Converting to DARTS TimeSeries...")
        ts_creator = DartsTimeSeriesCreator()
        timeseries = ts_creator.create_timeseries(df)
        logger.info(f"Created TimeSeries with {len(timeseries)} points")
        
        return timeseries, custom_freq
        
    except Exception as e:
        logger.error(f"Error in TimeSeries creation: {e}")
        raise

def split_and_scale_data(timeseries: TimeSeries, config: Dict, logger: logging.Logger) -> Tuple[TimeSeries, TimeSeries, TimeSeries, Any]:
    """Split and scale the TimeSeries data."""
    logger.info("=== Data Splitting and Scaling Phase ===")
    
    try:
        # Split data temporally (uses fixed 70/15/15 ratios)
        logger.info("Splitting data temporally...")
        splitter = DataSplitter()
        train_ts, val_ts, test_ts = splitter.split_data(timeseries)
        logger.info(f"Split sizes - Train: {len(train_ts)}, Val: {len(val_ts)}, Test: {len(test_ts)}")
        
        # Scale features
        logger.info("Scaling features...")
        scaler = DataScaler()
        scaled_train, scaled_val, scaled_test, fitted_scaler = scaler.scale_data(train_ts, val_ts, test_ts)
        logger.info("Data scaling completed")
        
        return scaled_train, scaled_val, scaled_test, fitted_scaler
        
    except Exception as e:
        logger.error(f"Error in data splitting/scaling: {e}")
        raise

def create_targets(timeseries: TimeSeries, config: Dict, logger: logging.Logger) -> TimeSeries:
    """Create target variables for prediction."""
    logger.info("=== Target Creation Phase ===")
    
    try:
        target_creator = TargetCreator(config['prediction_horizon'])
        # Use column index 0 for adjusted_close (first column after preprocessing)
        targets = target_creator.create_targets(timeseries, "0")
        logger.info(f"Created targets with {len(targets)} points")
        return targets
        
    except Exception as e:
        logger.error(f"Error in target creation: {e}")
        raise

def train_all_models(train_ts: TimeSeries, val_ts: TimeSeries, config: Dict, logger: logging.Logger) -> Dict[str, Any]:
    """Train all available DARTS models."""
    logger.info("=== Model Factory and Training Phase ===")
    
    try:
        # Create models
        logger.info("Creating model instances...")
        model_factory = ModelFactory()
        models = model_factory.create_models()
        logger.info(f"Created {len(models)} models: {list(models.keys())}")
        
        # Train models
        logger.info("Training models...")
        trainer = ModelTrainer()
        training_results = {}
        
        for model_name, model in models.items():
            logger.info(f"Training {model_name}...")
            try:
                result = trainer.train_model(model, train_ts, val_ts)
                training_results[model_name] = {
                    'model': model,
                    'results': result
                }
                logger.info(f"{model_name} training completed - Final loss: {result.final_val_loss:.4f}")
            except Exception as e:
                logger.warning(f"Failed to train {model_name}: {e}")
                continue
        
        logger.info(f"Successfully trained {len(training_results)} models")
        return training_results
        
    except Exception as e:
        logger.error(f"Error in model training: {e}")
        raise

def evaluate_and_visualize(training_results: Dict, test_ts: TimeSeries, paths: Dict[str, Path], logger: logging.Logger) -> Dict[str, Any]:
    """Evaluate models and create visualizations."""
    logger.info("=== Model Evaluation and Visualization Phase ===")
    
    try:
        evaluator = ModelEvaluator()
        visualizer = ResultsVisualizer()
        evaluation_results = {}
        
        # Evaluate each model
        for model_name, model_data in training_results.items():
            logger.info(f"Evaluating {model_name}...")
            try:
                eval_result = evaluator.evaluate_model(model_data['model'], test_ts)
                evaluation_results[model_name] = eval_result
                logger.info(f"{model_name} - MAE: {eval_result.mae:.4f}, RMSE: {eval_result.rmse:.4f}")
            except Exception as e:
                logger.warning(f"Failed to evaluate {model_name}: {e}")
                continue
        
        # Create visualizations
        logger.info("Creating visualizations...")
        visualizer.visualize_results(evaluation_results, str(paths['plots']))
        logger.info("Visualizations saved")
        
        return evaluation_results
        
    except Exception as e:
        logger.error(f"Error in evaluation/visualization: {e}")
        raise

def save_artifacts(training_results: Dict, scaler: Any, paths: Dict[str, Path], logger: logging.Logger):
    """Save model artifacts for future use."""
    logger.info("=== Model Artifact Management Phase ===")
    
    try:
        artifact_saver = ModelArtifactSaver()
        
        for model_name, model_data in training_results.items():
            logger.info(f"Saving artifacts for {model_name}...")
            try:
                metadata = {
                    'model_name': model_name,
                    'training_results': model_data['results'].__dict__,
                    'timestamp': datetime.now().isoformat()
                }
                
                artifact_path = artifact_saver.save_artifacts(
                    model_data['model'], 
                    scaler, 
                    metadata,
                    str(paths['artifacts'])
                )
                logger.info(f"Saved {model_name} artifacts to {artifact_path}")
            except Exception as e:
                logger.warning(f"Failed to save artifacts for {model_name}: {e}")
                continue
        
    except Exception as e:
        logger.error(f"Error in artifact saving: {e}")
        raise

def validate_data_integrity(timeseries: TimeSeries, scaler: Any, logger: logging.Logger):
    """Run comprehensive data integrity validation."""
    logger.info("=== Data Integrity Validation Phase ===")
    
    try:
        validator = DataIntegrityValidator()
        validation_report = validator.validate_data_integrity(timeseries, scaler)
        
        if validation_report.data_integrity_passed:
            logger.info("✓ Data integrity validation PASSED")
        else:
            logger.warning("✗ Data integrity validation FAILED")
            for issue in validation_report.issues:
                logger.warning(f"  Issue: {issue}")
        
        for warning in validation_report.warnings:
            logger.warning(f"  Warning: {warning}")
            
    except Exception as e:
        logger.error(f"Error in data integrity validation: {e}")
        raise

def generate_final_report(training_results: Dict, evaluation_results: Dict, logger: logging.Logger):
    """Generate final performance report."""
    logger.info("=== Final Results and Summary Phase ===")
    
    try:
        logger.info("DARTS Stock Forecasting Pipeline Complete!")
        logger.info("=" * 60)
        logger.info("FINAL PERFORMANCE SUMMARY")
        logger.info("=" * 60)
        
        # Sort models by performance (MAE)
        sorted_results = sorted(
            evaluation_results.items(), 
            key=lambda x: x[1].mae
        )
        
        logger.info("Model Performance Ranking (by MAE):")
        for i, (model_name, result) in enumerate(sorted_results, 1):
            logger.info(f"{i:2d}. {model_name:15s} - MAE: {result.mae:.4f}, RMSE: {result.rmse:.4f}, MAPE: {result.mape:.2f}%")
        
        # Training convergence summary
        logger.info("\nTraining Convergence Summary:")
        for model_name, model_data in training_results.items():
            result = model_data['results']
            status = "✓ Converged" if result.convergence_achieved else "✗ No convergence"
            logger.info(f"{model_name:15s} - {status}, Final Val Loss: {result.final_val_loss:.4f}")
        
        logger.info("\nCheck output directories for:")
        logger.info("- Saved models and artifacts: model_artifacts/")
        logger.info("- Visualization plots: output/plots/")
        logger.info("- Training logs: darts_forecasting.log")
        
    except Exception as e:
        logger.error(f"Error generating final report: {e}")
        raise

# === CELL: code ===
# ## Main Execution Pipeline

def run_full_pipeline(data_path: str, log_level: str = "INFO", models_subset: Optional[List[str]] = None) -> bool:
    """
    Run the complete DARTS forecasting pipeline.
    
    Args:
        data_path: Path to the CSV data file
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        models_subset: Optional list of specific models to train
        
    Returns:
        bool: True if pipeline completed successfully
    """
    # Setup
    logger = setup_logging(log_level)
    paths = setup_directories()
    
    # Override data path if provided
    if data_path:
        paths['data'] = Path(data_path)
    
    logger.info("Starting DARTS Stock Forecasting System...")
    logger.info(f"Data path: {paths['data']}")
    logger.info(f"DARTS available: {DARTS_AVAILABLE}")
    
    if not DARTS_AVAILABLE:
        logger.error("DARTS library not available. Please install DARTS to run the pipeline.")
        return False
    
    try:
        # 1. Load and preprocess data
        processed_df = load_and_preprocess_data(paths['data'], logger)
        
        # 2. Create TimeSeries with custom calendar
        timeseries, custom_freq = create_timeseries_with_calendar(processed_df, logger)
        
        # 3. Split and scale data
        train_ts, val_ts, test_ts, scaler = split_and_scale_data(timeseries, CONFIG, logger)
        
        # 4. Create targets (for validation purposes)
        targets = create_targets(timeseries, CONFIG, logger)
        
        # 5. Train models
        training_results = train_all_models(train_ts, val_ts, CONFIG, logger)
        
        if not training_results:
            logger.error("No models were successfully trained. Pipeline failed.")
            return False
        
        # 6. Evaluate and visualize
        evaluation_results = evaluate_and_visualize(training_results, test_ts, paths, logger)
        
        # 7. Save artifacts
        save_artifacts(training_results, scaler, paths, logger)
        
        # 8. Validate data integrity
        validate_data_integrity(timeseries, scaler, logger)
        
        # 9. Generate final report
        generate_final_report(training_results, evaluation_results, logger)
        
        logger.info("Pipeline completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def run_quick_test(data_path: str, log_level: str = "INFO") -> bool:
    """
    Run a quick test with a subset of models for faster execution.
    
    Args:
        data_path: Path to the CSV data file
        log_level: Logging level
        
    Returns:
        bool: True if test completed successfully
    """
    logger = setup_logging(log_level)
    logger.info("Running quick test with subset of models...")
    
    # Use only fast models for quick testing
    quick_models = ['DLinearModel', 'NLinearModel']
    
    # Temporarily reduce epochs for quick testing
    original_epochs = CONFIG['epochs']
    CONFIG['epochs'] = 10
    
    try:
        success = run_full_pipeline(data_path, log_level, quick_models)
        return success
    finally:
        # Restore original configuration
        CONFIG['epochs'] = original_epochs

# === CELL: code ===
# ## Command Line Interface

def main():
    """Main entry point with command line interface."""
    parser = argparse.ArgumentParser(description='DARTS Stock Forecasting System')
    parser.add_argument('--data-path', '-d', 
                       default='Data/covariaterawdata1.csv',
                       help='Path to CSV data file (default: Data/covariaterawdata1.csv)')
    parser.add_argument('--log-level', '-l', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO',
                       help='Logging level (default: INFO)')
    parser.add_argument('--quick-test', '-q', 
                       action='store_true',
                       help='Run quick test with subset of models')
    parser.add_argument('--models-subset', '-m',
                       nargs='+',
                       help='Specific models to train (e.g., RNNModel TCNModel)')
    
    args = parser.parse_args()
    
    try:
        if args.quick_test:
            success = run_quick_test(args.data_path, args.log_level)
        else:
            success = run_full_pipeline(args.data_path, args.log_level, args.models_subset)
        
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()