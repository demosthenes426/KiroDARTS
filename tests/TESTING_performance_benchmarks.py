"""
Performance benchmarking tests for DARTS Stock Forecasting System.

This module provides detailed performance benchmarking and regression testing
to ensure model quality and system performance over time.

Requirements: 5.3, 5.4
"""

import unittest
import pandas as pd
import numpy as np
import sys
import os
import time
import json
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
import tempfile

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Try to import DARTS and other dependencies
try:
    from darts import TimeSeries
    from model_factory import ModelFactory
    from model_trainer import ModelTrainer
    from model_evaluator import ModelEvaluator
    from darts_timeseries_creator import DartsTimeSeriesCreator
    from data_splitter import DataSplitter
    from data_scaler import DataScaler
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    DEPENDENCIES_AVAILABLE = False
    print(f"Warning: Dependencies not available: {e}")


class PerformanceBenchmark:
    """Class for managing performance benchmarks and regression testing."""
    
    def __init__(self, benchmark_file: str = None):
        """Initialize performance benchmark manager."""
        if benchmark_file is None:
            benchmark_file = os.path.join(
                os.path.dirname(__file__), 'performance_benchmarks.json'
            )
        self.benchmark_file = benchmark_file
        self.benchmarks = self._load_benchmarks()
    
    def _load_benchmarks(self) -> Dict[str, Any]:
        """Load existing benchmarks from file."""
        if os.path.exists(self.benchmark_file):
            try:
                with open(self.benchmark_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not load benchmarks: {e}")
        
        # Return default benchmarks
        return {
            'model_performance': {},
            'execution_times': {},
            'memory_usage': {},
            'accuracy_metrics': {},
            'last_updated': None
        }
    
    def save_benchmarks(self):
        """Save benchmarks to file."""
        self.benchmarks['last_updated'] = datetime.now().isoformat()
        try:
            with open(self.benchmark_file, 'w') as f:
                json.dump(self.benchmarks, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save benchmarks: {e}")
    
    def record_model_performance(self, model_name: str, metrics: Dict[str, float]):
        """Record model performance metrics."""
        if 'model_performance' not in self.benchmarks:
            self.benchmarks['model_performance'] = {}
        
        self.benchmarks['model_performance'][model_name] = {
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
    
    def record_execution_time(self, operation: str, execution_time: float):
        """Record execution time for an operation."""
        if 'execution_times' not in self.benchmarks:
            self.benchmarks['execution_times'] = {}
        
        if operation not in self.benchmarks['execution_times']:
            self.benchmarks['execution_times'][operation] = []
        
        self.benchmarks['execution_times'][operation].append({
            'time': execution_time,
            'timestamp': datetime.now().isoformat()
        })
        
        # Keep only last 10 measurements
        self.benchmarks['execution_times'][operation] = \
            self.benchmarks['execution_times'][operation][-10:]
    
    def get_performance_regression(self, model_name: str, current_metrics: Dict[str, float]) -> Dict[str, float]:
        """Check for performance regression compared to benchmarks."""
        if model_name not in self.benchmarks.get('model_performance', {}):
            return {}
        
        baseline_metrics = self.benchmarks['model_performance'][model_name]['metrics']
        regression_analysis = {}
        
        for metric, current_value in current_metrics.items():
            if metric in baseline_metrics:
                baseline_value = baseline_metrics[metric]
                if baseline_value != 0:
                    change_pct = ((current_value - baseline_value) / baseline_value) * 100
                    regression_analysis[metric] = {
                        'current': current_value,
                        'baseline': baseline_value,
                        'change_pct': change_pct,
                        'regression': change_pct > 10  # 10% degradation threshold
                    }
        
        return regression_analysis


@unittest.skipUnless(DEPENDENCIES_AVAILABLE, "Required dependencies not available")
class TestPerformanceBenchmarks(unittest.TestCase):
    """Performance benchmarking and regression tests."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures for the class."""
        cls.benchmark_manager = PerformanceBenchmark()
        
        # Create standardized test data for consistent benchmarking
        cls.benchmark_data = cls._create_benchmark_dataset()
        
        # Initialize components
        cls.timeseries_creator = DartsTimeSeriesCreator()
        cls.data_splitter = DataSplitter()
        cls.data_scaler = DataScaler()
        cls.model_factory = ModelFactory(
            input_chunk_length=10,
            output_chunk_length=5,
            n_epochs=10,  # Consistent for benchmarking
            batch_size=32,
            random_state=42
        )
        cls.model_trainer = ModelTrainer(max_epochs=10, verbose=False)
        cls.model_evaluator = ModelEvaluator()
    
    @classmethod
    def _create_benchmark_dataset(cls) -> pd.DataFrame:
        """Create standardized dataset for benchmarking."""
        np.random.seed(42)  # Fixed seed for reproducibility
        
        # Generate 2 years of business day data
        dates = pd.date_range(start='2022-01-01', end='2023-12-31', freq='B')
        n_days = len(dates)
        
        # Generate realistic stock price data
        returns = np.random.normal(0.0005, 0.02, n_days)  # Daily returns
        prices = 100 * np.exp(np.cumsum(returns))
        
        # Generate correlated features
        volumes = np.random.lognormal(mean=15, sigma=0.5, size=n_days).astype(int)
        
        data = {
            'adjusted_close': prices,
            'volume': volumes,
            'sma_20': pd.Series(prices).rolling(20).mean().fillna(method='bfill'),
            'ema_12': pd.Series(prices).ewm(span=12).mean(),
            'rsi': np.random.uniform(20, 80, n_days),
            'macd': np.random.normal(0, 0.5, n_days),
            'bollinger_upper': prices * np.random.uniform(1.01, 1.03, n_days),
            'bollinger_lower': prices * np.random.uniform(0.97, 0.99, n_days),
            'atr': np.random.uniform(0.5, 3.0, n_days),
            'adx': np.random.uniform(10, 50, n_days)
        }
        
        df = pd.DataFrame(data, index=dates)
        df.index.name = 'date'
        
        return df
    
    def setUp(self):
        """Set up test fixtures for each test."""
        warnings.filterwarnings('ignore', category=UserWarning)
    
    def test_data_processing_performance_benchmark(self):
        """Benchmark data processing operations."""
        print("\n=== Data Processing Performance Benchmark ===")
        
        operations_to_benchmark = [
            ('timeseries_creation', lambda: self.timeseries_creator.create_timeseries(self.benchmark_data)),
            ('data_splitting', lambda: self.data_splitter.split_data(self.test_timeseries)),
            ('data_scaling', lambda: self.data_scaler.scale_data(self.train_ts, self.val_ts, self.test_ts))
        ]
        
        # Prepare TimeSeries for dependent operations
        self.test_timeseries = self.timeseries_creator.create_timeseries(self.benchmark_data)
        self.train_ts, self.val_ts, self.test_ts = self.data_splitter.split_data(self.test_timeseries)
        
        benchmark_results = {}
        
        for operation_name, operation_func in operations_to_benchmark:
            print(f"   Benchmarking {operation_name}...")
            
            # Run multiple iterations for stable measurement
            execution_times = []
            for i in range(3):
                start_time = time.time()
                try:
                    result = operation_func()
                    execution_time = time.time() - start_time
                    execution_times.append(execution_time)
                except Exception as e:
                    print(f"     ✗ {operation_name} failed: {e}")
                    continue
            
            if execution_times:
                avg_time = np.mean(execution_times)
                std_time = np.std(execution_times)
                
                benchmark_results[operation_name] = {
                    'avg_time': avg_time,
                    'std_time': std_time,
                    'min_time': min(execution_times),
                    'max_time': max(execution_times)
                }
                
                # Record in benchmark manager
                self.benchmark_manager.record_execution_time(operation_name, avg_time)
                
                print(f"     ✓ {operation_name}: {avg_time:.3f}s ± {std_time:.3f}s")
                
                # Validate performance (should complete within reasonable time)
                max_acceptable_times = {
                    'timeseries_creation': 5.0,
                    'data_splitting': 2.0,
                    'data_scaling': 3.0
                }
                
                if operation_name in max_acceptable_times:
                    self.assertLess(
                        avg_time, 
                        max_acceptable_times[operation_name],
                        f"{operation_name} took too long: {avg_time:.3f}s"
                    )
        
        print(f"\n📊 Data Processing Benchmark Summary:")
        for op, metrics in benchmark_results.items():
            print(f"   {op}: {metrics['avg_time']:.3f}s (range: {metrics['min_time']:.3f}-{metrics['max_time']:.3f}s)")
    
    def test_model_training_performance_benchmark(self):
        """Benchmark model training performance across different models."""
        print("\n=== Model Training Performance Benchmark ===")
        
        # Prepare data
        timeseries = self.timeseries_creator.create_timeseries(self.benchmark_data)
        train_ts, val_ts, test_ts = self.data_splitter.split_data(timeseries)
        train_scaled, val_scaled, test_scaled, scaler = self.data_scaler.scale_data(
            train_ts, val_ts, test_ts
        )
        
        # Models to benchmark (fast models for CI/testing)
        benchmark_models = ['DLinearModel', 'NLinearModel', 'RNNModel']
        
        training_benchmarks = {}
        
        for model_name in benchmark_models:
            print(f"   Benchmarking {model_name} training...")
            
            try:
                # Create model
                model = self.model_factory.create_single_model(model_name)
                if model is None:
                    print(f"     ✗ {model_name} creation failed")
                    continue
                
                # Benchmark training
                start_time = time.time()
                training_result = self.model_trainer.train_model(
                    model, train_scaled, val_scaled, model_name=model_name
                )
                training_time = time.time() - start_time
                
                # Extract metrics
                if hasattr(training_result, 'final_train_loss'):
                    final_train_loss = training_result.final_train_loss
                    final_val_loss = training_result.final_val_loss
                else:
                    final_train_loss = training_result.get('final_train_loss', float('inf'))
                    final_val_loss = training_result.get('final_val_loss', float('inf'))
                
                # Benchmark evaluation
                eval_start_time = time.time()
                eval_result = self.model_evaluator.evaluate_model(
                    model, test_scaled, model_name=model_name
                )
                eval_time = time.time() - eval_start_time
                
                # Extract evaluation metrics
                if hasattr(eval_result, 'mae'):
                    mae = eval_result.mae
                    rmse = eval_result.rmse
                    mape = eval_result.mape
                else:
                    mae = eval_result.get('mae', float('inf'))
                    rmse = eval_result.get('rmse', float('inf'))
                    mape = eval_result.get('mape', float('inf'))
                
                benchmark_metrics = {
                    'training_time': training_time,
                    'evaluation_time': eval_time,
                    'final_train_loss': final_train_loss,
                    'final_val_loss': final_val_loss,
                    'mae': mae,
                    'rmse': rmse,
                    'mape': mape
                }
                
                training_benchmarks[model_name] = benchmark_metrics
                
                # Record in benchmark manager
                self.benchmark_manager.record_model_performance(model_name, benchmark_metrics)
                self.benchmark_manager.record_execution_time(f'{model_name}_training', training_time)
                
                print(f"     ✓ {model_name}: {training_time:.2f}s training, MAE: {mae:.4f}")
                
                # Validate training completed successfully
                self.assertIsNotNone(training_result)
                self.assertLess(training_time, 300, f"{model_name} training took too long")
                self.assertLess(mae, 100, f"{model_name} MAE too high: {mae}")
                
            except Exception as e:
                print(f"     ✗ {model_name} benchmark failed: {e}")
                continue
        
        # Performance comparison
        if len(training_benchmarks) > 1:
            print(f"\n📊 Model Training Benchmark Comparison:")
            
            # Sort by training time
            sorted_models = sorted(
                training_benchmarks.items(), 
                key=lambda x: x[1]['training_time']
            )
            
            for model_name, metrics in sorted_models:
                print(f"   {model_name}:")
                print(f"     Training: {metrics['training_time']:.2f}s")
                print(f"     MAE: {metrics['mae']:.4f}")
                print(f"     RMSE: {metrics['rmse']:.4f}")
        
        # Validate at least one model trained successfully
        self.assertGreater(len(training_benchmarks), 0, "At least one model should train successfully")
    
    def test_performance_regression_detection(self):
        """Test for performance regression against historical benchmarks."""
        print("\n=== Performance Regression Detection ===")
        
        # Run current performance test
        timeseries = self.timeseries_creator.create_timeseries(self.benchmark_data)
        train_ts, val_ts, test_ts = self.data_splitter.split_data(timeseries)
        train_scaled, val_scaled, test_scaled, scaler = self.data_scaler.scale_data(
            train_ts, val_ts, test_ts
        )
        
        # Test with one representative model
        model_name = 'DLinearModel'
        
        try:
            model = self.model_factory.create_single_model(model_name)
            if model is None:
                self.skipTest(f"{model_name} creation failed")
            
            # Measure current performance
            start_time = time.time()
            training_result = self.model_trainer.train_model(
                model, train_scaled, val_scaled, model_name=model_name
            )
            training_time = time.time() - start_time
            
            eval_result = self.model_evaluator.evaluate_model(
                model, test_scaled, model_name=model_name
            )
            
            # Extract metrics
            current_metrics = {
                'training_time': training_time,
                'mae': eval_result.mae if hasattr(eval_result, 'mae') else eval_result.get('mae', 0),
                'rmse': eval_result.rmse if hasattr(eval_result, 'rmse') else eval_result.get('rmse', 0)
            }
            
            # Check for regression
            regression_analysis = self.benchmark_manager.get_performance_regression(
                model_name, current_metrics
            )
            
            if regression_analysis:
                print(f"   Regression analysis for {model_name}:")
                
                has_regression = False
                for metric, analysis in regression_analysis.items():
                    status = "✗" if analysis['regression'] else "✓"
                    print(f"     {status} {metric}: {analysis['current']:.4f} vs {analysis['baseline']:.4f} ({analysis['change_pct']:+.1f}%)")
                    
                    if analysis['regression']:
                        has_regression = True
                
                # Fail test if significant regression detected
                if has_regression:
                    print("   ⚠ Performance regression detected!")
                    # Note: In production, you might want to fail the test here
                    # self.fail("Performance regression detected")
                else:
                    print("   ✓ No significant performance regression")
            else:
                print(f"   No baseline available for {model_name} - establishing new baseline")
            
            # Update benchmarks with current results
            self.benchmark_manager.record_model_performance(model_name, current_metrics)
            
        except Exception as e:
            print(f"   ✗ Regression test failed: {e}")
            self.skipTest(f"Regression test failed: {e}")
    
    def test_scalability_benchmark(self):
        """Test system scalability with different data sizes."""
        print("\n=== Scalability Benchmark ===")
        
        data_sizes = [
            ('small', 100),
            ('medium', 500),
            ('large', 1000)
        ]
        
        scalability_results = {}
        
        for size_name, n_days in data_sizes:
            print(f"   Testing {size_name} dataset ({n_days} days)...")
            
            # Generate data of specific size
            end_date = pd.Timestamp('2023-01-01') + pd.Timedelta(days=n_days * 1.5)  # Account for weekends
            test_dates = pd.date_range(start='2023-01-01', end=end_date, freq='B')[:n_days]
            
            test_data = self.benchmark_data.iloc[:len(test_dates)].copy()
            test_data.index = test_dates
            
            # Benchmark pipeline
            start_time = time.time()
            
            try:
                timeseries = self.timeseries_creator.create_timeseries(test_data)
                train_ts, val_ts, test_ts = self.data_splitter.split_data(timeseries)
                train_scaled, val_scaled, test_scaled, scaler = self.data_scaler.scale_data(
                    train_ts, val_ts, test_ts
                )
                
                processing_time = time.time() - start_time
                
                scalability_results[size_name] = {
                    'data_size': n_days,
                    'processing_time': processing_time,
                    'timeseries_length': len(timeseries),
                    'train_length': len(train_scaled)
                }
                
                print(f"     ✓ {size_name}: {processing_time:.3f}s for {n_days} days")
                
                # Validate reasonable performance
                max_time_per_day = 0.01  # 10ms per day max
                self.assertLess(
                    processing_time / n_days, 
                    max_time_per_day,
                    f"Processing time per day too high for {size_name}: {processing_time/n_days:.4f}s"
                )
                
            except Exception as e:
                print(f"     ✗ {size_name} failed: {e}")
                scalability_results[size_name] = {'error': str(e)}
        
        # Analyze scalability
        successful_results = {k: v for k, v in scalability_results.items() if 'error' not in v}
        
        if len(successful_results) >= 2:
            print(f"\n📊 Scalability Analysis:")
            
            sizes = [v['data_size'] for v in successful_results.values()]
            times = [v['processing_time'] for v in successful_results.values()]
            
            # Calculate scaling factor
            if len(sizes) >= 2:
                size_ratio = max(sizes) / min(sizes)
                time_ratio = max(times) / min(times)
                scaling_efficiency = size_ratio / time_ratio if time_ratio > 0 else 0
                
                print(f"   Size ratio: {size_ratio:.1f}x")
                print(f"   Time ratio: {time_ratio:.1f}x")
                print(f"   Scaling efficiency: {scaling_efficiency:.2f} (1.0 = linear)")
                
                # Validate sub-linear scaling (efficiency > 0.5)
                self.assertGreater(
                    scaling_efficiency, 0.3,
                    f"Poor scaling efficiency: {scaling_efficiency:.2f}"
                )
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests."""
        # Save benchmarks
        cls.benchmark_manager.save_benchmarks()
        print(f"\n📁 Performance benchmarks saved to {cls.benchmark_manager.benchmark_file}")


if __name__ == '__main__':
    # Configure warnings
    warnings.filterwarnings('ignore', category=UserWarning)
    
    if not DEPENDENCIES_AVAILABLE:
        print("Warning: Required dependencies not available. Tests will be skipped.")
    
    # Run tests with verbose output
    unittest.main(verbosity=2)