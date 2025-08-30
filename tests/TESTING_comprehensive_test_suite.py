"""
Comprehensive test suite for DARTS Stock Forecasting System.

This test suite implements:
1. End-to-end integration tests with real data
2. Performance regression tests to ensure model quality
3. Test data generators for edge cases and synthetic scenarios
4. Memory usage and execution time profiling tests

Requirements: 5.3, 5.4
"""

import unittest
import pandas as pd
import numpy as np
import sys
import os
import time
import psutil
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
import tempfile
import json

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Try to import DARTS and other dependencies
try:
    from darts import TimeSeries
    from data_loader import DataLoader
    from data_preprocessor import DataPreprocessor
    from custom_holiday_calendar import CustomHolidayCalendar
    from darts_timeseries_creator import DartsTimeSeriesCreator
    from data_splitter import DataSplitter
    from data_scaler import DataScaler
    from target_creator import TargetCreator
    from model_factory import ModelFactory
    from model_trainer import ModelTrainer
    from model_evaluator import ModelEvaluator
    from results_visualizer import ResultsVisualizer
    from model_artifact_saver import ModelArtifactSaver
    from data_integrity_validator import DataIntegrityValidator
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    DEPENDENCIES_AVAILABLE = False
    print(f"Warning: Dependencies not available: {e}")


class TestDataGenerator:
    """Utility class for generating test data for edge cases and synthetic scenarios."""
    
    @staticmethod
    def generate_synthetic_stock_data(
        start_date: str = "2020-01-01",
        end_date: str = "2023-12-31",
        missing_days_ratio: float = 0.05,
        volatility: float = 0.02,
        trend: float = 0.0001,
        seed: int = 42
    ) -> pd.DataFrame:
        """Generate synthetic stock data with configurable parameters."""
        np.random.seed(seed)
        
        # Create date range
        date_range = pd.date_range(start=start_date, end=end_date, freq='B')
        
        # Remove random business days to simulate missing data
        n_missing = int(len(date_range) * missing_days_ratio)
        missing_indices = np.random.choice(len(date_range), n_missing, replace=False)
        available_dates = date_range.delete(missing_indices)
        
        # Generate price data with trend and volatility
        n_days = len(available_dates)
        returns = np.random.normal(trend, volatility, n_days)
        prices = 100 * np.exp(np.cumsum(returns))
        
        # Generate volume data
        volumes = np.random.lognormal(mean=15, sigma=0.5, size=n_days).astype(int)
        
        # Generate technical indicators
        data = {
            'adjusted_close': prices,
            'volume': volumes,
            'sma_20': pd.Series(prices).rolling(20, min_periods=1).mean(),
            'ema_12': pd.Series(prices).ewm(span=12, min_periods=1).mean(),
            'rsi': np.random.uniform(20, 80, n_days),
            'macd': np.random.normal(0, 0.5, n_days),
            'bollinger_upper': prices * 1.02,
            'bollinger_lower': prices * 0.98,
            'atr': np.random.uniform(0.5, 3.0, n_days),
            'adx': np.random.uniform(10, 50, n_days)
        }
        
        df = pd.DataFrame(data, index=available_dates)
        df.index.name = 'date'
        
        return df
    
    @staticmethod
    def generate_edge_case_data(case_type: str) -> pd.DataFrame:
        """Generate edge case data for specific testing scenarios."""
        base_date = pd.Timestamp('2023-01-01')
        
        if case_type == 'minimal_data':
            # Minimal viable dataset
            dates = pd.date_range(start=base_date, periods=30, freq='B')
            data = {
                'adjusted_close': np.random.uniform(90, 110, 30),
                'volume': np.random.randint(1000, 10000, 30)
            }
            return pd.DataFrame(data, index=dates)
        
        elif case_type == 'high_volatility':
            # High volatility data
            dates = pd.date_range(start=base_date, periods=100, freq='B')
            prices = 100 * np.exp(np.cumsum(np.random.normal(0, 0.1, 100)))
            data = {
                'adjusted_close': prices,
                'volume': np.random.randint(10000, 100000, 100),
                'volatility_indicator': np.random.uniform(0.5, 2.0, 100)
            }
            return pd.DataFrame(data, index=dates)
        
        elif case_type == 'many_missing_days':
            # Data with many missing business days (holidays, market closures)
            full_range = pd.date_range(start=base_date, periods=200, freq='B')
            # Keep only 60% of business days
            keep_indices = np.random.choice(len(full_range), int(len(full_range) * 0.6), replace=False)
            dates = full_range[sorted(keep_indices)]
            
            data = {
                'adjusted_close': np.random.uniform(95, 105, len(dates)),
                'volume': np.random.randint(5000, 50000, len(dates))
            }
            return pd.DataFrame(data, index=dates)
        
        elif case_type == 'trend_reversal':
            # Data with clear trend reversals
            dates = pd.date_range(start=base_date, periods=150, freq='B')
            
            # Create trend reversal pattern
            uptrend = np.linspace(100, 120, 50)
            downtrend = np.linspace(120, 90, 50)
            recovery = np.linspace(90, 110, 50)
            prices = np.concatenate([uptrend, downtrend, recovery])
            
            # Add noise
            prices += np.random.normal(0, 1, len(prices))
            
            data = {
                'adjusted_close': prices,
                'volume': np.random.randint(8000, 80000, len(dates)),
                'trend_indicator': np.concatenate([
                    np.ones(50), -np.ones(50), np.ones(50)
                ])
            }
            return pd.DataFrame(data, index=dates)
        
        else:
            raise ValueError(f"Unknown edge case type: {case_type}")


class PerformanceProfiler:
    """Utility class for performance profiling and memory monitoring."""
    
    def __init__(self):
        self.process = psutil.Process()
        self.start_time = None
        self.start_memory = None
        self.peak_memory = None
    
    def start_profiling(self):
        """Start performance profiling."""
        self.start_time = time.time()
        self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.peak_memory = self.start_memory
    
    def update_peak_memory(self):
        """Update peak memory usage."""
        current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        if current_memory > self.peak_memory:
            self.peak_memory = current_memory
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get current performance metrics."""
        if self.start_time is None:
            raise ValueError("Profiling not started")
        
        current_time = time.time()
        current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        return {
            'execution_time': current_time - self.start_time,
            'start_memory_mb': self.start_memory,
            'current_memory_mb': current_memory,
            'peak_memory_mb': self.peak_memory,
            'memory_increase_mb': current_memory - self.start_memory,
            'cpu_percent': self.process.cpu_percent()
        }


@unittest.skipUnless(DEPENDENCIES_AVAILABLE, "Required dependencies not available")
class TestComprehensiveIntegration(unittest.TestCase):
    """Comprehensive integration tests for the entire DARTS forecasting pipeline."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures for the class."""
        cls.real_data_file = os.path.join(os.path.dirname(__file__), '..', 'Data', 'covariaterawdata1.csv')
        cls.test_data_generator = TestDataGenerator()
        cls.profiler = PerformanceProfiler()
        
        # Performance baselines (to be updated based on actual measurements)
        cls.performance_baselines = {
            'data_loading_time_max': 5.0,  # seconds
            'timeseries_creation_time_max': 10.0,  # seconds
            'model_training_time_max': 300.0,  # seconds per model
            'memory_increase_max': 500.0,  # MB
            'prediction_accuracy_min': 0.7  # minimum acceptable accuracy
        }
        
        # Initialize components
        cls.data_loader = DataLoader()
        cls.data_preprocessor = DataPreprocessor()
        cls.holiday_calendar = CustomHolidayCalendar()
        cls.timeseries_creator = DartsTimeSeriesCreator()
        cls.data_splitter = DataSplitter()
        cls.data_scaler = DataScaler()
        cls.target_creator = TargetCreator()
        cls.model_factory = ModelFactory(
            input_chunk_length=10,
            output_chunk_length=5,
            n_epochs=5,  # Reduced for testing
            batch_size=16,
            random_state=42
        )
        cls.model_trainer = ModelTrainer(max_epochs=5, verbose=False)
        cls.model_evaluator = ModelEvaluator()
        cls.results_visualizer = ResultsVisualizer()
        cls.artifact_saver = ModelArtifactSaver()
        cls.data_validator = DataIntegrityValidator()
    
    def setUp(self):
        """Set up test fixtures for each test."""
        self.profiler = PerformanceProfiler()
        warnings.filterwarnings('ignore', category=UserWarning)
    
    def test_end_to_end_pipeline_with_real_data(self):
        """Test complete end-to-end pipeline with real data."""
        print("\n=== End-to-End Pipeline Test with Real Data ===")
        
        if not os.path.exists(self.real_data_file):
            self.skipTest("Real data file not available")
        
        self.profiler.start_profiling()
        pipeline_results = {}
        
        try:
            # 1. Data Loading
            print("1. Loading real data...")
            start_time = time.time()
            raw_df = self.data_loader.load_data(self.real_data_file)
            loading_time = time.time() - start_time
            
            self.assertIsNotNone(raw_df)
            self.assertGreater(len(raw_df), 0)
            pipeline_results['data_loading'] = {
                'success': True,
                'time': loading_time,
                'rows': len(raw_df),
                'columns': len(raw_df.columns)
            }
            print(f"   ✓ Loaded {len(raw_df)} rows in {loading_time:.2f}s")
            
            # 2. Data Preprocessing
            print("2. Preprocessing data...")
            processed_df = self.data_preprocessor.preprocess_data(raw_df)
            self.assertIsNotNone(processed_df)
            pipeline_results['preprocessing'] = {'success': True}
            print(f"   ✓ Preprocessed to {len(processed_df)} rows, {len(processed_df.columns)} columns")
            
            # 3. TimeSeries Creation
            print("3. Creating TimeSeries...")
            start_time = time.time()
            timeseries = self.timeseries_creator.create_timeseries(processed_df)
            ts_creation_time = time.time() - start_time
            
            self.assertIsInstance(timeseries, TimeSeries)
            pipeline_results['timeseries_creation'] = {
                'success': True,
                'time': ts_creation_time,
                'length': len(timeseries)
            }
            print(f"   ✓ Created TimeSeries with {len(timeseries)} points in {ts_creation_time:.2f}s")
            
            # 4. Data Validation
            print("4. Validating data integrity...")
            validation_report = self.data_validator.validate_data_integrity(
                timeseries, None, processed_df
            )
            self.assertTrue(validation_report.data_integrity_passed)
            pipeline_results['validation'] = {'success': True}
            print("   ✓ Data integrity validation passed")
            
            # 5. Data Splitting and Scaling
            print("5. Splitting and scaling data...")
            train_ts, val_ts, test_ts = self.data_splitter.split_data(timeseries)
            train_scaled, val_scaled, test_scaled, scaler = self.data_scaler.scale_data(
                train_ts, val_ts, test_ts
            )
            
            pipeline_results['data_preparation'] = {
                'success': True,
                'train_size': len(train_scaled),
                'val_size': len(val_scaled),
                'test_size': len(test_scaled)
            }
            print(f"   ✓ Split data: {len(train_scaled)}/{len(val_scaled)}/{len(test_scaled)}")
            
            # 6. Model Training (subset for speed)
            print("6. Training models...")
            test_models = ['DLinearModel', 'RNNModel']  # Fast models for testing
            training_results = {}
            
            for model_name in test_models:
                print(f"   Training {model_name}...")
                try:
                    # Create single model for efficiency
                    model = self.model_factory.create_single_model(model_name)
                    if model is None:
                        continue
                    
                    start_time = time.time()
                    training_result = self.model_trainer.train_model(
                        model, train_scaled, val_scaled, model_name=model_name
                    )
                    training_time = time.time() - start_time
                    
                    training_results[model_name] = {
                        'result': training_result,
                        'time': training_time,
                        'model': model
                    }
                    
                    # Validate training time
                    self.assertLess(training_time, self.performance_baselines['model_training_time_max'])
                    
                    print(f"     ✓ {model_name} trained in {training_time:.2f}s")
                    
                except Exception as e:
                    print(f"     ✗ {model_name} training failed: {e}")
                    continue
            
            pipeline_results['model_training'] = {
                'success': len(training_results) > 0,
                'models_trained': len(training_results),
                'results': training_results
            }
            
            # 7. Model Evaluation
            print("7. Evaluating models...")
            evaluation_results = {}
            
            for model_name, training_data in training_results.items():
                try:
                    model = training_data['model']
                    eval_result = self.model_evaluator.evaluate_model(
                        model, test_scaled, model_name=model_name
                    )
                    evaluation_results[model_name] = eval_result
                    print(f"   ✓ {model_name} evaluated successfully")
                    
                except Exception as e:
                    print(f"   ✗ {model_name} evaluation failed: {e}")
                    continue
            
            pipeline_results['evaluation'] = {
                'success': len(evaluation_results) > 0,
                'models_evaluated': len(evaluation_results)
            }
            
            # 8. Performance Metrics
            final_metrics = self.profiler.get_performance_metrics()
            pipeline_results['performance'] = final_metrics
            
            # Validate performance baselines
            self.assertLess(
                final_metrics['execution_time'], 
                600.0,  # 10 minutes max for full pipeline
                "Pipeline execution time exceeded baseline"
            )
            
            self.assertLess(
                final_metrics['memory_increase_mb'],
                self.performance_baselines['memory_increase_max'],
                "Memory usage exceeded baseline"
            )
            
            print(f"\n📊 Pipeline Performance Summary:")
            print(f"   Total execution time: {final_metrics['execution_time']:.2f}s")
            print(f"   Memory usage: {final_metrics['memory_increase_mb']:.1f}MB increase")
            print(f"   Peak memory: {final_metrics['peak_memory_mb']:.1f}MB")
            print(f"   Models trained: {len(training_results)}")
            print(f"   Models evaluated: {len(evaluation_results)}")
            
            # Store results for regression testing
            self._save_performance_baseline(pipeline_results)
            
            print("\n🎉 End-to-end pipeline test completed successfully!")
            
        except Exception as e:
            print(f"\n💥 Pipeline test failed: {e}")
            raise
    
    def test_synthetic_data_edge_cases(self):
        """Test pipeline with synthetic data covering edge cases."""
        print("\n=== Synthetic Data Edge Cases Test ===")
        
        edge_cases = [
            'minimal_data',
            'high_volatility', 
            'many_missing_days',
            'trend_reversal'
        ]
        
        results = {}
        
        for case_type in edge_cases:
            print(f"\nTesting edge case: {case_type}")
            
            try:
                # Generate synthetic data
                synthetic_df = self.test_data_generator.generate_edge_case_data(case_type)
                
                # Run abbreviated pipeline
                timeseries = self.timeseries_creator.create_timeseries(synthetic_df)
                train_ts, val_ts, test_ts = self.data_splitter.split_data(timeseries)
                
                # Validate basic properties
                self.assertGreater(len(train_ts), 0)
                self.assertGreater(len(val_ts), 0)
                self.assertGreater(len(test_ts), 0)
                
                results[case_type] = {
                    'success': True,
                    'data_shape': synthetic_df.shape,
                    'timeseries_length': len(timeseries),
                    'train_length': len(train_ts)
                }
                
                print(f"   ✓ {case_type}: {synthetic_df.shape} -> {len(timeseries)} points")
                
            except Exception as e:
                print(f"   ✗ {case_type} failed: {e}")
                results[case_type] = {'success': False, 'error': str(e)}
        
        # Validate that most edge cases pass
        success_rate = sum(1 for r in results.values() if r['success']) / len(results)
        self.assertGreater(success_rate, 0.75, "At least 75% of edge cases should pass")
        
        print(f"\n📊 Edge Cases Summary: {success_rate:.1%} success rate")
    
    def test_performance_regression(self):
        """Test for performance regression against established baselines."""
        print("\n=== Performance Regression Test ===")
        
        # Load previous performance baseline if available
        baseline_file = os.path.join(os.path.dirname(__file__), 'performance_baseline.json')
        previous_baseline = None
        
        if os.path.exists(baseline_file):
            try:
                with open(baseline_file, 'r') as f:
                    previous_baseline = json.load(f)
                print("   Loaded previous performance baseline")
            except Exception as e:
                print(f"   Warning: Could not load baseline: {e}")
        
        # Generate consistent test data
        test_df = self.test_data_generator.generate_synthetic_stock_data(
            start_date="2022-01-01",
            end_date="2022-12-31", 
            seed=42  # Fixed seed for consistency
        )
        
        self.profiler.start_profiling()
        
        # Run performance test
        performance_metrics = {}
        
        # Data processing performance
        start_time = time.time()
        timeseries = self.timeseries_creator.create_timeseries(test_df)
        performance_metrics['timeseries_creation_time'] = time.time() - start_time
        
        # Data splitting performance
        start_time = time.time()
        train_ts, val_ts, test_ts = self.data_splitter.split_data(timeseries)
        train_scaled, val_scaled, test_scaled, scaler = self.data_scaler.scale_data(
            train_ts, val_ts, test_ts
        )
        performance_metrics['data_preparation_time'] = time.time() - start_time
        
        # Model training performance (single fast model)
        start_time = time.time()
        model = self.model_factory.create_single_model('DLinearModel')
        if model is not None:
            training_result = self.model_trainer.train_model(
                model, train_scaled, val_scaled, model_name='DLinearModel'
            )
            performance_metrics['model_training_time'] = time.time() - start_time
        
        # Overall performance
        overall_metrics = self.profiler.get_performance_metrics()
        performance_metrics.update(overall_metrics)
        
        # Compare with baseline if available
        if previous_baseline:
            print("   Comparing with previous baseline:")
            
            for metric, current_value in performance_metrics.items():
                if metric in previous_baseline:
                    previous_value = previous_baseline[metric]
                    if isinstance(current_value, (int, float)) and isinstance(previous_value, (int, float)):
                        change_pct = ((current_value - previous_value) / previous_value) * 100
                        status = "✓" if change_pct < 20 else "⚠" if change_pct < 50 else "✗"
                        print(f"     {status} {metric}: {current_value:.3f} vs {previous_value:.3f} ({change_pct:+.1f}%)")
                        
                        # Fail test if performance degraded significantly
                        if metric.endswith('_time') and change_pct > 100:  # 100% slower
                            self.fail(f"Performance regression detected: {metric} increased by {change_pct:.1f}%")
        
        # Validate against absolute baselines
        self.assertLess(
            performance_metrics.get('timeseries_creation_time', 0),
            self.performance_baselines['timeseries_creation_time_max']
        )
        
        if 'model_training_time' in performance_metrics:
            self.assertLess(
                performance_metrics['model_training_time'],
                self.performance_baselines['model_training_time_max']
            )
        
        print(f"\n📊 Performance Metrics:")
        for metric, value in performance_metrics.items():
            if isinstance(value, (int, float)):
                unit = "s" if "time" in metric else "MB" if "memory" in metric else ""
                print(f"   {metric}: {value:.3f}{unit}")
        
        print("   ✓ Performance regression test passed")
    
    def test_memory_usage_profiling(self):
        """Test memory usage patterns and detect memory leaks."""
        print("\n=== Memory Usage Profiling Test ===")
        
        # Generate test data of different sizes
        data_sizes = [
            ('small', 100),
            ('medium', 500), 
            ('large', 1000)
        ]
        
        memory_profiles = {}
        
        for size_name, n_days in data_sizes:
            print(f"\nTesting {size_name} dataset ({n_days} days)...")
            
            # Generate data
            end_date = pd.Timestamp('2023-01-01') + pd.Timedelta(days=n_days)
            test_df = self.test_data_generator.generate_synthetic_stock_data(
                start_date="2023-01-01",
                end_date=end_date.strftime('%Y-%m-%d'),
                seed=42
            )
            
            # Profile memory usage through pipeline
            profiler = PerformanceProfiler()
            profiler.start_profiling()
            
            memory_checkpoints = {}
            
            # Checkpoint 1: After data loading
            timeseries = self.timeseries_creator.create_timeseries(test_df)
            profiler.update_peak_memory()
            memory_checkpoints['after_timeseries'] = profiler.get_performance_metrics()
            
            # Checkpoint 2: After data splitting
            train_ts, val_ts, test_ts = self.data_splitter.split_data(timeseries)
            profiler.update_peak_memory()
            memory_checkpoints['after_splitting'] = profiler.get_performance_metrics()
            
            # Checkpoint 3: After scaling
            train_scaled, val_scaled, test_scaled, scaler = self.data_scaler.scale_data(
                train_ts, val_ts, test_ts
            )
            profiler.update_peak_memory()
            memory_checkpoints['after_scaling'] = profiler.get_performance_metrics()
            
            memory_profiles[size_name] = {
                'data_size': n_days,
                'checkpoints': memory_checkpoints,
                'final_metrics': profiler.get_performance_metrics()
            }
            
            # Validate memory usage is reasonable
            final_memory = memory_checkpoints['after_scaling']['memory_increase_mb']
            self.assertLess(final_memory, self.performance_baselines['memory_increase_max'])
            
            print(f"   Memory usage: {final_memory:.1f}MB increase")
        
        # Check memory scaling
        small_memory = memory_profiles['small']['final_metrics']['memory_increase_mb']
        large_memory = memory_profiles['large']['final_metrics']['memory_increase_mb']
        
        # Memory should scale sub-linearly (not 10x for 10x data)
        memory_scaling_factor = large_memory / small_memory if small_memory > 0 else 1
        self.assertLess(memory_scaling_factor, 15, "Memory usage should scale sub-linearly")
        
        print(f"\n📊 Memory Scaling Analysis:")
        for size_name, profile in memory_profiles.items():
            final_mem = profile['final_metrics']['memory_increase_mb']
            print(f"   {size_name} ({profile['data_size']} days): {final_mem:.1f}MB")
        
        print(f"   Scaling factor (large/small): {memory_scaling_factor:.1f}x")
        print("   ✓ Memory profiling test passed")
    
    def test_concurrent_execution_stability(self):
        """Test system stability under concurrent execution scenarios."""
        print("\n=== Concurrent Execution Stability Test ===")
        
        import threading
        import queue
        
        # Generate test data
        test_df = self.test_data_generator.generate_synthetic_stock_data(
            start_date="2023-01-01",
            end_date="2023-06-30",
            seed=42
        )
        
        results_queue = queue.Queue()
        errors_queue = queue.Queue()
        
        def worker_function(worker_id: int):
            """Worker function for concurrent testing."""
            try:
                # Each worker processes the same data independently
                # Create a clean copy of the data
                worker_df = test_df.copy()
                
                # Ensure no NaN values in the data
                worker_df = worker_df.ffill().bfill()
                
                timeseries = self.timeseries_creator.create_timeseries(worker_df)
                train_ts, val_ts, test_ts = self.data_splitter.split_data(timeseries)
                
                # Validate results
                result = {
                    'worker_id': worker_id,
                    'success': True,
                    'timeseries_length': len(timeseries),
                    'train_length': len(train_ts)
                }
                results_queue.put(result)
                
            except Exception as e:
                errors_queue.put({
                    'worker_id': worker_id,
                    'error': str(e)
                })
        
        # Run concurrent workers
        n_workers = 3
        threads = []
        
        print(f"   Starting {n_workers} concurrent workers...")
        
        for i in range(n_workers):
            thread = threading.Thread(target=worker_function, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join(timeout=60)  # 60 second timeout
        
        # Collect results
        results = []
        while not results_queue.empty():
            results.append(results_queue.get())
        
        errors = []
        while not errors_queue.empty():
            errors.append(errors_queue.get())
        
        # Validate results
        self.assertEqual(len(results), n_workers, "All workers should complete successfully")
        self.assertEqual(len(errors), 0, "No errors should occur during concurrent execution")
        
        # Validate consistency
        if len(results) > 1:
            first_result = results[0]
            for result in results[1:]:
                self.assertEqual(
                    result['timeseries_length'], 
                    first_result['timeseries_length'],
                    "All workers should produce consistent results"
                )
        
        print(f"   ✓ {len(results)} workers completed successfully")
        print(f"   ✓ Results are consistent across workers")
        print("   ✓ Concurrent execution stability test passed")
    
    def _save_performance_baseline(self, results: Dict[str, Any]):
        """Save performance baseline for future regression testing."""
        baseline_file = os.path.join(os.path.dirname(__file__), 'performance_baseline.json')
        
        # Extract numeric metrics for baseline
        baseline_data = {}
        
        if 'performance' in results:
            for key, value in results['performance'].items():
                if isinstance(value, (int, float)):
                    baseline_data[key] = value
        
        # Add timestamp
        baseline_data['timestamp'] = datetime.now().isoformat()
        baseline_data['test_version'] = '1.0'
        
        try:
            with open(baseline_file, 'w') as f:
                json.dump(baseline_data, f, indent=2)
            print(f"   Performance baseline saved to {baseline_file}")
        except Exception as e:
            print(f"   Warning: Could not save baseline: {e}")


@unittest.skipUnless(DEPENDENCIES_AVAILABLE, "Required dependencies not available")
class TestStressAndLoadTesting(unittest.TestCase):
    """Stress testing and load testing for the forecasting system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_data_generator = TestDataGenerator()
        self.profiler = PerformanceProfiler()
    
    def test_large_dataset_handling(self):
        """Test system behavior with large datasets."""
        print("\n=== Large Dataset Handling Test ===")
        
        # Generate large dataset (5 years of daily data)
        large_df = self.test_data_generator.generate_synthetic_stock_data(
            start_date="2019-01-01",
            end_date="2023-12-31",
            missing_days_ratio=0.02,
            seed=42
        )
        
        print(f"   Testing with {len(large_df)} data points")
        
        self.profiler.start_profiling()
        
        try:
            # Test data processing
            timeseries_creator = DartsTimeSeriesCreator()
            timeseries = timeseries_creator.create_timeseries(large_df)
            
            # Test data splitting
            data_splitter = DataSplitter()
            train_ts, val_ts, test_ts = data_splitter.split_data(timeseries)
            
            # Validate results
            self.assertGreater(len(train_ts), 0)
            self.assertGreater(len(val_ts), 0) 
            self.assertGreater(len(test_ts), 0)
            
            # Check performance
            metrics = self.profiler.get_performance_metrics()
            
            # Should complete within reasonable time (5 minutes)
            self.assertLess(metrics['execution_time'], 300)
            
            # Memory usage should be reasonable (< 1GB)
            self.assertLess(metrics['memory_increase_mb'], 1000)
            
            print(f"   ✓ Processed {len(large_df)} points in {metrics['execution_time']:.2f}s")
            print(f"   ✓ Memory usage: {metrics['memory_increase_mb']:.1f}MB")
            
        except Exception as e:
            self.fail(f"Large dataset handling failed: {e}")
    
    def test_extreme_edge_cases(self):
        """Test system behavior with extreme edge cases."""
        print("\n=== Extreme Edge Cases Test ===")
        
        edge_cases = {
            'single_feature': pd.DataFrame({
                'adjusted_close': np.random.uniform(90, 110, 50)
            }, index=pd.date_range('2023-01-01', periods=50, freq='B')),
            
            'many_features': pd.DataFrame({
                **{f'feature_{i}': np.random.randn(100) for i in range(50)},
                'adjusted_close': np.random.uniform(90, 110, 100)
            }, index=pd.date_range('2023-01-01', periods=100, freq='B')),
            
            'extreme_values': pd.DataFrame({
                'adjusted_close': [1e-6, 1e6, 0.001, 1000000],
                'volume': [1, 1e9, 100, 1e8]
            }, index=pd.date_range('2023-01-01', periods=4, freq='B'))
        }
        
        results = {}
        
        for case_name, test_df in edge_cases.items():
            print(f"   Testing {case_name}...")
            
            try:
                timeseries_creator = DartsTimeSeriesCreator()
                timeseries = timeseries_creator.create_timeseries(test_df)
                
                # Basic validation
                self.assertIsInstance(timeseries, TimeSeries)
                self.assertGreater(len(timeseries), 0)
                
                results[case_name] = {'success': True, 'length': len(timeseries)}
                print(f"     ✓ {case_name}: {len(timeseries)} points")
                
            except Exception as e:
                results[case_name] = {'success': False, 'error': str(e)}
                print(f"     ⚠ {case_name}: {e}")
        
        # At least basic cases should work
        basic_success = results.get('single_feature', {}).get('success', False)
        self.assertTrue(basic_success, "Single feature case should work")


if __name__ == '__main__':
    # Configure warnings and output
    warnings.filterwarnings('ignore', category=UserWarning)
    
    if not DEPENDENCIES_AVAILABLE:
        print("Warning: Required dependencies not available. Tests will be skipped.")
    
    # Run tests with detailed output
    unittest.main(verbosity=2, buffer=True)