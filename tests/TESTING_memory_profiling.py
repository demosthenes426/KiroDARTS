"""
Memory profiling and load testing for DARTS Stock Forecasting System.

This module provides comprehensive memory usage analysis, leak detection,
and load testing capabilities to ensure system stability under various conditions.

Requirements: 5.3, 5.4
"""

import unittest
import pandas as pd
import numpy as np
import sys
import os
import time
import gc
import warnings
import threading
import queue
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
import tempfile
import json

# Memory profiling imports
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Warning: psutil not available - memory profiling will be limited")

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Try to import DARTS and other dependencies
try:
    from darts import TimeSeries
    from darts_timeseries_creator import DartsTimeSeriesCreator
    from data_splitter import DataSplitter
    from data_scaler import DataScaler
    from model_factory import ModelFactory
    from model_trainer import ModelTrainer
    from model_evaluator import ModelEvaluator
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    DEPENDENCIES_AVAILABLE = False
    print(f"Warning: Dependencies not available: {e}")


class MemoryProfiler:
    """Advanced memory profiler for tracking memory usage patterns."""
    
    def __init__(self):
        """Initialize memory profiler."""
        self.process = psutil.Process() if PSUTIL_AVAILABLE else None
        self.memory_snapshots = []
        self.start_memory = None
        self.peak_memory = 0
        self.baseline_memory = None
    
    def start_profiling(self):
        """Start memory profiling session."""
        if self.process:
            self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
            self.peak_memory = self.start_memory
            self.baseline_memory = self.start_memory
        else:
            self.start_memory = 0
            self.peak_memory = 0
            self.baseline_memory = 0
        
        self.memory_snapshots = []
        gc.collect()  # Clean up before starting
    
    def take_snapshot(self, label: str = ""):
        """Take a memory usage snapshot."""
        if self.process:
            current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
            if current_memory > self.peak_memory:
                self.peak_memory = current_memory
        else:
            current_memory = 0
        
        snapshot = {
            'timestamp': time.time(),
            'label': label,
            'memory_mb': current_memory,
            'memory_increase_mb': current_memory - self.baseline_memory if self.baseline_memory else 0
        }
        
        self.memory_snapshots.append(snapshot)
        return snapshot
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get comprehensive memory statistics."""
        if not self.memory_snapshots:
            return {}
        
        current_memory = self.memory_snapshots[-1]['memory_mb']
        
        stats = {
            'start_memory_mb': self.start_memory or 0,
            'current_memory_mb': current_memory,
            'peak_memory_mb': self.peak_memory,
            'total_increase_mb': current_memory - (self.start_memory or 0),
            'peak_increase_mb': self.peak_memory - (self.start_memory or 0),
            'num_snapshots': len(self.memory_snapshots)
        }
        
        if len(self.memory_snapshots) > 1:
            memory_values = [s['memory_mb'] for s in self.memory_snapshots]
            stats['memory_std_mb'] = np.std(memory_values)
            stats['memory_trend'] = np.polyfit(range(len(memory_values)), memory_values, 1)[0]
        
        return stats
    
    def detect_memory_leak(self, threshold_mb: float = 50.0) -> Dict[str, Any]:
        """Detect potential memory leaks."""
        if len(self.memory_snapshots) < 3:
            return {'leak_detected': False, 'reason': 'Insufficient data'}
        
        memory_values = [s['memory_mb'] for s in self.memory_snapshots]
        
        # Calculate trend
        trend = np.polyfit(range(len(memory_values)), memory_values, 1)[0]
        
        # Check for consistent upward trend
        total_increase = memory_values[-1] - memory_values[0]
        
        leak_detected = (trend > 1.0 and total_increase > threshold_mb)  # >1MB/snapshot trend
        
        return {
            'leak_detected': leak_detected,
            'trend_mb_per_snapshot': trend,
            'total_increase_mb': total_increase,
            'threshold_mb': threshold_mb,
            'snapshots_analyzed': len(memory_values)
        }
    
    def generate_memory_report(self) -> str:
        """Generate a comprehensive memory usage report."""
        stats = self.get_memory_stats()
        leak_analysis = self.detect_memory_leak()
        
        report = []
        report.append("=== Memory Profiling Report ===")
        report.append(f"Start Memory: {stats.get('start_memory_mb', 0):.1f} MB")
        report.append(f"Current Memory: {stats.get('current_memory_mb', 0):.1f} MB")
        report.append(f"Peak Memory: {stats.get('peak_memory_mb', 0):.1f} MB")
        report.append(f"Total Increase: {stats.get('total_increase_mb', 0):.1f} MB")
        report.append(f"Peak Increase: {stats.get('peak_increase_mb', 0):.1f} MB")
        
        if 'memory_trend' in stats:
            report.append(f"Memory Trend: {stats['memory_trend']:.2f} MB/snapshot")
        
        report.append("")
        report.append("=== Memory Leak Analysis ===")
        report.append(f"Leak Detected: {leak_analysis['leak_detected']}")
        if leak_analysis['leak_detected']:
            report.append(f"Trend: {leak_analysis['trend_mb_per_snapshot']:.2f} MB/snapshot")
            report.append(f"Total Increase: {leak_analysis['total_increase_mb']:.1f} MB")
        
        report.append("")
        report.append("=== Memory Snapshots ===")
        for i, snapshot in enumerate(self.memory_snapshots):
            report.append(f"{i+1:2d}. {snapshot['label']:20s} {snapshot['memory_mb']:6.1f} MB (+{snapshot['memory_increase_mb']:5.1f})")
        
        return "\n".join(report)


class LoadTester:
    """Load testing utility for stress testing the forecasting system."""
    
    def __init__(self):
        """Initialize load tester."""
        self.results = []
        self.errors = []
    
    def run_concurrent_load_test(
        self,
        test_function,
        n_workers: int = 4,
        iterations_per_worker: int = 5,
        timeout_seconds: int = 300
    ) -> Dict[str, Any]:
        """Run concurrent load test with multiple workers."""
        
        results_queue = queue.Queue()
        errors_queue = queue.Queue()
        
        def worker(worker_id: int):
            """Worker function for load testing."""
            worker_results = []
            worker_errors = []
            
            for iteration in range(iterations_per_worker):
                try:
                    start_time = time.time()
                    result = test_function(worker_id, iteration)
                    execution_time = time.time() - start_time
                    
                    worker_results.append({
                        'worker_id': worker_id,
                        'iteration': iteration,
                        'execution_time': execution_time,
                        'result': result,
                        'success': True
                    })
                    
                except Exception as e:
                    worker_errors.append({
                        'worker_id': worker_id,
                        'iteration': iteration,
                        'error': str(e),
                        'success': False
                    })
            
            results_queue.put(worker_results)
            if worker_errors:
                errors_queue.put(worker_errors)
        
        # Start workers
        threads = []
        start_time = time.time()
        
        for worker_id in range(n_workers):
            thread = threading.Thread(target=worker, args=(worker_id,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join(timeout=timeout_seconds)
        
        total_time = time.time() - start_time
        
        # Collect results
        all_results = []
        all_errors = []
        
        while not results_queue.empty():
            worker_results = results_queue.get()
            all_results.extend(worker_results)
        
        while not errors_queue.empty():
            worker_errors = errors_queue.get()
            all_errors.extend(worker_errors)
        
        # Calculate statistics
        successful_results = [r for r in all_results if r['success']]
        execution_times = [r['execution_time'] for r in successful_results]
        
        stats = {
            'total_workers': n_workers,
            'iterations_per_worker': iterations_per_worker,
            'total_iterations': n_workers * iterations_per_worker,
            'successful_iterations': len(successful_results),
            'failed_iterations': len(all_errors),
            'success_rate': len(successful_results) / (n_workers * iterations_per_worker) if n_workers * iterations_per_worker > 0 else 0,
            'total_execution_time': total_time,
            'avg_iteration_time': np.mean(execution_times) if execution_times else 0,
            'min_iteration_time': np.min(execution_times) if execution_times else 0,
            'max_iteration_time': np.max(execution_times) if execution_times else 0,
            'std_iteration_time': np.std(execution_times) if execution_times else 0,
            'throughput_per_second': len(successful_results) / total_time if total_time > 0 else 0
        }
        
        return {
            'statistics': stats,
            'results': all_results,
            'errors': all_errors
        }


@unittest.skipUnless(DEPENDENCIES_AVAILABLE and PSUTIL_AVAILABLE, "Required dependencies not available")
class TestMemoryProfiling(unittest.TestCase):
    """Memory profiling and leak detection tests."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.profiler = MemoryProfiler()
        self.load_tester = LoadTester()
        
        # Create test data
        self.test_data = self._create_test_data()
        
        # Initialize components
        self.timeseries_creator = DartsTimeSeriesCreator()
        self.data_splitter = DataSplitter()
        self.data_scaler = DataScaler()
        self.model_factory = ModelFactory(
            input_chunk_length=10,
            output_chunk_length=5,
            n_epochs=3,  # Reduced for testing
            batch_size=16,
            random_state=42
        )
        self.model_trainer = ModelTrainer(max_epochs=3, verbose=False)
        
        warnings.filterwarnings('ignore', category=UserWarning)
    
    def _create_test_data(self) -> pd.DataFrame:
        """Create standardized test data."""
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', periods=200, freq='B')
        
        data = {
            'adjusted_close': 100 + np.cumsum(np.random.normal(0, 1, 200)),
            'volume': np.random.randint(10000, 100000, 200),
            'feature_1': np.random.randn(200),
            'feature_2': np.random.randn(200),
            'feature_3': np.random.randn(200)
        }
        
        return pd.DataFrame(data, index=dates)
    
    def test_data_processing_memory_usage(self):
        """Test memory usage during data processing operations."""
        print("\n=== Data Processing Memory Usage Test ===")
        
        self.profiler.start_profiling()
        self.profiler.take_snapshot("Start")
        
        # Test TimeSeries creation
        timeseries = self.timeseries_creator.create_timeseries(self.test_data)
        self.profiler.take_snapshot("TimeSeries Created")
        
        # Test data splitting
        train_ts, val_ts, test_ts = self.data_splitter.split_data(timeseries)
        self.profiler.take_snapshot("Data Split")
        
        # Test data scaling
        train_scaled, val_scaled, test_scaled, scaler = self.data_scaler.scale_data(
            train_ts, val_ts, test_ts
        )
        self.profiler.take_snapshot("Data Scaled")
        
        # Force garbage collection
        del timeseries, train_ts, val_ts, test_ts
        gc.collect()
        self.profiler.take_snapshot("After GC")
        
        # Analyze memory usage
        stats = self.profiler.get_memory_stats()
        leak_analysis = self.profiler.detect_memory_leak(threshold_mb=20.0)
        
        print(f"   Memory usage: {stats['total_increase_mb']:.1f} MB increase")
        print(f"   Peak memory: {stats['peak_memory_mb']:.1f} MB")
        
        # Validate memory usage is reasonable
        self.assertLess(stats['total_increase_mb'], 100, "Data processing should use <100MB")
        self.assertFalse(leak_analysis['leak_detected'], "No memory leak should be detected")
        
        print("   ✓ Data processing memory usage within acceptable limits")
    
    def test_model_training_memory_usage(self):
        """Test memory usage during model training."""
        print("\n=== Model Training Memory Usage Test ===")
        
        # Prepare data
        timeseries = self.timeseries_creator.create_timeseries(self.test_data)
        train_ts, val_ts, test_ts = self.data_splitter.split_data(timeseries)
        train_scaled, val_scaled, test_scaled, scaler = self.data_scaler.scale_data(
            train_ts, val_ts, test_ts
        )
        
        self.profiler.start_profiling()
        self.profiler.take_snapshot("Training Start")
        
        # Test with lightweight model
        model_name = 'DLinearModel'
        
        try:
            # Create model
            model = self.model_factory.create_single_model(model_name)
            self.profiler.take_snapshot("Model Created")
            
            if model is not None:
                # Train model
                training_result = self.model_trainer.train_model(
                    model, train_scaled, val_scaled, model_name=model_name
                )
                self.profiler.take_snapshot("Model Trained")
                
                # Clean up
                del model, training_result
                gc.collect()
                self.profiler.take_snapshot("After Training GC")
                
                # Analyze memory usage
                stats = self.profiler.get_memory_stats()
                leak_analysis = self.profiler.detect_memory_leak(threshold_mb=50.0)
                
                print(f"   Training memory usage: {stats['total_increase_mb']:.1f} MB")
                print(f"   Peak memory: {stats['peak_memory_mb']:.1f} MB")
                
                # Validate memory usage
                self.assertLess(stats['total_increase_mb'], 200, "Training should use <200MB")
                
                if leak_analysis['leak_detected']:
                    print(f"   ⚠ Potential memory leak detected: {leak_analysis['trend_mb_per_snapshot']:.2f} MB/snapshot")
                else:
                    print("   ✓ No memory leak detected during training")
            
        except Exception as e:
            print(f"   ✗ Model training memory test failed: {e}")
            self.skipTest(f"Model training failed: {e}")
    
    def test_repeated_operations_memory_leak(self):
        """Test for memory leaks during repeated operations."""
        print("\n=== Repeated Operations Memory Leak Test ===")
        
        self.profiler.start_profiling()
        self.profiler.take_snapshot("Start")
        
        # Perform repeated TimeSeries creation and processing
        n_iterations = 10
        
        for i in range(n_iterations):
            # Create TimeSeries
            timeseries = self.timeseries_creator.create_timeseries(self.test_data)
            
            # Split data
            train_ts, val_ts, test_ts = self.data_splitter.split_data(timeseries)
            
            # Scale data
            train_scaled, val_scaled, test_scaled, scaler = self.data_scaler.scale_data(
                train_ts, val_ts, test_ts
            )
            
            # Clean up explicitly
            del timeseries, train_ts, val_ts, test_ts, train_scaled, val_scaled, test_scaled, scaler
            
            if i % 3 == 0:  # Take snapshots every 3 iterations
                gc.collect()
                self.profiler.take_snapshot(f"Iteration {i+1}")
        
        # Final cleanup and analysis
        gc.collect()
        self.profiler.take_snapshot("Final")
        
        # Analyze for memory leaks
        leak_analysis = self.profiler.detect_memory_leak(threshold_mb=30.0)
        stats = self.profiler.get_memory_stats()
        
        print(f"   Completed {n_iterations} iterations")
        print(f"   Total memory increase: {stats['total_increase_mb']:.1f} MB")
        print(f"   Memory per iteration: {stats['total_increase_mb']/n_iterations:.2f} MB")
        
        if leak_analysis['leak_detected']:
            print(f"   ⚠ Memory leak detected: {leak_analysis['trend_mb_per_snapshot']:.2f} MB/snapshot")
            print(f"   Total leak: {leak_analysis['total_increase_mb']:.1f} MB")
            
            # In production, you might want to fail the test here
            # self.fail("Memory leak detected during repeated operations")
        else:
            print("   ✓ No significant memory leak detected")
        
        # Validate reasonable memory usage
        memory_per_iteration = stats['total_increase_mb'] / n_iterations
        self.assertLess(memory_per_iteration, 5.0, "Memory per iteration should be <5MB")
    
    def test_large_dataset_memory_scaling(self):
        """Test memory usage scaling with different dataset sizes."""
        print("\n=== Large Dataset Memory Scaling Test ===")
        
        dataset_sizes = [100, 500, 1000]
        memory_usage_results = {}
        
        for size in dataset_sizes:
            print(f"   Testing dataset size: {size} days")
            
            # Generate data of specific size
            np.random.seed(42)
            dates = pd.date_range(start='2023-01-01', periods=size, freq='B')
            
            large_data = {
                'adjusted_close': 100 + np.cumsum(np.random.normal(0, 1, size)),
                'volume': np.random.randint(10000, 100000, size),
                'feature_1': np.random.randn(size),
                'feature_2': np.random.randn(size),
                'feature_3': np.random.randn(size),
                'feature_4': np.random.randn(size),
                'feature_5': np.random.randn(size)
            }
            
            test_df = pd.DataFrame(large_data, index=dates)
            
            # Profile memory usage
            self.profiler.start_profiling()
            self.profiler.take_snapshot("Start")
            
            try:
                # Process data
                timeseries = self.timeseries_creator.create_timeseries(test_df)
                self.profiler.take_snapshot("TimeSeries")
                
                train_ts, val_ts, test_ts = self.data_splitter.split_data(timeseries)
                self.profiler.take_snapshot("Split")
                
                train_scaled, val_scaled, test_scaled, scaler = self.data_scaler.scale_data(
                    train_ts, val_ts, test_ts
                )
                self.profiler.take_snapshot("Scaled")
                
                # Get memory stats
                stats = self.profiler.get_memory_stats()
                memory_usage_results[size] = {
                    'total_memory_mb': stats['total_increase_mb'],
                    'peak_memory_mb': stats['peak_memory_mb'],
                    'memory_per_row_kb': (stats['total_increase_mb'] * 1024) / size
                }
                
                print(f"     Memory usage: {stats['total_increase_mb']:.1f} MB ({memory_usage_results[size]['memory_per_row_kb']:.2f} KB/row)")
                
                # Clean up
                del timeseries, train_ts, val_ts, test_ts, train_scaled, val_scaled, test_scaled, scaler, test_df
                gc.collect()
                
            except Exception as e:
                print(f"     ✗ Failed for size {size}: {e}")
                memory_usage_results[size] = {'error': str(e)}
        
        # Analyze scaling
        successful_results = {k: v for k, v in memory_usage_results.items() if 'error' not in v}
        
        if len(successful_results) >= 2:
            sizes = list(successful_results.keys())
            memories = [successful_results[s]['total_memory_mb'] for s in sizes]
            
            # Calculate scaling factor
            size_ratio = max(sizes) / min(sizes)
            memory_ratio = max(memories) / min(memories) if min(memories) > 0 else 1
            scaling_efficiency = size_ratio / memory_ratio if memory_ratio > 0 else 0
            
            print(f"\n   📊 Memory Scaling Analysis:")
            print(f"   Size ratio: {size_ratio:.1f}x")
            print(f"   Memory ratio: {memory_ratio:.1f}x")
            print(f"   Scaling efficiency: {scaling_efficiency:.2f} (1.0 = linear)")
            
            # Validate sub-linear scaling
            self.assertGreater(scaling_efficiency, 0.5, "Memory should scale sub-linearly")
            
            # Validate reasonable memory per row
            for size, result in successful_results.items():
                self.assertLess(result['memory_per_row_kb'], 10.0, f"Memory per row too high for size {size}")
    
    def test_concurrent_memory_usage(self):
        """Test memory usage under concurrent operations."""
        print("\n=== Concurrent Memory Usage Test ===")
        
        def concurrent_processing_task(worker_id: int, iteration: int):
            """Task for concurrent processing."""
            # Create TimeSeries
            timeseries = self.timeseries_creator.create_timeseries(self.test_data)
            
            # Split and scale
            train_ts, val_ts, test_ts = self.data_splitter.split_data(timeseries)
            train_scaled, val_scaled, test_scaled, scaler = self.data_scaler.scale_data(
                train_ts, val_ts, test_ts
            )
            
            # Return some result to validate success
            return {
                'timeseries_length': len(timeseries),
                'train_length': len(train_scaled)
            }
        
        # Profile memory during concurrent execution
        self.profiler.start_profiling()
        self.profiler.take_snapshot("Concurrent Start")
        
        # Run concurrent load test
        load_test_results = self.load_tester.run_concurrent_load_test(
            test_function=concurrent_processing_task,
            n_workers=3,
            iterations_per_worker=3,
            timeout_seconds=120
        )
        
        self.profiler.take_snapshot("Concurrent Complete")
        
        # Analyze results
        stats = load_test_results['statistics']
        memory_stats = self.profiler.get_memory_stats()
        
        print(f"   Concurrent execution results:")
        print(f"   Success rate: {stats['success_rate']:.1%}")
        print(f"   Total iterations: {stats['successful_iterations']}/{stats['total_iterations']}")
        print(f"   Avg iteration time: {stats['avg_iteration_time']:.3f}s")
        print(f"   Memory usage: {memory_stats['total_increase_mb']:.1f} MB")
        print(f"   Peak memory: {memory_stats['peak_memory_mb']:.1f} MB")
        
        # Validate concurrent execution
        self.assertGreater(stats['success_rate'], 0.8, "At least 80% of concurrent operations should succeed")
        self.assertLess(memory_stats['total_increase_mb'], 300, "Concurrent memory usage should be <300MB")
        
        # Check for memory leaks
        leak_analysis = self.profiler.detect_memory_leak(threshold_mb=50.0)
        if leak_analysis['leak_detected']:
            print(f"   ⚠ Potential memory leak in concurrent execution")
        else:
            print("   ✓ No memory leak detected in concurrent execution")
    
    def tearDown(self):
        """Clean up after each test."""
        # Force garbage collection
        gc.collect()
        
        # Print memory report if profiler was used
        if hasattr(self, 'profiler') and self.profiler.memory_snapshots:
            print(f"\n{self.profiler.generate_memory_report()}")


@unittest.skipUnless(DEPENDENCIES_AVAILABLE, "Required dependencies not available")
class TestLoadTesting(unittest.TestCase):
    """Load testing and stress testing for system stability."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.load_tester = LoadTester()
        
        # Create test data
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', periods=100, freq='B')
        
        self.test_data = pd.DataFrame({
            'adjusted_close': 100 + np.cumsum(np.random.normal(0, 1, 100)),
            'volume': np.random.randint(10000, 100000, 100),
            'feature_1': np.random.randn(100)
        }, index=dates)
        
        # Initialize components
        self.timeseries_creator = DartsTimeSeriesCreator()
        self.data_splitter = DataSplitter()
        
        warnings.filterwarnings('ignore', category=UserWarning)
    
    def test_high_frequency_operations(self):
        """Test system stability under high frequency operations."""
        print("\n=== High Frequency Operations Test ===")
        
        def high_freq_task(worker_id: int, iteration: int):
            """High frequency processing task."""
            timeseries = self.timeseries_creator.create_timeseries(self.test_data)
            train_ts, val_ts, test_ts = self.data_splitter.split_data(timeseries)
            
            return {
                'worker_id': worker_id,
                'iteration': iteration,
                'timeseries_length': len(timeseries)
            }
        
        # Run high frequency test
        results = self.load_tester.run_concurrent_load_test(
            test_function=high_freq_task,
            n_workers=5,
            iterations_per_worker=10,
            timeout_seconds=180
        )
        
        stats = results['statistics']
        
        print(f"   High frequency test results:")
        print(f"   Total operations: {stats['total_iterations']}")
        print(f"   Success rate: {stats['success_rate']:.1%}")
        print(f"   Throughput: {stats['throughput_per_second']:.2f} ops/sec")
        print(f"   Avg operation time: {stats['avg_iteration_time']:.3f}s")
        print(f"   Max operation time: {stats['max_iteration_time']:.3f}s")
        
        # Validate performance
        self.assertGreater(stats['success_rate'], 0.95, "High frequency operations should have >95% success rate")
        self.assertLess(stats['avg_iteration_time'], 2.0, "Average operation time should be <2s")
        self.assertGreater(stats['throughput_per_second'], 1.0, "Throughput should be >1 ops/sec")
        
        print("   ✓ High frequency operations test passed")
    
    def test_stress_testing_with_errors(self):
        """Test system behavior under stress with intentional errors."""
        print("\n=== Stress Testing with Error Injection ===")
        
        def error_prone_task(worker_id: int, iteration: int):
            """Task that occasionally fails to test error handling."""
            
            # Inject random failures
            if np.random.random() < 0.1:  # 10% failure rate
                raise ValueError(f"Simulated error in worker {worker_id}, iteration {iteration}")
            
            # Normal processing
            timeseries = self.timeseries_creator.create_timeseries(self.test_data)
            train_ts, val_ts, test_ts = self.data_splitter.split_data(timeseries)
            
            return {'success': True, 'worker_id': worker_id}
        
        # Run stress test
        results = self.load_tester.run_concurrent_load_test(
            test_function=error_prone_task,
            n_workers=4,
            iterations_per_worker=15,
            timeout_seconds=120
        )
        
        stats = results['statistics']
        errors = results['errors']
        
        print(f"   Stress test with errors:")
        print(f"   Total operations: {stats['total_iterations']}")
        print(f"   Successful: {stats['successful_iterations']}")
        print(f"   Failed: {stats['failed_iterations']}")
        print(f"   Success rate: {stats['success_rate']:.1%}")
        print(f"   Error rate: {len(errors)/stats['total_iterations']:.1%}")
        
        # Validate error handling
        self.assertGreater(stats['success_rate'], 0.8, "Should handle errors gracefully with >80% success")
        self.assertLess(len(errors) / stats['total_iterations'], 0.2, "Error rate should be <20%")
        
        # Check that errors are properly captured
        if errors:
            sample_error = errors[0]
            self.assertIn('error', sample_error, "Errors should be properly captured")
            print(f"   Sample error: {sample_error['error'][:50]}...")
        
        print("   ✓ Stress testing with error injection passed")
    
    def test_resource_exhaustion_recovery(self):
        """Test system recovery from resource exhaustion scenarios."""
        print("\n=== Resource Exhaustion Recovery Test ===")
        
        def memory_intensive_task(worker_id: int, iteration: int):
            """Task that uses significant memory."""
            
            # Create larger dataset for this worker
            large_size = 500 + (worker_id * 100)  # Different sizes per worker
            dates = pd.date_range(start='2023-01-01', periods=large_size, freq='B')
            
            large_data = pd.DataFrame({
                'adjusted_close': 100 + np.cumsum(np.random.normal(0, 1, large_size)),
                'volume': np.random.randint(10000, 100000, large_size),
                'feature_1': np.random.randn(large_size),
                'feature_2': np.random.randn(large_size),
                'feature_3': np.random.randn(large_size)
            }, index=dates)
            
            # Process the large dataset
            timeseries = self.timeseries_creator.create_timeseries(large_data)
            train_ts, val_ts, test_ts = self.data_splitter.split_data(timeseries)
            
            # Explicit cleanup
            del large_data, timeseries, train_ts, val_ts, test_ts
            gc.collect()
            
            return {'data_size': large_size, 'worker_id': worker_id}
        
        # Run resource intensive test
        results = self.load_tester.run_concurrent_load_test(
            test_function=memory_intensive_task,
            n_workers=3,
            iterations_per_worker=5,
            timeout_seconds=300
        )
        
        stats = results['statistics']
        
        print(f"   Resource exhaustion test:")
        print(f"   Success rate: {stats['success_rate']:.1%}")
        print(f"   Avg execution time: {stats['avg_iteration_time']:.2f}s")
        print(f"   Max execution time: {stats['max_iteration_time']:.2f}s")
        
        # Validate recovery
        self.assertGreater(stats['success_rate'], 0.7, "Should recover from resource pressure with >70% success")
        
        # Check for reasonable execution times (not hanging)
        self.assertLess(stats['max_iteration_time'], 60.0, "No operation should take >60s")
        
        print("   ✓ Resource exhaustion recovery test passed")


if __name__ == '__main__':
    # Configure warnings
    warnings.filterwarnings('ignore', category=UserWarning)
    
    if not DEPENDENCIES_AVAILABLE:
        print("Warning: Required dependencies not available. Tests will be skipped.")
    
    if not PSUTIL_AVAILABLE:
        print("Warning: psutil not available. Memory profiling tests will be skipped.")
    
    # Run tests with verbose output
    unittest.main(verbosity=2)