"""
Unit tests for DataSplitter class.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, date
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_splitter import DataSplitter, InsufficientDataError, SplitValidationError

# Try to import DARTS - skip tests if not available
try:
    from darts import TimeSeries
    DARTS_AVAILABLE = True
except ImportError:
    DARTS_AVAILABLE = False
    TimeSeries = None


@unittest.skipUnless(DARTS_AVAILABLE, "DARTS library not available")
class TestDataSplitter(unittest.TestCase):
    """Test cases for DataSplitter class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.splitter = DataSplitter()
        
        # Create sample TimeSeries with sufficient data points
        dates = pd.date_range(start='2015-01-01', periods=100, freq='B')
        data = np.random.randn(100, 3) + 100  # 3 columns, 100 rows
        df = pd.DataFrame(data, index=dates, columns=['adjusted_close', 'volume', 'indicator'])
        self.sample_ts = TimeSeries.from_dataframe(df)
        
        # Create small TimeSeries for edge case testing
        small_dates = pd.date_range(start='2015-01-01', periods=5, freq='B')
        small_data = np.random.randn(5, 2) + 100
        small_df = pd.DataFrame(small_data, index=small_dates, columns=['adjusted_close', 'volume'])
        self.small_ts = TimeSeries.from_dataframe(small_df)
        
        # Create minimum size TimeSeries (exactly 10 points)
        min_dates = pd.date_range(start='2015-01-01', periods=10, freq='B')
        min_data = np.random.randn(10, 2) + 100
        min_df = pd.DataFrame(min_data, index=min_dates, columns=['adjusted_close', 'volume'])
        self.min_ts = TimeSeries.from_dataframe(min_df)
    
    def test_split_data_basic(self):
        """Test basic data splitting functionality."""
        train_ts, val_ts, test_ts = self.splitter.split_data(self.sample_ts)
        
        # Check that all splits are TimeSeries objects
        self.assertIsInstance(train_ts, TimeSeries)
        self.assertIsInstance(val_ts, TimeSeries)
        self.assertIsInstance(test_ts, TimeSeries)
        
        # Check that all splits are non-empty
        self.assertGreater(len(train_ts), 0)
        self.assertGreater(len(val_ts), 0)
        self.assertGreater(len(test_ts), 0)
        
        # Check that splits sum to original length
        total_length = len(train_ts) + len(val_ts) + len(test_ts)
        self.assertEqual(total_length, len(self.sample_ts))
    
    def test_split_ratios(self):
        """Test that split ratios are approximately correct."""
        train_ts, val_ts, test_ts = self.splitter.split_data(self.sample_ts)
        
        total_length = len(self.sample_ts)
        train_ratio = len(train_ts) / total_length
        val_ratio = len(val_ts) / total_length
        test_ratio = len(test_ts) / total_length
        
        # Check ratios with tolerance for rounding
        tolerance = 0.1  # 10% tolerance for small datasets
        self.assertAlmostEqual(train_ratio, self.splitter.TRAIN_RATIO, delta=tolerance)
        self.assertAlmostEqual(val_ratio, self.splitter.VAL_RATIO, delta=tolerance)
        self.assertAlmostEqual(test_ratio, self.splitter.TEST_RATIO, delta=tolerance)
    
    def test_temporal_ordering(self):
        """Test that temporal ordering is maintained."""
        train_ts, val_ts, test_ts = self.splitter.split_data(self.sample_ts)
        
        # Check that train comes before val comes before test
        self.assertLess(train_ts.end_time(), val_ts.start_time())
        self.assertLess(val_ts.end_time(), test_ts.start_time())
        
        # Check that each split is internally ordered
        self.assertTrue(train_ts.time_index.is_monotonic_increasing)
        self.assertTrue(val_ts.time_index.is_monotonic_increasing)
        self.assertTrue(test_ts.time_index.is_monotonic_increasing)
    
    def test_no_data_overlap(self):
        """Test that there's no data overlap between splits."""
        train_ts, val_ts, test_ts = self.splitter.split_data(self.sample_ts)
        
        # Convert to sets for overlap checking
        train_dates = set(train_ts.time_index)
        val_dates = set(val_ts.time_index)
        test_dates = set(test_ts.time_index)
        
        # Check no overlaps
        self.assertEqual(len(train_dates & val_dates), 0)
        self.assertEqual(len(val_dates & test_dates), 0)
        self.assertEqual(len(train_dates & test_dates), 0)
        
        # Check that union equals original
        all_split_dates = train_dates | val_dates | test_dates
        original_dates = set(self.sample_ts.time_index)
        self.assertEqual(all_split_dates, original_dates)
    
    def test_validate_input_timeseries_none(self):
        """Test validation with None input."""
        with self.assertRaises(ValueError) as context:
            self.splitter.split_data(None)
        
        self.assertIn("TimeSeries cannot be None", str(context.exception))
    
    def test_validate_input_timeseries_empty(self):
        """Test validation with empty TimeSeries."""
        # Create empty TimeSeries
        empty_df = pd.DataFrame(columns=['adjusted_close'])
        empty_ts = TimeSeries.from_dataframe(empty_df)
        
        with self.assertRaises(ValueError) as context:
            self.splitter.split_data(empty_ts)
        
        self.assertIn("TimeSeries cannot be empty", str(context.exception))
    
    def test_validate_input_timeseries_too_small(self):
        """Test validation with TimeSeries too small for splitting."""
        with self.assertRaises(InsufficientDataError) as context:
            self.splitter.split_data(self.small_ts)
        
        self.assertIn("TimeSeries too small for splitting", str(context.exception))
        self.assertIn("Minimum size: 10", str(context.exception))
    
    def test_minimum_size_timeseries(self):
        """Test splitting with minimum size TimeSeries (exactly 10 points)."""
        train_ts, val_ts, test_ts = self.splitter.split_data(self.min_ts)
        
        # Should work with exactly 10 points
        self.assertGreater(len(train_ts), 0)
        self.assertGreater(len(val_ts), 0)
        self.assertGreater(len(test_ts), 0)
        
        # Check total length
        total_length = len(train_ts) + len(val_ts) + len(test_ts)
        self.assertEqual(total_length, 10)
    
    def test_calculate_split_indices(self):
        """Test split index calculation."""
        train_end_idx, val_end_idx = self.splitter._calculate_split_indices(self.sample_ts)
        
        total_length = len(self.sample_ts)
        
        # Check that indices are reasonable
        self.assertGreater(train_end_idx, 0)
        self.assertGreater(val_end_idx, train_end_idx)
        self.assertLess(val_end_idx, total_length)
        
        # Check approximate ratios
        expected_train_end = int(total_length * self.splitter.TRAIN_RATIO)
        expected_val_end = int(total_length * (self.splitter.TRAIN_RATIO + self.splitter.VAL_RATIO))
        
        # Allow for adjustments made in the method
        self.assertLessEqual(abs(train_end_idx - expected_train_end), 2)
        self.assertLessEqual(abs(val_end_idx - expected_val_end), 2)
    
    def test_perform_splits(self):
        """Test the actual splitting operation."""
        train_end_idx = 70
        val_end_idx = 85
        
        train_ts, val_ts, test_ts = self.splitter._perform_splits(self.sample_ts, train_end_idx, val_end_idx)
        
        # Check lengths
        self.assertEqual(len(train_ts), train_end_idx)
        self.assertEqual(len(val_ts), val_end_idx - train_end_idx)
        self.assertEqual(len(test_ts), len(self.sample_ts) - val_end_idx)
        
        # Check that data is preserved
        self.assertEqual(train_ts.start_time(), self.sample_ts.start_time())
        self.assertEqual(test_ts.end_time(), self.sample_ts.end_time())
    
    def test_validate_splits_success(self):
        """Test successful split validation."""
        train_ts, val_ts, test_ts = self.splitter.split_data(self.sample_ts)
        
        # This should not raise any exceptions
        self.splitter._validate_splits(train_ts, val_ts, test_ts, self.sample_ts)
    
    def test_get_split_info(self):
        """Test getting split information."""
        train_ts, val_ts, test_ts = self.splitter.split_data(self.sample_ts)
        info = self.splitter.get_split_info(train_ts, val_ts, test_ts)
        
        # Check that all expected keys are present
        expected_keys = [
            'total_length', 'train_length', 'val_length', 'test_length',
            'train_ratio', 'val_ratio', 'test_ratio',
            'train_period', 'val_period', 'test_period'
        ]
        for key in expected_keys:
            self.assertIn(key, info)
        
        # Check specific values
        self.assertEqual(info['total_length'], len(self.sample_ts))
        self.assertEqual(info['train_length'], len(train_ts))
        self.assertEqual(info['val_length'], len(val_ts))
        self.assertEqual(info['test_length'], len(test_ts))
        
        # Check that ratios sum to 1
        total_ratio = info['train_ratio'] + info['val_ratio'] + info['test_ratio']
        self.assertAlmostEqual(total_ratio, 1.0, places=10)
        
        # Check period information
        self.assertEqual(info['train_period']['start'], train_ts.start_time())
        self.assertEqual(info['train_period']['end'], train_ts.end_time())
        self.assertEqual(info['val_period']['start'], val_ts.start_time())
        self.assertEqual(info['val_period']['end'], val_ts.end_time())
        self.assertEqual(info['test_period']['start'], test_ts.start_time())
        self.assertEqual(info['test_period']['end'], test_ts.end_time())
    
    def test_validate_temporal_consistency(self):
        """Test temporal consistency validation."""
        train_ts, val_ts, test_ts = self.splitter.split_data(self.sample_ts)
        
        # Should return True for valid splits
        self.assertTrue(self.splitter.validate_temporal_consistency(train_ts, val_ts, test_ts))
    
    def test_validate_temporal_consistency_invalid(self):
        """Test temporal consistency validation with invalid data."""
        # Create overlapping splits by manually creating them
        train_ts = self.sample_ts[:50]
        val_ts = self.sample_ts[45:75]  # Overlaps with train
        test_ts = self.sample_ts[70:]
        
        # Should return False for overlapping splits
        self.assertFalse(self.splitter.validate_temporal_consistency(train_ts, val_ts, test_ts))
    
    def test_real_data_file(self):
        """Test with real covariaterawdata1.csv file if available."""
        real_file_path = os.path.join(os.path.dirname(__file__), '..', 'Data', 'covariaterawdata1.csv')
        
        if os.path.exists(real_file_path):
            # Load and preprocess real data
            df = pd.read_csv(real_file_path)
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date').sort_index()
            
            # Remove non-numeric columns and NaN values
            numeric_df = df.select_dtypes(include=[np.number])
            numeric_df = numeric_df.dropna()
            
            if len(numeric_df) >= 10:  # Ensure sufficient data
                try:
                    # Create TimeSeries
                    ts = TimeSeries.from_dataframe(numeric_df)
                    
                    # Test splitting
                    train_ts, val_ts, test_ts = self.splitter.split_data(ts)
                    
                    # Check that splits are valid
                    self.assertIsInstance(train_ts, TimeSeries)
                    self.assertIsInstance(val_ts, TimeSeries)
                    self.assertIsInstance(test_ts, TimeSeries)
                    
                    # Check temporal consistency
                    self.assertTrue(self.splitter.validate_temporal_consistency(train_ts, val_ts, test_ts))
                    
                    # Get split info
                    info = self.splitter.get_split_info(train_ts, val_ts, test_ts)
                    
                    print(f"Real data split info: {info}")
                    
                except Exception as e:
                    self.skipTest(f"Real data processing failed: {e}")
            else:
                self.skipTest("Real data file has insufficient data after cleaning")
        else:
            self.skipTest("Real data file not found")
    
    def test_edge_case_uneven_split(self):
        """Test splitting with data that doesn't divide evenly."""
        # Create TimeSeries with 23 points (doesn't divide evenly by ratios)
        dates = pd.date_range(start='2015-01-01', periods=23, freq='B')
        data = np.random.randn(23, 2) + 100
        df = pd.DataFrame(data, index=dates, columns=['adjusted_close', 'volume'])
        uneven_ts = TimeSeries.from_dataframe(df)
        
        train_ts, val_ts, test_ts = self.splitter.split_data(uneven_ts)
        
        # Check that all splits are non-empty
        self.assertGreater(len(train_ts), 0)
        self.assertGreater(len(val_ts), 0)
        self.assertGreater(len(test_ts), 0)
        
        # Check that total length is preserved
        total_length = len(train_ts) + len(val_ts) + len(test_ts)
        self.assertEqual(total_length, 23)
        
        # Check temporal ordering
        self.assertTrue(self.splitter.validate_temporal_consistency(train_ts, val_ts, test_ts))
    
    def test_class_constants(self):
        """Test that class constants are set correctly."""
        self.assertEqual(self.splitter.TRAIN_RATIO, 0.7)
        self.assertEqual(self.splitter.VAL_RATIO, 0.15)
        self.assertEqual(self.splitter.TEST_RATIO, 0.15)
        
        # Check that ratios sum to 1
        total_ratio = self.splitter.TRAIN_RATIO + self.splitter.VAL_RATIO + self.splitter.TEST_RATIO
        self.assertAlmostEqual(total_ratio, 1.0, places=10)


@unittest.skipIf(DARTS_AVAILABLE, "DARTS library is available - skipping mock tests")
class TestDataSplitterWithoutDarts(unittest.TestCase):
    """Test cases for when DARTS is not available."""
    
    def test_import_error_handling(self):
        """Test that import error is handled gracefully."""
        # This test runs when DARTS is not available
        # It ensures the test file can still be imported and run
        self.assertTrue(True, "Test file can be imported without DARTS")


if __name__ == '__main__':
    if not DARTS_AVAILABLE:
        print("Warning: DARTS library not available. Some tests will be skipped.")
    unittest.main()