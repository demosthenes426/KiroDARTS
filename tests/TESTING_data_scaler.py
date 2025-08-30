"""
Unit tests for DataScaler class.

This module tests the scaling functionality for DARTS TimeSeries objects,
ensuring proper StandardScaler usage and validation of scaling statistics.
"""

import unittest
import pandas as pd
import numpy as np
from darts import TimeSeries
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_scaler import DataScaler, ScalingValidationError, ScalerFitError
from data_splitter import DataSplitter


def get_dataframe_from_timeseries(ts):
    """Helper function to get DataFrame from TimeSeries, handling different DARTS versions."""
    try:
        return ts.pd_dataframe()
    except AttributeError:
        try:
            return ts.to_dataframe()
        except AttributeError:
            return ts.pd_series().to_frame()


class TestDataScaler(unittest.TestCase):
    """Test cases for DataScaler class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.scaler = DataScaler()
        self.splitter = DataSplitter()
        
        # Create test data
        self.create_test_data()
    
    def create_test_data(self):
        """Create various test datasets for scaling tests."""
        # Create TimeSeries with known statistics for testing
        np.random.seed(42)  # For reproducible tests
        
        # Medium-sized dataset with multiple features
        dates = pd.date_range(start='2020-01-01', periods=1000, freq='D')
        
        # Create features with different scales and distributions
        feature1 = np.random.normal(100, 20, 1000)  # Mean=100, std=20
        feature2 = np.random.normal(-50, 5, 1000)   # Mean=-50, std=5
        feature3 = np.random.normal(0, 1, 1000)     # Mean=0, std=1
        
        df = pd.DataFrame({
            'price': feature1,
            'volume': feature2,
            'indicator': feature3
        }, index=dates)
        
        self.ts_multi = TimeSeries.from_dataframe(df)
        
        # Split the data for testing
        self.train_ts, self.val_ts, self.test_ts = self.splitter.split_data(self.ts_multi)
        
        # Create single feature TimeSeries
        dates_single = pd.date_range(start='2020-01-01', periods=100, freq='D')
        values_single = np.random.normal(1000, 100, 100)
        df_single = pd.DataFrame({'value': values_single}, index=dates_single)
        self.ts_single = TimeSeries.from_dataframe(df_single)
        
        # Split single feature data
        self.train_single, self.val_single, self.test_single = self.splitter.split_data(self.ts_single)
        
        # Create small dataset for edge case testing
        dates_small = pd.date_range(start='2020-01-01', periods=30, freq='D')
        values_small = np.random.normal(0, 1, (30, 2))
        df_small = pd.DataFrame(values_small, index=dates_small, columns=['a', 'b'])
        self.ts_small = TimeSeries.from_dataframe(df_small)
        self.train_small, self.val_small, self.test_small = self.splitter.split_data(self.ts_small)
    
    def test_basic_scaling_functionality(self):
        """Test basic scaling functionality."""
        scaled_train, scaled_val, scaled_test, scaler = self.scaler.scale_data(
            self.train_ts, self.val_ts, self.test_ts
        )
        
        # Check that all outputs are returned
        self.assertIsInstance(scaled_train, TimeSeries)
        self.assertIsInstance(scaled_val, TimeSeries)
        self.assertIsInstance(scaled_test, TimeSeries)
        self.assertIsInstance(scaler, StandardScaler)
        
        # Check that lengths are preserved
        self.assertEqual(len(scaled_train), len(self.train_ts))
        self.assertEqual(len(scaled_val), len(self.val_ts))
        self.assertEqual(len(scaled_test), len(self.test_ts))
        
        # Check that feature count is preserved
        self.assertEqual(len(scaled_train.columns), len(self.train_ts.columns))
        self.assertEqual(len(scaled_val.columns), len(self.val_ts.columns))
        self.assertEqual(len(scaled_test.columns), len(self.test_ts.columns))
    
    def test_training_data_scaling_statistics(self):
        """Test that training data has mean ≈ 0 and std ≈ 1 after scaling."""
        scaled_train, scaled_val, scaled_test, scaler = self.scaler.scale_data(
            self.train_ts, self.val_ts, self.test_ts
        )
        
        # Get training data statistics
        train_df = get_dataframe_from_timeseries(scaled_train)
        means = train_df.mean()
        stds = train_df.std()
        
        # Check means are approximately 0
        for col in train_df.columns:
            self.assertAlmostEqual(means[col], 0.0, places=10)
        
        # Check standard deviations are approximately 1 (allow some tolerance for small samples)
        for col in train_df.columns:
            self.assertAlmostEqual(stds[col], 1.0, delta=0.05)
    
    def test_single_feature_scaling(self):
        """Test scaling with single feature TimeSeries."""
        scaled_train, scaled_val, scaled_test, scaler = self.scaler.scale_data(
            self.train_single, self.val_single, self.test_single
        )
        
        # Check that scaling works with single feature
        train_df = get_dataframe_from_timeseries(scaled_train)
        self.assertEqual(len(train_df.columns), 1)
        
        # Check statistics (allow some tolerance for small samples)
        self.assertAlmostEqual(train_df.mean().iloc[0], 0.0, places=10)
        self.assertAlmostEqual(train_df.std().iloc[0], 1.0, delta=0.05)
    
    def test_scaler_fitted_only_on_training_data(self):
        """Test that scaler is fitted only on training data."""
        # Get original training statistics
        train_df = get_dataframe_from_timeseries(self.train_ts)
        original_train_mean = train_df.mean()
        original_train_std = train_df.std()
        
        # Scale data
        scaled_train, scaled_val, scaled_test, scaler = self.scaler.scale_data(
            self.train_ts, self.val_ts, self.test_ts
        )
        
        # Check that scaler parameters match training data statistics (allow small differences due to numerical precision)
        for i, col in enumerate(train_df.columns):
            self.assertAlmostEqual(scaler.mean_[i], original_train_mean[col], delta=0.01)
            self.assertAlmostEqual(scaler.scale_[i], original_train_std[col], delta=0.1)
    
    def test_validation_and_test_data_not_used_for_fitting(self):
        """Test that validation and test data don't affect scaler fitting."""
        # Scale with original splits
        _, _, _, scaler1 = self.scaler.scale_data(
            self.train_ts, self.val_ts, self.test_ts
        )
        
        # Create modified validation and test data with extreme values
        val_df = get_dataframe_from_timeseries(self.val_ts)
        test_df = get_dataframe_from_timeseries(self.test_ts)
        
        # Add extreme values to val and test
        val_df_extreme = val_df * 1000  # Multiply by 1000
        test_df_extreme = test_df * 1000
        
        val_ts_extreme = TimeSeries.from_dataframe(val_df_extreme)
        test_ts_extreme = TimeSeries.from_dataframe(test_df_extreme)
        
        # Scale with extreme val/test data
        _, _, _, scaler2 = self.scaler.scale_data(
            self.train_ts, val_ts_extreme, test_ts_extreme
        )
        
        # Scaler parameters should be identical (only training data used)
        np.testing.assert_array_almost_equal(scaler1.mean_, scaler2.mean_)
        np.testing.assert_array_almost_equal(scaler1.scale_, scaler2.scale_)
    
    def test_feature_names_preserved(self):
        """Test that feature names are preserved after scaling."""
        original_columns = list(self.train_ts.columns)
        
        scaled_train, scaled_val, scaled_test, scaler = self.scaler.scale_data(
            self.train_ts, self.val_ts, self.test_ts
        )
        
        # Check that column names are preserved
        self.assertEqual(list(scaled_train.columns), original_columns)
        self.assertEqual(list(scaled_val.columns), original_columns)
        self.assertEqual(list(scaled_test.columns), original_columns)
    
    def test_time_index_preserved(self):
        """Test that time indices are preserved after scaling."""
        scaled_train, scaled_val, scaled_test, scaler = self.scaler.scale_data(
            self.train_ts, self.val_ts, self.test_ts
        )
        
        # Check that time indices are preserved
        pd.testing.assert_index_equal(scaled_train.time_index, self.train_ts.time_index)
        pd.testing.assert_index_equal(scaled_val.time_index, self.val_ts.time_index)
        pd.testing.assert_index_equal(scaled_test.time_index, self.test_ts.time_index)
    
    def test_no_nan_values_after_scaling(self):
        """Test that no NaN values are introduced during scaling."""
        scaled_train, scaled_val, scaled_test, scaler = self.scaler.scale_data(
            self.train_ts, self.val_ts, self.test_ts
        )
        
        # Check for NaN values
        train_df = get_dataframe_from_timeseries(scaled_train)
        val_df = get_dataframe_from_timeseries(scaled_val)
        test_df = get_dataframe_from_timeseries(scaled_test)
        
        self.assertFalse(train_df.isna().any().any())
        self.assertFalse(val_df.isna().any().any())
        self.assertFalse(test_df.isna().any().any())
    
    def test_no_infinite_values_after_scaling(self):
        """Test that no infinite values are introduced during scaling."""
        scaled_train, scaled_val, scaled_test, scaler = self.scaler.scale_data(
            self.train_ts, self.val_ts, self.test_ts
        )
        
        # Check for infinite values
        train_df = get_dataframe_from_timeseries(scaled_train)
        val_df = get_dataframe_from_timeseries(scaled_val)
        test_df = get_dataframe_from_timeseries(scaled_test)
        
        self.assertFalse(np.isinf(train_df.values).any())
        self.assertFalse(np.isinf(val_df.values).any())
        self.assertFalse(np.isinf(test_df.values).any())
    
    def test_input_validation_none_timeseries(self):
        """Test input validation with None TimeSeries."""
        with self.assertRaises(ValueError):
            self.scaler.scale_data(None, self.val_ts, self.test_ts)
        
        with self.assertRaises(ValueError):
            self.scaler.scale_data(self.train_ts, None, self.test_ts)
        
        with self.assertRaises(ValueError):
            self.scaler.scale_data(self.train_ts, self.val_ts, None)
    
    def test_input_validation_mismatched_features(self):
        """Test input validation with mismatched feature counts."""
        # Create TimeSeries with different number of features
        dates = pd.date_range(start='2020-01-01', periods=50, freq='D')
        df_different = pd.DataFrame({
            'feature1': np.random.randn(50),
            'feature2': np.random.randn(50),
            'feature3': np.random.randn(50),
            'feature4': np.random.randn(50)  # Extra feature
        }, index=dates)
        ts_different = TimeSeries.from_dataframe(df_different)
        
        with self.assertRaises(ValueError):
            self.scaler.scale_data(self.train_ts, self.val_ts, ts_different)
    
    def test_get_scaling_info(self):
        """Test get_scaling_info method."""
        scaled_train, scaled_val, scaled_test, scaler = self.scaler.scale_data(
            self.train_ts, self.val_ts, self.test_ts
        )
        
        info = self.scaler.get_scaling_info(scaler, scaled_train, scaled_val, scaled_test)
        
        # Check that info contains expected keys
        expected_keys = [
            'scaler_mean', 'scaler_scale', 'feature_names', 
            'splits_statistics', 'scaling_validation_passed'
        ]
        for key in expected_keys:
            self.assertIn(key, info)
        
        # Check that feature names are correct
        expected_features = list(self.train_ts.columns)
        self.assertEqual(info['feature_names'], expected_features)
        
        # Check that splits statistics are present
        self.assertIn('train', info['splits_statistics'])
        self.assertIn('val', info['splits_statistics'])
        self.assertIn('test', info['splits_statistics'])
        
        # Check that validation passed
        self.assertTrue(info['scaling_validation_passed'])
    
    def test_inverse_transform_timeseries(self):
        """Test inverse transformation of scaled TimeSeries."""
        scaled_train, scaled_val, scaled_test, scaler = self.scaler.scale_data(
            self.train_ts, self.val_ts, self.test_ts
        )
        
        # Inverse transform training data
        reconstructed_train = self.scaler.inverse_transform_timeseries(scaled_train, scaler)
        
        # Check that reconstruction is close to original
        original_df = get_dataframe_from_timeseries(self.train_ts)
        reconstructed_df = get_dataframe_from_timeseries(reconstructed_train)
        
        np.testing.assert_allclose(
            original_df.values, reconstructed_df.values, rtol=1e-10
        )
    
    def test_validate_scaling_consistency(self):
        """Test validate_scaling_consistency method."""
        scaled_train, scaled_val, scaled_test, scaler = self.scaler.scale_data(
            self.train_ts, self.val_ts, self.test_ts
        )
        
        # Should return True for consistent scaling
        is_consistent = self.scaler.validate_scaling_consistency(
            self.train_ts, scaled_train, scaler
        )
        self.assertTrue(is_consistent)
    
    def test_small_dataset_scaling(self):
        """Test scaling with small dataset."""
        scaled_train, scaled_val, scaled_test, scaler = self.scaler.scale_data(
            self.train_small, self.val_small, self.test_small
        )
        
        # Should work without errors
        train_df = get_dataframe_from_timeseries(scaled_train)
        
        # Check basic statistics (allow tolerance for small samples)
        for col in train_df.columns:
            self.assertAlmostEqual(train_df[col].mean(), 0.0, places=10)
            self.assertAlmostEqual(train_df[col].std(), 1.0, delta=0.05)
    
    def test_scaling_with_constant_feature(self):
        """Test scaling behavior with constant feature (std = 0)."""
        # Create TimeSeries with one constant feature
        dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
        df_constant = pd.DataFrame({
            'variable': np.random.randn(100),
            'constant': np.ones(100) * 5.0  # Constant value
        }, index=dates)
        ts_constant = TimeSeries.from_dataframe(df_constant)
        
        train_const, val_const, test_const = self.splitter.split_data(ts_constant)
        
        # Scaling should handle constant feature (sklearn sets scale to 1.0 for constant features)
        scaled_train, scaled_val, scaled_test, scaler = self.scaler.scale_data(
            train_const, val_const, test_const
        )
        
        # Should not raise errors
        self.assertIsInstance(scaled_train, TimeSeries)
        
        # Constant feature should be transformed to (constant - mean) / 1.0 = 0
        train_df = get_dataframe_from_timeseries(scaled_train)
        constant_values = train_df['constant'].values
        np.testing.assert_allclose(constant_values, 0.0, atol=1e-10)
    
    def test_scaling_preserves_temporal_order(self):
        """Test that scaling preserves temporal ordering."""
        scaled_train, scaled_val, scaled_test, scaler = self.scaler.scale_data(
            self.train_ts, self.val_ts, self.test_ts
        )
        
        # Check that temporal ordering is preserved
        self.assertTrue(scaled_train.time_index.is_monotonic_increasing)
        self.assertTrue(scaled_val.time_index.is_monotonic_increasing)
        self.assertTrue(scaled_test.time_index.is_monotonic_increasing)
    
    def test_scaling_deterministic(self):
        """Test that scaling is deterministic (same input produces same output)."""
        scaled_train1, scaled_val1, scaled_test1, scaler1 = self.scaler.scale_data(
            self.train_ts, self.val_ts, self.test_ts
        )
        
        scaled_train2, scaled_val2, scaled_test2, scaler2 = self.scaler.scale_data(
            self.train_ts, self.val_ts, self.test_ts
        )
        
        # Check that results are identical
        train_df1 = get_dataframe_from_timeseries(scaled_train1)
        train_df2 = get_dataframe_from_timeseries(scaled_train2)
        
        np.testing.assert_array_equal(train_df1.values, train_df2.values)
        np.testing.assert_array_equal(scaler1.mean_, scaler2.mean_)
        np.testing.assert_array_equal(scaler1.scale_, scaler2.scale_)


def run_tests():
    """Run all tests and return results."""
    # Create test suite
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestDataScaler)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"DataScaler Tests Summary")
    print(f"{'='*50}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print(f"\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    exit(0 if success else 1)