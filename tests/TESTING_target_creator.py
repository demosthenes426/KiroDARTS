"""
Unit tests for TargetCreator class.

Tests the creation of 5-day future price prediction targets and validates
that no data leakage occurs in the target creation process.
"""

import unittest
import pandas as pd
import numpy as np
from darts import TimeSeries
import sys
import os

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from target_creator import TargetCreator


class TestTargetCreator(unittest.TestCase):
    """Test cases for TargetCreator class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create sample time series data
        dates = pd.date_range('2023-01-01', periods=20, freq='B')  # 20 business days
        
        # Create sample data with predictable pattern for testing
        adjusted_close = [100 + i for i in range(20)]  # 100, 101, 102, ..., 119
        volume = [1000 + i * 10 for i in range(20)]
        
        self.sample_df = pd.DataFrame({
            'adjusted_close': adjusted_close,
            'volume': volume
        }, index=dates)
        
        self.sample_ts = TimeSeries.from_dataframe(self.sample_df)
        self.target_creator = TargetCreator(prediction_horizon=5)
    
    def test_init(self):
        """Test TargetCreator initialization."""
        creator = TargetCreator()
        self.assertEqual(creator.prediction_horizon, 5)
        
        creator_custom = TargetCreator(prediction_horizon=3)
        self.assertEqual(creator_custom.prediction_horizon, 3)
    
    def test_create_targets_basic(self):
        """Test basic target creation functionality."""
        targets = self.target_creator.create_targets(self.sample_ts)
        
        # Check that target length is correct (original length - prediction horizon)
        expected_length = len(self.sample_ts) - 5
        self.assertEqual(len(targets), expected_length)
        
        # Check that target has correct column name
        expected_col_name = "adjusted_close_target_5d"
        self.assertIn(expected_col_name, targets.columns)
    
    def test_create_targets_values(self):
        """Test that target values are correct (no data leakage)."""
        targets = self.target_creator.create_targets(self.sample_ts)
        try:
            target_df = targets.pd_dataframe()
        except AttributeError:
            try:
                target_df = targets.to_dataframe()
            except AttributeError:
                target_df = targets.pd_series().to_frame()
        
        try:
            original_df = self.sample_ts.pd_dataframe()
        except AttributeError:
            try:
                original_df = self.sample_ts.to_dataframe()
            except AttributeError:
                original_df = self.sample_ts.pd_series().to_frame()
        
        # Check specific target values
        target_col = target_df.columns[0]
        
        # First target should be the 6th value in original data (index 5)
        self.assertEqual(target_df.iloc[0][target_col], original_df.iloc[5]['adjusted_close'])
        
        # Second target should be the 7th value in original data (index 6)
        self.assertEqual(target_df.iloc[1][target_col], original_df.iloc[6]['adjusted_close'])
        
        # Last target should be the last value in original data
        last_target_idx = len(target_df) - 1
        last_original_idx = len(original_df) - 1
        self.assertEqual(target_df.iloc[last_target_idx][target_col], 
                        original_df.iloc[last_original_idx]['adjusted_close'])
    
    def test_create_targets_custom_column(self):
        """Test target creation with custom target column."""
        targets = self.target_creator.create_targets(self.sample_ts, target_column='volume')
        
        # Check column name
        expected_col_name = "volume_target_5d"
        self.assertIn(expected_col_name, targets.columns)
        
        # Check values
        try:
            target_df = targets.pd_dataframe()
        except AttributeError:
            try:
                target_df = targets.to_dataframe()
            except AttributeError:
                target_df = targets.pd_series().to_frame()
        
        try:
            original_df = self.sample_ts.pd_dataframe()
        except AttributeError:
            try:
                original_df = self.sample_ts.to_dataframe()
            except AttributeError:
                original_df = self.sample_ts.pd_series().to_frame()
        
        target_col = target_df.columns[0]
        self.assertEqual(target_df.iloc[0][target_col], original_df.iloc[5]['volume'])
    
    def test_create_targets_invalid_column(self):
        """Test error handling for invalid target column."""
        with self.assertRaises(ValueError) as context:
            self.target_creator.create_targets(self.sample_ts, target_column='nonexistent')
        
        self.assertIn("Target column 'nonexistent' not found", str(context.exception))
    
    def test_create_targets_insufficient_data(self):
        """Test error handling for insufficient data."""
        # Create very short time series
        short_dates = pd.date_range('2023-01-01', periods=3, freq='B')
        short_df = pd.DataFrame({
            'adjusted_close': [100, 101, 102]
        }, index=short_dates)
        short_ts = TimeSeries.from_dataframe(short_df)
        
        with self.assertRaises(ValueError) as context:
            self.target_creator.create_targets(short_ts)
        
        self.assertIn("must be greater than prediction horizon", str(context.exception))
    
    def test_validate_no_data_leakage_valid(self):
        """Test validation passes for correctly created targets."""
        targets = self.target_creator.create_targets(self.sample_ts)
        
        is_valid = self.target_creator.validate_no_data_leakage(
            self.sample_ts, targets, 'adjusted_close'
        )
        
        self.assertTrue(is_valid)
    
    def test_validate_no_data_leakage_invalid_length(self):
        """Test validation fails for incorrect target length."""
        targets = self.target_creator.create_targets(self.sample_ts)
        
        # Manually create targets with wrong length
        try:
            wrong_df = targets.pd_dataframe().iloc[:-1]  # Remove one row
        except AttributeError:
            try:
                wrong_df = targets.to_dataframe().iloc[:-1]
            except AttributeError:
                wrong_df = targets.pd_series().to_frame().iloc[:-1]
        wrong_targets = TimeSeries.from_dataframe(wrong_df)
        
        is_valid = self.target_creator.validate_no_data_leakage(
            self.sample_ts, wrong_targets, 'adjusted_close'
        )
        
        self.assertFalse(is_valid)
    
    def test_validate_no_data_leakage_wrong_values(self):
        """Test validation fails for incorrect target values."""
        targets = self.target_creator.create_targets(self.sample_ts)
        
        # Manually corrupt target values
        try:
            corrupted_df = targets.pd_dataframe().copy()
        except AttributeError:
            try:
                corrupted_df = targets.to_dataframe().copy()
            except AttributeError:
                corrupted_df = targets.pd_series().to_frame().copy()
        corrupted_df.iloc[0, 0] = 999.0  # Wrong value
        corrupted_targets = TimeSeries.from_dataframe(corrupted_df)
        
        is_valid = self.target_creator.validate_no_data_leakage(
            self.sample_ts, corrupted_targets, 'adjusted_close'
        )
        
        self.assertFalse(is_valid)
    
    def test_get_aligned_features_and_targets(self):
        """Test creation of aligned features and targets."""
        aligned_features, targets = self.target_creator.get_aligned_features_and_targets(
            self.sample_ts, 'adjusted_close'
        )
        
        # Check lengths match
        self.assertEqual(len(aligned_features), len(targets))
        
        # Check that features are properly truncated
        expected_length = len(self.sample_ts) - 5
        self.assertEqual(len(aligned_features), expected_length)
        self.assertEqual(len(targets), expected_length)
        
        # Check that features contain all original columns
        for col in self.sample_ts.columns:
            self.assertIn(col, aligned_features.columns)
        
        # Check that targets have the correct column
        self.assertIn("adjusted_close_target_5d", targets.columns)
    
    def test_different_prediction_horizons(self):
        """Test target creation with different prediction horizons."""
        # Test 3-day horizon
        creator_3d = TargetCreator(prediction_horizon=3)
        targets_3d = creator_3d.create_targets(self.sample_ts)
        
        expected_length_3d = len(self.sample_ts) - 3
        self.assertEqual(len(targets_3d), expected_length_3d)
        
        # Test 1-day horizon
        creator_1d = TargetCreator(prediction_horizon=1)
        targets_1d = creator_1d.create_targets(self.sample_ts)
        
        expected_length_1d = len(self.sample_ts) - 1
        self.assertEqual(len(targets_1d), expected_length_1d)
        
        # Verify different horizons produce different results
        self.assertNotEqual(len(targets_3d), len(targets_1d))
    
    def test_real_data_pattern(self):
        """Test with more realistic stock price data pattern."""
        # Create data with more realistic stock price movements
        dates = pd.date_range('2023-01-01', periods=50, freq='B')
        
        # Simulate stock price with some volatility
        np.random.seed(42)  # For reproducible tests
        base_price = 100
        returns = np.random.normal(0.001, 0.02, 50)  # Daily returns
        prices = [base_price]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        realistic_df = pd.DataFrame({
            'adjusted_close': prices,
            'volume': np.random.randint(1000, 10000, 50),
            'high': [p * 1.02 for p in prices],
            'low': [p * 0.98 for p in prices]
        }, index=dates)
        
        realistic_ts = TimeSeries.from_dataframe(realistic_df)
        
        # Test target creation
        targets = self.target_creator.create_targets(realistic_ts)
        
        # Validate no data leakage
        is_valid = self.target_creator.validate_no_data_leakage(
            realistic_ts, targets, 'adjusted_close'
        )
        
        self.assertTrue(is_valid)
        
        # Check that targets are reasonable (not all the same)
        try:
            target_values = targets.pd_dataframe().iloc[:, 0].values
        except AttributeError:
            try:
                target_values = targets.to_dataframe().iloc[:, 0].values
            except AttributeError:
                target_values = targets.pd_series().to_frame().iloc[:, 0].values
        self.assertGreater(np.std(target_values), 0)  # Should have some variation
    
    def test_edge_case_minimum_data(self):
        """Test with minimum required data (prediction_horizon + 1 points)."""
        min_dates = pd.date_range('2023-01-01', periods=6, freq='B')  # 5 + 1
        min_df = pd.DataFrame({
            'adjusted_close': [100, 101, 102, 103, 104, 105]
        }, index=min_dates)
        min_ts = TimeSeries.from_dataframe(min_df)
        
        targets = self.target_creator.create_targets(min_ts)
        
        # Should have exactly 1 target value
        self.assertEqual(len(targets), 1)
        
        # Target should be the last value (105)
        try:
            target_value = targets.pd_dataframe().iloc[0, 0]
        except AttributeError:
            try:
                target_value = targets.to_dataframe().iloc[0, 0]
            except AttributeError:
                target_value = targets.pd_series().to_frame().iloc[0, 0]
        self.assertEqual(target_value, 105)


if __name__ == '__main__':
    unittest.main()