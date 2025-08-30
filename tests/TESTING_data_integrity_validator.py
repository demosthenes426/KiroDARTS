"""
Unit tests for DataIntegrityValidator class.
"""

import unittest
import pandas as pd
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch
import sys
from datetime import datetime, timedelta

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_integrity_validator import DataIntegrityValidator, ValidationReport
from darts import TimeSeries
from sklearn.preprocessing import StandardScaler


class TestDataIntegrityValidator(unittest.TestCase):
    """Test cases for DataIntegrityValidator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.validator = DataIntegrityValidator()
        
        # Create sample data
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        self.sample_df = pd.DataFrame({
            'adjusted_close': np.random.randn(100) * 10 + 100,
            'volume': np.random.randint(1000000, 10000000, 100),
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100)
        }, index=dates)
        
        # Create valid TimeSeries
        self.valid_ts = TimeSeries.from_dataframe(self.sample_df[['adjusted_close', 'feature1', 'feature2']])
        
        # Create fitted scaler
        self.fitted_scaler = StandardScaler()
        sample_data = np.random.randn(50, 3)
        self.fitted_scaler.fit(sample_data)
        
        # Create scaled TimeSeries (approximately mean=0, std=1)
        scaled_data = self.fitted_scaler.transform(self.valid_ts.values())
        self.scaled_ts = TimeSeries.from_times_and_values(
            self.valid_ts.time_index, 
            scaled_data
        )
    
    def test_validate_data_integrity_valid_data(self):
        """Test validation with valid data."""
        report = self.validator.validate_data_integrity(
            self.scaled_ts, 
            self.fitted_scaler, 
            self.sample_df
        )
        
        self.assertTrue(report.data_integrity_passed)
        self.assertTrue(report.timeseries_validation_passed)
        self.assertTrue(report.scaling_validation_passed)
        self.assertEqual(len(report.issues), 0)
    
    def test_validate_data_integrity_without_scaler(self):
        """Test validation without scaler."""
        report = self.validator.validate_data_integrity(self.valid_ts)
        
        self.assertTrue(report.data_integrity_passed)
        self.assertTrue(report.timeseries_validation_passed)
        self.assertTrue(report.scaling_validation_passed)  # Should be True when no scaler provided
    
    def test_timeseries_with_nan_values(self):
        """Test TimeSeries validation with NaN values."""
        # Use mock since DARTS doesn't allow NaN values in creation
        mock_ts = Mock(spec=TimeSeries)
        mock_ts.values.return_value = np.array([[1.0, np.nan], [2.0, 3.0]])
        mock_ts.time_index = pd.date_range('2020-01-01', periods=2, freq='D')
        mock_ts.__len__ = Mock(return_value=2)
        
        report = self.validator.validate_data_integrity(mock_ts)
        
        self.assertFalse(report.timeseries_validation_passed)
        self.assertIn("TimeSeries contains NaN values", report.issues)
    
    def test_timeseries_with_non_monotonic_index(self):
        """Test TimeSeries validation with non-monotonic index."""
        # Use mock since DARTS may not allow non-monotonic creation
        mock_ts = Mock(spec=TimeSeries)
        mock_ts.values.return_value = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        # Create non-monotonic index
        non_monotonic_index = pd.to_datetime(['2020-01-03', '2020-01-01', '2020-01-02'])
        mock_ts.time_index = non_monotonic_index
        mock_ts.__len__ = Mock(return_value=3)
        
        report = self.validator.validate_data_integrity(mock_ts)
        
        self.assertFalse(report.timeseries_validation_passed)
        self.assertIn("TimeSeries index is not strictly increasing", report.issues)
    
    def test_timeseries_with_infinite_values(self):
        """Test TimeSeries validation with infinite values."""
        # Use mock since DARTS may not allow infinite values in creation
        mock_ts = Mock(spec=TimeSeries)
        mock_ts.values.return_value = np.array([[1.0, 2.0], [np.inf, 4.0]])
        mock_ts.time_index = pd.date_range('2020-01-01', periods=2, freq='D')
        mock_ts.__len__ = Mock(return_value=2)
        
        report = self.validator.validate_data_integrity(mock_ts)
        
        self.assertFalse(report.timeseries_validation_passed)
        self.assertIn("TimeSeries contains infinite values", report.issues)
    
    def test_empty_timeseries(self):
        """Test validation with empty TimeSeries."""
        # Use mock since DARTS may not allow empty TimeSeries creation
        mock_ts = Mock(spec=TimeSeries)
        mock_ts.values.return_value = np.array([]).reshape(0, 1)
        mock_ts.time_index = pd.DatetimeIndex([])
        mock_ts.__len__ = Mock(return_value=0)
        
        report = self.validator.validate_data_integrity(mock_ts)
        
        self.assertFalse(report.timeseries_validation_passed)
        self.assertIn("TimeSeries is empty", report.issues)
    
    def test_timeseries_length_mismatch_with_dataframe(self):
        """Test TimeSeries length validation against original DataFrame."""
        # Create shorter TimeSeries
        shorter_ts = TimeSeries.from_dataframe(self.sample_df[['adjusted_close']].iloc[:50])
        
        report = self.validator.validate_data_integrity(
            shorter_ts, 
            original_df=self.sample_df
        )
        
        self.assertFalse(report.timeseries_validation_passed)
        self.assertIn("TimeSeries length (50) does not match DataFrame length (100)", report.issues)
    
    def test_scaling_statistics_validation_good_scaling(self):
        """Test scaling statistics validation with properly scaled data."""
        # Create perfectly scaled data (mean=0, std=1)
        n_samples, n_features = 1000, 3
        scaled_data = np.random.randn(n_samples, n_features)
        
        dates = pd.date_range('2020-01-01', periods=n_samples, freq='D')
        scaled_df = pd.DataFrame(scaled_data, index=dates, columns=['col1', 'col2', 'col3'])
        scaled_ts = TimeSeries.from_dataframe(scaled_df, fill_missing_dates=True, freq=None)
        
        report = self.validator.validate_data_integrity(scaled_ts, self.fitted_scaler)
        
        self.assertTrue(report.scaling_validation_passed)
    
    def test_scaling_statistics_validation_poor_scaling(self):
        """Test scaling statistics validation with poorly scaled data."""
        # Create poorly scaled data (mean=5, std=2)
        n_samples, n_features = 100, 2
        poorly_scaled_data = np.random.randn(n_samples, n_features) * 2 + 5
        
        dates = pd.date_range('2020-01-01', periods=n_samples, freq='D')
        poorly_scaled_df = pd.DataFrame(poorly_scaled_data, index=dates, columns=['col1', 'col2'])
        poorly_scaled_ts = TimeSeries.from_dataframe(poorly_scaled_df, fill_missing_dates=True, freq=None)
        
        report = self.validator.validate_data_integrity(poorly_scaled_ts, self.fitted_scaler)
        
        # Should have warnings about mean and std
        self.assertTrue(any("mean" in warning for warning in report.warnings))
        self.assertTrue(any("std" in warning for warning in report.warnings))
    
    def test_unfitted_scaler(self):
        """Test validation with unfitted scaler."""
        unfitted_scaler = StandardScaler()
        
        report = self.validator.validate_data_integrity(self.valid_ts, unfitted_scaler)
        
        self.assertFalse(report.scaling_validation_passed)
        self.assertIn("StandardScaler has not been fitted", report.issues)
    
    def test_scaler_with_invalid_scale(self):
        """Test validation with scaler having invalid scale values."""
        invalid_scaler = StandardScaler()
        invalid_scaler.mean_ = np.array([0, 0])
        invalid_scaler.scale_ = np.array([0, -1])  # Invalid: zero and negative
        
        report = self.validator.validate_data_integrity(self.valid_ts, invalid_scaler)
        
        self.assertFalse(report.scaling_validation_passed)
        self.assertIn("StandardScaler has non-positive scale values", report.issues)
    
    def test_non_numeric_data_validation(self):
        """Test validation with non-numeric data."""
        # Create TimeSeries with object dtype (should not happen in practice)
        dates = pd.date_range('2020-01-01', periods=5, freq='D')
        
        # Mock TimeSeries with non-numeric values
        mock_ts = Mock(spec=TimeSeries)
        mock_ts.values.return_value = np.array([['a', 'b'], ['c', 'd']], dtype=object)
        mock_ts.time_index = dates
        mock_ts.__len__ = Mock(return_value=5)
        
        report = self.validator.validate_data_integrity(mock_ts)
        
        self.assertFalse(report.data_integrity_passed)
        self.assertIn("TimeSeries contains non-numeric data", report.issues)
    
    def test_very_large_values_warning(self):
        """Test warning for very large values."""
        large_data = np.array([[1e12, 1e11], [1e12, 1e11]])
        dates = pd.date_range('2020-01-01', periods=2, freq='D')
        
        large_df = pd.DataFrame(large_data, index=dates, columns=['col1', 'col2'])
        large_ts = TimeSeries.from_dataframe(large_df, fill_missing_dates=True, freq=None)
        
        report = self.validator.validate_data_integrity(large_ts)
        
        self.assertTrue(any("Very large values detected" in warning for warning in report.warnings))
    
    def test_constant_values_warning(self):
        """Test warning for constant feature values."""
        constant_data = np.array([[1, 5], [1, 5], [1, 5]])  # First feature is constant
        dates = pd.date_range('2020-01-01', periods=3, freq='D')
        
        constant_df = pd.DataFrame(constant_data, index=dates, columns=['col1', 'col2'])
        constant_ts = TimeSeries.from_dataframe(constant_df, fill_missing_dates=True, freq=None)
        
        report = self.validator.validate_data_integrity(constant_ts)
        
        self.assertTrue(any("Feature 0 has constant values" in warning for warning in report.warnings))
    
    def test_few_data_points_warning(self):
        """Test warning for very few data points."""
        small_data = np.array([[1, 2], [3, 4]])  # Only 2 points
        dates = pd.date_range('2020-01-01', periods=2, freq='D')
        
        small_df = pd.DataFrame(small_data, index=dates, columns=['col1', 'col2'])
        small_ts = TimeSeries.from_dataframe(small_df, fill_missing_dates=True, freq=None)
        
        report = self.validator.validate_data_integrity(small_ts)
        
        self.assertTrue(any("TimeSeries has very few data points" in warning for warning in report.warnings))
    
    def test_validate_holiday_calendar_match_perfect_match(self):
        """Test holiday calendar validation with perfect match."""
        missing_dates = [pd.Timestamp('2020-01-01'), pd.Timestamp('2020-01-02')]
        
        # Create simple holiday objects with date attribute
        class SimpleHoliday:
            def __init__(self, date):
                self.date = date
        
        custom_holidays = [
            SimpleHoliday(pd.Timestamp('2020-01-01')),
            SimpleHoliday(pd.Timestamp('2020-01-02'))
        ]
        
        report = self.validator.validate_holiday_calendar_match(missing_dates, custom_holidays)
        
        self.assertTrue(report.data_integrity_passed)
        self.assertEqual(len(report.issues), 0)
    
    def test_validate_holiday_calendar_match_mismatch(self):
        """Test holiday calendar validation with mismatch."""
        missing_dates = [pd.Timestamp('2020-01-01'), pd.Timestamp('2020-01-02')]
        
        # Create simple holiday objects with date attribute
        class SimpleHoliday:
            def __init__(self, date):
                self.date = date
        
        custom_holidays = [
            SimpleHoliday(pd.Timestamp('2020-01-01')),
            SimpleHoliday(pd.Timestamp('2020-01-03'))  # Different date
        ]
        
        report = self.validator.validate_holiday_calendar_match(missing_dates, custom_holidays)
        
        self.assertFalse(report.data_integrity_passed)
        self.assertTrue(any("Extra holidays not in missing dates" in issue for issue in report.issues))
        self.assertTrue(any("Missing dates not in holidays" in issue for issue in report.issues))
    
    def test_validate_holiday_calendar_with_date_range(self):
        """Test holiday calendar validation with date range holidays."""
        missing_dates = [pd.Timestamp('2020-01-01'), pd.Timestamp('2020-01-02')]
        
        # Create holiday with date range
        class DateRangeHoliday:
            def __init__(self, start_date, end_date):
                self.start_date = start_date
                self.end_date = end_date
        
        custom_holidays = [DateRangeHoliday(pd.Timestamp('2020-01-01'), pd.Timestamp('2020-01-02'))]
        
        report = self.validator.validate_holiday_calendar_match(missing_dates, custom_holidays)
        
        self.assertTrue(report.data_integrity_passed)
    
    def test_validate_holiday_calendar_unparseable_holiday(self):
        """Test holiday calendar validation with unparseable holiday."""
        missing_dates = [pd.Timestamp('2020-01-01')]
        
        # Mock unparseable holiday
        unparseable_holiday = object()  # Can't be converted to timestamp
        
        custom_holidays = [unparseable_holiday]
        
        report = self.validator.validate_holiday_calendar_match(missing_dates, custom_holidays)
        
        self.assertTrue(any("Could not parse holiday" in warning for warning in report.warnings))
    
    def test_validate_feature_columns_numeric_all_numeric(self):
        """Test feature column validation with all numeric columns."""
        numeric_df = pd.DataFrame({
            'adjusted_close': [100.0, 101.0, 102.0],
            'volume': [1000000, 1100000, 1200000],
            'feature1': [1.5, 2.5, 3.5],
            'symbol': ['SPY', 'SPY', 'SPY']  # Will be excluded
        })
        
        report = self.validator.validate_feature_columns_numeric(numeric_df)
        
        self.assertTrue(report.data_integrity_passed)
        self.assertEqual(len(report.issues), 0)
    
    def test_validate_feature_columns_numeric_with_non_numeric(self):
        """Test feature column validation with non-numeric columns."""
        mixed_df = pd.DataFrame({
            'adjusted_close': [100.0, 101.0, 102.0],
            'volume': [1000000, 1100000, 1200000],
            'text_feature': ['a', 'b', 'c'],  # Non-numeric
            'symbol': ['SPY', 'SPY', 'SPY']  # Will be excluded
        })
        
        report = self.validator.validate_feature_columns_numeric(mixed_df)
        
        self.assertFalse(report.data_integrity_passed)
        self.assertTrue(any("text_feature" in issue and "not numeric" in issue for issue in report.issues))
    
    def test_validate_feature_columns_numeric_custom_exclude(self):
        """Test feature column validation with custom exclude list."""
        df = pd.DataFrame({
            'adjusted_close': [100.0, 101.0, 102.0],
            'text_col': ['a', 'b', 'c'],  # Non-numeric but will be excluded
            'numeric_col': [1.0, 2.0, 3.0]
        })
        
        report = self.validator.validate_feature_columns_numeric(
            df, 
            exclude_columns=['text_col', 'symbol', 'date']
        )
        
        self.assertTrue(report.data_integrity_passed)
        self.assertEqual(len(report.issues), 0)
    
    def test_validation_report_dataclass(self):
        """Test ValidationReport dataclass functionality."""
        report = ValidationReport(
            data_integrity_passed=True,
            timeseries_validation_passed=False,
            scaling_validation_passed=True,
            issues=['Test issue'],
            warnings=['Test warning']
        )
        
        self.assertTrue(report.data_integrity_passed)
        self.assertFalse(report.timeseries_validation_passed)
        self.assertTrue(report.scaling_validation_passed)
        self.assertEqual(report.issues, ['Test issue'])
        self.assertEqual(report.warnings, ['Test warning'])
    
    def test_tolerance_setting(self):
        """Test that tolerance setting works correctly."""
        validator = DataIntegrityValidator()
        self.assertEqual(validator.tolerance, 1e-6)
        
        # Test with custom tolerance
        validator.tolerance = 1e-3
        self.assertEqual(validator.tolerance, 1e-3)


if __name__ == '__main__':
    unittest.main()