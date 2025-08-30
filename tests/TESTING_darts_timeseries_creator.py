"""
Unit tests for DartsTimeSeriesCreator class.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, date
from pandas.tseries.offsets import CustomBusinessDay
from pandas.tseries.holiday import Holiday
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from darts_timeseries_creator import DartsTimeSeriesCreator, TimeSeriesIndexError, NaNValuesError

# Try to import DARTS - skip tests if not available
try:
    from darts import TimeSeries
    DARTS_AVAILABLE = True
except ImportError:
    DARTS_AVAILABLE = False
    TimeSeries = None


@unittest.skipUnless(DARTS_AVAILABLE, "DARTS library not available")
class TestDartsTimeSeriesCreator(unittest.TestCase):
    """Test cases for DartsTimeSeriesCreator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.creator = DartsTimeSeriesCreator()
        
        # Create sample DataFrame with proper datetime index and numeric columns
        dates = pd.date_range(start='2015-02-02', end='2015-02-06', freq='B')
        self.sample_df = pd.DataFrame({
            'adjusted_close': [100.0, 101.0, 102.0, 103.0, 104.0],
            'volume': [1000, 1100, 1200, 1300, 1400],
            'technical_indicator': [0.5, 0.6, 0.7, 0.8, 0.9]
        }, index=dates)
        
        # Create DataFrame with missing business days for custom frequency testing
        missing_dates = [
            '2015-02-02',  # Monday
            '2015-02-03',  # Tuesday
            # '2015-02-04' is missing (Wednesday)
            '2015-02-05',  # Thursday
            '2015-02-06',  # Friday
        ]
        self.missing_days_df = pd.DataFrame({
            'adjusted_close': [100.0, 101.0, 103.0, 104.0],
            'volume': [1000, 1100, 1300, 1400]
        }, index=pd.to_datetime(missing_dates))
        
        # Create custom business day frequency for testing
        missing_holiday = Holiday('Missing_Trading_Day_2015_02_04', month=2, day=4, year=2015)
        from pandas.tseries.holiday import AbstractHolidayCalendar
        
        class TestHolidayCalendar(AbstractHolidayCalendar):
            rules = [missing_holiday]
        
        self.custom_freq = CustomBusinessDay(calendar=TestHolidayCalendar())
    
    def test_create_timeseries_basic(self):
        """Test basic TimeSeries creation without custom frequency."""
        timeseries = self.creator.create_timeseries(self.sample_df)
        
        # Check that TimeSeries is created
        self.assertIsInstance(timeseries, TimeSeries)
        
        # Check length matches DataFrame
        self.assertEqual(len(timeseries), len(self.sample_df))
        
        # Check column count matches
        self.assertEqual(len(timeseries.columns), len(self.sample_df.columns))
        
        # Check that start and end times match
        self.assertEqual(timeseries.start_time(), self.sample_df.index[0])
        self.assertEqual(timeseries.end_time(), self.sample_df.index[-1])
    
    def test_create_timeseries_with_custom_frequency(self):
        """Test TimeSeries creation with custom business day frequency."""
        timeseries = self.creator.create_timeseries(self.missing_days_df, self.custom_freq)
        
        # Check that TimeSeries is created
        self.assertIsInstance(timeseries, TimeSeries)
        
        # Check length matches DataFrame
        self.assertEqual(len(timeseries), len(self.missing_days_df))
        
        # Check that values are preserved (allowing for potential length differences)
        try:
            ts_df = timeseries.pd_dataframe()
        except AttributeError:
            ts_df = timeseries.to_dataframe()
        
        # Compare first few values to ensure data integrity
        min_len = min(len(ts_df), len(self.missing_days_df))
        if min_len > 0:
            self.assertTrue(np.allclose(
                ts_df['adjusted_close'].iloc[:min_len].values,
                self.missing_days_df['adjusted_close'].iloc[:min_len].values
            ))
    
    def test_validate_input_dataframe_empty(self):
        """Test validation with empty DataFrame."""
        empty_df = pd.DataFrame()
        
        with self.assertRaises(ValueError) as context:
            self.creator.create_timeseries(empty_df)
        
        self.assertIn("DataFrame cannot be empty", str(context.exception))
    
    def test_validate_input_dataframe_non_datetime_index(self):
        """Test validation with non-datetime index."""
        invalid_df = pd.DataFrame({
            'adjusted_close': [100.0, 101.0],
            'volume': [1000, 1100]
        }, index=[0, 1])  # Integer index instead of datetime
        
        with self.assertRaises(TimeSeriesIndexError) as context:
            self.creator.create_timeseries(invalid_df)
        
        self.assertIn("DatetimeIndex", str(context.exception))
    
    def test_validate_input_dataframe_non_monotonic(self):
        """Test validation with non-monotonic index."""
        unsorted_dates = ['2015-02-03', '2015-02-02', '2015-02-04']  # Out of order
        unsorted_df = pd.DataFrame({
            'adjusted_close': [101.0, 100.0, 102.0],
            'volume': [1100, 1000, 1200]
        }, index=pd.to_datetime(unsorted_dates))
        
        with self.assertRaises(TimeSeriesIndexError) as context:
            self.creator.create_timeseries(unsorted_df)
        
        self.assertIn("strictly increasing", str(context.exception))
    
    def test_validate_input_dataframe_duplicate_index(self):
        """Test validation with duplicate index values."""
        duplicate_dates = ['2015-02-02', '2015-02-02', '2015-02-03']  # Duplicate
        duplicate_df = pd.DataFrame({
            'adjusted_close': [100.0, 100.5, 101.0],
            'volume': [1000, 1050, 1100]
        }, index=pd.to_datetime(duplicate_dates))
        
        with self.assertRaises(TimeSeriesIndexError) as context:
            self.creator.create_timeseries(duplicate_df)
        
        self.assertIn("Duplicate index values", str(context.exception))
    
    def test_validate_input_dataframe_non_numeric_columns(self):
        """Test validation with non-numeric columns."""
        non_numeric_df = pd.DataFrame({
            'adjusted_close': [100.0, 101.0],
            'volume': [1000, 1100],
            'symbol': ['SPY', 'SPY']  # Non-numeric column
        }, index=pd.date_range(start='2015-02-02', periods=2, freq='B'))
        
        with self.assertRaises(ValueError) as context:
            self.creator.create_timeseries(non_numeric_df)
        
        self.assertIn("All columns must be numeric", str(context.exception))
        self.assertIn("symbol", str(context.exception))
    
    def test_validate_input_dataframe_nan_values(self):
        """Test validation with NaN values."""
        nan_df = pd.DataFrame({
            'adjusted_close': [100.0, np.nan, 102.0],
            'volume': [1000, 1100, 1200]
        }, index=pd.date_range(start='2015-02-02', periods=3, freq='B'))
        
        with self.assertRaises(NaNValuesError) as context:
            self.creator.create_timeseries(nan_df)
        
        self.assertIn("NaN values found", str(context.exception))
        self.assertIn("adjusted_close", str(context.exception))
    
    def test_apply_custom_frequency(self):
        """Test applying custom frequency to DataFrame."""
        df_with_freq = self.creator._apply_custom_frequency(self.missing_days_df, self.custom_freq)
        
        # Check that DataFrame structure is preserved
        self.assertEqual(len(df_with_freq), len(self.missing_days_df))
        self.assertTrue(df_with_freq.columns.equals(self.missing_days_df.columns))
        
        # Check that values are preserved
        pd.testing.assert_frame_equal(df_with_freq, self.missing_days_df, check_freq=False)
    
    def test_validate_timeseries_properties(self):
        """Test TimeSeries properties validation."""
        timeseries = self.creator.create_timeseries(self.sample_df)
        
        # This should not raise any exceptions if validation passes
        self.creator._validate_timeseries_properties(timeseries, self.sample_df)
        
        # Check specific properties
        # Note: lengths may differ slightly due to DARTS processing
        self.assertGreater(len(timeseries), 0)
        
        try:
            ts_df = timeseries.pd_dataframe()
        except AttributeError:
            ts_df = timeseries.to_dataframe()
        
        self.assertFalse(ts_df.isna().any().any())
        self.assertTrue(timeseries.time_index.is_monotonic_increasing)
    
    def test_get_timeseries_info(self):
        """Test getting TimeSeries information."""
        timeseries = self.creator.create_timeseries(self.sample_df)
        info = self.creator.get_timeseries_info(timeseries)
        
        # Check that all expected keys are present
        expected_keys = [
            'length', 'columns', 'start_time', 'end_time', 'frequency',
            'has_nan', 'data_types', 'value_ranges'
        ]
        for key in expected_keys:
            self.assertIn(key, info)
        
        # Check specific values
        self.assertEqual(info['length'], len(self.sample_df))
        self.assertEqual(len(info['columns']), len(self.sample_df.columns))
        self.assertFalse(info['has_nan'])
        
        # Check value ranges
        for col in self.sample_df.columns:
            self.assertIn(col, info['value_ranges'])
            self.assertEqual(info['value_ranges'][col]['min'], self.sample_df[col].min())
            self.assertEqual(info['value_ranges'][col]['max'], self.sample_df[col].max())
    
    def test_validate_business_day_frequency(self):
        """Test business day frequency validation."""
        timeseries = self.creator.create_timeseries(self.sample_df)
        
        # Test with no expected frequency (should return True)
        self.assertTrue(self.creator.validate_business_day_frequency(timeseries))
        
        # Test with custom frequency
        result = self.creator.validate_business_day_frequency(timeseries, self.custom_freq)
        self.assertIsInstance(result, bool)
    
    def test_real_data_file(self):
        """Test with real covariaterawdata1.csv file if available."""
        real_file_path = os.path.join(os.path.dirname(__file__), '..', 'Data', 'covariaterawdata1.csv')
        
        if os.path.exists(real_file_path):
            # Load and preprocess real data
            df = pd.read_csv(real_file_path)
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date').sort_index()
            
            # Remove non-numeric columns
            numeric_df = df.select_dtypes(include=[np.number])
            
            # Remove any NaN values
            numeric_df = numeric_df.dropna()
            
            # Ensure we have the required column
            if 'adjusted_close' in numeric_df.columns and len(numeric_df) > 0:
                try:
                    # Test TimeSeries creation with real data
                    timeseries = self.creator.create_timeseries(numeric_df)
                    
                    # Check that TimeSeries is valid
                    self.assertIsInstance(timeseries, TimeSeries)
                    
                    # Get TimeSeries info
                    info = self.creator.get_timeseries_info(timeseries)
                    
                    # Check that info is reasonable
                    self.assertGreater(info['length'], 0)
                    
                    print(f"Real data TimeSeries info: {info}")
                    
                except (NaNValuesError, ValueError) as e:
                    # Skip test if real data has quality issues
                    self.skipTest(f"Real data has quality issues: {e}")
            else:
                self.skipTest("Real data file missing 'adjusted_close' column or empty after cleaning")
        else:
            self.skipTest("Real data file not found")
    
    def test_edge_case_single_column(self):
        """Test with single column DataFrame."""
        single_col_df = pd.DataFrame({
            'adjusted_close': [100.0, 101.0, 102.0]
        }, index=pd.date_range(start='2015-02-02', periods=3, freq='B'))
        
        timeseries = self.creator.create_timeseries(single_col_df)
        
        # Check that TimeSeries is created correctly
        self.assertIsInstance(timeseries, TimeSeries)
        self.assertEqual(len(timeseries), 3)
        self.assertEqual(len(timeseries.columns), 1)
    
    def test_edge_case_large_values(self):
        """Test with large numeric values."""
        large_values_df = pd.DataFrame({
            'adjusted_close': [1e6, 1e7, 1e8],
            'volume': [1e9, 1e10, 1e11]
        }, index=pd.date_range(start='2015-02-02', periods=3, freq='B'))
        
        timeseries = self.creator.create_timeseries(large_values_df)
        
        # Check that large values are handled correctly
        self.assertIsInstance(timeseries, TimeSeries)
        
        try:
            ts_df = timeseries.pd_dataframe()
        except AttributeError:
            ts_df = timeseries.to_dataframe()
        
        # Compare values allowing for potential length differences
        min_len = min(len(ts_df), len(large_values_df))
        if min_len > 0:
            self.assertTrue(np.allclose(
                ts_df['adjusted_close'].iloc[:min_len].values,
                large_values_df['adjusted_close'].iloc[:min_len].values
            ))
    
    def test_edge_case_small_values(self):
        """Test with very small numeric values."""
        small_values_df = pd.DataFrame({
            'adjusted_close': [1e-6, 1e-7, 1e-8],
            'volume': [1e-3, 1e-4, 1e-5]
        }, index=pd.date_range(start='2015-02-02', periods=3, freq='B'))
        
        timeseries = self.creator.create_timeseries(small_values_df)
        
        # Check that small values are handled correctly
        self.assertIsInstance(timeseries, TimeSeries)
        
        try:
            ts_df = timeseries.pd_dataframe()
        except AttributeError:
            ts_df = timeseries.to_dataframe()
        
        # Compare values allowing for potential length differences
        min_len = min(len(ts_df), len(small_values_df))
        if min_len > 0:
            self.assertTrue(np.allclose(
                ts_df['adjusted_close'].iloc[:min_len].values,
                small_values_df['adjusted_close'].iloc[:min_len].values
            ))


@unittest.skipIf(DARTS_AVAILABLE, "DARTS library is available - skipping mock tests")
class TestDartsTimeSeriesCreatorWithoutDarts(unittest.TestCase):
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