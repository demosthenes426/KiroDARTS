"""
Unit tests for CustomHolidayCalendar class.
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

from custom_holiday_calendar import CustomHolidayCalendar, CalendarMismatchError, InsufficientDataError


class TestCustomHolidayCalendar(unittest.TestCase):
    """Test cases for CustomHolidayCalendar class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.calendar = CustomHolidayCalendar()
        
        # Create sample data with missing business days
        dates = [
            '2015-02-02',  # Monday
            '2015-02-03',  # Tuesday
            # '2015-02-04' is missing (Wednesday)
            '2015-02-05',  # Thursday
            '2015-02-06',  # Friday
            # Weekend (2015-02-07, 2015-02-08)
            '2015-02-09',  # Monday
            # '2015-02-10' is missing (Tuesday)
            '2015-02-11',  # Wednesday
        ]
        
        self.sample_df = pd.DataFrame({
            'adjusted_close': [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
            'volume': [1000, 1100, 1200, 1300, 1400, 1500]
        }, index=pd.to_datetime(dates))
        
        # Create continuous data (no missing days)
        continuous_dates = pd.bdate_range(start='2015-02-02', end='2015-02-11', freq='B')
        self.continuous_df = pd.DataFrame({
            'adjusted_close': np.random.randn(len(continuous_dates)) + 100,
            'volume': np.random.randint(1000, 2000, len(continuous_dates))
        }, index=continuous_dates)
    
    def test_create_custom_calendar_with_missing_days(self):
        """Test creating custom calendar with missing business days."""
        custom_freq, holidays = self.calendar.create_custom_calendar(self.sample_df)
        
        # Check that custom frequency is returned
        self.assertIsInstance(custom_freq, CustomBusinessDay)
        
        # Check that holidays list is returned
        self.assertIsInstance(holidays, list)
        
        # Should have 2 missing business days (2015-02-04 and 2015-02-10)
        self.assertEqual(len(holidays), 2)
        
        # Check holiday names contain missing dates
        holiday_names = [h.name for h in holidays]
        self.assertTrue(any('2015_02_04' in name for name in holiday_names))
        self.assertTrue(any('2015_02_10' in name for name in holiday_names))
    
    def test_create_custom_calendar_continuous_data(self):
        """Test creating custom calendar with continuous data (no missing days)."""
        custom_freq, holidays = self.calendar.create_custom_calendar(self.continuous_df)
        
        # Check that custom frequency is returned
        self.assertIsInstance(custom_freq, CustomBusinessDay)
        
        # Should have no holidays for continuous data
        self.assertEqual(len(holidays), 0)
    
    def test_validate_dataframe_empty(self):
        """Test validation with empty DataFrame."""
        empty_df = pd.DataFrame()
        
        with self.assertRaises(InsufficientDataError) as context:
            self.calendar.create_custom_calendar(empty_df)
        
        self.assertIn("DataFrame is empty", str(context.exception))
    
    def test_validate_dataframe_non_datetime_index(self):
        """Test validation with non-datetime index."""
        invalid_df = pd.DataFrame({
            'adjusted_close': [100.0, 101.0],
            'volume': [1000, 1100]
        }, index=[0, 1])  # Integer index instead of datetime
        
        with self.assertRaises(InsufficientDataError) as context:
            self.calendar.create_custom_calendar(invalid_df)
        
        self.assertIn("datetime index", str(context.exception))
    
    def test_validate_dataframe_insufficient_rows(self):
        """Test validation with insufficient rows."""
        single_row_df = pd.DataFrame({
            'adjusted_close': [100.0],
            'volume': [1000]
        }, index=pd.to_datetime(['2015-02-02']))
        
        with self.assertRaises(InsufficientDataError) as context:
            self.calendar.create_custom_calendar(single_row_df)
        
        self.assertIn("at least 2 rows", str(context.exception))
    
    def test_validate_dataframe_unsorted_index(self):
        """Test validation with unsorted datetime index."""
        unsorted_dates = ['2015-02-03', '2015-02-02', '2015-02-04']  # Out of order
        unsorted_df = pd.DataFrame({
            'adjusted_close': [101.0, 100.0, 102.0],
            'volume': [1100, 1000, 1200]
        }, index=pd.to_datetime(unsorted_dates))
        
        with self.assertRaises(InsufficientDataError) as context:
            self.calendar.create_custom_calendar(unsorted_df)
        
        self.assertIn("sorted in ascending order", str(context.exception))
    
    def test_create_complete_business_day_range(self):
        """Test creating complete business day range."""
        min_date = pd.Timestamp('2015-02-02')
        max_date = pd.Timestamp('2015-02-11')
        
        business_days = self.calendar._create_complete_business_day_range(min_date, max_date)
        
        # Check that result is DatetimeIndex
        self.assertIsInstance(business_days, pd.DatetimeIndex)
        
        # Check that weekends are excluded
        for date in business_days:
            self.assertNotIn(date.weekday(), [5, 6])  # Saturday=5, Sunday=6
        
        # Should have 8 business days from 2015-02-02 to 2015-02-11
        self.assertEqual(len(business_days), 8)
    
    def test_identify_missing_business_days(self):
        """Test identifying missing business days."""
        complete_range = pd.bdate_range(start='2015-02-02', end='2015-02-11', freq='B')
        actual_dates = self.sample_df.index
        
        missing_dates = self.calendar._identify_missing_business_days(actual_dates, complete_range)
        
        # Should identify 2 missing dates
        self.assertEqual(len(missing_dates), 2)
        
        # Check specific missing dates
        missing_dates_list = sorted(list(missing_dates))
        self.assertEqual(missing_dates_list[0].strftime('%Y-%m-%d'), '2015-02-04')
        self.assertEqual(missing_dates_list[1].strftime('%Y-%m-%d'), '2015-02-10')
    
    def test_create_holiday_list(self):
        """Test creating holiday list from missing dates."""
        missing_dates = {
            pd.Timestamp('2015-02-04'),
            pd.Timestamp('2015-02-10')
        }
        
        holidays = self.calendar._create_holiday_list(missing_dates)
        
        # Check that holidays are created
        self.assertEqual(len(holidays), 2)
        
        # Check that all items are Holiday objects
        for holiday in holidays:
            self.assertIsInstance(holiday, Holiday)
        
        # Check holiday names
        holiday_names = [h.name for h in holidays]
        self.assertTrue(any('2015_02_04' in name for name in holiday_names))
        self.assertTrue(any('2015_02_10' in name for name in holiday_names))
    
    def test_create_custom_business_day(self):
        """Test creating CustomBusinessDay frequency."""
        # Create sample holidays
        holidays = [
            Holiday('Test_Holiday_1', month=2, day=4, year=2015),
            Holiday('Test_Holiday_2', month=2, day=10, year=2015)
        ]
        
        custom_freq = self.calendar._create_custom_business_day(holidays)
        
        # Check that CustomBusinessDay is returned
        self.assertIsInstance(custom_freq, CustomBusinessDay)
        
        # Test that the frequency works by generating a date range
        date_range = pd.date_range(start='2015-02-02', end='2015-02-11', freq=custom_freq)
        
        # Should exclude the holiday dates
        date_strings = [d.strftime('%Y-%m-%d') for d in date_range]
        self.assertNotIn('2015-02-04', date_strings)
        self.assertNotIn('2015-02-10', date_strings)
    
    def test_get_calendar_info(self):
        """Test getting calendar information."""
        info = self.calendar.get_calendar_info(self.sample_df)
        
        # Check that all expected keys are present
        expected_keys = [
            'min_date', 'max_date', 'total_actual_days', 'total_business_days',
            'missing_business_days', 'missing_dates', 'coverage_percentage'
        ]
        for key in expected_keys:
            self.assertIn(key, info)
        
        # Check specific values
        self.assertEqual(info['total_actual_days'], 6)
        self.assertEqual(info['total_business_days'], 8)
        self.assertEqual(info['missing_business_days'], 2)
        self.assertEqual(info['coverage_percentage'], 75.0)  # 6/8 * 100
        
        # Check missing dates
        missing_dates_str = [d.strftime('%Y-%m-%d') for d in info['missing_dates']]
        self.assertIn('2015-02-04', missing_dates_str)
        self.assertIn('2015-02-10', missing_dates_str)
    
    def test_calendar_validation_success(self):
        """Test successful calendar validation."""
        # This should not raise any exceptions
        custom_freq, holidays = self.calendar.create_custom_calendar(self.sample_df)
        
        # If we get here, validation passed
        self.assertIsInstance(custom_freq, CustomBusinessDay)
        self.assertIsInstance(holidays, list)
    
    def test_real_data_file(self):
        """Test with real covariaterawdata1.csv file if available."""
        real_file_path = os.path.join(os.path.dirname(__file__), '..', 'Data', 'covariaterawdata1.csv')
        
        if os.path.exists(real_file_path):
            # Load real data using a simple approach
            df = pd.read_csv(real_file_path)
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date').sort_index()
            
            # Test calendar creation with real data
            custom_freq, holidays = self.calendar.create_custom_calendar(df)
            
            # Check that results are valid
            self.assertIsInstance(custom_freq, CustomBusinessDay)
            self.assertIsInstance(holidays, list)
            
            # Get calendar info
            info = self.calendar.get_calendar_info(df)
            
            # Check that coverage is reasonable (should be high for real stock data)
            self.assertGreater(info['coverage_percentage'], 50.0)
            
            print(f"Real data calendar info: {info}")
        else:
            self.skipTest("Real data file not found")
    
    def test_edge_case_single_missing_day(self):
        """Test with single missing business day."""
        dates = ['2015-02-02', '2015-02-03', '2015-02-05']  # Missing 2015-02-04
        single_missing_df = pd.DataFrame({
            'adjusted_close': [100.0, 101.0, 102.0],
            'volume': [1000, 1100, 1200]
        }, index=pd.to_datetime(dates))
        
        custom_freq, holidays = self.calendar.create_custom_calendar(single_missing_df)
        
        # Should have exactly 1 holiday
        self.assertEqual(len(holidays), 1)
        self.assertIn('2015_02_04', holidays[0].name)
    
    def test_edge_case_weekend_boundaries(self):
        """Test with data that spans weekends."""
        # Friday to Monday (weekend in between)
        dates = ['2015-02-06', '2015-02-09']  # Friday to Monday
        weekend_df = pd.DataFrame({
            'adjusted_close': [100.0, 101.0],
            'volume': [1000, 1100]
        }, index=pd.to_datetime(dates))
        
        custom_freq, holidays = self.calendar.create_custom_calendar(weekend_df)
        
        # Should have no holidays (weekends are naturally excluded from business days)
        self.assertEqual(len(holidays), 0)


if __name__ == '__main__':
    unittest.main()