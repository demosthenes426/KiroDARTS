"""
CustomHolidayCalendar class for handling missing business days in stock data.

This module creates custom holiday calendars that treat missing trading days
as holidays, enabling proper DARTS TimeSeries creation with business day frequency.
"""

import pandas as pd
from pandas.tseries.holiday import Holiday, AbstractHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
from typing import List, Tuple, Set
from datetime import datetime, date


class CalendarMismatchError(Exception):
    """Custom exception for holiday calendar mismatch errors."""
    pass


class InsufficientDataError(Exception):
    """Custom exception for insufficient data errors."""
    pass


class CustomHolidayCalendar:
    """
    Creates custom holiday calendars for stock data with missing trading days.
    
    This class identifies missing business days in a date range and creates
    a pandas CustomBusinessDay frequency that excludes those dates as holidays.
    """
    
    def __init__(self):
        """Initialize CustomHolidayCalendar."""
        pass
    
    def create_custom_calendar(self, df: pd.DataFrame) -> Tuple[CustomBusinessDay, List[Holiday]]:
        """
        Create custom business day frequency from missing dates in DataFrame.
        
        Identifies missing business days between min and max dates in the DataFrame
        and creates a CustomBusinessDay frequency that excludes those dates.
        
        Args:
            df (pd.DataFrame): DataFrame with datetime index
            
        Returns:
            Tuple[CustomBusinessDay, List[Holiday]]: Custom business day frequency and holiday list
            
        Raises:
            InsufficientDataError: If DataFrame has insufficient data
            CalendarMismatchError: If holiday validation fails
        """
        # Validate input DataFrame
        self._validate_dataframe(df)
        
        # Get date range from DataFrame
        min_date = df.index.min()
        max_date = df.index.max()
        
        # Create complete business day range
        complete_business_days = self._create_complete_business_day_range(min_date, max_date)
        
        # Identify missing business days
        missing_dates = self._identify_missing_business_days(df.index, complete_business_days)
        
        # Create holiday list from missing dates
        holidays = self._create_holiday_list(missing_dates)
        
        # Create custom business day frequency
        custom_freq = self._create_custom_business_day(holidays)
        
        # Validate that holidays match missing business days exactly
        self._validate_holiday_calendar(df.index, complete_business_days, missing_dates, custom_freq)
        
        return custom_freq, holidays
    
    def _validate_dataframe(self, df: pd.DataFrame) -> None:
        """
        Validate that DataFrame has proper datetime index and sufficient data.
        
        Args:
            df (pd.DataFrame): DataFrame to validate
            
        Raises:
            InsufficientDataError: If DataFrame validation fails
        """
        if df.empty:
            raise InsufficientDataError("DataFrame is empty")
        
        if not isinstance(df.index, pd.DatetimeIndex):
            raise InsufficientDataError("DataFrame must have datetime index")
        
        if len(df) < 2:
            raise InsufficientDataError("DataFrame must have at least 2 rows for date range analysis")
        
        # Check if index is sorted
        if not df.index.is_monotonic_increasing:
            raise InsufficientDataError("DataFrame index must be sorted in ascending order")
    
    def _create_complete_business_day_range(self, min_date: pd.Timestamp, max_date: pd.Timestamp) -> pd.DatetimeIndex:
        """
        Create complete business day range between min and max dates.
        
        Args:
            min_date (pd.Timestamp): Start date
            max_date (pd.Timestamp): End date
            
        Returns:
            pd.DatetimeIndex: Complete business day range
        """
        # Create business day range (excludes weekends by default)
        business_days = pd.bdate_range(start=min_date, end=max_date, freq='B')
        return business_days
    
    def _identify_missing_business_days(self, actual_dates: pd.DatetimeIndex, 
                                      complete_business_days: pd.DatetimeIndex) -> Set[pd.Timestamp]:
        """
        Identify missing business days by comparing actual dates with complete range.
        
        Args:
            actual_dates (pd.DatetimeIndex): Actual dates in DataFrame
            complete_business_days (pd.DatetimeIndex): Complete business day range
            
        Returns:
            Set[pd.Timestamp]: Set of missing business day dates
        """
        # Convert to sets for efficient set operations
        actual_dates_set = set(actual_dates)
        complete_dates_set = set(complete_business_days)
        
        # Find missing dates (in complete range but not in actual data)
        missing_dates = complete_dates_set - actual_dates_set
        
        return missing_dates
    
    def _create_holiday_list(self, missing_dates: Set[pd.Timestamp]) -> List[Holiday]:
        """
        Create pandas Holiday objects from missing dates.
        
        Args:
            missing_dates (Set[pd.Timestamp]): Set of missing dates
            
        Returns:
            List[Holiday]: List of Holiday objects for missing dates
        """
        holidays = []
        
        for missing_date in sorted(missing_dates):
            # Create Holiday object with observance rule for weekends
            holiday = Holiday(
                name=f"Missing_Trading_Day_{missing_date.strftime('%Y_%m_%d')}",
                month=missing_date.month,
                day=missing_date.day,
                year=missing_date.year,
                observance=lambda x: x  # No observance rule - use exact date
            )
            holidays.append(holiday)
        
        return holidays
    
    def _create_custom_business_day(self, holidays: List[Holiday]) -> CustomBusinessDay:
        """
        Create CustomBusinessDay frequency with specified holidays.
        
        Args:
            holidays (List[Holiday]): List of holidays to exclude
            
        Returns:
            CustomBusinessDay: Custom business day frequency
        """
        # Create custom holiday calendar
        class StockTradingCalendar(AbstractHolidayCalendar):
            rules = holidays
        
        # Create CustomBusinessDay with the holiday calendar
        custom_freq = CustomBusinessDay(calendar=StockTradingCalendar())
        
        return custom_freq
    
    def _validate_holiday_calendar(self, actual_dates: pd.DatetimeIndex, 
                                 complete_business_days: pd.DatetimeIndex,
                                 missing_dates: Set[pd.Timestamp],
                                 custom_freq: CustomBusinessDay) -> None:
        """
        Validate that the custom holiday calendar matches missing business days exactly.
        
        Args:
            actual_dates (pd.DatetimeIndex): Actual dates in DataFrame
            complete_business_days (pd.DatetimeIndex): Complete business day range
            missing_dates (Set[pd.Timestamp]): Missing dates identified
            custom_freq (CustomBusinessDay): Custom business day frequency
            
        Raises:
            CalendarMismatchError: If validation fails
        """
        # Test the custom frequency by generating dates
        min_date = actual_dates.min()
        max_date = actual_dates.max()
        
        try:
            # Generate dates using custom frequency
            generated_dates = pd.date_range(start=min_date, end=max_date, freq=custom_freq)
            
            # Convert to set for comparison
            generated_dates_set = set(generated_dates)
            actual_dates_set = set(actual_dates)
            
            # Check if generated dates match actual dates
            if generated_dates_set != actual_dates_set:
                extra_in_generated = generated_dates_set - actual_dates_set
                missing_in_generated = actual_dates_set - generated_dates_set
                
                error_msg = "Holiday calendar validation failed:\n"
                if extra_in_generated:
                    error_msg += f"Extra dates in generated: {sorted(extra_in_generated)}\n"
                if missing_in_generated:
                    error_msg += f"Missing dates in generated: {sorted(missing_in_generated)}\n"
                
                raise CalendarMismatchError(error_msg)
            
            # Validate that missing dates count matches
            expected_missing_count = len(complete_business_days) - len(actual_dates)
            actual_missing_count = len(missing_dates)
            
            if expected_missing_count != actual_missing_count:
                raise CalendarMismatchError(
                    f"Missing dates count mismatch: expected {expected_missing_count}, "
                    f"got {actual_missing_count}"
                )
                
        except Exception as e:
            if isinstance(e, CalendarMismatchError):
                raise
            else:
                raise CalendarMismatchError(f"Error validating holiday calendar: {e}")
    
    def get_calendar_info(self, df: pd.DataFrame) -> dict:
        """
        Get information about the calendar for the given DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame with datetime index
            
        Returns:
            dict: Calendar information including date ranges and missing days
        """
        min_date = df.index.min()
        max_date = df.index.max()
        
        complete_business_days = self._create_complete_business_day_range(min_date, max_date)
        missing_dates = self._identify_missing_business_days(df.index, complete_business_days)
        
        return {
            'min_date': min_date,
            'max_date': max_date,
            'total_actual_days': len(df),
            'total_business_days': len(complete_business_days),
            'missing_business_days': len(missing_dates),
            'missing_dates': sorted(missing_dates),
            'coverage_percentage': (len(df) / len(complete_business_days)) * 100
        }