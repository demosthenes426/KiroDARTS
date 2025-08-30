"""
Unit tests for DataPreprocessor class.
"""

import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_preprocessor import DataPreprocessor, DataTypeError


class TestDataPreprocessor(unittest.TestCase):
    """Test cases for DataPreprocessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.preprocessor = DataPreprocessor()
        
        # Create sample data with datetime index
        dates = pd.date_range('2015-02-02', periods=5, freq='D')
        self.valid_df = pd.DataFrame({
            'adjusted_close': [168.49, 170.92, 170.27, 171.99, 171.51],
            'volume': [163106969, 124212881, 134306728, 97953181, 125672026],
            'price': [168.5, 170.9, 170.3, 172.0, 171.5],
            'symbol': ['SPY', 'SPY', 'SPY', 'SPY', 'SPY']  # Non-numeric column
        }, index=dates)
        
        # Unsorted data
        unsorted_dates = [dates[2], dates[0], dates[4], dates[1], dates[3]]
        self.unsorted_df = pd.DataFrame({
            'adjusted_close': [170.27, 168.49, 171.51, 170.92, 171.99],
            'volume': [134306728, 163106969, 125672026, 124212881, 97953181]
        }, index=unsorted_dates)
        
        # Data with NaN values
        self.nan_df = pd.DataFrame({
            'adjusted_close': [168.49, np.nan, 170.27, 171.99, 171.51],
            'volume': [163106969, 124212881, 134306728, 97953181, 125672026]
        }, index=dates)
        
        # Data with invalid values
        self.invalid_df = pd.DataFrame({
            'adjusted_close': [168.49, -170.92, 0, 171.99, 171.51],
            'volume': [163106969, 124212881, 134306728, 97953181, 125672026]
        }, index=dates)
    
    def test_preprocess_valid_data(self):
        """Test preprocessing valid data."""
        result_df = self.preprocessor.preprocess_data(self.valid_df)
        
        # Check that non-numeric columns are removed
        self.assertNotIn('symbol', result_df.columns)
        
        # Check that numeric columns remain
        self.assertIn('adjusted_close', result_df.columns)
        self.assertIn('volume', result_df.columns)
        self.assertIn('price', result_df.columns)
        
        # Check that data is sorted
        self.assertTrue(result_df.index.is_monotonic_increasing)
        
        # Check that all remaining columns are numeric
        for col in result_df.columns:
            self.assertTrue(pd.api.types.is_numeric_dtype(result_df[col]))
    
    def test_preprocess_unsorted_data(self):
        """Test preprocessing unsorted data."""
        result_df = self.preprocessor.preprocess_data(self.unsorted_df)
        
        # Check that data is sorted after preprocessing
        self.assertTrue(result_df.index.is_monotonic_increasing)
        
        # Check that the first date is the earliest
        expected_first_date = self.unsorted_df.index.min()
        self.assertEqual(result_df.index[0], expected_first_date)
    
    def test_remove_non_numeric_columns(self):
        """Test _remove_non_numeric_columns method."""
        result_df = self.preprocessor._remove_non_numeric_columns(self.valid_df)
        
        # Check that symbol column is removed
        self.assertNotIn('symbol', result_df.columns)
        
        # Check that numeric columns remain
        numeric_cols = ['adjusted_close', 'volume', 'price']
        for col in numeric_cols:
            self.assertIn(col, result_df.columns)
    
    def test_sort_by_date(self):
        """Test _sort_by_date method."""
        result_df = self.preprocessor._sort_by_date(self.unsorted_df)
        
        # Check that data is sorted
        self.assertTrue(result_df.index.is_monotonic_increasing)
        
        # Check that values are in correct order
        expected_first_value = self.unsorted_df.loc[self.unsorted_df.index.min(), 'adjusted_close']
        self.assertEqual(result_df.iloc[0]['adjusted_close'], expected_first_value)
    
    def test_sort_by_date_invalid_index(self):
        """Test _sort_by_date with non-datetime index."""
        df_with_int_index = self.valid_df.reset_index()
        
        with self.assertRaises(DataTypeError) as context:
            self.preprocessor._sort_by_date(df_with_int_index)
        
        self.assertIn('DataFrame index must be DatetimeIndex', str(context.exception))
    
    def test_validate_data_integrity_valid(self):
        """Test _validate_data_integrity with valid data."""
        # Remove non-numeric columns first
        clean_df = self.preprocessor._remove_non_numeric_columns(self.valid_df)
        
        # Should not raise exception
        self.preprocessor._validate_data_integrity(clean_df)
    
    def test_validate_data_integrity_empty_df(self):
        """Test _validate_data_integrity with empty DataFrame."""
        empty_df = pd.DataFrame()
        
        with self.assertRaises(DataTypeError) as context:
            self.preprocessor._validate_data_integrity(empty_df)
        
        self.assertIn('DataFrame is empty', str(context.exception))
    
    def test_validate_data_integrity_nan_values(self):
        """Test _validate_data_integrity with NaN values."""
        with self.assertRaises(DataTypeError) as context:
            self.preprocessor._validate_data_integrity(self.nan_df)
        
        self.assertIn('NaN values found in adjusted_close', str(context.exception))
    
    def test_validate_data_integrity_invalid_values(self):
        """Test _validate_data_integrity with invalid values."""
        with self.assertRaises(DataTypeError) as context:
            self.preprocessor._validate_data_integrity(self.invalid_df)
        
        self.assertIn('Invalid values (<=0) found in adjusted_close', str(context.exception))
    
    def test_validate_data_integrity_duplicate_dates(self):
        """Test _validate_data_integrity with duplicate dates."""
        dates = pd.date_range('2015-02-02', periods=3, freq='D')
        duplicate_dates = [dates[0], dates[1], dates[1]]  # Duplicate second date
        
        duplicate_df = pd.DataFrame({
            'adjusted_close': [168.49, 170.92, 170.27],
            'volume': [163106969, 124212881, 134306728]
        }, index=duplicate_dates)
        
        with self.assertRaises(DataTypeError) as context:
            self.preprocessor._validate_data_integrity(duplicate_df)
        
        self.assertIn('Duplicate dates found', str(context.exception))
    
    def test_validate_data_integrity_missing_adjusted_close(self):
        """Test _validate_data_integrity without adjusted_close column."""
        df_no_adjusted_close = self.valid_df[['volume', 'price']].copy()
        
        with self.assertRaises(DataTypeError) as context:
            self.preprocessor._validate_data_integrity(df_no_adjusted_close)
        
        self.assertIn("Required column 'adjusted_close' not found", str(context.exception))
    
    def test_get_numeric_columns(self):
        """Test get_numeric_columns method."""
        numeric_cols = self.preprocessor.get_numeric_columns(self.valid_df)
        
        expected_numeric = ['adjusted_close', 'volume', 'price']
        self.assertEqual(set(numeric_cols), set(expected_numeric))
        self.assertNotIn('symbol', numeric_cols)
    
    def test_get_non_numeric_columns(self):
        """Test get_non_numeric_columns method."""
        non_numeric_cols = self.preprocessor.get_non_numeric_columns(self.valid_df)
        
        self.assertEqual(non_numeric_cols, ['symbol'])
        self.assertNotIn('adjusted_close', non_numeric_cols)
    
    def test_preprocess_with_real_data_structure(self):
        """Test preprocessing with structure similar to real data."""
        # Create data similar to covariaterawdata1.csv structure
        dates = pd.date_range('2015-02-02', periods=3, freq='D')
        real_like_df = pd.DataFrame({
            'adjusted_close': [168.49, 170.92, 170.27],
            'volume': [163106969, 124212881, 134306728],
            'Chaikin A/D': [25426872651, 25550005420, 25494702650],
            'ADX': [19.9664, 19.4378, 18.7114],
            'ATR': [2.564, 2.5555, 2.4844],
            'symbol': ['SPY', 'SPY', 'SPY']
        }, index=dates)
        
        result_df = self.preprocessor.preprocess_data(real_like_df)
        
        # Check that symbol is removed
        self.assertNotIn('symbol', result_df.columns)
        
        # Check that all technical indicators remain
        expected_cols = ['adjusted_close', 'volume', 'Chaikin A/D', 'ADX', 'ATR']
        for col in expected_cols:
            self.assertIn(col, result_df.columns)
        
        # Check data integrity
        self.assertTrue(result_df.index.is_monotonic_increasing)
        for col in result_df.columns:
            self.assertTrue(pd.api.types.is_numeric_dtype(result_df[col]))


if __name__ == '__main__':
    unittest.main()