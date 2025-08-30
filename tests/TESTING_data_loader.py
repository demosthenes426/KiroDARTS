"""
Unit tests for DataLoader class.
"""

import unittest
import pandas as pd
import tempfile
import os
from unittest.mock import patch
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_loader import DataLoader, DataValidationError, DateParsingError


class TestDataLoader(unittest.TestCase):
    """Test cases for DataLoader class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.loader = DataLoader()
        
        # Sample valid CSV data
        self.valid_csv_data = """date,adjusted_close,volume,symbol
2015-02-02,168.49,163106969,SPY
2015-02-03,170.92,124212881,SPY
2015-02-04,170.27,134306728,SPY"""
        
        # Invalid CSV data (missing required column)
        self.invalid_csv_data = """date,volume,symbol
2015-02-02,163106969,SPY
2015-02-03,124212881,SPY"""
        
        # Invalid date format
        self.invalid_date_csv = """date,adjusted_close,volume,symbol
invalid-date,168.49,163106969,SPY
2015-02-03,170.92,124212881,SPY"""
    
    def test_load_valid_data(self):
        """Test loading valid CSV data."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(self.valid_csv_data)
            temp_file = f.name
        
        try:
            df = self.loader.load_data(temp_file)
            
            # Check that data is loaded correctly
            self.assertIsInstance(df, pd.DataFrame)
            self.assertEqual(len(df), 3)
            
            # Check that date is index
            self.assertIsInstance(df.index, pd.DatetimeIndex)
            
            # Check required columns exist
            self.assertIn('adjusted_close', df.columns)
            
            # Check data types
            self.assertTrue(pd.api.types.is_numeric_dtype(df['adjusted_close']))
            
        finally:
            os.unlink(temp_file)
    
    def test_load_real_data_file(self):
        """Test loading the real covariaterawdata1.csv file."""
        real_file_path = os.path.join(os.path.dirname(__file__), '..', 'Data', 'covariaterawdata1.csv')
        
        if os.path.exists(real_file_path):
            df = self.loader.load_data(real_file_path)
            
            # Check that data is loaded correctly
            self.assertIsInstance(df, pd.DataFrame)
            self.assertGreater(len(df), 0)
            
            # Check that date is index
            self.assertIsInstance(df.index, pd.DatetimeIndex)
            
            # Check required columns exist
            self.assertIn('adjusted_close', df.columns)
            
            # Check data is sorted by date
            self.assertTrue(df.index.is_monotonic_increasing)
            
            # Check data types
            self.assertTrue(pd.api.types.is_numeric_dtype(df['adjusted_close']))
        else:
            self.skipTest("Real data file not found")
    
    def test_file_not_found(self):
        """Test handling of non-existent file."""
        with self.assertRaises(FileNotFoundError):
            self.loader.load_data('non_existent_file.csv')
    
    def test_missing_required_columns(self):
        """Test handling of missing required columns."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(self.invalid_csv_data)
            temp_file = f.name
        
        try:
            with self.assertRaises(DataValidationError) as context:
                self.loader.load_data(temp_file)
            
            self.assertIn('Missing required columns', str(context.exception))
            self.assertIn('adjusted_close', str(context.exception))
            
        finally:
            os.unlink(temp_file)
    
    def test_invalid_date_format(self):
        """Test handling of invalid date format."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(self.invalid_date_csv)
            temp_file = f.name
        
        try:
            with self.assertRaises(DateParsingError):
                self.loader.load_data(temp_file)
            
        finally:
            os.unlink(temp_file)
    
    def test_empty_file(self):
        """Test handling of empty CSV file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write('')
            temp_file = f.name
        
        try:
            with self.assertRaises(DataValidationError) as context:
                self.loader.load_data(temp_file)
            
            self.assertIn('Empty CSV file', str(context.exception))
            
        finally:
            os.unlink(temp_file)
    
    def test_validate_required_columns(self):
        """Test _validate_required_columns method."""
        # Valid DataFrame
        valid_df = pd.DataFrame({
            'date': ['2015-02-02'],
            'adjusted_close': [168.49],
            'volume': [163106969]
        })
        
        # Should not raise exception
        self.loader._validate_required_columns(valid_df)
        
        # Invalid DataFrame
        invalid_df = pd.DataFrame({
            'date': ['2015-02-02'],
            'volume': [163106969]
        })
        
        with self.assertRaises(DataValidationError):
            self.loader._validate_required_columns(invalid_df)
    
    def test_parse_datetime_index(self):
        """Test _parse_datetime_index method."""
        df = pd.DataFrame({
            'date': ['2015-02-03', '2015-02-02', '2015-02-04'],
            'adjusted_close': [170.92, 168.49, 170.27]
        })
        
        result_df = self.loader._parse_datetime_index(df)
        
        # Check that index is datetime
        self.assertIsInstance(result_df.index, pd.DatetimeIndex)
        
        # Check that data is sorted by date
        self.assertTrue(result_df.index.is_monotonic_increasing)
        
        # Check that date column is removed from columns
        self.assertNotIn('date', result_df.columns)
    
    def test_convert_data_types(self):
        """Test _convert_data_types method."""
        df = pd.DataFrame({
            'adjusted_close': ['168.49', '170.92'],
            'volume': ['163106969', '124212881'],
            'symbol': ['SPY', 'SPY']
        })
        
        result_df = self.loader._convert_data_types(df)
        
        # Check that adjusted_close is numeric
        self.assertTrue(pd.api.types.is_numeric_dtype(result_df['adjusted_close']))
        
        # Check that volume is numeric
        self.assertTrue(pd.api.types.is_numeric_dtype(result_df['volume']))
        
        # Check that symbol remains as string
        self.assertTrue(pd.api.types.is_object_dtype(result_df['symbol']))


if __name__ == '__main__':
    unittest.main()