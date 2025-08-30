"""
DataLoader class for CSV processing with proper datetime parsing and validation.
"""

import pandas as pd
from typing import Optional
import os


class DataValidationError(Exception):
    """Custom exception for data validation errors."""
    pass


class DateParsingError(Exception):
    """Custom exception for date parsing errors."""
    pass


class DataLoader:
    """
    DataLoader class for loading and validating CSV files with stock data.
    
    Handles proper datetime parsing and validates required columns.
    """
    
    REQUIRED_COLUMNS = ['date', 'adjusted_close']
    
    def __init__(self):
        """Initialize DataLoader."""
        pass
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load CSV data with proper datetime parsing and validation.
        
        Args:
            file_path (str): Path to the CSV file
            
        Returns:
            pd.DataFrame: Loaded and validated DataFrame with datetime index
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            DataValidationError: If required columns are missing
            DateParsingError: If date parsing fails
        """
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        try:
            # Load CSV with date parsing
            df = pd.read_csv(file_path)
            
            # Validate required columns exist
            self._validate_required_columns(df)
            
            # Parse and set datetime index
            df = self._parse_datetime_index(df)
            
            # Convert data types
            df = self._convert_data_types(df)
            
            return df
            
        except pd.errors.EmptyDataError:
            raise DataValidationError(f"Empty CSV file: {file_path}")
        except pd.errors.ParserError as e:
            raise DataValidationError(f"Error parsing CSV file: {e}")
        except DateParsingError:
            # Re-raise DateParsingError as is
            raise
        except Exception as e:
            raise DataValidationError(f"Unexpected error loading data: {e}")
    
    def _validate_required_columns(self, df: pd.DataFrame) -> None:
        """
        Validate that required columns are present in the DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame to validate
            
        Raises:
            DataValidationError: If required columns are missing
        """
        missing_columns = []
        
        for col in self.REQUIRED_COLUMNS:
            if col not in df.columns:
                missing_columns.append(col)
        
        if missing_columns:
            raise DataValidationError(
                f"Missing required columns: {missing_columns}. "
                f"Available columns: {list(df.columns)}"
            )
    
    def _parse_datetime_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Parse date column and set as datetime index.
        
        Args:
            df (pd.DataFrame): DataFrame with date column
            
        Returns:
            pd.DataFrame: DataFrame with datetime index
            
        Raises:
            DateParsingError: If date parsing fails
        """
        try:
            # Parse date column
            df['date'] = pd.to_datetime(df['date'])
            
            # Set as index
            df = df.set_index('date')
            
            # Sort by date ascending
            df = df.sort_index()
            
            return df
            
        except (ValueError, TypeError) as e:
            raise DateParsingError(f"Error parsing date column: {e}")
    
    def _convert_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert data types for numeric columns.
        
        Args:
            df (pd.DataFrame): DataFrame to convert
            
        Returns:
            pd.DataFrame: DataFrame with converted data types
        """
        # Convert adjusted_close to float
        try:
            df['adjusted_close'] = pd.to_numeric(df['adjusted_close'], errors='coerce')
        except Exception as e:
            raise DataValidationError(f"Error converting adjusted_close to numeric: {e}")
        
        # Convert other numeric columns
        numeric_columns = df.select_dtypes(include=['object']).columns
        for col in numeric_columns:
            if col != 'symbol':  # Skip symbol column
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except Exception:
                    # If conversion fails, leave as is
                    pass
        
        return df