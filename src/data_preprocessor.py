"""
DataPreprocessor class for data cleaning and preprocessing.
"""

import pandas as pd
from typing import List, Optional
import numpy as np


class DataTypeError(Exception):
    """Custom exception for data type errors."""
    pass


class DataPreprocessor:
    """
    DataPreprocessor class for cleaning and preprocessing stock data.
    
    Handles removal of non-numeric columns, date sorting, and data integrity validation.
    """
    
    def __init__(self):
        """Initialize DataPreprocessor."""
        pass
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess data by cleaning and validating.
        
        Args:
            df (pd.DataFrame): Input DataFrame with datetime index
            
        Returns:
            pd.DataFrame: Cleaned and preprocessed DataFrame
            
        Raises:
            DataTypeError: If data integrity issues are found
        """
        # Make a copy to avoid modifying original
        processed_df = df.copy()
        
        # Remove non-numeric columns (like symbol)
        processed_df = self._remove_non_numeric_columns(processed_df)
        
        # Sort by date (should already be sorted from DataLoader)
        processed_df = self._sort_by_date(processed_df)
        
        # Validate data integrity
        self._validate_data_integrity(processed_df)
        
        return processed_df
    
    def _remove_non_numeric_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove non-numeric columns from DataFrame.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with only numeric columns
        """
        # Identify non-numeric columns
        non_numeric_columns = []
        
        for col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                non_numeric_columns.append(col)
        
        # Remove non-numeric columns
        if non_numeric_columns:
            df = df.drop(columns=non_numeric_columns)
            print(f"Removed non-numeric columns: {non_numeric_columns}")
        
        return df
    
    def _sort_by_date(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Sort DataFrame by date index in ascending order.
        
        Args:
            df (pd.DataFrame): Input DataFrame with datetime index
            
        Returns:
            pd.DataFrame: Sorted DataFrame
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            raise DataTypeError("DataFrame index must be DatetimeIndex")
        
        # Sort by index (date)
        df_sorted = df.sort_index()
        
        return df_sorted
    
    def _validate_data_integrity(self, df: pd.DataFrame) -> None:
        """
        Validate data integrity and consistency.
        
        Args:
            df (pd.DataFrame): DataFrame to validate
            
        Raises:
            DataTypeError: If data integrity issues are found
        """
        # Check if DataFrame is empty
        if df.empty:
            raise DataTypeError("DataFrame is empty after preprocessing")
        
        # Check if index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            raise DataTypeError("DataFrame index must be DatetimeIndex")
        
        # Check if index is sorted ascending
        if not df.index.is_monotonic_increasing:
            raise DataTypeError("DataFrame index must be sorted in ascending order")
        
        # Check for duplicate dates
        if df.index.duplicated().any():
            duplicate_dates = df.index[df.index.duplicated()].tolist()
            raise DataTypeError(f"Duplicate dates found: {duplicate_dates}")
        
        # Check that all columns are numeric
        non_numeric_cols = []
        for col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                non_numeric_cols.append(col)
        
        if non_numeric_cols:
            raise DataTypeError(f"Non-numeric columns found after preprocessing: {non_numeric_cols}")
        
        # Check for required column (adjusted_close)
        if 'adjusted_close' not in df.columns:
            raise DataTypeError("Required column 'adjusted_close' not found")
        
        # Check for NaN values in adjusted_close
        if df['adjusted_close'].isna().any():
            nan_count = df['adjusted_close'].isna().sum()
            raise DataTypeError(f"NaN values found in adjusted_close column: {nan_count} values")
        
        # Check for negative or zero values in adjusted_close
        if (df['adjusted_close'] <= 0).any():
            invalid_count = (df['adjusted_close'] <= 0).sum()
            raise DataTypeError(f"Invalid values (<=0) found in adjusted_close: {invalid_count} values")
    
    def get_numeric_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Get list of numeric columns in DataFrame.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            List[str]: List of numeric column names
        """
        numeric_columns = []
        
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                numeric_columns.append(col)
        
        return numeric_columns
    
    def get_non_numeric_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Get list of non-numeric columns in DataFrame.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            List[str]: List of non-numeric column names
        """
        non_numeric_columns = []
        
        for col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                non_numeric_columns.append(col)
        
        return non_numeric_columns