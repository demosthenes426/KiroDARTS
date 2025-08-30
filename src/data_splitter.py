"""
DataSplitter class for temporal splitting of DARTS TimeSeries objects.

This module handles the temporal splitting of TimeSeries data into train/validation/test sets
while maintaining temporal ordering and ensuring no data leakage.
"""

import pandas as pd
import numpy as np
from darts import TimeSeries
from typing import Tuple
import warnings


class InsufficientDataError(Exception):
    """Custom exception for insufficient data for splitting."""
    pass


class SplitValidationError(Exception):
    """Custom exception for split validation errors."""
    pass


class DataSplitter:
    """
    Splits DARTS TimeSeries objects into train/validation/test sets with temporal ordering.
    
    This class implements temporal splitting with 70/15/15 ratios while ensuring
    no data leakage and maintaining temporal consistency.
    """
    
    # Split ratios as class constants
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    
    def __init__(self):
        """Initialize DataSplitter."""
        pass
    
    def split_data(self, ts: TimeSeries) -> Tuple[TimeSeries, TimeSeries, TimeSeries]:
        """
        Split TimeSeries into train/validation/test sets with 70/15/15 ratios.
        
        Args:
            ts (TimeSeries): Input TimeSeries to split
            
        Returns:
            Tuple[TimeSeries, TimeSeries, TimeSeries]: Train, validation, and test TimeSeries
            
        Raises:
            InsufficientDataError: If TimeSeries is too small for splitting
            SplitValidationError: If split validation fails
            ValueError: If input validation fails
        """
        # Validate input TimeSeries
        self._validate_input_timeseries(ts)
        
        # Calculate split indices
        train_end_idx, val_end_idx = self._calculate_split_indices(ts)
        
        # Perform the splits
        train_ts, val_ts, test_ts = self._perform_splits(ts, train_end_idx, val_end_idx)
        
        # Validate splits
        self._validate_splits(train_ts, val_ts, test_ts, ts)
        
        return train_ts, val_ts, test_ts
    
    def _validate_input_timeseries(self, ts: TimeSeries) -> None:
        """
        Validate input TimeSeries for splitting.
        
        Args:
            ts (TimeSeries): TimeSeries to validate
            
        Raises:
            ValueError: If TimeSeries validation fails
            InsufficientDataError: If TimeSeries is too small
        """
        if ts is None:
            raise ValueError("TimeSeries cannot be None")
        
        if len(ts) == 0:
            raise ValueError("TimeSeries cannot be empty")
        
        # Check minimum size for meaningful splits
        # Need at least 10 points to have reasonable train/val/test splits
        min_size = 10
        if len(ts) < min_size:
            raise InsufficientDataError(
                f"TimeSeries too small for splitting. "
                f"Minimum size: {min_size}, actual size: {len(ts)}"
            )
        
        # Check that index is strictly increasing (temporal ordering)
        time_index = ts.time_index
        if not time_index.is_monotonic_increasing:
            raise ValueError("TimeSeries index must be strictly increasing for temporal splitting")
        
        # Check for duplicates in time index
        if time_index.duplicated().any():
            raise ValueError("TimeSeries index contains duplicate timestamps")
    
    def _calculate_split_indices(self, ts: TimeSeries) -> Tuple[int, int]:
        """
        Calculate split indices based on ratios.
        
        Args:
            ts (TimeSeries): Input TimeSeries
            
        Returns:
            Tuple[int, int]: Train end index and validation end index
        """
        total_length = len(ts)
        
        # Calculate split points
        train_end_idx = int(total_length * self.TRAIN_RATIO)
        val_end_idx = int(total_length * (self.TRAIN_RATIO + self.VAL_RATIO))
        
        # Ensure we have at least 1 point in each split
        train_end_idx = max(1, train_end_idx)
        val_end_idx = max(train_end_idx + 1, val_end_idx)
        val_end_idx = min(total_length - 1, val_end_idx)  # Leave at least 1 for test
        
        return train_end_idx, val_end_idx
    
    def _perform_splits(self, ts: TimeSeries, train_end_idx: int, val_end_idx: int) -> Tuple[TimeSeries, TimeSeries, TimeSeries]:
        """
        Perform the actual splitting of TimeSeries.
        
        Args:
            ts (TimeSeries): Input TimeSeries
            train_end_idx (int): End index for training set
            val_end_idx (int): End index for validation set
            
        Returns:
            Tuple[TimeSeries, TimeSeries, TimeSeries]: Split TimeSeries
        """
        # Split using DARTS slicing
        # Training set: from start to train_end_idx
        train_ts = ts[:train_end_idx]
        
        # Validation set: from train_end_idx to val_end_idx
        val_ts = ts[train_end_idx:val_end_idx]
        
        # Test set: from val_end_idx to end
        test_ts = ts[val_end_idx:]
        
        return train_ts, val_ts, test_ts
    
    def _validate_splits(self, train_ts: TimeSeries, val_ts: TimeSeries, 
                        test_ts: TimeSeries, original_ts: TimeSeries) -> None:
        """
        Validate that splits are correct and don't overlap.
        
        Args:
            train_ts (TimeSeries): Training TimeSeries
            val_ts (TimeSeries): Validation TimeSeries
            test_ts (TimeSeries): Test TimeSeries
            original_ts (TimeSeries): Original TimeSeries
            
        Raises:
            SplitValidationError: If validation fails
        """
        # Check that all splits are non-empty
        if len(train_ts) == 0:
            raise SplitValidationError("Training set is empty")
        if len(val_ts) == 0:
            raise SplitValidationError("Validation set is empty")
        if len(test_ts) == 0:
            raise SplitValidationError("Test set is empty")
        
        # Check that splits cover the full dataset
        total_split_length = len(train_ts) + len(val_ts) + len(test_ts)
        if total_split_length != len(original_ts):
            raise SplitValidationError(
                f"Split lengths don't sum to original length. "
                f"Original: {len(original_ts)}, Sum of splits: {total_split_length}"
            )
        
        # Check temporal ordering (no overlap)
        train_end = train_ts.end_time()
        val_start = val_ts.start_time()
        val_end = val_ts.end_time()
        test_start = test_ts.start_time()
        
        if train_end >= val_start:
            raise SplitValidationError(
                f"Training and validation sets overlap. "
                f"Train end: {train_end}, Val start: {val_start}"
            )
        
        if val_end >= test_start:
            raise SplitValidationError(
                f"Validation and test sets overlap. "
                f"Val end: {val_end}, Test start: {test_start}"
            )
        
        # Check that temporal ordering is maintained within each split
        for split_name, split_ts in [("train", train_ts), ("val", val_ts), ("test", test_ts)]:
            time_index = split_ts.time_index
            if not time_index.is_monotonic_increasing:
                raise SplitValidationError(f"{split_name} set temporal ordering is not maintained")
        
        # Validate split ratios are approximately correct
        original_length = len(original_ts)
        actual_train_ratio = len(train_ts) / original_length
        actual_val_ratio = len(val_ts) / original_length
        actual_test_ratio = len(test_ts) / original_length
        
        # Allow for small deviations due to rounding
        tolerance = 0.05  # 5% tolerance
        
        if abs(actual_train_ratio - self.TRAIN_RATIO) > tolerance:
            warnings.warn(
                f"Training ratio deviation: expected {self.TRAIN_RATIO:.2f}, "
                f"actual {actual_train_ratio:.2f}"
            )
        
        if abs(actual_val_ratio - self.VAL_RATIO) > tolerance:
            warnings.warn(
                f"Validation ratio deviation: expected {self.VAL_RATIO:.2f}, "
                f"actual {actual_val_ratio:.2f}"
            )
        
        if abs(actual_test_ratio - self.TEST_RATIO) > tolerance:
            warnings.warn(
                f"Test ratio deviation: expected {self.TEST_RATIO:.2f}, "
                f"actual {actual_test_ratio:.2f}"
            )
    
    def get_split_info(self, train_ts: TimeSeries, val_ts: TimeSeries, 
                      test_ts: TimeSeries) -> dict:
        """
        Get information about the splits.
        
        Args:
            train_ts (TimeSeries): Training TimeSeries
            val_ts (TimeSeries): Validation TimeSeries
            test_ts (TimeSeries): Test TimeSeries
            
        Returns:
            dict: Split information
        """
        total_length = len(train_ts) + len(val_ts) + len(test_ts)
        
        return {
            'total_length': total_length,
            'train_length': len(train_ts),
            'val_length': len(val_ts),
            'test_length': len(test_ts),
            'train_ratio': len(train_ts) / total_length,
            'val_ratio': len(val_ts) / total_length,
            'test_ratio': len(test_ts) / total_length,
            'train_period': {
                'start': train_ts.start_time(),
                'end': train_ts.end_time()
            },
            'val_period': {
                'start': val_ts.start_time(),
                'end': val_ts.end_time()
            },
            'test_period': {
                'start': test_ts.start_time(),
                'end': test_ts.end_time()
            }
        }
    
    def validate_temporal_consistency(self, train_ts: TimeSeries, val_ts: TimeSeries, 
                                    test_ts: TimeSeries) -> bool:
        """
        Validate temporal consistency across splits.
        
        Args:
            train_ts (TimeSeries): Training TimeSeries
            val_ts (TimeSeries): Validation TimeSeries
            test_ts (TimeSeries): Test TimeSeries
            
        Returns:
            bool: True if temporal consistency is maintained
        """
        try:
            # Check that train comes before val comes before test
            train_end = train_ts.end_time()
            val_start = val_ts.start_time()
            val_end = val_ts.end_time()
            test_start = test_ts.start_time()
            
            if train_end >= val_start:
                return False
            
            if val_end >= test_start:
                return False
            
            # Check that each split is internally ordered
            for split_ts in [train_ts, val_ts, test_ts]:
                if not split_ts.time_index.is_monotonic_increasing:
                    return False
            
            return True
            
        except Exception:
            return False