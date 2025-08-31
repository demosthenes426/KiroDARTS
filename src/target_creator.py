"""
TargetCreator module for creating 5-day future price prediction targets.

This module implements the TargetCreator class that generates target values
for 5-day future adjusted_close predictions while ensuring no data leakage.
"""

import pandas as pd
import numpy as np
from darts import TimeSeries
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class TargetCreator:
    """
    Creates 5-day future price prediction targets from time series data.
    
    This class ensures no data leakage by only using data available at
    prediction time to create future targets.
    """
    
    def __init__(self, prediction_horizon: int = 5):
        """
        Initialize TargetCreator.
        
        Args:
            prediction_horizon (int): Number of days to predict into the future.
                                    Defaults to 5 business days.
        """
        self.prediction_horizon = prediction_horizon
        logger.info(f"TargetCreator initialized with {prediction_horizon}-day prediction horizon")
    
    def create_targets(self, ts: TimeSeries, target_column: str = "adjusted_close") -> TimeSeries:
        """
        Create target values for future price prediction.
        
        This method creates a new TimeSeries where each value represents the
        adjusted_close price that will occur prediction_horizon days in the future.
        The method ensures no data leakage by only using data available at each
        prediction time point.
        
        Args:
            ts (TimeSeries): Input time series with historical data
            target_column (str): Name of the column to use for target creation.
                               Defaults to "adjusted_close".
        
        Returns:
            TimeSeries: Time series with target values, shortened by prediction_horizon
                       to avoid future data leakage
        
        Raises:
            ValueError: If target_column is not found in the time series
            ValueError: If time series is too short for the prediction horizon
        """
        logger.info(f"Creating targets with {self.prediction_horizon}-day horizon")
        
        # Validate inputs
        if target_column not in ts.columns:
            available_columns = list(ts.columns)
            raise ValueError(f"Target column '{target_column}' not found. Available columns: {available_columns}")
        
        if len(ts) <= self.prediction_horizon:
            raise ValueError(f"Time series length ({len(ts)}) must be greater than prediction horizon ({self.prediction_horizon})")
        
        # Convert to pandas DataFrame for easier manipulation
        try:
            df = ts.pd_dataframe()
        except AttributeError:
            try:
                df = ts.to_dataframe()
            except AttributeError:
                df = ts.pd_series().to_frame()
        
        # Create target column by shifting the target_column forward by prediction_horizon
        # This means each row will contain the value that occurs prediction_horizon days later
        target_values = df[target_column].shift(-self.prediction_horizon)
        
        # Remove the last prediction_horizon rows since they would contain NaN values
        # (we don't have future data for these points)
        valid_length = len(df) - self.prediction_horizon
        target_values = target_values.iloc[:valid_length]
        
        # Create a new DataFrame with just the target values
        target_df = pd.DataFrame({
            f"{target_column}_target_{self.prediction_horizon}d": target_values
        }, index=df.index[:valid_length])
        
        # Convert back to DARTS TimeSeries
        target_ts = TimeSeries.from_dataframe(target_df)
        
        logger.info(f"Created targets: original length {len(ts)}, target length {len(target_ts)}")
        logger.info(f"Target column: {target_ts.columns[0]}")
        
        return target_ts
    
    def validate_no_data_leakage(self, original_ts: TimeSeries, target_ts: TimeSeries, 
                                target_column: str = "adjusted_close") -> bool:
        """
        Validate that target creation doesn't introduce data leakage.
        
        This method ensures that each target value corresponds to the correct future value
        and that no future information is used to create targets at any given time point.
        
        Args:
            original_ts (TimeSeries): Original time series data
            target_ts (TimeSeries): Created target time series
            target_column (str): Name of the target column used
            
        Returns:
            bool: True if no data leakage is detected
            
        Raises:
            ValueError: If data leakage is detected
        """
        logger.info("Validating target creation for data leakage...")
        
        try:
            # Convert to pandas for easier validation
            original_df = original_ts.pd_dataframe()
            target_df = target_ts.pd_dataframe()
            
            # Check that target series is shorter by prediction_horizon
            expected_target_length = len(original_ts) - self.prediction_horizon
            if len(target_ts) != expected_target_length:
                raise ValueError(
                    f"Target series length ({len(target_ts)}) should be "
                    f"{expected_target_length} (original length - prediction horizon)"
                )
            
            # Validate that each target value matches the correct future value
            target_col_name = target_df.columns[0]  # Get the actual target column name
            
            for i in range(len(target_ts)):
                # Get the target value at position i
                target_value = target_df.iloc[i, 0]
                
                # Get the corresponding future value from original data
                future_index = i + self.prediction_horizon
                if future_index < len(original_df):
                    expected_value = original_df[target_column].iloc[future_index]
                    
                    # Check if values match (with small tolerance for floating point)
                    if not np.isclose(target_value, expected_value, rtol=1e-10):
                        raise ValueError(
                            f"Data leakage detected at index {i}: "
                            f"target value {target_value} != expected future value {expected_value}"
                        )
            
            # Validate time alignment
            original_index = original_ts.time_index
            target_index = target_ts.time_index
            
            # Target index should match the first part of original index
            expected_target_index = original_index[:len(target_ts)]
            if not target_index.equals(expected_target_index):
                raise ValueError("Target time index doesn't match expected alignment with original data")
            
            logger.info("✓ No data leakage detected in target creation")
            return True
            
        except Exception as e:
            logger.error(f"Data leakage validation failed: {e}")
            raise ValueError(f"Data leakage validation failed: {e}")
    
    def get_aligned_features_and_targets(self, features_ts: TimeSeries, 
                                       target_column: str = "adjusted_close") -> Tuple[TimeSeries, TimeSeries]:
        """
        Create aligned features and targets for model training.
        
        This method creates properly aligned feature and target TimeSeries where:
        - Features contain all available data at each time point
        - Targets contain the corresponding future values
        - Both series have the same time index (shortened to avoid data leakage)
        
        Args:
            features_ts (TimeSeries): Input features time series
            target_column (str): Name of the column to use for target creation
            
        Returns:
            Tuple[TimeSeries, TimeSeries]: Aligned features and targets
        """
        logger.info("Creating aligned features and targets...")
        
        # Create targets
        targets_ts = self.create_targets(features_ts, target_column)
        
        # Align features to match target length (remove last prediction_horizon points)
        aligned_features_ts = features_ts[:-self.prediction_horizon]
        
        # Validate alignment
        if len(aligned_features_ts) != len(targets_ts):
            raise ValueError(
                f"Feature and target lengths don't match: "
                f"features={len(aligned_features_ts)}, targets={len(targets_ts)}"
            )
        
        if not aligned_features_ts.time_index.equals(targets_ts.time_index):
            raise ValueError("Feature and target time indices don't match")
        
        logger.info(f"Created aligned features and targets with {len(aligned_features_ts)} points")
        return aligned_features_ts, targets_ts
        """
        Validate that target creation doesn't introduce data leakage.
        
        This method checks that each target value corresponds to a future value
        that would actually be available at the prediction time.
        
        Args:
            original_ts (TimeSeries): Original time series data
            target_ts (TimeSeries): Generated target time series
            target_column (str): Name of the target column
        
        Returns:
            bool: True if validation passes, False otherwise
        """
        logger.info("Validating no data leakage in target creation")
        
        try:
            original_df = original_ts.pd_dataframe()
        except AttributeError:
            try:
                original_df = original_ts.to_dataframe()
            except AttributeError:
                original_df = original_ts.pd_series().to_frame()
        
        try:
            target_df = target_ts.pd_dataframe()
        except AttributeError:
            try:
                target_df = target_ts.to_dataframe()
            except AttributeError:
                target_df = target_ts.pd_series().to_frame()
        
        # Check that target length is correct
        expected_length = len(original_df) - self.prediction_horizon
        if len(target_df) != expected_length:
            logger.error(f"Target length mismatch: expected {expected_length}, got {len(target_df)}")
            return False
        
        # Check that each target value matches the corresponding future value
        target_col_name = target_df.columns[0]
        
        for i in range(len(target_df)):
            target_value = target_df.iloc[i][target_col_name]
            future_index = i + self.prediction_horizon
            
            if future_index < len(original_df):
                expected_value = original_df.iloc[future_index][target_column]
                
                if not np.isclose(target_value, expected_value, rtol=1e-10):
                    logger.error(f"Data leakage detected at index {i}: target={target_value}, expected={expected_value}")
                    return False
        
        logger.info("No data leakage detected - validation passed")
        return True
    
    def get_aligned_features_and_targets(self, features_ts: TimeSeries, 
                                       target_column: str = "adjusted_close") -> Tuple[TimeSeries, TimeSeries]:
        """
        Create aligned features and targets for model training.
        
        This method creates targets from the features and returns both
        aligned time series that can be used for training.
        
        Args:
            features_ts (TimeSeries): Time series with feature data
            target_column (str): Name of the column to use for target creation
        
        Returns:
            Tuple[TimeSeries, TimeSeries]: (aligned_features, targets)
                - aligned_features: Features truncated to match target length
                - targets: Target values for prediction
        """
        logger.info("Creating aligned features and targets")
        
        # Create targets
        targets = self.create_targets(features_ts, target_column)
        
        # Align features to match target length
        # Features should be truncated to the same length as targets
        aligned_features = features_ts[:len(targets)]
        
        logger.info(f"Aligned features length: {len(aligned_features)}, targets length: {len(targets)}")
        
        # Validate alignment
        if len(aligned_features) != len(targets):
            raise ValueError(f"Feature and target lengths don't match: {len(aligned_features)} vs {len(targets)}")
        
        return aligned_features, targets