"""
DataScaler class for scaling DARTS TimeSeries objects using StandardScaler.

This module handles the scaling of TimeSeries data using StandardScaler fitted only
on training data, with validation of scaling statistics.
"""

import pandas as pd
import numpy as np
from darts import TimeSeries
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, List
import warnings


class ScalingValidationError(Exception):
    """Custom exception for scaling validation errors."""
    pass


class ScalerFitError(Exception):
    """Custom exception for scaler fit errors."""
    pass


class DataScaler:
    """
    Scales DARTS TimeSeries objects using StandardScaler fitted only on training data.
    
    This class implements proper scaling where the StandardScaler is fitted only on
    training data and then applied to all splits to prevent data leakage.
    """
    
    def __init__(self):
        """Initialize DataScaler."""
        pass
    
    def scale_data(self, train_ts: TimeSeries, val_ts: TimeSeries, 
                   test_ts: TimeSeries) -> Tuple[TimeSeries, TimeSeries, TimeSeries, StandardScaler]:
        """
        Scale TimeSeries data using StandardScaler fitted only on training data.
        
        Args:
            train_ts (TimeSeries): Training TimeSeries
            val_ts (TimeSeries): Validation TimeSeries
            test_ts (TimeSeries): Test TimeSeries
            
        Returns:
            Tuple[TimeSeries, TimeSeries, TimeSeries, StandardScaler]: 
                Scaled train, validation, test TimeSeries and fitted scaler
                
        Raises:
            ValueError: If input validation fails
            ScalerFitError: If scaler fitting fails
            ScalingValidationError: If scaling validation fails
        """
        # Validate input TimeSeries
        self._validate_input_timeseries(train_ts, val_ts, test_ts)
        
        # Fit scaler on training data only
        scaler = self._fit_scaler_on_training_data(train_ts)
        
        # Transform all splits using the fitted scaler
        scaled_train_ts, scaled_val_ts, scaled_test_ts = self._transform_all_splits(
            train_ts, val_ts, test_ts, scaler
        )
        
        # Validate scaling statistics
        self._validate_scaling_statistics(scaled_train_ts, scaled_val_ts, scaled_test_ts)
        
        return scaled_train_ts, scaled_val_ts, scaled_test_ts, scaler
    
    def _validate_input_timeseries(self, train_ts: TimeSeries, val_ts: TimeSeries, 
                                  test_ts: TimeSeries) -> None:
        """
        Validate input TimeSeries for scaling.
        
        Args:
            train_ts (TimeSeries): Training TimeSeries
            val_ts (TimeSeries): Validation TimeSeries
            test_ts (TimeSeries): Test TimeSeries
            
        Raises:
            ValueError: If validation fails
        """
        # Check that all TimeSeries are not None
        if train_ts is None or val_ts is None or test_ts is None:
            raise ValueError("All TimeSeries must be non-None")
        
        # Check that all TimeSeries are not empty
        if len(train_ts) == 0 or len(val_ts) == 0 or len(test_ts) == 0:
            raise ValueError("All TimeSeries must be non-empty")
        
        # Check that all TimeSeries have the same number of features
        train_features = len(train_ts.columns)
        val_features = len(val_ts.columns)
        test_features = len(test_ts.columns)
        
        if not (train_features == val_features == test_features):
            raise ValueError(
                f"All TimeSeries must have the same number of features. "
                f"Train: {train_features}, Val: {val_features}, Test: {test_features}"
            )
        
        # Check that feature names match
        train_columns = list(train_ts.columns)
        val_columns = list(val_ts.columns)
        test_columns = list(test_ts.columns)
        
        if not (train_columns == val_columns == test_columns):
            raise ValueError("All TimeSeries must have the same feature names")
        
        # Check for NaN values
        for name, ts in [("train", train_ts), ("val", val_ts), ("test", test_ts)]:
            try:
                ts_df = ts.pd_dataframe()
            except AttributeError:
                try:
                    ts_df = ts.to_dataframe()
                except AttributeError:
                    ts_df = ts.pd_series().to_frame()
            
            if ts_df.isna().any().any():
                nan_cols = ts_df.columns[ts_df.isna().any()].tolist()
                raise ValueError(f"NaN values found in {name} TimeSeries columns: {nan_cols}")
    
    def _fit_scaler_on_training_data(self, train_ts: TimeSeries) -> StandardScaler:
        """
        Fit StandardScaler on training data only.
        
        Args:
            train_ts (TimeSeries): Training TimeSeries
            
        Returns:
            StandardScaler: Fitted scaler
            
        Raises:
            ScalerFitError: If scaler fitting fails
        """
        try:
            # Convert TimeSeries to DataFrame
            try:
                train_df = train_ts.pd_dataframe()
            except AttributeError:
                try:
                    train_df = train_ts.to_dataframe()
                except AttributeError:
                    train_df = train_ts.pd_series().to_frame()
            
            # Create and fit StandardScaler
            scaler = StandardScaler()
            scaler.fit(train_df.values)
            
            return scaler
            
        except Exception as e:
            raise ScalerFitError(f"Failed to fit scaler on training data: {e}")
    
    def _transform_all_splits(self, train_ts: TimeSeries, val_ts: TimeSeries, 
                             test_ts: TimeSeries, scaler: StandardScaler) -> Tuple[TimeSeries, TimeSeries, TimeSeries]:
        """
        Transform all splits using the fitted scaler.
        
        Args:
            train_ts (TimeSeries): Training TimeSeries
            val_ts (TimeSeries): Validation TimeSeries
            test_ts (TimeSeries): Test TimeSeries
            scaler (StandardScaler): Fitted scaler
            
        Returns:
            Tuple[TimeSeries, TimeSeries, TimeSeries]: Scaled TimeSeries
            
        Raises:
            ValueError: If transformation fails
        """
        try:
            # Transform each split
            scaled_train_ts = self._transform_single_timeseries(train_ts, scaler)
            scaled_val_ts = self._transform_single_timeseries(val_ts, scaler)
            scaled_test_ts = self._transform_single_timeseries(test_ts, scaler)
            
            return scaled_train_ts, scaled_val_ts, scaled_test_ts
            
        except Exception as e:
            raise ValueError(f"Failed to transform TimeSeries: {e}")
    
    def _transform_single_timeseries(self, ts: TimeSeries, scaler: StandardScaler) -> TimeSeries:
        """
        Transform a single TimeSeries using the fitted scaler.
        
        Args:
            ts (TimeSeries): TimeSeries to transform
            scaler (StandardScaler): Fitted scaler
            
        Returns:
            TimeSeries: Scaled TimeSeries
        """
        # Convert to DataFrame
        try:
            ts_df = ts.pd_dataframe()
        except AttributeError:
            try:
                ts_df = ts.to_dataframe()
            except AttributeError:
                ts_df = ts.pd_series().to_frame()
        
        # Transform values
        scaled_values = scaler.transform(ts_df.values)
        
        # Create new DataFrame with scaled values
        scaled_df = pd.DataFrame(
            scaled_values,
            index=ts_df.index,
            columns=ts_df.columns
        )
        
        # Convert back to TimeSeries
        try:
            # Try to preserve frequency if it exists
            scaled_ts = TimeSeries.from_dataframe(scaled_df, fill_missing_dates=False)
        except ValueError:
            # If frequency issues, try without frequency specification
            try:
                scaled_ts = TimeSeries.from_dataframe(scaled_df, fill_missing_dates=False, freq=None)
            except:
                # Last resort
                scaled_ts = TimeSeries.from_dataframe(scaled_df)
        
        return scaled_ts
    
    def _validate_scaling_statistics(self, scaled_train_ts: TimeSeries, 
                                   scaled_val_ts: TimeSeries, scaled_test_ts: TimeSeries) -> None:
        """
        Validate that scaled features have correct mean (≈0) and std (≈1).
        
        Args:
            scaled_train_ts (TimeSeries): Scaled training TimeSeries
            scaled_val_ts (TimeSeries): Scaled validation TimeSeries
            scaled_test_ts (TimeSeries): Scaled test TimeSeries
            
        Raises:
            ScalingValidationError: If validation fails
        """
        # Validate training data statistics (should be exactly mean=0, std=1)
        self._validate_training_statistics(scaled_train_ts)
        
        # Validate that all splits have reasonable statistics
        self._validate_all_splits_statistics(scaled_train_ts, scaled_val_ts, scaled_test_ts)
    
    def _validate_training_statistics(self, scaled_train_ts: TimeSeries) -> None:
        """
        Validate that training data has mean ≈ 0 and std ≈ 1.
        
        Args:
            scaled_train_ts (TimeSeries): Scaled training TimeSeries
            
        Raises:
            ScalingValidationError: If validation fails
        """
        try:
            train_df = scaled_train_ts.pd_dataframe()
        except AttributeError:
            try:
                train_df = scaled_train_ts.to_dataframe()
            except AttributeError:
                train_df = scaled_train_ts.pd_series().to_frame()
        
        # Calculate statistics
        means = train_df.mean()
        stds = train_df.std()
        
        # Tolerance for floating point comparison
        # Use more reasonable tolerance for numerical precision
        mean_tolerance = 1e-12
        std_tolerance = 0.05  # Allow 5% deviation for std due to sample vs population std and small sample effects
        
        # Check means are approximately 0
        for col in train_df.columns:
            if abs(means[col]) > mean_tolerance:
                raise ScalingValidationError(
                    f"Training data mean for column '{col}' is not approximately 0. "
                    f"Actual mean: {means[col]}"
                )
        
        # Check standard deviations are approximately 1
        # Note: For constant features, sklearn sets scale to 1.0, resulting in std=0 after scaling
        for col in train_df.columns:
            if stds[col] == 0.0:
                # This is expected for constant features - sklearn handles them by setting scale=1.0
                continue
            elif abs(stds[col] - 1.0) > std_tolerance:
                raise ScalingValidationError(
                    f"Training data std for column '{col}' is not approximately 1. "
                    f"Actual std: {stds[col]}"
                )
    
    def _validate_all_splits_statistics(self, scaled_train_ts: TimeSeries, 
                                      scaled_val_ts: TimeSeries, scaled_test_ts: TimeSeries) -> None:
        """
        Validate statistics across all splits for reasonableness.
        
        Args:
            scaled_train_ts (TimeSeries): Scaled training TimeSeries
            scaled_val_ts (TimeSeries): Scaled validation TimeSeries
            scaled_test_ts (TimeSeries): Scaled test TimeSeries
        """
        splits = [
            ("train", scaled_train_ts),
            ("val", scaled_val_ts),
            ("test", scaled_test_ts)
        ]
        
        for split_name, ts in splits:
            try:
                ts_df = ts.pd_dataframe()
            except AttributeError:
                try:
                    ts_df = ts.to_dataframe()
                except AttributeError:
                    ts_df = ts.pd_series().to_frame()
            
            # Check for infinite values
            if np.isinf(ts_df.values).any():
                raise ScalingValidationError(f"Infinite values found in {split_name} split after scaling")
            
            # Check for NaN values
            if ts_df.isna().any().any():
                raise ScalingValidationError(f"NaN values found in {split_name} split after scaling")
            
            # Check that values are in reasonable range (warn if outside [-10, 10])
            min_val = ts_df.min().min()
            max_val = ts_df.max().max()
            
            if min_val < -10 or max_val > 10:
                warnings.warn(
                    f"{split_name} split has values outside typical scaled range [-10, 10]. "
                    f"Range: [{min_val:.2f}, {max_val:.2f}]. This may indicate outliers."
                )
    
    def get_scaling_info(self, scaler: StandardScaler, 
                        scaled_train_ts: TimeSeries, scaled_val_ts: TimeSeries, 
                        scaled_test_ts: TimeSeries) -> Dict:
        """
        Get information about the scaling process and results.
        
        Args:
            scaler (StandardScaler): Fitted scaler
            scaled_train_ts (TimeSeries): Scaled training TimeSeries
            scaled_val_ts (TimeSeries): Scaled validation TimeSeries
            scaled_test_ts (TimeSeries): Scaled test TimeSeries
            
        Returns:
            Dict: Scaling information
        """
        # Get feature names
        try:
            feature_names = list(scaled_train_ts.columns)
        except:
            feature_names = [f"feature_{i}" for i in range(len(scaler.mean_))]
        
        # Calculate statistics for each split
        splits_stats = {}
        splits = [
            ("train", scaled_train_ts),
            ("val", scaled_val_ts),
            ("test", scaled_test_ts)
        ]
        
        for split_name, ts in splits:
            try:
                ts_df = ts.pd_dataframe()
            except AttributeError:
                try:
                    ts_df = ts.to_dataframe()
                except AttributeError:
                    ts_df = ts.pd_series().to_frame()
            
            splits_stats[split_name] = {
                'mean': ts_df.mean().to_dict(),
                'std': ts_df.std().to_dict(),
                'min': ts_df.min().to_dict(),
                'max': ts_df.max().to_dict(),
                'length': len(ts_df)
            }
        
        return {
            'scaler_mean': dict(zip(feature_names, scaler.mean_)),
            'scaler_scale': dict(zip(feature_names, scaler.scale_)),
            'feature_names': feature_names,
            'splits_statistics': splits_stats,
            'scaling_validation_passed': True  # If we get here, validation passed
        }
    
    def inverse_transform_timeseries(self, scaled_ts: TimeSeries, scaler: StandardScaler) -> TimeSeries:
        """
        Inverse transform a scaled TimeSeries back to original scale.
        
        Args:
            scaled_ts (TimeSeries): Scaled TimeSeries
            scaler (StandardScaler): Fitted scaler used for original scaling
            
        Returns:
            TimeSeries: TimeSeries in original scale
            
        Raises:
            ValueError: If inverse transformation fails
        """
        try:
            # Convert to DataFrame
            try:
                scaled_df = scaled_ts.pd_dataframe()
            except AttributeError:
                try:
                    scaled_df = scaled_ts.to_dataframe()
                except AttributeError:
                    scaled_df = scaled_ts.pd_series().to_frame()
            
            # Inverse transform values
            original_values = scaler.inverse_transform(scaled_df.values)
            
            # Create new DataFrame with original values
            original_df = pd.DataFrame(
                original_values,
                index=scaled_df.index,
                columns=scaled_df.columns
            )
            
            # Convert back to TimeSeries
            try:
                original_ts = TimeSeries.from_dataframe(original_df, fill_missing_dates=False)
            except ValueError:
                try:
                    original_ts = TimeSeries.from_dataframe(original_df, fill_missing_dates=False, freq=None)
                except:
                    original_ts = TimeSeries.from_dataframe(original_df)
            
            return original_ts
            
        except Exception as e:
            raise ValueError(f"Failed to inverse transform TimeSeries: {e}")
    
    def validate_scaling_consistency(self, original_train_ts: TimeSeries, 
                                   scaled_train_ts: TimeSeries, scaler: StandardScaler) -> bool:
        """
        Validate that scaling and inverse scaling are consistent.
        
        Args:
            original_train_ts (TimeSeries): Original training TimeSeries
            scaled_train_ts (TimeSeries): Scaled training TimeSeries
            scaler (StandardScaler): Fitted scaler
            
        Returns:
            bool: True if scaling is consistent
        """
        try:
            # Inverse transform the scaled data
            reconstructed_ts = self.inverse_transform_timeseries(scaled_train_ts, scaler)
            
            # Compare with original
            try:
                original_df = original_train_ts.pd_dataframe()
                reconstructed_df = reconstructed_ts.pd_dataframe()
            except AttributeError:
                try:
                    original_df = original_train_ts.to_dataframe()
                    reconstructed_df = reconstructed_ts.to_dataframe()
                except AttributeError:
                    original_df = original_train_ts.pd_series().to_frame()
                    reconstructed_df = reconstructed_ts.pd_series().to_frame()
            
            # Check if values are approximately equal
            return np.allclose(original_df.values, reconstructed_df.values, rtol=1e-10)
            
        except Exception:
            return False