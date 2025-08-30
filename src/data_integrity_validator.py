"""
DataIntegrityValidator class for comprehensive data validation.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from darts import TimeSeries
from sklearn.preprocessing import StandardScaler


@dataclass
class ValidationReport:
    """Data validation report containing results and issues."""
    data_integrity_passed: bool
    timeseries_validation_passed: bool
    scaling_validation_passed: bool
    issues: List[str]
    warnings: List[str]


class DataIntegrityValidator:
    """Validates data integrity throughout the pipeline."""
    
    def __init__(self):
        """Initialize the validator."""
        self.tolerance = 1e-6  # Tolerance for floating point comparisons
    
    def validate_data_integrity(self, ts: TimeSeries, scaler: Optional[StandardScaler] = None, 
                              original_df: Optional[pd.DataFrame] = None) -> ValidationReport:
        """
        Validate data integrity using comprehensive checklist.
        
        Args:
            ts: DARTS TimeSeries object to validate
            scaler: Optional StandardScaler to validate scaling statistics
            original_df: Optional original DataFrame for comparison
            
        Returns:
            ValidationReport with validation results
        """
        issues = []
        warnings = []
        
        # Validate TimeSeries properties
        timeseries_passed = self._validate_timeseries_properties(ts, original_df, issues, warnings)
        
        # Validate scaling statistics if scaler provided
        scaling_passed = True
        if scaler is not None:
            scaling_passed = self._validate_scaling_statistics(ts, scaler, issues, warnings)
        
        # Overall data integrity check
        data_integrity_passed = self._validate_general_data_integrity(ts, issues, warnings)
        
        # Overall validation result
        overall_passed = timeseries_passed and scaling_passed and data_integrity_passed
        
        return ValidationReport(
            data_integrity_passed=overall_passed,
            timeseries_validation_passed=timeseries_passed,
            scaling_validation_passed=scaling_passed,
            issues=issues,
            warnings=warnings
        )
    
    def _validate_timeseries_properties(self, ts: TimeSeries, original_df: Optional[pd.DataFrame], 
                                      issues: List[str], warnings: List[str]) -> bool:
        """
        Validate TimeSeries properties according to requirements.
        
        Requirements covered: 1.4, 6.2, 6.3
        """
        passed = True
        
        # Get values for NaN checking
        values = ts.values()
        
        # Check for NaN values (Requirement 1.4)
        try:
            if np.any(np.isnan(values)):
                issues.append("TimeSeries contains NaN values")
                passed = False
        except TypeError:
            # Handle non-numeric data types that can't be checked for NaN
            if not np.issubdtype(values.dtype, np.number):
                issues.append("TimeSeries contains non-numeric data")
                passed = False
        
        # Check if index is strictly increasing (Requirement 1.4)
        time_index = ts.time_index
        # Use is_monotonic_increasing and check for duplicates for strict increasing
        if not time_index.is_monotonic_increasing or time_index.has_duplicates:
            issues.append("TimeSeries index is not strictly increasing")
            passed = False
        
        # Check if datetime index is properly sorted (Requirement 6.2)
        if not isinstance(time_index, pd.DatetimeIndex):
            issues.append("TimeSeries index is not a DatetimeIndex")
            passed = False
        
        # Validate TimeSeries point count matches DataFrame rows (Requirement 6.3)
        if original_df is not None:
            if len(ts) != len(original_df):
                issues.append(f"TimeSeries length ({len(ts)}) does not match DataFrame length ({len(original_df)})")
                passed = False
        
        # Check for reasonable data range
        try:
            if np.any(np.isinf(values)):
                issues.append("TimeSeries contains infinite values")
                passed = False
        except TypeError:
            # Skip infinite check for non-numeric data
            pass
        
        # Check for empty TimeSeries
        if len(ts) == 0:
            issues.append("TimeSeries is empty")
            passed = False
        
        return passed
    
    def _validate_scaling_statistics(self, ts: TimeSeries, scaler: StandardScaler, 
                                   issues: List[str], warnings: List[str]) -> bool:
        """
        Validate scaling statistics according to requirements.
        
        Requirements covered: 6.6
        """
        passed = True
        
        # Get scaled values
        values = ts.values()
        
        # Check if scaler was fitted
        if not hasattr(scaler, 'mean_'):
            issues.append("StandardScaler has not been fitted")
            return False
        
        # Validate mean is approximately 0 for each feature (Requirement 6.6)
        means = np.mean(values, axis=0)
        for i, mean_val in enumerate(means):
            if abs(mean_val) > self.tolerance:
                warnings.append(f"Feature {i} mean ({mean_val:.6f}) is not close to 0")
        
        # Validate standard deviation is approximately 1 for each feature (Requirement 6.6)
        stds = np.std(values, axis=0, ddof=1)  # Use sample std
        for i, std_val in enumerate(stds):
            if abs(std_val - 1.0) > self.tolerance:
                warnings.append(f"Feature {i} std ({std_val:.6f}) is not close to 1")
        
        # Check scaler parameters are reasonable
        if hasattr(scaler, 'scale_'):
            if np.any(scaler.scale_ <= 0):
                issues.append("StandardScaler has non-positive scale values")
                passed = False
        
        return passed
    
    def _validate_general_data_integrity(self, ts: TimeSeries, issues: List[str], 
                                       warnings: List[str]) -> bool:
        """
        Validate general data integrity requirements.
        
        Requirements covered: 6.1, 6.5
        """
        passed = True
        
        # Check for data consistency (Requirement 6.1)
        values = ts.values()
        
        # Check for reasonable value ranges
        if values.size > 0 and np.issubdtype(values.dtype, np.number):
            # Check for extremely large or small values that might indicate data issues
            try:
                max_val = np.max(np.abs(values))
                if max_val > 1e10:
                    warnings.append(f"Very large values detected (max absolute: {max_val:.2e})")
            except (TypeError, ValueError):
                # Skip for non-numeric data
                pass
            
            # Check for constant values across all features
            if values.shape[1] > 1:  # Multi-variate
                for i in range(values.shape[1]):
                    try:
                        feature_values = values[:, i]
                        if np.all(feature_values == feature_values[0]):
                            warnings.append(f"Feature {i} has constant values")
                    except (TypeError, ValueError):
                        # Skip for non-comparable data
                        pass
        
        # Validate numeric data types (Requirement 6.5)
        if not np.issubdtype(values.dtype, np.number):
            issues.append("TimeSeries contains non-numeric data")
            passed = False
        
        # Check for minimum data requirements
        if len(ts) < 10:
            warnings.append(f"TimeSeries has very few data points ({len(ts)})")
        
        return passed
    
    def validate_holiday_calendar_match(self, missing_dates: List[pd.Timestamp], 
                                      custom_holidays: List[Any]) -> ValidationReport:
        """
        Validate that custom holidays exactly match missing business days.
        
        Requirements covered: 6.4
        """
        issues = []
        warnings = []
        
        # Extract dates from custom holidays
        holiday_dates = []
        for holiday in custom_holidays:
            if hasattr(holiday, 'start_date') and hasattr(holiday, 'end_date'):
                # Handle date range holidays
                current_date = holiday.start_date
                while current_date <= holiday.end_date:
                    holiday_dates.append(current_date)
                    current_date += pd.Timedelta(days=1)
            elif hasattr(holiday, 'date'):
                holiday_dates.append(holiday.date)
            else:
                # Try to convert to timestamp
                try:
                    holiday_dates.append(pd.Timestamp(holiday))
                except:
                    warnings.append(f"Could not parse holiday: {holiday}")
        
        # Convert to sets for comparison
        missing_set = set(pd.to_datetime(missing_dates).date)
        holiday_set = set(pd.to_datetime(holiday_dates).date)
        
        # Check if they match exactly (Requirement 6.4)
        if missing_set != holiday_set:
            extra_holidays = holiday_set - missing_set
            missing_holidays = missing_set - holiday_set
            
            if extra_holidays:
                issues.append(f"Extra holidays not in missing dates: {extra_holidays}")
            if missing_holidays:
                issues.append(f"Missing dates not in holidays: {missing_holidays}")
        
        passed = len(issues) == 0
        
        return ValidationReport(
            data_integrity_passed=passed,
            timeseries_validation_passed=passed,
            scaling_validation_passed=True,  # Not applicable for this validation
            issues=issues,
            warnings=warnings
        )
    
    def validate_feature_columns_numeric(self, df: pd.DataFrame, 
                                       exclude_columns: Optional[List[str]] = None) -> ValidationReport:
        """
        Validate that all feature columns are numeric before model training.
        
        Requirements covered: 6.5
        """
        issues = []
        warnings = []
        
        exclude_columns = exclude_columns or ['symbol', 'date']
        
        for column in df.columns:
            if column not in exclude_columns:
                if not pd.api.types.is_numeric_dtype(df[column]):
                    issues.append(f"Column '{column}' is not numeric: {df[column].dtype}")
        
        passed = len(issues) == 0
        
        return ValidationReport(
            data_integrity_passed=passed,
            timeseries_validation_passed=passed,
            scaling_validation_passed=True,  # Not applicable for this validation
            issues=issues,
            warnings=warnings
        )