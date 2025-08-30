"""
Synthetic data generators for comprehensive testing of edge cases and scenarios.

This module provides utilities for generating various types of synthetic financial
data to test the robustness and reliability of the forecasting system.

Requirements: 5.3, 5.4
"""

import unittest
import pandas as pd
import numpy as np
import sys
import os
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
import tempfile

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Try to import DARTS and other dependencies
try:
    from darts import TimeSeries
    from darts_timeseries_creator import DartsTimeSeriesCreator
    from data_splitter import DataSplitter
    from data_scaler import DataScaler
    from data_integrity_validator import DataIntegrityValidator
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    DEPENDENCIES_AVAILABLE = False
    print(f"Warning: Dependencies not available: {e}")


class AdvancedDataGenerator:
    """Advanced synthetic data generator for comprehensive testing scenarios."""
    
    def __init__(self, seed: int = 42):
        """Initialize the data generator with a random seed."""
        self.seed = seed
        np.random.seed(seed)
    
    def generate_market_regime_data(
        self,
        start_date: str = "2020-01-01",
        end_date: str = "2023-12-31",
        regimes: List[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Generate stock data with different market regimes (bull, bear, sideways).
        
        Args:
            start_date: Start date for data generation
            end_date: End date for data generation
            regimes: List of regime definitions with parameters
        
        Returns:
            DataFrame with synthetic stock data exhibiting different market regimes
        """
        # Reset random seed to ensure reproducibility
        np.random.seed(self.seed)
        
        if regimes is None:
            regimes = [
                {'type': 'bull', 'duration_days': 200, 'trend': 0.001, 'volatility': 0.015},
                {'type': 'bear', 'duration_days': 150, 'trend': -0.0008, 'volatility': 0.025},
                {'type': 'sideways', 'duration_days': 100, 'trend': 0.0001, 'volatility': 0.012},
                {'type': 'bull', 'duration_days': 180, 'trend': 0.0012, 'volatility': 0.018}
            ]
        
        # Create business day range
        full_range = pd.date_range(start=start_date, end=end_date, freq='B')
        
        # Generate regime-based data
        all_prices = []
        all_volumes = []
        all_regimes = []
        current_price = 100.0
        
        date_idx = 0
        for regime in regimes:
            regime_days = min(regime['duration_days'], len(full_range) - date_idx)
            if regime_days <= 0:
                break
            
            # Generate returns for this regime
            returns = np.random.normal(
                regime['trend'], 
                regime['volatility'], 
                regime_days
            )
            
            # Apply regime-specific patterns
            if regime['type'] == 'bull':
                # Bull market: occasional strong positive days
                strong_days = np.random.choice(regime_days, regime_days // 20, replace=False)
                returns[strong_days] += np.random.uniform(0.02, 0.05, len(strong_days))
            
            elif regime['type'] == 'bear':
                # Bear market: occasional crash days
                crash_days = np.random.choice(regime_days, regime_days // 30, replace=False)
                returns[crash_days] -= np.random.uniform(0.03, 0.08, len(crash_days))
            
            elif regime['type'] == 'sideways':
                # Sideways market: mean reversion
                for i in range(1, len(returns)):
                    if abs(returns[i-1]) > 0.02:  # If previous day was extreme
                        returns[i] *= -0.3  # Partial mean reversion
            
            # Calculate prices
            regime_prices = []
            for ret in returns:
                current_price *= (1 + ret)
                regime_prices.append(current_price)
            
            # Generate volumes (higher in volatile periods)
            base_volume = 50000
            volume_multiplier = 1 + regime['volatility'] * 10
            regime_volumes = np.random.lognormal(
                np.log(base_volume * volume_multiplier), 
                0.3, 
                regime_days
            ).astype(int)
            
            all_prices.extend(regime_prices)
            all_volumes.extend(regime_volumes)
            all_regimes.extend([regime['type']] * regime_days)
            
            date_idx += regime_days
        
        # Trim to actual date range
        n_days = min(len(all_prices), len(full_range))
        dates = full_range[:n_days]
        prices = all_prices[:n_days]
        volumes = all_volumes[:n_days]
        regimes_list = all_regimes[:n_days]
        
        # Generate technical indicators
        price_series = pd.Series(prices)
        
        # Calculate indicators with proper NaN handling
        sma_20 = price_series.rolling(20, min_periods=1).mean()
        sma_50 = price_series.rolling(50, min_periods=1).mean()
        ema_12 = price_series.ewm(span=12).mean()
        
        # Bollinger bands
        bb_mean = price_series.rolling(20, min_periods=1).mean()
        bb_std = price_series.rolling(20, min_periods=1).std().fillna(0)
        
        data = {
            'adjusted_close': prices,
            'volume': volumes,
            'regime': regimes_list,
            'sma_20': sma_20.bfill().fillna(100),
            'sma_50': sma_50.bfill().fillna(100),
            'ema_12': ema_12.bfill().fillna(100),
            'rsi': self._calculate_rsi(price_series),
            'macd': self._calculate_macd(price_series),
            'bollinger_upper': (bb_mean + 2 * bb_std).bfill().fillna(100),
            'bollinger_lower': (bb_mean - 2 * bb_std).bfill().fillna(100),
            'atr': self._calculate_atr(price_series),
            'adx': np.random.uniform(10, 50, n_days)  # Simplified ADX
        }
        
        df = pd.DataFrame(data, index=dates)
        df.index.name = 'date'
        
        # Post-process to ensure no NaN values remain
        for col in df.select_dtypes(include=[np.number]).columns:
            if df[col].isnull().all():
                # If all values are NaN, fill with reasonable defaults
                if any(x in col.lower() for x in ['sma', 'ema', 'bollinger', 'price']):
                    df[col] = df['adjusted_close'].iloc[0] if len(df) > 0 else 100
                elif 'volume' in col.lower():
                    df[col] = 50000
                elif 'rsi' in col.lower():
                    df[col] = 50
                elif 'atr' in col.lower():
                    df[col] = 1.0
                else:
                    df[col] = 0
            elif df[col].isnull().any():
                # Fill remaining NaN values
                df[col] = df[col].ffill().bfill().fillna(0)
        
        return df
    
    def generate_crisis_data(
        self,
        start_date: str = "2020-01-01",
        periods: int = 500,
        crisis_start: int = 100,
        crisis_duration: int = 50
    ) -> pd.DataFrame:
        """
        Generate data with a financial crisis period.
        
        Args:
            start_date: Start date for data generation
            periods: Total number of business days
            crisis_start: Day when crisis begins
            crisis_duration: Duration of crisis in days
        
        Returns:
            DataFrame with crisis period data
        """
        # Reset random seed to ensure reproducibility
        np.random.seed(self.seed)
        
        dates = pd.date_range(start=start_date, periods=periods, freq='B')
        
        # Pre-crisis: normal market
        pre_crisis_days = crisis_start
        pre_crisis_returns = np.random.normal(0.0005, 0.015, pre_crisis_days)
        
        # Crisis: high volatility, negative trend
        crisis_returns = np.random.normal(-0.003, 0.04, crisis_duration)
        # Add some extreme negative days
        extreme_days = np.random.choice(crisis_duration, crisis_duration // 5, replace=False)
        crisis_returns[extreme_days] -= np.random.uniform(0.05, 0.15, len(extreme_days))
        
        # Post-crisis: recovery with high volatility
        post_crisis_days = periods - crisis_start - crisis_duration
        post_crisis_returns = np.random.normal(0.002, 0.025, post_crisis_days)
        
        # Combine all periods
        all_returns = np.concatenate([pre_crisis_returns, crisis_returns, post_crisis_returns])
        
        # Calculate prices
        prices = 100 * np.exp(np.cumsum(all_returns))
        
        # Generate volumes (higher during crisis)
        volumes = []
        for i in range(periods):
            if crisis_start <= i < crisis_start + crisis_duration:
                # Crisis period: higher volume
                base_vol = 200000
            else:
                # Normal period
                base_vol = 80000
            
            vol = np.random.lognormal(np.log(base_vol), 0.4)
            volumes.append(int(vol))
        
        # Create crisis indicator
        crisis_indicator = np.zeros(periods)
        crisis_indicator[crisis_start:crisis_start + crisis_duration] = 1
        
        # Generate technical indicators
        price_series = pd.Series(prices)
        
        # Calculate indicators with proper NaN handling
        sma_20 = price_series.rolling(20, min_periods=1).mean()
        ema_12 = price_series.ewm(span=12).mean()
        volatility = price_series.rolling(20, min_periods=1).std()
        
        data = {
            'adjusted_close': prices,
            'volume': volumes,
            'crisis_period': crisis_indicator,
            'sma_20': sma_20.bfill().fillna(100),
            'ema_12': ema_12.bfill().fillna(100),
            'rsi': self._calculate_rsi(price_series),
            'volatility': volatility.bfill().fillna(0.02),  # Default volatility
            'returns': all_returns
        }
        
        df = pd.DataFrame(data, index=dates)
        df.index.name = 'date'
        
        # Post-process to ensure no NaN values remain
        for col in df.select_dtypes(include=[np.number]).columns:
            if df[col].isnull().all():
                # If all values are NaN, fill with reasonable defaults
                if any(x in col.lower() for x in ['sma', 'ema', 'bollinger', 'price']):
                    df[col] = df['adjusted_close'].iloc[0] if len(df) > 0 else 100
                elif 'volume' in col.lower():
                    df[col] = 50000
                elif 'rsi' in col.lower():
                    df[col] = 50
                elif 'atr' in col.lower():
                    df[col] = 1.0
                elif 'volatility' in col.lower():
                    df[col] = 0.02
                else:
                    df[col] = 0
            elif df[col].isnull().any():
                # Fill remaining NaN values
                df[col] = df[col].ffill().bfill().fillna(0)
        
        return df
    
    def generate_missing_data_scenarios(
        self,
        base_data: pd.DataFrame,
        missing_patterns: List[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate datasets with various missing data patterns.
        
        Args:
            base_data: Base DataFrame to create missing data from
            missing_patterns: List of missing data patterns to generate
        
        Returns:
            Dictionary of DataFrames with different missing data patterns
        """
        if missing_patterns is None:
            missing_patterns = [
                'random_missing',
                'consecutive_missing',
                'weekend_extensions',
                'holiday_clusters',
                'data_gaps'
            ]
        
        scenarios = {}
        
        for pattern in missing_patterns:
            df = base_data.copy()
            
            if pattern == 'random_missing':
                # Randomly remove 5% of days
                n_remove = int(len(df) * 0.05)
                remove_indices = np.random.choice(len(df), n_remove, replace=False)
                df = df.drop(df.index[remove_indices])
            
            elif pattern == 'consecutive_missing':
                # Remove consecutive periods (market closures)
                n_gaps = 3
                for _ in range(n_gaps):
                    start_idx = np.random.randint(0, len(df) - 10)
                    gap_length = np.random.randint(3, 8)
                    end_idx = min(start_idx + gap_length, len(df))
                    df = df.drop(df.index[start_idx:end_idx])
            
            elif pattern == 'weekend_extensions':
                # Extend some weekends (Friday or Monday missing)
                fridays = df[df.index.dayofweek == 4]  # Friday = 4
                mondays = df[df.index.dayofweek == 0]  # Monday = 0
                
                # Remove some Fridays and Mondays
                remove_fridays = np.random.choice(len(fridays), len(fridays) // 10, replace=False)
                remove_mondays = np.random.choice(len(mondays), len(mondays) // 10, replace=False)
                
                df = df.drop(fridays.index[remove_fridays])
                df = df.drop(mondays.index[remove_mondays])
            
            elif pattern == 'holiday_clusters':
                # Remove clusters around typical holiday periods
                holiday_periods = [
                    ('2020-12-20', '2021-01-05'),  # Christmas/New Year
                    ('2021-07-01', '2021-07-10'),  # July 4th period
                    ('2021-11-20', '2021-11-30'),  # Thanksgiving period
                ]
                
                for start_date, end_date in holiday_periods:
                    try:
                        mask = (df.index >= start_date) & (df.index <= end_date)
                        # Remove 70% of days in holiday periods
                        holiday_days = df[mask]
                        if len(holiday_days) > 0:
                            n_remove = int(len(holiday_days) * 0.7)
                            remove_indices = np.random.choice(len(holiday_days), n_remove, replace=False)
                            df = df.drop(holiday_days.index[remove_indices])
                    except:
                        continue
            
            elif pattern == 'data_gaps':
                # Create larger gaps in data (system outages)
                n_gaps = 2
                for _ in range(n_gaps):
                    start_idx = np.random.randint(0, len(df) - 20)
                    gap_length = np.random.randint(10, 20)
                    end_idx = min(start_idx + gap_length, len(df))
                    df = df.drop(df.index[start_idx:end_idx])
            
            scenarios[pattern] = df.sort_index()
        
        return scenarios
    
    def generate_extreme_value_data(
        self,
        start_date: str = "2020-01-01",
        periods: int = 200
    ) -> pd.DataFrame:
        """
        Generate data with extreme values to test robustness.
        
        Args:
            start_date: Start date for data generation
            periods: Number of business days
        
        Returns:
            DataFrame with extreme values
        """
        # Reset random seed to ensure reproducibility
        np.random.seed(self.seed)
        
        dates = pd.date_range(start=start_date, periods=periods, freq='B')
        
        # Generate base data
        base_returns = np.random.normal(0.001, 0.02, periods)
        base_prices = 100 * np.exp(np.cumsum(base_returns))
        
        # Add extreme events
        extreme_events = {
            'flash_crash': (50, -0.20),  # Day 50: 20% drop
            'bubble_burst': (100, -0.15),  # Day 100: 15% drop
            'short_squeeze': (150, 0.25),  # Day 150: 25% gain
        }
        
        prices = base_prices.copy()
        for day, return_change in extreme_events.values():
            if day < len(prices):
                prices[day:] *= (1 + return_change)
        
        # Generate extreme volumes
        volumes = np.random.lognormal(15, 0.5, periods).astype(int)
        for day, _ in extreme_events.values():
            if day < len(volumes):
                volumes[day] *= np.random.randint(5, 20)  # Extreme volume spike
        
        # Add some data quality issues
        data = {
            'adjusted_close': prices,
            'volume': volumes,
            'high': prices * np.random.uniform(1.001, 1.05, periods),
            'low': prices * np.random.uniform(0.95, 0.999, periods),
            'open': prices * np.random.uniform(0.98, 1.02, periods),
        }
        
        # Add some extreme technical indicators
        data['extreme_rsi'] = np.random.choice([0, 100], periods // 20).tolist() + \
                             np.random.uniform(20, 80, periods - periods // 20).tolist()
        np.random.shuffle(data['extreme_rsi'])
        
        data['extreme_macd'] = np.random.normal(0, 2, periods)
        # Add some extreme MACD values
        extreme_macd_days = np.random.choice(periods, periods // 30, replace=False)
        data['extreme_macd'][extreme_macd_days] = np.random.choice([-50, 50], len(extreme_macd_days))
        
        df = pd.DataFrame(data, index=dates)
        df.index.name = 'date'
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> np.ndarray:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=1).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50).values
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26) -> np.ndarray:
        """Calculate MACD indicator."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        return macd.fillna(0).values
    
    def _calculate_atr(self, prices: pd.Series, window: int = 14) -> np.ndarray:
        """Calculate ATR indicator (simplified)."""
        returns = prices.pct_change().abs()
        atr = returns.rolling(window=window, min_periods=1).mean()
        return (atr * prices).fillna(1.0).values


@unittest.skipUnless(DEPENDENCIES_AVAILABLE, "Required dependencies not available")
class TestSyntheticDataGenerators(unittest.TestCase):
    """Test synthetic data generators and validate their output."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.data_generator = AdvancedDataGenerator(seed=42)
        self.timeseries_creator = DartsTimeSeriesCreator()
        self.data_splitter = DataSplitter()
        self.data_scaler = DataScaler()
        self.data_validator = DataIntegrityValidator()
        warnings.filterwarnings('ignore', category=UserWarning)
    
    def test_market_regime_data_generation(self):
        """Test market regime data generation."""
        print("\n=== Market Regime Data Generation Test ===")
        
        # Generate market regime data
        regime_data = self.data_generator.generate_market_regime_data(
            start_date="2020-01-01",
            end_date="2022-12-31"
        )
        
        # Validate basic properties
        self.assertIsInstance(regime_data, pd.DataFrame)
        self.assertGreater(len(regime_data), 0)
        self.assertIn('adjusted_close', regime_data.columns)
        self.assertIn('regime', regime_data.columns)
        
        # Check regime diversity
        unique_regimes = regime_data['regime'].unique()
        self.assertGreater(len(unique_regimes), 1, "Should have multiple market regimes")
        
        # Validate data quality
        self.assertFalse(regime_data['adjusted_close'].isna().any(), "No NaN prices")
        self.assertTrue((regime_data['adjusted_close'] > 0).all(), "All prices should be positive")
        
        # Test TimeSeries creation
        try:
            timeseries = self.timeseries_creator.create_timeseries(
                regime_data.drop('regime', axis=1)  # Remove non-numeric column
            )
            self.assertIsInstance(timeseries, TimeSeries)
            print(f"   ✓ Generated {len(regime_data)} days with {len(unique_regimes)} regimes")
            print(f"   ✓ TimeSeries created successfully: {len(timeseries)} points")
        except Exception as e:
            self.fail(f"TimeSeries creation failed: {e}")
    
    def test_crisis_data_generation(self):
        """Test financial crisis data generation."""
        print("\n=== Crisis Data Generation Test ===")
        
        # Generate crisis data
        crisis_data = self.data_generator.generate_crisis_data(
            start_date="2020-01-01",
            periods=300,
            crisis_start=100,
            crisis_duration=50
        )
        
        # Validate basic properties
        self.assertIsInstance(crisis_data, pd.DataFrame)
        self.assertEqual(len(crisis_data), 300)
        self.assertIn('crisis_period', crisis_data.columns)
        
        # Check crisis period
        crisis_days = crisis_data[crisis_data['crisis_period'] == 1]
        self.assertEqual(len(crisis_days), 50, "Crisis period should be 50 days")
        
        # Validate higher volatility during crisis
        pre_crisis = crisis_data.iloc[:100]['returns'].std()
        crisis_period = crisis_data.iloc[100:150]['returns'].std()
        post_crisis = crisis_data.iloc[150:]['returns'].std()
        
        self.assertGreater(crisis_period, pre_crisis, "Crisis should have higher volatility")
        
        # Test pipeline compatibility
        try:
            numeric_data = crisis_data.select_dtypes(include=[np.number])
            timeseries = self.timeseries_creator.create_timeseries(numeric_data)
            train_ts, val_ts, test_ts = self.data_splitter.split_data(timeseries)
            
            self.assertGreater(len(train_ts), 0)
            self.assertGreater(len(val_ts), 0)
            self.assertGreater(len(test_ts), 0)
            
            print(f"   ✓ Generated crisis data: {len(crisis_data)} days")
            print(f"   ✓ Crisis volatility: {crisis_period:.4f} vs normal: {pre_crisis:.4f}")
            print(f"   ✓ Pipeline compatibility validated")
        except Exception as e:
            self.fail(f"Pipeline compatibility test failed: {e}")
    
    def test_missing_data_scenarios(self):
        """Test various missing data scenarios."""
        print("\n=== Missing Data Scenarios Test ===")
        
        # Generate base data
        base_data = self.data_generator.generate_market_regime_data(
            start_date="2020-01-01",
            end_date="2021-12-31"
        )
        
        # Remove non-numeric columns for testing
        numeric_base = base_data.select_dtypes(include=[np.number])
        
        # Generate missing data scenarios
        missing_scenarios = self.data_generator.generate_missing_data_scenarios(numeric_base)
        
        self.assertIsInstance(missing_scenarios, dict)
        self.assertGreater(len(missing_scenarios), 0)
        
        scenario_results = {}
        
        for scenario_name, scenario_data in missing_scenarios.items():
            print(f"   Testing {scenario_name}...")
            
            try:
                # Validate data reduction
                self.assertLess(len(scenario_data), len(numeric_base), 
                               f"{scenario_name} should have fewer rows")
                
                # Test TimeSeries creation
                timeseries = self.timeseries_creator.create_timeseries(scenario_data)
                self.assertIsInstance(timeseries, TimeSeries)
                
                # Test data splitting
                train_ts, val_ts, test_ts = self.data_splitter.split_data(timeseries)
                
                scenario_results[scenario_name] = {
                    'success': True,
                    'original_length': len(numeric_base),
                    'reduced_length': len(scenario_data),
                    'reduction_pct': (1 - len(scenario_data) / len(numeric_base)) * 100,
                    'timeseries_length': len(timeseries)
                }
                
                print(f"     ✓ {scenario_name}: {len(scenario_data)} days ({scenario_results[scenario_name]['reduction_pct']:.1f}% reduction)")
                
            except Exception as e:
                scenario_results[scenario_name] = {'success': False, 'error': str(e)}
                print(f"     ✗ {scenario_name} failed: {e}")
        
        # Validate that most scenarios work
        success_rate = sum(1 for r in scenario_results.values() if r.get('success', False)) / len(scenario_results)
        self.assertGreater(success_rate, 0.8, "At least 80% of missing data scenarios should work")
        
        print(f"   📊 Missing data scenarios: {success_rate:.1%} success rate")
    
    def test_extreme_value_data_generation(self):
        """Test extreme value data generation and system robustness."""
        print("\n=== Extreme Value Data Test ===")
        
        # Generate extreme value data
        extreme_data = self.data_generator.generate_extreme_value_data(
            start_date="2020-01-01",
            periods=200
        )
        
        # Validate basic properties
        self.assertIsInstance(extreme_data, pd.DataFrame)
        self.assertEqual(len(extreme_data), 200)
        
        # Check for extreme values
        price_changes = extreme_data['adjusted_close'].pct_change().abs()
        extreme_changes = price_changes[price_changes > 0.1]  # >10% changes
        self.assertGreater(len(extreme_changes), 0, "Should have extreme price changes")
        
        # Check volume spikes
        volume_ratios = extreme_data['volume'] / extreme_data['volume'].median()
        volume_spikes = volume_ratios[volume_ratios > 5]  # >5x median volume
        self.assertGreater(len(volume_spikes), 0, "Should have volume spikes")
        
        # Test system robustness with extreme data
        try:
            # Remove non-standard columns for TimeSeries creation
            standard_columns = ['adjusted_close', 'volume', 'high', 'low', 'open']
            test_data = extreme_data[standard_columns]
            
            timeseries = self.timeseries_creator.create_timeseries(test_data)
            train_ts, val_ts, test_ts = self.data_splitter.split_data(timeseries)
            train_scaled, val_scaled, test_scaled, scaler = self.data_scaler.scale_data(
                train_ts, val_ts, test_ts
            )
            
            # Validate scaling handled extreme values
            self.assertFalse(np.isnan(train_scaled.values()).any(), "Scaling should handle extreme values")
            self.assertFalse(np.isinf(train_scaled.values()).any(), "No infinite values after scaling")
            
            print(f"   ✓ Generated extreme data: {len(extreme_data)} days")
            print(f"   ✓ Extreme price changes: {len(extreme_changes)} events")
            print(f"   ✓ Volume spikes: {len(volume_spikes)} events")
            print(f"   ✓ System robustness validated")
            
        except Exception as e:
            self.fail(f"System failed to handle extreme values: {e}")
    
    def test_data_quality_validation(self):
        """Test data quality validation across all generated datasets."""
        print("\n=== Data Quality Validation Test ===")
        
        # Generate various datasets
        datasets = {
            'market_regimes': self.data_generator.generate_market_regime_data(),
            'crisis_data': self.data_generator.generate_crisis_data(),
            'extreme_values': self.data_generator.generate_extreme_value_data()
        }
        
        validation_results = {}
        
        for dataset_name, dataset in datasets.items():
            print(f"   Validating {dataset_name}...")
            
            try:
                # Select numeric columns only
                numeric_data = dataset.select_dtypes(include=[np.number])
                
                # Create TimeSeries
                timeseries = self.timeseries_creator.create_timeseries(numeric_data)
                
                # Run data integrity validation
                validation_report = self.data_validator.validate_data_integrity(
                    timeseries, None, numeric_data
                )
                
                validation_results[dataset_name] = {
                    'data_integrity_passed': validation_report.data_integrity_passed,
                    'timeseries_validation_passed': validation_report.timeseries_validation_passed,
                    'issues': validation_report.issues,
                    'warnings': validation_report.warnings
                }
                
                # Basic validations
                self.assertTrue(validation_report.data_integrity_passed, 
                               f"{dataset_name} should pass data integrity validation")
                
                status = "✓" if validation_report.data_integrity_passed else "✗"
                print(f"     {status} {dataset_name}: Data integrity validation")
                
                if validation_report.warnings:
                    print(f"       Warnings: {len(validation_report.warnings)}")
                
                if validation_report.issues:
                    print(f"       Issues: {len(validation_report.issues)}")
                
            except Exception as e:
                validation_results[dataset_name] = {'error': str(e)}
                print(f"     ✗ {dataset_name} validation failed: {e}")
        
        # Summary
        passed_validations = sum(1 for r in validation_results.values() 
                               if r.get('data_integrity_passed', False))
        total_validations = len(validation_results)
        
        print(f"   📊 Data quality validation: {passed_validations}/{total_validations} datasets passed")
        
        # Require at least 80% to pass
        self.assertGreaterEqual(passed_validations / total_validations, 0.8, 
                               "At least 80% of generated datasets should pass validation")
    
    def test_generator_reproducibility(self):
        """Test that generators produce reproducible results with same seed."""
        print("\n=== Generator Reproducibility Test ===")
        
        # Generate data with same parameters twice
        generator1 = AdvancedDataGenerator(seed=42)
        generator2 = AdvancedDataGenerator(seed=42)
        
        data1 = generator1.generate_market_regime_data(
            start_date="2020-01-01",
            end_date="2020-12-31"
        )
        
        data2 = generator2.generate_market_regime_data(
            start_date="2020-01-01", 
            end_date="2020-12-31"
        )
        
        # Validate reproducibility
        self.assertEqual(len(data1), len(data2), "Same length")
        
        # Check that numeric columns are identical
        numeric_cols = data1.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            np.testing.assert_array_almost_equal(
                data1[col].values, 
                data2[col].values,
                decimal=10,
                err_msg=f"Column {col} should be identical"
            )
        
        print(f"   ✓ Reproducibility validated: {len(data1)} rows, {len(numeric_cols)} numeric columns")
        
        # Test with different seeds produce different results
        generator3 = AdvancedDataGenerator(seed=123)
        data3 = generator3.generate_market_regime_data(
            start_date="2020-01-01",
            end_date="2020-12-31"
        )
        
        # Should be different
        price_diff = np.abs(data1['adjusted_close'].values - data3['adjusted_close'].values).mean()
        self.assertGreater(price_diff, 0.1, "Different seeds should produce different data")
        
        print(f"   ✓ Different seeds produce different results (avg price diff: {price_diff:.2f})")


if __name__ == '__main__':
    # Configure warnings
    warnings.filterwarnings('ignore', category=UserWarning)
    
    if not DEPENDENCIES_AVAILABLE:
        print("Warning: Required dependencies not available. Tests will be skipped.")
    
    # Run tests with verbose output
    unittest.main(verbosity=2)