"""
Demo script to test integration of data loading, preprocessing, custom calendar, and TimeSeries creation.
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_loader import DataLoader
from data_preprocessor import DataPreprocessor
from custom_holiday_calendar import CustomHolidayCalendar
from darts_timeseries_creator import DartsTimeSeriesCreator

def main():
    """Test integration of all data components."""
    print("=== DARTS Stock Forecasting - Data Components Integration Test ===\n")
    
    # Initialize components
    loader = DataLoader()
    preprocessor = DataPreprocessor()
    calendar_creator = CustomHolidayCalendar()
    ts_creator = DartsTimeSeriesCreator()
    
    # Test data file path
    data_file = Path("Data/covariaterawdata1.csv")
    
    if not data_file.exists():
        print(f"❌ Data file not found: {data_file}")
        return
    
    try:
        # Step 1: Load data
        print("1. Loading data...")
        df = loader.load_data(data_file)
        print(f"   ✓ Loaded {len(df)} rows with columns: {list(df.columns)}")
        print(f"   ✓ Date range: {df.index.min()} to {df.index.max()}")
        print(f"   ✓ Data sorted: {df.index.is_monotonic_increasing}")
        print()
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    try:
        # Step 2: Preprocess data
        print("2. Preprocessing data...")
        processed_df = preprocessor.preprocess_data(df)
        print(f"   ✓ Processed {len(processed_df)} rows")
        print(f"   ✓ Remaining columns: {list(processed_df.columns)}")
        print(f"   ✓ All columns numeric: {all(processed_df[col].dtype.kind in 'biufc' for col in processed_df.columns)}")
        print(f"   ✓ Data sorted: {processed_df.index.is_monotonic_increasing}")
        print()
    except Exception as e:
        print(f"❌ Error preprocessing data: {e}")
        return
    
    try:
        # Step 3: Create custom calendar
        print("3. Creating custom holiday calendar...")
        custom_freq, holidays = calendar_creator.create_custom_calendar(processed_df)
        calendar_info = calendar_creator.get_calendar_info(processed_df)
        
        print(f"   ✓ Custom frequency created: {custom_freq}")
        print(f"   ✓ Found {len(holidays)} missing business days")
        print(f"   ✓ Coverage: {calendar_info['coverage_percentage']:.1f}%")
        print(f"   ✓ Missing dates: {len(calendar_info['missing_dates'])} days")
        print()
    except Exception as e:
        print(f"❌ Error creating custom calendar: {e}")
        return
    
    try:
        # Step 4: Create DARTS TimeSeries without custom frequency
        print("4. Creating DARTS TimeSeries (standard)...")
        timeseries_std = ts_creator.create_timeseries(processed_df)
        ts_info_std = ts_creator.get_timeseries_info(timeseries_std)
        
        print(f"   ✓ TimeSeries created: {len(timeseries_std)} points")
        print(f"   ✓ Columns: {ts_info_std['columns']}")
        print(f"   ✓ Frequency: {ts_info_std['frequency']}")
        print(f"   ✓ No NaN values: {not ts_info_std['has_nan']}")
        print()
    except Exception as e:
        print(f"❌ Error creating standard TimeSeries: {e}")
        return
    
    try:
        # Step 5: Create DARTS TimeSeries with custom frequency
        print("5. Creating DARTS TimeSeries (with custom frequency)...")
        timeseries_custom = ts_creator.create_timeseries(processed_df, custom_freq)
        ts_info_custom = ts_creator.get_timeseries_info(timeseries_custom)
        
        print(f"   ✓ TimeSeries created: {len(timeseries_custom)} points")
        print(f"   ✓ Columns: {ts_info_custom['columns']}")
        print(f"   ✓ Frequency: {ts_info_custom['frequency']}")
        print(f"   ✓ No NaN values: {not ts_info_custom['has_nan']}")
        print()
    except Exception as e:
        print(f"❌ Error creating custom TimeSeries: {e}")
        return
    
    # Step 6: Summary
    print("6. Integration Summary:")
    print(f"   ✓ Data pipeline completed successfully")
    print(f"   ✓ Original data: {len(df)} rows")
    print(f"   ✓ Processed data: {len(processed_df)} rows")
    print(f"   ✓ Standard TimeSeries: {len(timeseries_std)} points")
    print(f"   ✓ Custom TimeSeries: {len(timeseries_custom)} points")
    print(f"   ✓ Missing business days handled: {len(holidays)}")
    print()
    print("🎉 All components integrated successfully!")

if __name__ == "__main__":
    main()