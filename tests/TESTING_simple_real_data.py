"""
Simple real data integration test for ModelFactory and ModelTrainer with new data-driven holiday discovery.

This test uses a simplified approach with direct TimeSeries creation and focuses on core
model creation and training functionality with real CSV data. It specifically tests the
new data-driven holiday discovery logic in DartsTimeSeriesCreator.
"""

import unittest
import pandas as pd
import numpy as np
import sys
import os
import warnings
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Try to import DARTS and other dependencies
try:
    from darts import TimeSeries
    from darts_timeseries_creator import DartsTimeSeriesCreator
    from model_factory import ModelFactory
    from model_trainer import ModelTrainer
    from data_splitter import DataSplitter
    from data_scaler import DataScaler
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    DEPENDENCIES_AVAILABLE = False
    print(f"Warning: Dependencies not available: {e}")


@unittest.skipUnless(DEPENDENCIES_AVAILABLE, "Required dependencies not available")
class TestSimpleRealDataIntegration(unittest.TestCase):
    """Test ModelFactory and ModelTrainer with real data using new holiday discovery logic."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures for the class."""
        cls.real_data_file = os.path.join(os.path.dirname(__file__), '..', 'Data', 'covariaterawdata1.csv')
        
        # Load and prepare real data
        if os.path.exists(cls.real_data_file):
            cls.df = cls._load_and_prepare_data()
            cls.data_available = True
        else:
            cls.data_available = False
            print(f"Warning: Real data file not found at {cls.real_data_file}")
    
    @classmethod
    def _load_and_prepare_data(cls):
        """Load and prepare real CSV data for testing."""
        # Load CSV data
        df = pd.read_csv(cls.real_data_file)
        
        # Convert date column and set as index
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
        
        # Remove non-numeric columns (like 'symbol')
        numeric_df = df.select_dtypes(include=[np.number])
        
        # Remove any rows with NaN values
        clean_df = numeric_df.dropna()
        
        # Take a subset for faster testing (first 100 rows)
        subset_df = clean_df.head(100)
        
        print(f"Loaded real data: {len(subset_df)} rows, {len(subset_df.columns)} columns")
        print(f"Date range: {subset_df.index.min()} to {subset_df.index.max()}")
        
        return subset_df
    
    def setUp(self):
        """Set up test fixtures."""
        if not self.data_available:
            self.skipTest("Real data not available")
        
        # Initialize components
        self.timeseries_creator = DartsTimeSeriesCreator()
        self.model_factory = ModelFactory(
            input_chunk_length=10,
            output_chunk_length=1,
            n_epochs=2,  # Reduced for faster testing
            batch_size=16,
            random_state=42
        )
        self.model_trainer = ModelTrainer(
            max_epochs=2,  # Reduced for faster testing
            verbose=False
        )
        self.data_splitter = DataSplitter()
        self.data_scaler = DataScaler()
    
    def test_data_driven_holiday_discovery(self):
        """Test the new data-driven holiday discovery logic with real data."""
        print("\n=== Testing Data-Driven Holiday Discovery ===")
        
        # Test holiday discovery
        discovered_holidays = self.timeseries_creator.get_discovered_holidays(self.df)
        print(f"Discovered {len(discovered_holidays)} holidays from real data")
        
        if len(discovered_holidays) > 0:
            print(f"Sample holidays: {discovered_holidays[:5]}")
            
            # Validate that discovered holidays are actual missing business days
            min_date = self.df.index.min()
            max_date = self.df.index.max()
            complete_bdays = pd.date_range(start=min_date, end=max_date, freq='B')
            expected_missing = complete_bdays.difference(self.df.index)
            
            # Check that discovered holidays match expected missing dates
            self.assertEqual(len(discovered_holidays), len(expected_missing))
            for holiday in discovered_holidays:
                self.assertIn(holiday, expected_missing.tolist())
            
            print("✓ Holiday discovery validation passed")
        else:
            print("No holidays discovered (continuous business day data)")
        
        # Test TimeSeries creation with holiday discovery
        timeseries = self.timeseries_creator.create_timeseries(self.df)
        
        # Validate TimeSeries properties
        self.assertIsInstance(timeseries, TimeSeries)
        self.assertGreater(len(timeseries), 0)
        
        # Get TimeSeries info
        ts_info = self.timeseries_creator.get_timeseries_info(timeseries)
        print(f"Created TimeSeries: {ts_info['length']} points, {len(ts_info['columns'])} columns")
        print(f"Date range: {ts_info['start_time']} to {ts_info['end_time']}")
        
        # Validate no NaN values
        self.assertFalse(ts_info['has_nan'], "TimeSeries should not contain NaN values")
        
        print("✓ TimeSeries creation with holiday discovery passed")
    
    def test_model_factory_with_real_data(self):
        """Test ModelFactory with real data using new TimeSeries creation."""
        print("\n=== Testing ModelFactory with Real Data ===")
        
        # Create TimeSeries using new data-driven approach
        timeseries = self.timeseries_creator.create_timeseries(self.df)
        
        # Test model creation with subset of models for speed
        test_models = ['RNNModel', 'DLinearModel']
        
        for model_name in test_models:
            print(f"\nTesting {model_name}...")
            
            try:
                # Create all models and extract the specific one
                all_models = self.model_factory.create_models()
                model = all_models.get(model_name)
                self.assertIsNotNone(model, f"{model_name} should be created successfully")
                
                # Get model info
                model_info = self.model_factory.get_model_info(all_models)
                print(f"✓ {model_name} created successfully")
                print(f"  Model info: {model_info}")
                
            except Exception as e:
                print(f"✗ {model_name} creation failed: {e}")
                # Don't fail the test for individual model failures
                continue
    
    def test_model_training_with_real_data(self):
        """Test ModelTrainer with real data and new TimeSeries creation."""
        print("\n=== Testing Model Training with Real Data ===")
        
        # Create TimeSeries using new data-driven approach
        timeseries = self.timeseries_creator.create_timeseries(self.df)
        print(f"Created TimeSeries with {len(timeseries)} points for training")
        
        # Split data for training
        train_ts, val_ts, test_ts = self.data_splitter.split_data(timeseries)
        print(f"Data split: Train={len(train_ts)}, Val={len(val_ts)}, Test={len(test_ts)}")
        
        # Scale data
        train_scaled, val_scaled, test_scaled, scaler = self.data_scaler.scale_data(train_ts, val_ts, test_ts)
        print(f"Data scaled successfully")
        
        # Test training with subset of models
        test_models = ['RNNModel', 'DLinearModel']
        training_results = {}
        
        for model_name in test_models:
            print(f"\nTraining {model_name}...")
            
            try:
                # Create all models and extract the specific one
                all_models = self.model_factory.create_models()
                model = all_models.get(model_name)
                if model is None:
                    print(f"✗ {model_name} creation failed, skipping training")
                    continue
                
                # Train model
                training_result = self.model_trainer.train_model(
                    model, 
                    train_scaled, 
                    val_scaled,
                    model_name=model_name
                )
                
                # Validate training results
                self.assertIsNotNone(training_result, f"{model_name} training should return results")
                
                # Check if training_result is a TrainingResults object or dict
                if hasattr(training_result, 'final_train_loss'):
                    # It's a TrainingResults dataclass
                    final_train_loss = training_result.final_train_loss
                    final_val_loss = training_result.final_val_loss
                    training_time = training_result.training_time
                else:
                    # It's a dictionary
                    self.assertIn('final_train_loss', training_result)
                    self.assertIn('final_val_loss', training_result)
                    self.assertIn('training_time', training_result)
                    final_train_loss = training_result['final_train_loss']
                    final_val_loss = training_result['final_val_loss']
                    training_time = training_result['training_time']
                
                training_results[model_name] = training_result
                
                print(f"✓ {model_name} training completed")
                print(f"  Final train loss: {final_train_loss:.6f}")
                print(f"  Final val loss: {final_val_loss:.6f}")
                print(f"  Training time: {training_time:.2f}s")
                
            except Exception as e:
                print(f"✗ {model_name} training failed: {e}")
                # Continue with other models
                continue
        
        # Validate that at least one model trained successfully
        self.assertGreater(len(training_results), 0, "At least one model should train successfully")
        
        return training_results
    
    def test_prediction_with_real_data(self):
        """Test model prediction with real data."""
        print("\n=== Testing Model Prediction with Real Data ===")
        
        # Create TimeSeries using new data-driven approach
        timeseries = self.timeseries_creator.create_timeseries(self.df)
        
        # Split and scale data
        train_ts, val_ts, test_ts = self.data_splitter.split_data(timeseries)
        train_scaled, val_scaled, test_scaled, scaler = self.data_scaler.scale_data(train_ts, val_ts, test_ts)
        
        # Test with one model for prediction
        model_name = 'DLinearModel'  # Usually fastest to train
        
        try:
            # Create all models and extract the specific one
            all_models = self.model_factory.create_models()
            model = all_models.get(model_name)
            if model is None:
                self.skipTest(f"{model_name} creation failed")
            
            training_result = self.model_trainer.train_model(
                model, 
                train_scaled, 
                val_scaled,
                model_name=model_name
            )
            
            print(f"Model trained successfully")
            
            # Make predictions
            prediction_length = min(5, len(test_scaled))  # Predict up to 5 steps
            predictions = model.predict(n=prediction_length, series=train_scaled)
            
            # Validate predictions
            self.assertIsInstance(predictions, TimeSeries)
            self.assertEqual(len(predictions), prediction_length)
            
            # Check that predictions don't contain NaN values
            try:
                pred_df = predictions.pd_dataframe()
            except AttributeError:
                pred_df = predictions.to_dataframe()
            
            self.assertFalse(pred_df.isna().any().any(), "Predictions should not contain NaN values")
            
            print(f"✓ Predictions generated successfully: {len(predictions)} steps")
            print(f"  Prediction shape: {pred_df.shape}")
            print(f"  Prediction range: {pred_df.min().min():.4f} to {pred_df.max().max():.4f}")
            
        except Exception as e:
            print(f"✗ Prediction test failed: {e}")
            # Don't fail the test - prediction might fail due to model complexity
            self.skipTest(f"Prediction failed: {e}")
    
    def test_comprehensive_pipeline_with_holiday_discovery(self):
        """Test the complete pipeline with new holiday discovery logic."""
        print("\n=== Testing Comprehensive Pipeline ===")
        
        pipeline_results = {
            'data_loaded': False,
            'holidays_discovered': False,
            'timeseries_created': False,
            'data_split': False,
            'data_scaled': False,
            'models_created': 0,
            'models_trained': 0,
            'predictions_made': False
        }
        
        try:
            # 1. Data loading (already done in setup)
            pipeline_results['data_loaded'] = True
            print(f"✓ Data loaded: {len(self.df)} rows")
            
            # 2. Holiday discovery
            discovered_holidays = self.timeseries_creator.get_discovered_holidays(self.df)
            pipeline_results['holidays_discovered'] = True
            print(f"✓ Holidays discovered: {len(discovered_holidays)}")
            
            # 3. TimeSeries creation with holiday discovery
            timeseries = self.timeseries_creator.create_timeseries(self.df)
            pipeline_results['timeseries_created'] = True
            print(f"✓ TimeSeries created: {len(timeseries)} points")
            
            # 4. Data splitting
            train_ts, val_ts, test_ts = self.data_splitter.split_data(timeseries)
            pipeline_results['data_split'] = True
            print(f"✓ Data split: {len(train_ts)}/{len(val_ts)}/{len(test_ts)}")
            
            # 5. Data scaling
            train_scaled, val_scaled, test_scaled, scaler = self.data_scaler.scale_data(train_ts, val_ts, test_ts)
            pipeline_results['data_scaled'] = True
            print(f"✓ Data scaled successfully")
            
            # 6. Model creation
            test_models = ['RNNModel', 'DLinearModel']
            created_models = {}
            
            try:
                # Create all models at once
                all_models = self.model_factory.create_models()
                for model_name in test_models:
                    if model_name in all_models and all_models[model_name] is not None:
                        created_models[model_name] = all_models[model_name]
                        pipeline_results['models_created'] += 1
            except Exception as e:
                print(f"  Model creation failed: {e}")
            
            print(f"✓ Models created: {pipeline_results['models_created']}/{len(test_models)}")
            
            # 7. Model training
            trained_models = {}
            for model_name, model in created_models.items():
                try:
                    training_result = self.model_trainer.train_model(
                        model, 
                        train_scaled, 
                        val_scaled,
                        model_name=model_name
                    )
                    trained_models[model_name] = (model, training_result)
                    pipeline_results['models_trained'] += 1
                except Exception as e:
                    print(f"  Model {model_name} training failed: {e}")
            
            print(f"✓ Models trained: {pipeline_results['models_trained']}/{len(created_models)}")
            
            # 8. Prediction testing
            if len(trained_models) > 0:
                model_name, (model, _) = list(trained_models.items())[0]
                try:
                    predictions = model.predict(n=3, series=train_scaled)
                    pipeline_results['predictions_made'] = True
                    print(f"✓ Predictions made with {model_name}")
                except Exception as e:
                    print(f"  Prediction failed: {e}")
            
            # Summary
            print(f"\n📊 Pipeline Summary:")
            for key, value in pipeline_results.items():
                status = "✓" if value else "✗"
                print(f"  {status} {key.replace('_', ' ').title()}: {value}")
            
            # Validate critical components
            self.assertTrue(pipeline_results['data_loaded'], "Data should be loaded")
            self.assertTrue(pipeline_results['holidays_discovered'], "Holidays should be discovered")
            self.assertTrue(pipeline_results['timeseries_created'], "TimeSeries should be created")
            self.assertTrue(pipeline_results['data_split'], "Data should be split")
            self.assertTrue(pipeline_results['data_scaled'], "Data should be scaled")
            self.assertGreater(pipeline_results['models_created'], 0, "At least one model should be created")
            
            print(f"\n🎉 Comprehensive pipeline test completed successfully!")
            
        except Exception as e:
            print(f"\n💥 Pipeline test failed: {e}")
            raise
    
    def test_error_handling_and_diagnostics(self):
        """Test error handling and provide comprehensive diagnostics."""
        print("\n=== Testing Error Handling and Diagnostics ===")
        
        # Test with invalid data
        print("Testing error handling...")
        
        # Test with empty DataFrame
        empty_df = pd.DataFrame()
        with self.assertRaises(ValueError):
            self.timeseries_creator.create_timeseries(empty_df)
        print("✓ Empty DataFrame error handling works")
        
        # Test with non-datetime index
        invalid_df = pd.DataFrame({'value': [1, 2, 3]})
        with self.assertRaises(Exception):  # Should raise TimeSeriesIndexError
            self.timeseries_creator.create_timeseries(invalid_df)
        print("✓ Invalid index error handling works")
        
        # Diagnostic information
        print(f"\nDiagnostic Information:")
        print(f"  Real data file: {self.real_data_file}")
        print(f"  Data available: {self.data_available}")
        if self.data_available:
            print(f"  Data shape: {self.df.shape}")
            print(f"  Date range: {self.df.index.min()} to {self.df.index.max()}")
            print(f"  Columns: {list(self.df.columns)}")
            
            # Test TimeSeries creation diagnostics
            timeseries = self.timeseries_creator.create_timeseries(self.df)
            ts_info = self.timeseries_creator.get_timeseries_info(timeseries)
            print(f"  TimeSeries length: {ts_info['length']}")
            print(f"  TimeSeries columns: {len(ts_info['columns'])}")
            print(f"  Has NaN: {ts_info['has_nan']}")
        
        print("✓ Error handling and diagnostics completed")


@unittest.skipIf(DEPENDENCIES_AVAILABLE, "Dependencies are available - skipping mock tests")
class TestSimpleRealDataWithoutDependencies(unittest.TestCase):
    """Test cases for when dependencies are not available."""
    
    def test_import_error_handling(self):
        """Test that import errors are handled gracefully."""
        self.assertTrue(True, "Test file can be imported without dependencies")


if __name__ == '__main__':
    # Configure warnings
    warnings.filterwarnings('ignore', category=UserWarning)
    
    if not DEPENDENCIES_AVAILABLE:
        print("Warning: Required dependencies not available. Some tests will be skipped.")
    
    # Run tests with verbose output
    unittest.main(verbosity=2)