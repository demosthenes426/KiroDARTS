"""
Integration test for ModelArtifactSaver with other project components.

This test demonstrates the ModelArtifactSaver working with real scalers and metadata
from the training pipeline.
"""

import unittest
import tempfile
import shutil
import numpy as np
import pandas as pd
from pathlib import Path

from src.model_artifact_saver import ModelArtifactSaver
from src.data_scaler import DataScaler
from src.model_trainer import TrainingResults

# Try to import DARTS components
try:
    from darts import TimeSeries
    DARTS_AVAILABLE = True
except ImportError:
    DARTS_AVAILABLE = False


class TestModelArtifactSaverIntegration(unittest.TestCase):
    """Integration tests for ModelArtifactSaver with project components."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.saver = ModelArtifactSaver(base_artifacts_dir=self.test_dir)
        
        # Create test data
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        data = np.random.randn(100, 3)
        
        self.test_df = pd.DataFrame(
            data,
            index=dates,
            columns=['feature1', 'feature2', 'feature3']
        )
        
        if DARTS_AVAILABLE:
            # Create TimeSeries objects for testing
            self.train_ts = TimeSeries.from_dataframe(self.test_df[:70])
            self.val_ts = TimeSeries.from_dataframe(self.test_df[70:85])
            self.test_ts = TimeSeries.from_dataframe(self.test_df[85:])
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    @unittest.skipIf(not DARTS_AVAILABLE, "DARTS not available")
    def test_save_artifacts_with_real_scaler(self):
        """Test saving artifacts with a real DataScaler."""
        # Create and use DataScaler
        data_scaler = DataScaler()
        scaled_train, scaled_val, scaled_test, scaler = data_scaler.scale_data(
            self.train_ts, self.val_ts, self.test_ts
        )
        
        # Create realistic training results
        training_results = TrainingResults(
            model_name="RNNModel",
            train_loss=[1.0, 0.8, 0.6, 0.4, 0.2],
            val_loss=[1.1, 0.9, 0.7, 0.5, 0.3],
            training_time=123.45,
            final_train_loss=0.2,
            final_val_loss=0.3,
            epochs_completed=5,
            early_stopped=False,
            convergence_achieved=True
        )
        
        # Convert TrainingResults to metadata dict
        metadata = {
            'model_name': training_results.model_name,
            'train_loss': training_results.train_loss,
            'val_loss': training_results.val_loss,
            'training_time': training_results.training_time,
            'final_train_loss': training_results.final_train_loss,
            'final_val_loss': training_results.final_val_loss,
            'epochs_completed': training_results.epochs_completed,
            'early_stopped': training_results.early_stopped,
            'convergence_achieved': training_results.convergence_achieved,
            'scaler_info': data_scaler.get_scaling_info(scaler, scaled_train, scaled_val, scaled_test)
        }
        
        # Create mock model
        from unittest.mock import Mock
        mock_model = Mock()
        
        def mock_save(path):
            Path(path).touch()
        
        mock_model.save.side_effect = mock_save
        
        # Save artifacts
        saved_dir = self.saver.save_artifacts(
            model=mock_model,
            scaler=scaler,
            metadata=metadata,
            model_name="RNNModel"
        )
        
        # Verify artifacts were saved
        self.assertTrue(Path(saved_dir).exists())
        
        # Verify scaler file exists and is loadable
        scaler_file = Path(saved_dir) / "RNNModel_scaler.pkl"
        self.assertTrue(scaler_file.exists())
        
        # Load and verify scaler
        loaded_scaler = self.saver._load_scaler(scaler_file)
        np.testing.assert_array_equal(loaded_scaler.mean_, scaler.mean_)
        np.testing.assert_array_equal(loaded_scaler.scale_, scaler.scale_)
        
        # Verify metadata file exists and contains scaling info
        metadata_file = Path(saved_dir) / "RNNModel_metadata.json"
        self.assertTrue(metadata_file.exists())
        
        loaded_metadata = self.saver._load_metadata(metadata_file)
        self.assertIn('scaler_info', loaded_metadata)
        self.assertIn('scaler_mean', loaded_metadata['scaler_info'])
        self.assertIn('scaler_scale', loaded_metadata['scaler_info'])
        self.assertEqual(loaded_metadata['convergence_achieved'], True)
        self.assertEqual(loaded_metadata['final_val_loss'], 0.3)
    
    def test_save_artifacts_with_training_results_dataclass(self):
        """Test saving artifacts using TrainingResults dataclass directly."""
        # Create realistic training results
        training_results = TrainingResults(
            model_name="TCNModel",
            train_loss=[2.0, 1.5, 1.0, 0.8, 0.6],
            val_loss=[2.1, 1.6, 1.1, 0.9, 0.7],
            training_time=234.56,
            final_train_loss=0.6,
            final_val_loss=0.7,
            epochs_completed=5,
            early_stopped=True,
            convergence_achieved=True
        )
        
        # Create scaler
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.fit(self.test_df.values)
        
        # Create mock model
        from unittest.mock import Mock
        mock_model = Mock()
        
        def mock_save(path):
            Path(path).touch()
        
        mock_model.save.side_effect = mock_save
        
        # Save artifacts using dataclass directly as metadata
        # The _make_json_serializable method should handle the dataclass
        saved_dir = self.saver.save_artifacts(
            model=mock_model,
            scaler=scaler,
            metadata={'training_results': training_results},
            model_name="TCNModel"
        )
        
        # Verify artifacts were saved
        self.assertTrue(Path(saved_dir).exists())
        
        # Load and verify metadata
        metadata_file = Path(saved_dir) / "TCNModel_metadata.json"
        loaded_metadata = self.saver._load_metadata(metadata_file)
        
        # Verify training results were serialized correctly
        self.assertIn('training_results', loaded_metadata)
        training_data = loaded_metadata['training_results']
        
        # Should be converted to dict by _make_json_serializable
        if isinstance(training_data, dict):
            self.assertEqual(training_data['model_name'], "TCNModel")
            self.assertEqual(training_data['final_val_loss'], 0.7)
            self.assertEqual(training_data['early_stopped'], True)
        else:
            # If converted to string, should contain the data
            self.assertIn("TCNModel", str(training_data))
    
    def test_artifacts_summary_with_multiple_models(self):
        """Test getting artifacts summary with multiple saved models."""
        # Save multiple models
        from unittest.mock import Mock
        from sklearn.preprocessing import StandardScaler
        
        models_data = [
            ("RNNModel", 0.2, 0.3, True),
            ("TCNModel", 0.15, 0.25, True),
            ("DLinearModel", 0.18, 0.28, False)
        ]
        
        for model_name, train_loss, val_loss, converged in models_data:
            mock_model = Mock()
            mock_model.save.side_effect = lambda path: Path(path).touch()
            
            scaler = StandardScaler()
            scaler.fit(np.random.randn(50, 3))
            
            metadata = {
                'model_type': model_name,
                'final_train_loss': train_loss,
                'final_val_loss': val_loss,
                'convergence_achieved': converged,
                'training_time': np.random.uniform(100, 300)
            }
            
            self.saver.save_artifacts(
                model=mock_model,
                scaler=scaler,
                metadata=metadata,
                model_name=model_name
            )
        
        # Get artifacts summary
        summary = self.saver.get_artifacts_summary()
        
        # Verify summary contains all models
        self.assertEqual(summary['total_models'], 3)
        self.assertIn('models', summary)
        
        saved_models = summary['models']
        self.assertEqual(len(saved_models), 3)
        
        # Verify each model is present
        model_names = set(saved_models.keys())
        expected_names = {"RNNModel", "TCNModel", "DLinearModel"}
        self.assertEqual(model_names, expected_names)
        
        # Verify model information
        for model_name in expected_names:
            model_info = saved_models[model_name]
            self.assertIn('model_type', model_info)
            self.assertIn('final_val_loss', model_info)
            self.assertIn('saved_at', model_info)
    
    def test_validate_artifacts_integrity_comprehensive(self):
        """Test comprehensive artifact integrity validation."""
        from unittest.mock import Mock
        from sklearn.preprocessing import StandardScaler
        
        # Create and save a complete set of artifacts
        mock_model = Mock()
        mock_model.save.side_effect = lambda path: Path(path).touch()
        
        scaler = StandardScaler()
        scaler.fit(np.random.randn(50, 3))
        
        metadata = {
            'model_type': 'RNNModel',
            'final_train_loss': 0.2,
            'final_val_loss': 0.3,
            'convergence_achieved': True
        }
        
        saved_dir = self.saver.save_artifacts(
            model=mock_model,
            scaler=scaler,
            metadata=metadata,
            model_name="ValidationTest"
        )
        
        # Validate integrity
        validation = self.saver.validate_artifacts_integrity(
            "ValidationTest", saved_dir
        )
        
        # All files should exist
        self.assertTrue(validation['model_file_exists'])
        self.assertTrue(validation['scaler_file_exists'])
        self.assertTrue(validation['metadata_file_exists'])
        
        # Scaler and metadata should be loadable
        self.assertTrue(validation['scaler_loadable'])
        self.assertTrue(validation['metadata_loadable'])
        
        # Model loading will fail since it's a mock, but that's expected
        self.assertFalse(validation['model_loadable'])
        self.assertFalse(validation['all_valid'])


if __name__ == '__main__':
    unittest.main()