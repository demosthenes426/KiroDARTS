"""
Unit tests for ModelArtifactSaver class.

Tests the saving and loading functionality for DARTS models, scalers, and metadata.
"""

import unittest
import tempfile
import shutil
import json
import pickle
import os
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Import the class to test
from src.model_artifact_saver import ModelArtifactSaver, ModelArtifactError

# Try to import DARTS components for testing
try:
    from darts import TimeSeries
    from darts.models.forecasting.forecasting_model import ForecastingModel
    DARTS_AVAILABLE = True
except ImportError:
    TimeSeries = None
    ForecastingModel = None
    DARTS_AVAILABLE = False


class TestModelArtifactSaver(unittest.TestCase):
    """Test cases for ModelArtifactSaver class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for testing
        self.test_dir = tempfile.mkdtemp()
        self.saver = ModelArtifactSaver(base_artifacts_dir=self.test_dir)
        
        # Create mock model
        self.mock_model = Mock(spec=ForecastingModel if DARTS_AVAILABLE else object)
        self.mock_model.save = Mock()
        
        # Create real scaler with test data
        self.scaler = StandardScaler()
        test_data = np.random.randn(100, 3)
        self.scaler.fit(test_data)
        
        # Create test metadata
        self.test_metadata = {
            'model_type': 'RNNModel',
            'training_time': 123.45,
            'final_train_loss': 0.123,
            'final_val_loss': 0.145,
            'convergence_achieved': True,
            'epochs_completed': 50,
            'train_loss': [1.0, 0.8, 0.6, 0.4, 0.2],
            'val_loss': [1.1, 0.9, 0.7, 0.5, 0.3]
        }
        
        self.model_name = "test_model"
    
    def tearDown(self):
        """Clean up test fixtures."""
        # Remove temporary directory
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_init(self):
        """Test ModelArtifactSaver initialization."""
        # Test default initialization
        default_saver = ModelArtifactSaver()
        self.assertTrue(default_saver.base_artifacts_dir.exists())
        
        # Test custom directory initialization
        custom_dir = os.path.join(self.test_dir, "custom_artifacts")
        custom_saver = ModelArtifactSaver(base_artifacts_dir=custom_dir)
        self.assertTrue(custom_saver.base_artifacts_dir.exists())
        self.assertEqual(str(custom_saver.base_artifacts_dir), custom_dir)
    
    @patch('src.model_artifact_saver.DARTS_AVAILABLE', True)
    def test_save_artifacts_success(self):
        """Test successful artifact saving."""
        # Mock the model save method to actually create a file
        def mock_save(path):
            Path(path).touch()
        
        self.mock_model.save.side_effect = mock_save
        
        # Save artifacts
        saved_dir = self.saver.save_artifacts(
            model=self.mock_model,
            scaler=self.scaler,
            metadata=self.test_metadata,
            model_name=self.model_name
        )
        
        # Verify directory was created
        self.assertTrue(os.path.exists(saved_dir))
        
        # Verify files were created
        saved_path = Path(saved_dir)
        model_file = saved_path / f"{self.model_name}_model.pt"
        scaler_file = saved_path / f"{self.model_name}_scaler.pkl"
        metadata_file = saved_path / f"{self.model_name}_metadata.json"
        
        self.assertTrue(model_file.exists())
        self.assertTrue(scaler_file.exists())
        self.assertTrue(metadata_file.exists())
        
        # Verify model.save was called
        self.mock_model.save.assert_called_once()
        
        # Verify scaler was saved correctly
        with open(scaler_file, 'rb') as f:
            loaded_scaler = pickle.load(f)
        np.testing.assert_array_equal(loaded_scaler.mean_, self.scaler.mean_)
        np.testing.assert_array_equal(loaded_scaler.scale_, self.scaler.scale_)
        
        # Verify metadata was saved correctly
        with open(metadata_file, 'r') as f:
            loaded_metadata = json.load(f)
        
        self.assertEqual(loaded_metadata['model_name'], self.model_name)
        self.assertEqual(loaded_metadata['model_type'], 'RNNModel')
        self.assertEqual(loaded_metadata['training_time'], 123.45)
        self.assertTrue('saved_at' in loaded_metadata)
        self.assertTrue('artifact_version' in loaded_metadata)
    
    @patch('src.model_artifact_saver.DARTS_AVAILABLE', False)
    def test_save_artifacts_darts_unavailable(self):
        """Test artifact saving when DARTS is unavailable."""
        with self.assertRaises(ModelArtifactError) as context:
            self.saver.save_artifacts(
                model=self.mock_model,
                scaler=self.scaler,
                metadata=self.test_metadata,
                model_name=self.model_name
            )
        
        self.assertIn("DARTS library is not available", str(context.exception))
    
    def test_save_artifacts_model_save_failure(self):
        """Test artifact saving when model save fails."""
        # Mock model save to raise exception
        self.mock_model.save.side_effect = Exception("Model save failed")
        
        with patch('src.model_artifact_saver.DARTS_AVAILABLE', True):
            with patch('torch.save') as mock_torch_save:
                mock_torch_save.side_effect = Exception("Torch save failed")
                
                with self.assertRaises(ModelArtifactError) as context:
                    self.saver.save_artifacts(
                        model=self.mock_model,
                        scaler=self.scaler,
                        metadata=self.test_metadata,
                        model_name=self.model_name
                    )
                
                self.assertIn("Failed to save artifacts", str(context.exception))
    
    def test_create_model_directory(self):
        """Test model directory creation."""
        model_dir = self.saver._create_model_directory(self.model_name)
        
        # Verify directory exists
        self.assertTrue(model_dir.exists())
        
        # Verify directory name format
        self.assertTrue(model_dir.name.startswith(f"{self.model_name}_"))
        
        # Verify timestamp format in directory name
        timestamp_part = model_dir.name.split(f"{self.model_name}_")[1]
        # Should be in format YYYYMMDD_HHMMSS
        self.assertEqual(len(timestamp_part), 15)  # 8 + 1 + 6
    
    def test_save_scaler(self):
        """Test scaler saving."""
        model_dir = self.saver._create_model_directory(self.model_name)
        scaler_path = self.saver._save_scaler(self.scaler, model_dir, self.model_name)
        
        # Verify file was created
        self.assertTrue(scaler_path.exists())
        
        # Verify scaler can be loaded and is identical
        with open(scaler_path, 'rb') as f:
            loaded_scaler = pickle.load(f)
        
        np.testing.assert_array_equal(loaded_scaler.mean_, self.scaler.mean_)
        np.testing.assert_array_equal(loaded_scaler.scale_, self.scaler.scale_)
    
    def test_save_metadata(self):
        """Test metadata saving."""
        model_dir = self.saver._create_model_directory(self.model_name)
        model_path = model_dir / f"{self.model_name}_model.pt"
        scaler_path = model_dir / f"{self.model_name}_scaler.pkl"
        
        # Create dummy files
        model_path.touch()
        scaler_path.touch()
        
        enhanced_metadata = self.saver._enhance_metadata(
            self.test_metadata, self.model_name, model_path, scaler_path
        )
        
        metadata_path = self.saver._save_metadata(enhanced_metadata, model_dir, self.model_name)
        
        # Verify file was created
        self.assertTrue(metadata_path.exists())
        
        # Verify metadata content
        with open(metadata_path, 'r') as f:
            loaded_metadata = json.load(f)
        
        self.assertEqual(loaded_metadata['model_name'], self.model_name)
        self.assertEqual(loaded_metadata['model_type'], 'RNNModel')
        self.assertTrue('saved_at' in loaded_metadata)
        self.assertTrue('artifact_version' in loaded_metadata)
    
    def test_enhance_metadata(self):
        """Test metadata enhancement."""
        model_path = Path("test_model.pt")
        scaler_path = Path("test_scaler.pkl")
        
        enhanced = self.saver._enhance_metadata(
            self.test_metadata, self.model_name, model_path, scaler_path
        )
        
        # Verify original metadata is preserved
        self.assertEqual(enhanced['model_type'], 'RNNModel')
        self.assertEqual(enhanced['training_time'], 123.45)
        
        # Verify new fields are added
        self.assertEqual(enhanced['model_name'], self.model_name)
        self.assertEqual(enhanced['model_file'], model_path.name)
        self.assertEqual(enhanced['scaler_file'], scaler_path.name)
        self.assertTrue('saved_at' in enhanced)
        self.assertTrue('artifact_version' in enhanced)
    
    def test_make_json_serializable(self):
        """Test JSON serialization of complex objects."""
        # Test with various data types
        test_data = {
            'string': 'test',
            'int': 42,
            'float': 3.14,
            'bool': True,
            'none': None,
            'list': [1, 2, 3],
            'nested_dict': {'a': 1, 'b': 2},
            'numpy_array': np.array([1, 2, 3]),
            'custom_object': Mock()
        }
        
        serializable = self.saver._make_json_serializable(test_data)
        
        # Verify it can be JSON serialized
        json_str = json.dumps(serializable)
        self.assertIsInstance(json_str, str)
        
        # Verify basic types are preserved
        self.assertEqual(serializable['string'], 'test')
        self.assertEqual(serializable['int'], 42)
        self.assertEqual(serializable['float'], 3.14)
        self.assertEqual(serializable['bool'], True)
        self.assertIsNone(serializable['none'])
        
        # Verify complex types are converted to strings
        self.assertIsInstance(serializable['numpy_array'], str)
        self.assertIsInstance(serializable['custom_object'], str)
    
    def test_list_saved_models_empty(self):
        """Test listing saved models when none exist."""
        saved_models = self.saver.list_saved_models()
        self.assertEqual(saved_models, {})
    
    def test_list_saved_models_with_models(self):
        """Test listing saved models when models exist."""
        # Create a fake model directory with metadata
        model_dir = self.saver._create_model_directory(self.model_name)
        
        # Create metadata file
        enhanced_metadata = self.saver._enhance_metadata(
            self.test_metadata, self.model_name, 
            model_dir / "model.pt", model_dir / "scaler.pkl"
        )
        self.saver._save_metadata(enhanced_metadata, model_dir, self.model_name)
        
        # List saved models
        saved_models = self.saver.list_saved_models()
        
        self.assertEqual(len(saved_models), 1)
        self.assertIn(self.model_name, saved_models)
        
        model_info = saved_models[self.model_name]
        self.assertEqual(model_info['model_type'], 'RNNModel')
        self.assertEqual(model_info['training_time'], 123.45)
        self.assertTrue('saved_at' in model_info)
    
    def test_find_model_directory_not_found(self):
        """Test finding model directory when it doesn't exist."""
        with self.assertRaises(ModelArtifactError) as context:
            self.saver._find_model_directory("nonexistent_model")
        
        self.assertIn("No saved artifacts found", str(context.exception))
    
    def test_find_model_directory_success(self):
        """Test finding model directory when it exists."""
        # Create model directory
        created_dir = self.saver._create_model_directory(self.model_name)
        
        # Find the directory
        found_dir = self.saver._find_model_directory(self.model_name)
        
        self.assertEqual(created_dir, found_dir)
    
    def test_find_model_directory_multiple_versions(self):
        """Test finding most recent model directory when multiple exist."""
        # Create multiple directories with different timestamps
        import time
        
        dir1 = self.saver._create_model_directory(self.model_name)
        time.sleep(0.1)  # Ensure different timestamps
        dir2 = self.saver._create_model_directory(self.model_name)
        
        # Find directory should return the most recent one
        found_dir = self.saver._find_model_directory(self.model_name)
        
        # The most recent directory should be dir2
        self.assertEqual(found_dir, dir2)
    
    def test_delete_model_artifacts_not_found(self):
        """Test deleting artifacts when model doesn't exist."""
        result = self.saver.delete_model_artifacts("nonexistent_model")
        self.assertFalse(result)
    
    def test_delete_model_artifacts_success(self):
        """Test successful deletion of model artifacts."""
        # Create model directory with files
        model_dir = self.saver._create_model_directory(self.model_name)
        
        # Create some test files
        (model_dir / "test_file1.txt").touch()
        (model_dir / "test_file2.txt").touch()
        
        # Delete artifacts
        result = self.saver.delete_model_artifacts(self.model_name)
        
        self.assertTrue(result)
        # Directory should be removed or empty
        if model_dir.exists():
            self.assertEqual(len(list(model_dir.iterdir())), 0)
    
    def test_validate_artifacts_integrity_missing_files(self):
        """Test artifact integrity validation with missing files."""
        # Create directory but no files
        model_dir = self.saver._create_model_directory(self.model_name)
        
        validation = self.saver.validate_artifacts_integrity(
            self.model_name, str(model_dir)
        )
        
        self.assertFalse(validation['model_file_exists'])
        self.assertFalse(validation['scaler_file_exists'])
        self.assertFalse(validation['metadata_file_exists'])
        self.assertFalse(validation['all_valid'])
    
    def test_validate_artifacts_integrity_with_files(self):
        """Test artifact integrity validation with existing files."""
        # Create model directory
        model_dir = self.saver._create_model_directory(self.model_name)
        
        # Create scaler file
        scaler_path = self.saver._save_scaler(self.scaler, model_dir, self.model_name)
        
        # Create metadata file
        enhanced_metadata = self.saver._enhance_metadata(
            self.test_metadata, self.model_name, 
            model_dir / "model.pt", scaler_path
        )
        metadata_path = self.saver._save_metadata(enhanced_metadata, model_dir, self.model_name)
        
        # Create dummy model file
        model_path = model_dir / f"{self.model_name}_model.pt"
        model_path.touch()
        
        validation = self.saver.validate_artifacts_integrity(
            self.model_name, str(model_dir)
        )
        
        self.assertTrue(validation['model_file_exists'])
        self.assertTrue(validation['scaler_file_exists'])
        self.assertTrue(validation['metadata_file_exists'])
        self.assertTrue(validation['scaler_loadable'])
        self.assertTrue(validation['metadata_loadable'])
        # Model loading will fail since it's just a dummy file
        self.assertFalse(validation['model_loadable'])
        self.assertFalse(validation['all_valid'])
    
    def test_get_artifacts_summary(self):
        """Test getting artifacts summary."""
        # Create a model
        model_dir = self.saver._create_model_directory(self.model_name)
        enhanced_metadata = self.saver._enhance_metadata(
            self.test_metadata, self.model_name, 
            model_dir / "model.pt", model_dir / "scaler.pkl"
        )
        self.saver._save_metadata(enhanced_metadata, model_dir, self.model_name)
        
        summary = self.saver.get_artifacts_summary()
        
        self.assertEqual(summary['total_models'], 1)
        self.assertEqual(summary['base_directory'], str(self.saver.base_artifacts_dir))
        self.assertIn('models', summary)
        self.assertIn('directory_size_mb', summary)
        self.assertIn('last_updated', summary)
    
    def test_get_directory_size_mb(self):
        """Test directory size calculation."""
        # Create some test files
        model_dir = self.saver._create_model_directory(self.model_name)
        
        # Create a file with known size
        test_file = model_dir / "test_file.txt"
        test_content = "x" * 1024  # 1KB
        test_file.write_text(test_content)
        
        size_mb = self.saver._get_directory_size_mb()
        
        # Should be greater than 0
        self.assertGreater(size_mb, 0)
        # Should be approximately 1KB = 0.001MB (allowing for filesystem overhead)
        self.assertLess(size_mb, 0.1)  # Should be much less than 0.1MB
    
    @patch('src.model_artifact_saver.DARTS_AVAILABLE', False)
    def test_load_artifacts_darts_unavailable(self):
        """Test loading artifacts when DARTS is unavailable."""
        with self.assertRaises(ModelArtifactError) as context:
            self.saver.load_artifacts(self.model_name)
        
        self.assertIn("DARTS library is not available", str(context.exception))
    
    def test_load_artifacts_directory_not_found(self):
        """Test loading artifacts when directory doesn't exist."""
        with patch('src.model_artifact_saver.DARTS_AVAILABLE', True):
            with self.assertRaises(ModelArtifactError) as context:
                self.saver.load_artifacts("nonexistent_model")
            
            self.assertIn("No saved artifacts found", str(context.exception))


class TestModelArtifactSaverIntegration(unittest.TestCase):
    """Integration tests for ModelArtifactSaver with real components."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.saver = ModelArtifactSaver(base_artifacts_dir=self.test_dir)
        
        # Create real scaler with test data
        self.scaler = StandardScaler()
        test_data = np.random.randn(100, 3)
        self.scaler.fit(test_data)
        
        self.model_name = "integration_test_model"
    
    def tearDown(self):
        """Clean up integration test fixtures."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_scaler_save_load_roundtrip(self):
        """Test that scaler can be saved and loaded correctly."""
        # Create model directory
        model_dir = self.saver._create_model_directory(self.model_name)
        
        # Save scaler
        scaler_path = self.saver._save_scaler(self.scaler, model_dir, self.model_name)
        
        # Load scaler
        loaded_scaler = self.saver._load_scaler(scaler_path)
        
        # Verify they are equivalent
        np.testing.assert_array_equal(loaded_scaler.mean_, self.scaler.mean_)
        np.testing.assert_array_equal(loaded_scaler.scale_, self.scaler.scale_)
        
        # Test transformation consistency
        test_data = np.random.randn(10, 3)
        original_transform = self.scaler.transform(test_data)
        loaded_transform = loaded_scaler.transform(test_data)
        
        np.testing.assert_array_almost_equal(original_transform, loaded_transform)
    
    def test_metadata_save_load_roundtrip(self):
        """Test that metadata can be saved and loaded correctly."""
        # Create model directory
        model_dir = self.saver._create_model_directory(self.model_name)
        
        # Create test metadata with various data types
        test_metadata = {
            'model_type': 'RNNModel',
            'training_time': 123.45,
            'epochs': 50,
            'convergence': True,
            'losses': [1.0, 0.8, 0.6, 0.4, 0.2],
            'config': {'lr': 0.001, 'batch_size': 32}
        }
        
        # Enhance and save metadata
        enhanced_metadata = self.saver._enhance_metadata(
            test_metadata, self.model_name,
            model_dir / "model.pt", model_dir / "scaler.pkl"
        )
        metadata_path = self.saver._save_metadata(enhanced_metadata, model_dir, self.model_name)
        
        # Load metadata
        loaded_metadata = self.saver._load_metadata(metadata_path)
        
        # Verify original data is preserved
        self.assertEqual(loaded_metadata['model_type'], 'RNNModel')
        self.assertEqual(loaded_metadata['training_time'], 123.45)
        self.assertEqual(loaded_metadata['epochs'], 50)
        self.assertEqual(loaded_metadata['convergence'], True)
        self.assertEqual(loaded_metadata['losses'], [1.0, 0.8, 0.6, 0.4, 0.2])
        
        # Verify enhanced data is present
        self.assertEqual(loaded_metadata['model_name'], self.model_name)
        self.assertTrue('saved_at' in loaded_metadata)
        self.assertTrue('artifact_version' in loaded_metadata)


if __name__ == '__main__':
    unittest.main()