"""
ModelArtifactSaver class for persisting trained models and preprocessing artifacts.

This module provides functionality to save and load trained DARTS models along with
their associated scalers and metadata for future predictions.
"""

import os
import json
import pickle
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union
from dataclasses import asdict

# Try to import DARTS components, handle gracefully if not available
try:
    from darts.models.forecasting.forecasting_model import ForecastingModel
    DARTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: DARTS not available: {e}")
    ForecastingModel = Any
    DARTS_AVAILABLE = False

from sklearn.preprocessing import StandardScaler


class ModelArtifactError(Exception):
    """Custom exception for model artifact operations."""
    pass


class ModelArtifactSaver:
    """
    Saves and loads trained DARTS models with associated preprocessing artifacts.
    
    This class handles the persistence of trained models, scalers, and metadata
    in an organized directory structure for future predictions.
    """
    
    def __init__(self, base_artifacts_dir: str = "model_artifacts"):
        """
        Initialize ModelArtifactSaver with base directory for artifacts.
        
        Args:
            base_artifacts_dir (str): Base directory for storing model artifacts
        """
        self.base_artifacts_dir = Path(base_artifacts_dir)
        self.base_artifacts_dir.mkdir(parents=True, exist_ok=True)
        
    def save_artifacts(self, 
                      model: ForecastingModel,
                      scaler: StandardScaler,
                      metadata: Dict[str, Any],
                      model_name: str) -> str:
        """
        Save trained model, scaler, and metadata to organized directory structure.
        
        Args:
            model (ForecastingModel): Trained DARTS model
            scaler (StandardScaler): Fitted scaler used for preprocessing
            metadata (Dict[str, Any]): Model metadata and training information
            model_name (str): Name of the model for file naming
            
        Returns:
            str: Path to the saved model directory
            
        Raises:
            ModelArtifactError: If saving fails
        """
        if not DARTS_AVAILABLE:
            raise ModelArtifactError("DARTS library is not available. Please install darts.")
        
        try:
            # Create model-specific directory
            model_dir = self._create_model_directory(model_name)
            
            # Save model (.pt file)
            model_path = self._save_model(model, model_dir, model_name)
            
            # Save scaler (.pkl file)
            scaler_path = self._save_scaler(scaler, model_dir, model_name)
            
            # Enhance metadata with save information
            enhanced_metadata = self._enhance_metadata(metadata, model_name, model_path, scaler_path)
            
            # Save metadata (.json file)
            metadata_path = self._save_metadata(enhanced_metadata, model_dir, model_name)
            
            print(f"✓ Model artifacts saved successfully:")
            print(f"  📁 Directory: {model_dir}")
            print(f"  🤖 Model: {model_path.name}")
            print(f"  📊 Scaler: {scaler_path.name}")
            print(f"  📋 Metadata: {metadata_path.name}")
            
            return str(model_dir)
            
        except Exception as e:
            raise ModelArtifactError(f"Failed to save artifacts for {model_name}: {e}") from e
    
    def load_artifacts(self, model_name: str, 
                      model_dir: Optional[str] = None) -> Tuple[ForecastingModel, StandardScaler, Dict[str, Any]]:
        """
        Load trained model, scaler, and metadata from saved artifacts.
        
        Args:
            model_name (str): Name of the model to load
            model_dir (Optional[str]): Specific model directory path (if None, searches in base directory)
            
        Returns:
            Tuple[ForecastingModel, StandardScaler, Dict[str, Any]]: 
                Loaded model, scaler, and metadata
                
        Raises:
            ModelArtifactError: If loading fails
        """
        if not DARTS_AVAILABLE:
            raise ModelArtifactError("DARTS library is not available. Please install darts.")
        
        try:
            # Find model directory
            if model_dir is None:
                model_dir = self._find_model_directory(model_name)
            else:
                model_dir = Path(model_dir)
            
            if not model_dir.exists():
                raise ModelArtifactError(f"Model directory not found: {model_dir}")
            
            # Load metadata first to get file paths
            metadata_path = model_dir / f"{model_name}_metadata.json"
            metadata = self._load_metadata(metadata_path)
            
            # Load scaler
            scaler_path = model_dir / f"{model_name}_scaler.pkl"
            scaler = self._load_scaler(scaler_path)
            
            # Load model
            model_path = model_dir / f"{model_name}_model.pt"
            model = self._load_model(model_path, metadata)
            
            print(f"✓ Model artifacts loaded successfully:")
            print(f"  📁 Directory: {model_dir}")
            print(f"  🤖 Model: {model_path.name}")
            print(f"  📊 Scaler: {scaler_path.name}")
            print(f"  📋 Metadata: {metadata_path.name}")
            
            return model, scaler, metadata
            
        except Exception as e:
            raise ModelArtifactError(f"Failed to load artifacts for {model_name}: {e}") from e
    
    def list_saved_models(self) -> Dict[str, Dict[str, Any]]:
        """
        List all saved models with their metadata.
        
        Returns:
            Dict[str, Dict[str, Any]]: Dictionary of model names to their metadata
        """
        saved_models = {}
        
        try:
            for model_dir in self.base_artifacts_dir.iterdir():
                if model_dir.is_dir():
                    # Look for metadata files in the directory
                    metadata_files = list(model_dir.glob("*_metadata.json"))
                    
                    for metadata_file in metadata_files:
                        try:
                            # Extract model name from metadata filename
                            model_name = metadata_file.stem.replace("_metadata", "")
                            
                            # Load metadata
                            with open(metadata_file, 'r') as f:
                                metadata = json.load(f)
                            
                            saved_models[model_name] = {
                                'directory': str(model_dir),
                                'saved_at': metadata.get('saved_at', 'Unknown'),
                                'model_type': metadata.get('model_type', 'Unknown'),
                                'training_time': metadata.get('training_time', 'Unknown'),
                                'final_val_loss': metadata.get('final_val_loss', 'Unknown'),
                                'convergence_achieved': metadata.get('convergence_achieved', 'Unknown')
                            }
                            
                        except Exception as e:
                            warnings.warn(f"Failed to read metadata for {metadata_file}: {e}")
                            continue
            
            return saved_models
            
        except Exception as e:
            warnings.warn(f"Failed to list saved models: {e}")
            return {}
    
    def delete_model_artifacts(self, model_name: str, model_dir: Optional[str] = None) -> bool:
        """
        Delete all artifacts for a specific model.
        
        Args:
            model_name (str): Name of the model to delete
            model_dir (Optional[str]): Specific model directory path
            
        Returns:
            bool: True if deletion was successful
        """
        try:
            # Find model directory
            if model_dir is None:
                model_dir = self._find_model_directory(model_name)
            else:
                model_dir = Path(model_dir)
            
            if not model_dir.exists():
                warnings.warn(f"Model directory not found: {model_dir}")
                return False
            
            # Delete all files in the model directory
            for file_path in model_dir.iterdir():
                if file_path.is_file():
                    file_path.unlink()
            
            # Remove the directory if it's empty
            try:
                model_dir.rmdir()
                print(f"✓ Deleted model artifacts for {model_name}")
                return True
            except OSError:
                # Directory not empty (might contain other files)
                print(f"✓ Deleted model files for {model_name} (directory not empty)")
                return True
                
        except Exception as e:
            warnings.warn(f"Failed to delete artifacts for {model_name}: {e}")
            return False
    
    def _create_model_directory(self, model_name: str) -> Path:
        """
        Create directory for model artifacts.
        
        Args:
            model_name (str): Name of the model
            
        Returns:
            Path: Path to the created directory
        """
        # Create timestamp for unique directory naming
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = self.base_artifacts_dir / f"{model_name}_{timestamp}"
        model_dir.mkdir(parents=True, exist_ok=True)
        return model_dir
    
    def _save_model(self, model: ForecastingModel, model_dir: Path, model_name: str) -> Path:
        """
        Save DARTS model to .pt file.
        
        Args:
            model (ForecastingModel): Trained DARTS model
            model_dir (Path): Directory to save the model
            model_name (str): Name of the model
            
        Returns:
            Path: Path to the saved model file
        """
        model_path = model_dir / f"{model_name}_model.pt"
        
        try:
            # Use DARTS model's save method
            model.save(str(model_path))
            return model_path
            
        except Exception as e:
            # Fallback: try to save using torch if available
            try:
                import torch
                torch.save(model, str(model_path))
                return model_path
            except Exception as torch_e:
                raise ModelArtifactError(
                    f"Failed to save model using both DARTS and torch methods. "
                    f"DARTS error: {e}, Torch error: {torch_e}"
                )
    
    def _save_scaler(self, scaler: StandardScaler, model_dir: Path, model_name: str) -> Path:
        """
        Save StandardScaler to .pkl file.
        
        Args:
            scaler (StandardScaler): Fitted scaler
            model_dir (Path): Directory to save the scaler
            model_name (str): Name of the model
            
        Returns:
            Path: Path to the saved scaler file
        """
        scaler_path = model_dir / f"{model_name}_scaler.pkl"
        
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        
        return scaler_path
    
    def _save_metadata(self, metadata: Dict[str, Any], model_dir: Path, model_name: str) -> Path:
        """
        Save metadata to .json file.
        
        Args:
            metadata (Dict[str, Any]): Model metadata
            model_dir (Path): Directory to save the metadata
            model_name (str): Name of the model
            
        Returns:
            Path: Path to the saved metadata file
        """
        metadata_path = model_dir / f"{model_name}_metadata.json"
        
        # Convert any non-serializable objects to strings
        serializable_metadata = self._make_json_serializable(metadata)
        
        with open(metadata_path, 'w') as f:
            json.dump(serializable_metadata, f, indent=2)
        
        return metadata_path
    
    def _enhance_metadata(self, metadata: Dict[str, Any], model_name: str, 
                         model_path: Path, scaler_path: Path) -> Dict[str, Any]:
        """
        Enhance metadata with save information.
        
        Args:
            metadata (Dict[str, Any]): Original metadata
            model_name (str): Name of the model
            model_path (Path): Path to saved model
            scaler_path (Path): Path to saved scaler
            
        Returns:
            Dict[str, Any]: Enhanced metadata
        """
        enhanced = metadata.copy()
        
        # Add save information
        enhanced.update({
            'model_name': model_name,
            'saved_at': datetime.now().isoformat(),
            'model_file': model_path.name,
            'scaler_file': scaler_path.name,
            'artifact_version': '1.0',
            'darts_available': DARTS_AVAILABLE
        })
        
        return enhanced
    
    def _load_model(self, model_path: Path, metadata: Dict[str, Any]) -> ForecastingModel:
        """
        Load DARTS model from .pt file.
        
        Args:
            model_path (Path): Path to the model file
            metadata (Dict[str, Any]): Model metadata
            
        Returns:
            ForecastingModel: Loaded DARTS model
        """
        try:
            # Try to load using DARTS
            from darts.models.forecasting.forecasting_model import ForecastingModel
            model = ForecastingModel.load(str(model_path))
            return model
            
        except Exception as e:
            # Fallback: try to load using torch
            try:
                import torch
                model = torch.load(str(model_path), map_location='cpu')
                return model
            except Exception as torch_e:
                raise ModelArtifactError(
                    f"Failed to load model using both DARTS and torch methods. "
                    f"DARTS error: {e}, Torch error: {torch_e}"
                )
    
    def _load_scaler(self, scaler_path: Path) -> StandardScaler:
        """
        Load StandardScaler from .pkl file.
        
        Args:
            scaler_path (Path): Path to the scaler file
            
        Returns:
            StandardScaler: Loaded scaler
        """
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        return scaler
    
    def _load_metadata(self, metadata_path: Path) -> Dict[str, Any]:
        """
        Load metadata from .json file.
        
        Args:
            metadata_path (Path): Path to the metadata file
            
        Returns:
            Dict[str, Any]: Loaded metadata
        """
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        return metadata
    
    def _find_model_directory(self, model_name: str) -> Path:
        """
        Find the most recent directory for a given model name.
        
        Args:
            model_name (str): Name of the model
            
        Returns:
            Path: Path to the model directory
            
        Raises:
            ModelArtifactError: If no directory is found
        """
        # Look for directories that start with the model name
        matching_dirs = []
        
        for model_dir in self.base_artifacts_dir.iterdir():
            if model_dir.is_dir() and model_dir.name.startswith(f"{model_name}_"):
                matching_dirs.append(model_dir)
        
        if not matching_dirs:
            raise ModelArtifactError(f"No saved artifacts found for model: {model_name}")
        
        # Return the most recent directory (based on timestamp in name)
        matching_dirs.sort(key=lambda x: x.name, reverse=True)
        return matching_dirs[0]
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """
        Convert objects to JSON-serializable format.
        
        Args:
            obj: Object to convert
            
        Returns:
            JSON-serializable object
        """
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif hasattr(obj, '__dict__'):
            # Handle dataclass or custom objects
            try:
                if hasattr(obj, '__dataclass_fields__'):
                    return asdict(obj)
                else:
                    return str(obj)
            except:
                return str(obj)
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        else:
            # Convert everything else to string
            return str(obj)
    
    def validate_artifacts_integrity(self, model_name: str, 
                                   model_dir: Optional[str] = None) -> Dict[str, bool]:
        """
        Validate the integrity of saved artifacts.
        
        Args:
            model_name (str): Name of the model to validate
            model_dir (Optional[str]): Specific model directory path
            
        Returns:
            Dict[str, bool]: Validation results for each artifact type
        """
        validation_results = {
            'model_file_exists': False,
            'scaler_file_exists': False,
            'metadata_file_exists': False,
            'model_loadable': False,
            'scaler_loadable': False,
            'metadata_loadable': False,
            'all_valid': False
        }
        
        try:
            # Find model directory
            if model_dir is None:
                model_dir = self._find_model_directory(model_name)
            else:
                model_dir = Path(model_dir)
            
            # Check file existence
            model_path = model_dir / f"{model_name}_model.pt"
            scaler_path = model_dir / f"{model_name}_scaler.pkl"
            metadata_path = model_dir / f"{model_name}_metadata.json"
            
            validation_results['model_file_exists'] = model_path.exists()
            validation_results['scaler_file_exists'] = scaler_path.exists()
            validation_results['metadata_file_exists'] = metadata_path.exists()
            
            # Test loading
            if validation_results['metadata_file_exists']:
                try:
                    self._load_metadata(metadata_path)
                    validation_results['metadata_loadable'] = True
                except:
                    pass
            
            if validation_results['scaler_file_exists']:
                try:
                    self._load_scaler(scaler_path)
                    validation_results['scaler_loadable'] = True
                except:
                    pass
            
            if validation_results['model_file_exists'] and validation_results['metadata_loadable']:
                try:
                    metadata = self._load_metadata(metadata_path)
                    self._load_model(model_path, metadata)
                    validation_results['model_loadable'] = True
                except:
                    pass
            
            # Overall validation
            validation_results['all_valid'] = all([
                validation_results['model_file_exists'],
                validation_results['scaler_file_exists'],
                validation_results['metadata_file_exists'],
                validation_results['model_loadable'],
                validation_results['scaler_loadable'],
                validation_results['metadata_loadable']
            ])
            
        except Exception as e:
            warnings.warn(f"Validation failed for {model_name}: {e}")
        
        return validation_results
    
    def get_artifacts_summary(self) -> Dict[str, Any]:
        """
        Get summary of all saved artifacts.
        
        Returns:
            Dict[str, Any]: Summary of artifacts
        """
        saved_models = self.list_saved_models()
        
        summary = {
            'total_models': len(saved_models),
            'base_directory': str(self.base_artifacts_dir),
            'models': saved_models,
            'directory_size_mb': self._get_directory_size_mb(),
            'last_updated': datetime.now().isoformat()
        }
        
        return summary
    
    def _get_directory_size_mb(self) -> float:
        """
        Calculate total size of artifacts directory in MB.
        
        Returns:
            float: Directory size in MB
        """
        try:
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(self.base_artifacts_dir):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    if os.path.exists(filepath):
                        total_size += os.path.getsize(filepath)
            
            return total_size / (1024 * 1024)  # Convert to MB
            
        except Exception:
            return 0.0