"""
Demonstration script for ModelArtifactSaver usage.

This script shows how to use the ModelArtifactSaver in a typical workflow
with model training and artifact persistence.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

from src.model_artifact_saver import ModelArtifactSaver
from src.model_trainer import TrainingResults

# Try to import DARTS components
try:
    from darts import TimeSeries
    DARTS_AVAILABLE = True
except ImportError:
    print("DARTS not available - using mock components")
    DARTS_AVAILABLE = False


def create_sample_data():
    """Create sample time series data for demonstration."""
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    
    # Create synthetic stock-like data
    data = {
        'adjusted_close': 100 + np.cumsum(np.random.randn(100) * 0.5),
        'volume': np.random.randint(1000, 10000, 100),
        'rsi': np.random.uniform(20, 80, 100),
        'macd': np.random.randn(100) * 0.1
    }
    
    df = pd.DataFrame(data, index=dates)
    return df


def demonstrate_artifact_saving():
    """Demonstrate saving model artifacts."""
    print("🚀 ModelArtifactSaver Demonstration")
    print("=" * 50)
    
    # Initialize artifact saver
    saver = ModelArtifactSaver(base_artifacts_dir="demo_artifacts")
    
    # Create sample data
    print("\n📊 Creating sample data...")
    df = create_sample_data()
    print(f"   Created DataFrame with {len(df)} rows and columns: {list(df.columns)}")
    
    # Create and fit scaler
    print("\n🔧 Creating and fitting scaler...")
    scaler = StandardScaler()
    scaler.fit(df.values)
    print(f"   Scaler fitted with mean: {scaler.mean_[:2]}... and scale: {scaler.scale_[:2]}...")
    
    # Create mock training results
    print("\n🎯 Creating training results...")
    training_results = TrainingResults(
        model_name="DemoRNNModel",
        train_loss=[1.5, 1.2, 0.9, 0.7, 0.5, 0.4, 0.3],
        val_loss=[1.6, 1.3, 1.0, 0.8, 0.6, 0.5, 0.4],
        training_time=156.78,
        final_train_loss=0.3,
        final_val_loss=0.4,
        epochs_completed=7,
        early_stopped=False,
        convergence_achieved=True
    )
    print(f"   Training completed in {training_results.training_time:.2f}s")
    print(f"   Final validation loss: {training_results.final_val_loss}")
    
    # Create mock model
    from unittest.mock import Mock
    mock_model = Mock()
    mock_model.save = lambda path: Path(path).touch()
    
    # Prepare metadata
    metadata = {
        'model_type': 'RNNModel',
        'dataset_info': {
            'rows': len(df),
            'features': list(df.columns),
            'date_range': f"{df.index[0]} to {df.index[-1]}"
        },
        'training_results': training_results,
        'scaler_info': {
            'feature_count': len(scaler.mean_),
            'mean_values': scaler.mean_.tolist(),
            'scale_values': scaler.scale_.tolist()
        }
    }
    
    # Save artifacts
    print("\n💾 Saving model artifacts...")
    saved_dir = saver.save_artifacts(
        model=mock_model,
        scaler=scaler,
        metadata=metadata,
        model_name="DemoRNNModel"
    )
    
    # List saved models
    print("\n📋 Listing all saved models...")
    saved_models = saver.list_saved_models()
    for model_name, info in saved_models.items():
        print(f"   📁 {model_name}:")
        print(f"      Type: {info.get('model_type', 'Unknown')}")
        print(f"      Saved: {info.get('saved_at', 'Unknown')}")
        print(f"      Training time: {info.get('training_time', 'Unknown')}")
        print(f"      Final val loss: {info.get('final_val_loss', 'Unknown')}")
    
    # Get artifacts summary
    print("\n📊 Artifacts summary...")
    summary = saver.get_artifacts_summary()
    print(f"   Total models: {summary['total_models']}")
    print(f"   Directory size: {summary['directory_size_mb']:.3f} MB")
    print(f"   Base directory: {summary['base_directory']}")
    
    # Validate artifacts integrity
    print("\n🔍 Validating artifacts integrity...")
    validation = saver.validate_artifacts_integrity("DemoRNNModel")
    print(f"   Model file exists: {validation['model_file_exists']}")
    print(f"   Scaler file exists: {validation['scaler_file_exists']}")
    print(f"   Metadata file exists: {validation['metadata_file_exists']}")
    print(f"   Scaler loadable: {validation['scaler_loadable']}")
    print(f"   Metadata loadable: {validation['metadata_loadable']}")
    print(f"   Overall valid: {validation['all_valid']}")
    
    return saved_dir


def demonstrate_artifact_loading(saved_dir):
    """Demonstrate loading model artifacts."""
    print("\n🔄 Loading Model Artifacts")
    print("=" * 30)
    
    saver = ModelArtifactSaver(base_artifacts_dir="demo_artifacts")
    
    try:
        # Load scaler and metadata (skip model since it's a mock)
        print("\n📥 Loading scaler and metadata...")
        
        # Load scaler directly
        scaler_path = Path(saved_dir) / "DemoRNNModel_scaler.pkl"
        loaded_scaler = saver._load_scaler(scaler_path)
        print(f"   ✓ Scaler loaded successfully")
        print(f"   Mean values: {loaded_scaler.mean_[:2]}...")
        print(f"   Scale values: {loaded_scaler.scale_[:2]}...")
        
        # Load metadata
        metadata_path = Path(saved_dir) / "DemoRNNModel_metadata.json"
        loaded_metadata = saver._load_metadata(metadata_path)
        print(f"   ✓ Metadata loaded successfully")
        print(f"   Model type: {loaded_metadata.get('model_type', 'Unknown')}")
        print(f"   Dataset rows: {loaded_metadata.get('dataset_info', {}).get('rows', 'Unknown')}")
        
        # Test scaler functionality
        print("\n🧪 Testing loaded scaler...")
        test_data = np.random.randn(5, 4)
        scaled_data = loaded_scaler.transform(test_data)
        print(f"   ✓ Scaler transformation successful")
        print(f"   Original data shape: {test_data.shape}")
        print(f"   Scaled data shape: {scaled_data.shape}")
        print(f"   Scaled data mean: {scaled_data.mean():.6f} (should be ~0)")
        print(f"   Scaled data std: {scaled_data.std():.6f} (should be ~1)")
        
    except Exception as e:
        print(f"   ❌ Error loading artifacts: {e}")


def cleanup_demo_artifacts():
    """Clean up demonstration artifacts."""
    print("\n🧹 Cleaning up demo artifacts...")
    
    saver = ModelArtifactSaver(base_artifacts_dir="demo_artifacts")
    
    # Delete the demo model
    try:
        result = saver.delete_model_artifacts("DemoRNNModel")
        if result:
            print("   ✓ Demo artifacts cleaned up successfully")
        else:
            print("   ⚠️ Some artifacts may remain")
    except Exception as e:
        print(f"   ❌ Error during cleanup: {e}")
    
    # Remove demo directory if empty
    try:
        demo_dir = Path("demo_artifacts")
        if demo_dir.exists() and not any(demo_dir.iterdir()):
            demo_dir.rmdir()
            print("   ✓ Demo directory removed")
    except Exception as e:
        print(f"   ⚠️ Could not remove demo directory: {e}")


def main():
    """Run the complete demonstration."""
    print("ModelArtifactSaver Complete Demonstration")
    print("=" * 60)
    
    try:
        # Demonstrate saving
        saved_dir = demonstrate_artifact_saving()
        
        # Demonstrate loading
        demonstrate_artifact_loading(saved_dir)
        
        # Show final status
        print("\n✅ Demonstration completed successfully!")
        print("\nKey features demonstrated:")
        print("  • Saving trained models, scalers, and metadata")
        print("  • Organized directory structure with timestamps")
        print("  • JSON serialization of complex metadata")
        print("  • Artifact integrity validation")
        print("  • Loading and using saved artifacts")
        print("  • Comprehensive artifact management")
        
    except Exception as e:
        print(f"\n❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Always try to clean up
        cleanup_demo_artifacts()


if __name__ == "__main__":
    main()