"""
Integration tests for DataLoader and DataPreprocessor with real data.
"""

import unittest
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_loader import DataLoader
from data_preprocessor import DataPreprocessor


class TestDataLoadingIntegration(unittest.TestCase):
    """Integration test cases for data loading and preprocessing."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.loader = DataLoader()
        self.preprocessor = DataPreprocessor()
        self.real_file_path = os.path.join(
            os.path.dirname(__file__), '..', 'Data', 'covariaterawdata1.csv'
        )
    
    def test_full_data_pipeline(self):
        """Test complete data loading and preprocessing pipeline."""
        if not os.path.exists(self.real_file_path):
            self.skipTest("Real data file not found")
        
        # Load data
        raw_df = self.loader.load_data(self.real_file_path)
        
        # Preprocess data
        processed_df = self.preprocessor.preprocess_data(raw_df)
        
        # Verify pipeline results
        self.assertIsNotNone(processed_df)
        self.assertGreater(len(processed_df), 0)
        
        # Check that symbol column is removed
        self.assertNotIn('symbol', processed_df.columns)
        
        # Check that adjusted_close exists
        self.assertIn('adjusted_close', processed_df.columns)
        
        # Check data integrity
        self.assertTrue(processed_df.index.is_monotonic_increasing)
        
        # Check all columns are numeric
        for col in processed_df.columns:
            self.assertTrue(processed_df[col].dtype.kind in 'biufc')


if __name__ == '__main__':
    unittest.main()