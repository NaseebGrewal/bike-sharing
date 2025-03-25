# --- Unit Tests ---
import unittest
import pandas as pd
from src.preprocess_data import preprocess_data
from src.load_data import load_data

class TestBikeSharingAnalysis(unittest.TestCase):
    """Unit tests for key functions in the bike sharing analysis module."""
    
    def setUp(self):
        # Create a small sample dataframe for testing
        self.sample_data = pd.DataFrame({
            'season': [1, 2],
            'yr': [0, 1],
            'mnth': [1, 2],
            'hr': [0, 1],
            'holiday': [0, 1],
            'weekday': [1, 2],
            'workingday': [1, 0],
            'weathersit': [1, 2],
            'temp': [0.5, 0.7],
            'atemp': [0.4, 0.6],
            'hum': [0.8, 0.9],
            'windspeed': [0.1, 0.2],
            'cnt': [100, 150]
        })
    
    def test_preprocess_data(self):
        processed = preprocess_data(self.sample_data)
        # Check if all expected columns are present
        expected_cols = ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday',
                         'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed', 'cnt']
        self.assertTrue(all(col in processed.columns for col in expected_cols))
        # Check type conversion
        self.assertTrue(pd.api.types.is_categorical_dtype(processed['season']))
    
    def test_load_data_failure(self):
        with self.assertRaises(Exception):
            # Attempt to load a non-existent file
            load_data("non_existent_file.csv")


if __name__ == '__main__':
    unittest.main()