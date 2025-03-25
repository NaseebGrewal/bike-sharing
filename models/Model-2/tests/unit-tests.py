import unittest
import pandas as pd
import numpy as np
from src.data_loader import BikeShareDataLoader
from src.model import BikeRentalPredictor

class TestBikeShareDataLoader(unittest.TestCase):
    def setUp(self):
        self.loader = BikeShareDataLoader('data/hour.csv')
        self.loader.load_data()

    def test_data_loading(self):
        """Test that data is loaded correctly"""
        self.assertIsNotNone(self.loader.raw_data)
        self.assertTrue(len(self.loader.raw_data) > 0)

    def test_preprocessing(self):
        """Test data preprocessing steps"""
        processed_data, metadata = self.loader.preprocess_data()
        
        self.assertIsNotNone(processed_data)
        self.assertIn('cnt', processed_data.columns)
        self.assertTrue('total_features' in metadata)
        self.assertTrue(metadata['total_features'] > 0)

class TestBikeRentalPredictor(unittest.TestCase):
    def setUp(self):
        loader = BikeShareDataLoader('data/hour.csv')
        self.processed_data, self.metadata = loader.preprocess_data()
        self.predictor = BikeRentalPredictor()

    def test_data_preparation(self):
        """Test data preparation for model training"""
        X, y = self.predictor.prepare_data(
            self.processed_data, 
            self.metadata['target_column']
        )
        
        self.assertEqual(len(X) + 1, len(self.processed_data))
        self.assertEqual(len(y), len(self.processed_data))
        self.assertFalse(self.metadata['target_column'] in X.columns)

    def test_model_training(self):
        """Test model training and performance"""
        X, y = self.predictor.prepare_data(
            self.processed_data, 
            self.metadata['target_column']
        )
        
        results = self.predictor.train_model(X, y)
        
        self.assertIn('mad', results)
        self.assertIn('mape', results)
        self.assertTrue(results['mad'] >= 0)
        self.assertTrue(results['mape'] >= 0)

if __name__ == '__main__':
    unittest.main()