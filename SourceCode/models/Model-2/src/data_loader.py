import pandas as pd
import numpy as np
from typing import Tuple, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BikeShareDataLoader:
    def __init__(self, filepath: str):
        """
        Initialize the data loader with the filepath
        
        Args:
            filepath (str): Path to the hour.csv file
        """
        self.filepath = filepath
        self.raw_data = None
        self.processed_data = None

    def load_data(self) -> pd.DataFrame:
        """
        Load the bike sharing dataset
        
        Returns:
            pd.DataFrame: Raw dataset
        """
        try:
            self.raw_data = pd.read_csv(self.filepath)
            logger.info(f"Data loaded successfully. Shape: {self.raw_data.shape}")
            return self.raw_data
        except FileNotFoundError:
            logger.error(f"File not found: {self.filepath}")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def preprocess_data(self, 
                         drop_columns: list = ['instant', 'casual', 'registered'],
                         encoding_columns: list = ['season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit']) -> Tuple[pd.DataFrame, Dict]:
        """
        Preprocess the dataset by:
        1. Dropping unnecessary columns
        2. Encoding categorical variables
        3. Normalizing numerical features
        
        Args:
            drop_columns (list): Columns to drop
            encoding_columns (list): Columns to one-hot encode
        
        Returns:
            Tuple of processed DataFrame and metadata dictionary
        """
        if self.raw_data is None:
            self.load_data()

        # Create a copy to avoid modifying the original data
        df = self.raw_data.copy()

        # Drop specified columns
        df.drop(columns=drop_columns, inplace=True)

        # One-hot encode categorical columns
        df_encoded = pd.get_dummies(df, columns=encoding_columns)

        # Normalize numerical columns
        numerical_cols = ['temp', 'atemp', 'hum', 'windspeed']
        df_encoded[numerical_cols] = (df_encoded[numerical_cols] - df_encoded[numerical_cols].mean()) / df_encoded[numerical_cols].std()

        # Prepare metadata
        metadata = {
            'target_column': 'cnt',
            'numerical_columns': numerical_cols,
            'categorical_columns': [col for col in df_encoded.columns if col.startswith(tuple(encoding_columns))],
            'total_features': len(df_encoded.columns) - 1  # Excluding target
        }

        self.processed_data = df_encoded
        logger.info(f"Data preprocessed. New shape: {df_encoded.shape}")
        
        return df_encoded, metadata

def main():
    # Example usage
    loader = BikeShareDataLoader('data/hour.csv')
    processed_data, metadata = loader.preprocess_data()
    print(metadata)

if __name__ == "__main__":
    main()