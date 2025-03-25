#!/usr/bin/env python3
"""
bike_sharing_analysis.py

This module performs an in-depth exploratory data analysis (EDA) on the Capital Bikeshare hourly dataset
and builds a RandomForestRegressor model to predict hourly bike utilization. The code is organized into
modular functions with unit tests for critical components.

Dataset Details:
- hour.csv contains 17,379 records with hourly aggregated bike rental counts.
- Data includes temporal, weather, and seasonal features.

Usage:
    python bike_sharing_analysis.py

Author: [Your Name]
Date: [Current Date]
"""

import os
import unittest
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the CSV data file into a pandas DataFrame.
    
    Parameters:
        filepath (str): The path to the CSV file.
        
    Returns:
        pd.DataFrame: The loaded dataset.
    """
    try:
        data = pd.read_csv(filepath)
        logging.info("Data loaded successfully from %s", filepath)
        return data
    except Exception as e:
        logging.error("Failed to load data: %s", e)
        raise

def perform_eda(data: pd.DataFrame, output_dir: str = "plots") -> None:
    """
    Conduct Exploratory Data Analysis (EDA) on the dataset and save plots to the specified output directory.
    
    Parameters:
        data (pd.DataFrame): The dataset for analysis.
        output_dir (str): Directory where plots will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Distribution of total counts (cnt)
    plt.figure(figsize=(8, 5))
    sns.histplot(data['cnt'], kde=True, bins=50)
    plt.title('Distribution of Total Bike Rental Count')
    plt.xlabel('Total Count (cnt)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cnt_distribution.png'))
    plt.close()
    
    # Time-series plot: average count by hour of day
    plt.figure(figsize=(8, 5))
    hourly_avg = data.groupby('hr')['cnt'].mean().reset_index()
    sns.lineplot(x='hr', y='cnt', data=hourly_avg, marker='o')
    plt.title('Average Bike Rentals by Hour of Day')
    plt.xlabel('Hour of Day')
    plt.ylabel('Average Count')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'hourly_avg.png'))
    plt.close()
    
    # Box plot by weather situation
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='weathersit', y='cnt', data=data)
    plt.title('Bike Rentals by Weather Situation')
    plt.xlabel('Weather Situation')
    plt.ylabel('Total Count (cnt)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cnt_by_weathersit.png'))
    plt.close()
    
    # Correlation heatmap
    plt.figure(figsize=(10, 8))
    corr = data[['temp', 'atemp', 'hum', 'windspeed', 'cnt']].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'))
    plt.close()
    
    logging.info("EDA plots saved in the '%s' directory.", output_dir)

def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the dataset by selecting features and handling any necessary transformations.
    
    Parameters:
        data (pd.DataFrame): The raw dataset.
        
    Returns:
        pd.DataFrame: The preprocessed dataset ready for model training.
    """
    # Drop columns that are not used in prediction
    features = ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed']
    target = 'cnt'
    
    # Additional transformations can be added here (e.g., one-hot encoding categorical variables)
    df = data[features + [target]].copy()
    
    # Convert categorical features to type 'category'
    categorical_cols = ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit']
    for col in categorical_cols:
        df[col] = df[col].astype('category')
    
    logging.info("Data preprocessing completed.")
    return df

def train_model(data: pd.DataFrame, random_state: int = 42) -> (RandomForestRegressor, float):
    """
    Train a RandomForestRegressor model to predict hourly bike utilization.
    
    Parameters:
        data (pd.DataFrame): The preprocessed dataset.
        random_state (int): Seed for reproducibility.
    
    Returns:
        model (RandomForestRegressor): The trained model.
        mad (float): Mean Absolute Deviation on the test set.
    """
    # Separate features and target
    X = data.drop(columns='cnt')
    y = data['cnt']
    
    # For simplicity, perform a train-test split (80-20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    
    # One-hot encode categorical features
    X_train_enc = pd.get_dummies(X_train, drop_first=True)
    X_test_enc = pd.get_dummies(X_test, drop_first=True)
    # Align columns in case of missing dummy variables in test set
    X_test_enc = X_test_enc.reindex(columns=X_train_enc.columns, fill_value=0)
    
    model = RandomForestRegressor(n_estimators=100, random_state=random_state)
    model.fit(X_train_enc, y_train)
    
    predictions = model.predict(X_test_enc)
    mad = mean_absolute_error(y_test, predictions)
    
    logging.info("Model trained successfully. MAD: %.2f", mad)
    return model, mad

# --- Unit Tests ---
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

def main():
    # File path configuration (adjust the path as needed)
    data_filepath = "hour.csv"  # Ensure this file is in your working directory
    
    # Load the dataset
    data = load_data(data_filepath)
    
    # Conduct EDA and save plots
    perform_eda(data, output_dir="plots")
    
    # Preprocess the data
    processed_data = preprocess_data(data)
    
    # Train the prediction model and evaluate using MAD
    model, mad = train_model(processed_data)
    
    # Print final evaluation metric
    print(f"Mean Absolute Deviation (MAD) of the model: {mad:.2f}")
    
    # (Optional) Save the trained model if needed for production use.
    # For example: using joblib: joblib.dump(model, 'bike_model.pkl')

if __name__ == "__main__":
    # Run unit tests
    unittest.main(exit=False)
    
    # Run main analysis
    main()
