"""
preprocess_data.py
------------------
This module contains functions for preprocessing the dataset before model training.
"""

# Import required libraries
import pandas as pd
import logging


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
