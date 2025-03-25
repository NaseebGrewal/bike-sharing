"""
model.py
--------
This module contains the function to train a RandomForestRegressor model to predict hourly bike utilization.
"""

import logging
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

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

    model_columns = X_train_enc.columns
    # Align columns in case of missing dummy variables in test set
    X_test_enc = X_test_enc.reindex(columns=X_train_enc.columns, fill_value=0)
    
    model = RandomForestRegressor(n_estimators=100, random_state=random_state)
    model.fit(X_train_enc, y_train)
    
    predictions = model.predict(X_test_enc)
    mad = mean_absolute_error(y_test, predictions)
    
    logging.info("Model trained successfully. MAD: %.2f", mad)
    return model, mad, model_columns