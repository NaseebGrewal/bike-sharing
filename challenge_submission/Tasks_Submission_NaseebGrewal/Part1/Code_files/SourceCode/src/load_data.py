"""
load_data.py
------------
This module contains the function to load the data from a CSV file.
"""
# import required libraries
import logging
import pandas as pd


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
