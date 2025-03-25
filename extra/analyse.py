"""
This module provides a function to compute the average of the 'cnt' column from the bike-sharing dataset.
"""
import logging
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def get_average_cnt(filepath: str) -> float:
    """
    Calculate the average of the 'cnt' column from a CSV file.

    Parameters:
        filepath (str): Path to the CSV file.

    Returns:
        float: The average value of the 'cnt' column.
    """
    try:
        data = pd.read_csv(filepath)
        if 'cnt' not in data.columns:
            raise KeyError("Column 'cnt' not found in the dataset.")
        avg_cnt = data['cnt'].mean()
        logging.info("Computed average of 'cnt': %.2f", avg_cnt)
        return avg_cnt
    except Exception as e:
        logging.error("Error computing average of 'cnt': %s", e)
        raise

if __name__ == "__main__":
    # Define the path to the CSV file
    # In production, this could be set via a config or environment variable
    # data_filepath = os.path.join("data", "hour.csv")
    data_filepath = "data/hour.csv"
    
    avg = get_average_cnt(data_filepath)
    print(f"Average 'cnt': {avg:.2f}")
