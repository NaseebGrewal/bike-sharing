# Model Comparison file

"""
bike_sharing_model_comparison.py

This module performs data preprocessing, trains several regression models 
(RandomForest, Linear Regression, Gradient Boosting, SVR) on the Capital Bikeshare hourly dataset,
and compares their performance using Mean Absolute Deviation (MAD).

"""

import os
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the CSV data file into a pandas DataFrame.
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
    Conduct Exploratory Data Analysis (EDA) and save plots.
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
    
    # Time-series plot: average count by hour
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
    Preprocess the dataset by selecting features and handling necessary transformations.
    Excludes 'dteday', 'casual', and 'registered' to avoid target leakage and redundant information.
    """
    features = ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit', 
                'temp', 'atemp', 'hum', 'windspeed']
    target = 'cnt'
    
    df = data[features + [target]].copy()
    
    # Convert categorical features to type 'category'
    categorical_cols = ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit']
    for col in categorical_cols:
        df[col] = df[col].astype('category')
    
    logging.info("Data preprocessing completed.")
    return df

def train_and_evaluate_models(data: pd.DataFrame, random_state: int = 42) -> dict:
    """
    Train multiple regression models and evaluate them using Mean Absolute Deviation (MAD).
    
    Returns a dictionary with model names and their corresponding MAD.
    """
    # Separate features and target
    X = data.drop(columns='cnt')
    y = data['cnt']
    
    # One-hot encode categorical variables (drop_first to avoid dummy variable trap)
    X = pd.get_dummies(X, drop_first=True)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    
    # Define models to compare
    models = {
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=random_state),
        'LinearRegression': LinearRegression(),
        'GradientBoosting': GradientBoostingRegressor(random_state=random_state),
        'SVR': SVR()
    }
    
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        mad = mean_absolute_error(y_test, predictions)
        results[name] = mad
        logging.info("%s model trained successfully. MAD: %.2f", name, mad)
    
    return results

def main():
    # File path configuration
    data_filepath = "../data/hour.csv"  # Ensure this file is in your working directory
    
    # Load data
    data = load_data(data_filepath)
    
    # Perform EDA
    perform_eda(data, output_dir="plots")
    
    # Preprocess data
    processed_data = preprocess_data(data)
    
    # Train and evaluate models
    results = train_and_evaluate_models(processed_data)
    
    # Print evaluation results for all models
    print("Model Performance (MAD):")
    for model_name, mad in results.items():
        print(f"{model_name}: {mad:.2f}")

if __name__ == "__main__":
    main()
