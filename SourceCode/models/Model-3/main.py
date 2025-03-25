import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import unittest

def load_data(filepath):
    """Loads and preprocesses the bike sharing data.

    Args:
        filepath (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded and preprocessed DataFrame.
    """
    df = pd.read_csv(filepath)
    df['dteday'] = pd.to_datetime(df['dteday'])
    return df

def perform_eda(df):
    """Performs exploratory data analysis on the bike sharing data.

    Args:
        df (pd.DataFrame): DataFrame to analyze.

    Returns:
        None (generates plots and prints summaries).
    """
    # Summary statistics
    print("Summary Statistics:\n", df.describe())

    # Correlation matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title("Correlation Matrix")
    plt.show()

    # Hourly usage patterns
    plt.figure(figsize=(15, 6))
    sns.lineplot(x='hr', y='cnt', data=df)
    plt.title("Hourly Bike Usage")
    plt.xlabel("Hour of the Day")
    plt.ylabel("Bike Count")
    plt.show()

    # Usage by day of the week
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='weekday', y='cnt', data=df)
    plt.title("Bike Usage by Day of Week")
    plt.xlabel("Day of Week")
    plt.ylabel("Bike Count")
    plt.show()

    # Usage by season
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='season', y='cnt', data=df)
    plt.title("Bike Usage by Season")
    plt.xlabel("Season")
    plt.ylabel("Bike Count")
    plt.show()

    # Usage by weather situation
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='weathersit', y='cnt', data=df)
    plt.title("Bike Usage by Weather Situation")
    plt.xlabel("Weather Situation")
    plt.ylabel("Bike Count")
    plt.show()
    
    #Usage by year
    plt.figure(figsize=(10,6))
    sns.boxplot(x='yr', y='cnt', data=df)
    plt.title("Bike Usage by Year")
    plt.xlabel("Year")
    plt.ylabel("Bike Count")
    plt.show()

    # Usage by month
    plt.figure(figsize=(15,6))
    sns.boxplot(x='mnth', y='cnt', data=df)
    plt.title("Bike Usage by Month")
    plt.xlabel("Month")
    plt.ylabel("Bike Count")
    plt.show()

    # Usage by holiday
    plt.figure(figsize=(8,6))
    sns.boxplot(x='holiday', y='cnt', data=df)
    plt.title("Bike Usage by Holiday")
    plt.xlabel("Holiday")
    plt.ylabel("Bike Count")
    plt.show()

    # Usage by workingday
    plt.figure(figsize=(8,6))
    sns.boxplot(x='workingday', y='cnt', data=df)
    plt.title("Bike Usage by Working Day")
    plt.xlabel("Working Day")
    plt.ylabel("Bike Count")
    plt.show()

def build_model(df):
    """Builds and evaluates a prediction model for hourly bike utilization.

    Args:
        df (pd.DataFrame): DataFrame containing the data.

    Returns:
        float: Mean absolute deviation (MAD) of the model.
    """
    features = ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed']
    target = 'cnt'

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mad = mean_absolute_error(y_test, y_pred)
    print(f"Mean Absolute Deviation (MAD): {mad}")
    return mad

class TestBikeSharing(unittest.TestCase):

    def test_load_data(self):
        df = load_data('hour.csv')
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIn('dteday', df.columns)

    def test_build_model(self):
        df = load_data('hour.csv')
        mad = build_model(df)
        self.assertIsInstance(mad, float)
        self.assertTrue(mad >= 0)

if __name__ == "__main__":
    df = load_data('hour.csv')
    perform_eda(df)
    build_model(df)
    unittest.main(argv=['first-arg-is-ignored'], exit=False)