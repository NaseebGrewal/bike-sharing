import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

class BikeShareEDA:
    def __init__(self, filepath):
        """
        Initialize EDA for bike sharing dataset
        
        Args:
            filepath (str): Path to the hour.csv file
        """
        self.data = pd.read_csv(filepath)
        self.preprocessed_data = None

    def preprocess_for_visualization(self):
        """
        Preprocess data for visualization purposes
        Converts categorical encodings to more readable labels
        """
        data = self.data.copy()
        
        # Decode season
        season_map = {1: 'Spring', 2: 'Summer', 3: 'Fall', 4: 'Winter'}
        data['season_name'] = data['season'].map(season_map)
        
        # Decode weather situation
        weather_map = {
            1: 'Clear/Partly Cloudy', 
            2: 'Misty/Cloudy', 
            3: 'Light Snow/Rain', 
            4: 'Heavy Rain/Snow'
        }
        data['weather_desc'] = data['weathersit'].map(weather_map)
        
        self.preprocessed_data = data

    def hourly_rental_distribution(self):
        """
        Plot distribution of bike rentals by hour of the day
        """
        plt.figure(figsize=(12, 6))
        self.preprocessed_data.groupby('hr')['cnt'].mean().plot(kind='bar')
        plt.title('Average Bike Rentals by Hour of Day')
        plt.xlabel('Hour of Day')
        plt.ylabel('Average Number of Rentals')
        plt.tight_layout()
        plt.show()

    def seasonal_rental_analysis(self):
        """
        Analyze bike rentals across different seasons
        """
        plt.figure(figsize=(12, 6))
        seasonal_rentals = self.preprocessed_data.groupby(['season_name', 'yr'])['cnt'].mean().unstack()
        seasonal_rentals.plot(kind='bar', stacked=False)
        plt.title('Average Bike Rentals by Season and Year')
        plt.xlabel('Season')
        plt.ylabel('Average Number of Rentals')
        plt.legend(title='Year', labels=['2011', '2012'])
        plt.tight_layout()
        plt.show()

    def weather_impact_analysis(self):
        """
        Visualize how weather impacts bike rentals
        """
        plt.figure(figsize=(12, 6))
        weather_rentals = self.preprocessed_data.groupby('weather_desc')['cnt'].mean()
        weather_rentals.plot(kind='bar')
        plt.title('Average Bike Rentals by Weather Condition')
        plt.xlabel('Weather Description')
        plt.ylabel('Average Number of Rentals')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def correlation_heatmap(self):
        """
        Create correlation heatmap of numerical features
        """
        numerical_cols = ['temp', 'atemp', 'hum', 'windspeed', 'cnt']
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.preprocessed_data[numerical_cols].corr(), 
                    annot=True, 
                    cmap='coolwarm', 
                    center=0)
        plt.title('Correlation Heatmap of Numerical Features')
        plt.tight_layout()
        plt.show()

def main():
    eda = BikeShareEDA('data/hour.csv')
    eda.preprocess_for_visualization()
    
    # Generate visualizations
    eda.hourly_rental_distribution()
    eda.seasonal_rental_analysis()
    eda.weather_impact_analysis()
    eda.correlation_heatmap()

if __name__ == "__main__":
    main()