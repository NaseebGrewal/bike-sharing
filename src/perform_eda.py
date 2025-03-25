
import os
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



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
    plt.show()
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
    plt.show()
    # plt.close()
    
    # Box plot by weather situation
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='weathersit', y='cnt', data=data)
    plt.title('Bike Rentals by Weather Situation')
    plt.xlabel('Weather Situation')
    plt.ylabel('Total Count (cnt)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cnt_by_weathersit.png'))
    plt.show()
    plt.close()
    
    # Correlation heatmap
    plt.figure(figsize=(10, 8))
    corr = data[['temp', 'atemp', 'hum', 'windspeed', 'cnt']].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'))
    plt.show()
    plt.close()
    
    logging.info("EDA plots saved in the '%s' directory.", output_dir)
