import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BikeRentalPredictor:
    def __init__(self, random_state: int = 42):
        """
        Initialize the bike rental prediction model
        
        Args:
            random_state (int): Seed for reproducibility
        """
        self.random_state = random_state
        self.model = RandomForestRegressor(
            n_estimators=100, 
            random_state=random_state,
            max_depth=15,
            min_samples_split=10
        )
        self.scaler = StandardScaler()

    def prepare_data(self, data: pd.DataFrame, target_column: str):
        """
        Prepare data for model training
        
        Args:
            data (pd.DataFrame): Preprocessed DataFrame
            target_column (str): Name of the target variable
        
        Returns:
            Tuple of X and y
        """
        X = data.drop(columns=[target_column])
        y = data[target_column]
        return X, y

    def train_model(self, X: pd.DataFrame, y: pd.Series):
        """
        Train the Random Forest model
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target variable
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train model
        self.model.fit(X_train_scaled, y_train)

        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        
        mad = mean_absolute_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        
        logger.info(f"Model Performance:")
        logger.info(f"Mean Absolute Deviation (MAD): {mad:.2f}")
        logger.info(f"Mean Absolute Percentage Error (MAPE): {mape:.2%}")

        return {
            'mad': mad,
            'mape': mape,
            'test_size': len(X_test)
        }

    def feature_importance(self, feature_names: list) -> pd.DataFrame:
        """
        Get feature importances from the trained model
        
        Args:
            feature_names (list): Names of features
        
        Returns:
            pd.DataFrame of feature importances
        """
        importances = self.model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)

        return feature_importance_df

def main():
    # Example usage
    from data_loader import BikeShareDataLoader

    loader = BikeShareDataLoader('data/hour.csv')
    processed_data, metadata = loader.preprocess_data()

    predictor = BikeRentalPredictor()
    X, y = predictor.prepare_data(processed_data, metadata['target_column'])
    
    results = predictor.train_model(X, y)
    feature_importance = predictor.feature_importance(X.columns)
    
    print("\nFeature Importance:")
    print(feature_importance.head(10))

if __name__ == "__main__":
    main()