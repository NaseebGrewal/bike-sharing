"""
main.py
-------
This script is the main entry point for the analysis. It loads the data, conducts EDA, preprocesses the data, trains the model, and evaluates the model using Mean Absolute Deviation (MAD).
"""
from load_data import load_data
from perform_eda import perform_eda
from preprocess_data import preprocess_data
from model import train_model

def main():
    # File path configuration (adjust the path as needed)
    data_filepath = "../data/hour.csv"  # Ensure this file is in your working directory
    
    # Load the dataset
    data = load_data(data_filepath)
    
    # Conduct EDA and save plots
    perform_eda(data, output_dir="plots")
    
    # Preprocess the data
    processed_data = preprocess_data(data)
    
    # Train the prediction model and evaluate using MAD
    model, mad, model_columns = train_model(processed_data)
    
    
    # Print final evaluation metric
    print(f"Mean Absolute Deviation (MAD) of the model: {mad:.2f}")
    
    # (Optional) Save the trained model if needed for production use.
    # For example: using joblib: joblib.dump(model, 'bike_model.pkl')
    return model, model_columns


if __name__ == "__main__":
    
    # Run main analysis
    model, model_columns = main()