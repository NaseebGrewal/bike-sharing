# Bike Sharing Data Analysis and Prediction


## Bike Sharing Project

## Overview
This project analyzes bike-sharing data (from the Capital Bikeshare dataset) to compute key statistics and build predictive models. It demonstrates how to preprocess data, perform exploratory data analysis (EDA), and, optionally, train models. The code is organized into clear, modular components and includes unit tests to ensure reliability.

## Project Structure
```
Bike-sharing-project
|___SourceCode/
|   ├── data/
|   │   ├── hour.csv             # Hourly bike-sharing data
|   │   └── day.csv              # Daily bike-sharing data (if needed)
|   ├── extra/
|   │   ├── analyse.py           # calcute avg of cnt column 
|   │   └── model_comparison.py  # compared results of different models (like: RandomForest, Linear Regression, Gradient Boosting, SVR) 
|   ├── src/
|   │   ├── notebooks/         
|   │   ├     └── tests.ipynb    # Jupyter notebook for testing functions
|   │   ├── load_data.py         # function that reads the CSV file
|   │   ├── main.py              # Contains the get_average_cnt function (and other analysis code)
|   │   └── models.py            # Function that Trains the RandomForestRegressor and computes the Mean Absolute Deviation.
|   │   └── perform_eda.py       # Function that Performs exploratory data analysis and saves plots.
|   │   └── preprocess_data.py   # Function that Prepares and transforms the data for modeling.
|   ├── .gitignore               # Specifies files/directories to be ignored by Git
|   ├── bike+sharing+dataset.zip # raw data zip file 
|   ├── requirements.txt         # List of Python dependencies
├───.gitignore                   # Specifies files/directories to be ignored by Git
├───LICENSE                      # Project license 
|___README.md                    # Documentation and usage instructions
```

- **data/**: Contains the raw CSV files.
- **extra/**: Contains model comparison code
- **src/**: Holds all source code including data processing and model code, and unit test.
- **requirements.txt**: List all required packages (e.g., pandas, matplotlib, seaborn, scikit-learn).
- **.gitignore**: To exclude files (like virtual environment folders, logs, etc.) from version control.

---

## Getting Started

### Prerequisites
- Python 3.x installed on your machine
- pip (Python package manager)

### Installation
1. **Clone the Repository**
   ```bash
   git clone https://github.com/NaseebGrewal/bike-sharing.git
   cd bike_sharing_project
   ```
2. **Create and Activate a Virtual Environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate    # On Windows use: venv\Scripts\activate
   ```
3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Running the Project

### Execution
The main script loads data from the data/hour.csv, perform eda, preprocess the data, and trains the model
```bash
python src/main.py
```
The script will load data from the `data/` folder, perform the analysis, and output results to the console.

### Plots

<p align="center">
<img src="https://github.com/NaseebGrewal/bike-sharing/blob/main/SourceCode/src/plots/cnt_by_weathersit.png" alt="cnt_by_weathersit.png" width="45%" />
<img src="https://github.com/NaseebGrewal/bike-sharing/blob/main/SourceCode/src/plots/cnt_distribution.png" alt="cnt_distribution" width="45%" />
</p>

<p align="center">
<img src="https://github.com/NaseebGrewal/bike-sharing/blob/main/SourceCode/src/plots/correlation_heatmap.png" alt="correlation_heatmap.png" width="45%" />
<img src="https://github.com/NaseebGrewal/bike-sharing/blob/main/SourceCode/src/plots/hourly_avg.png" alt="hourly_avg.png" width="45%" />
</p>

## Contributing
New contributors are welcome! Please fork the repository, create a new branch for your feature or fix, and submit a pull request. Ensure that changes are well documented and tests pass before submitting your PR.

## License
This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.
