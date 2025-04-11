import pandas as pd

def load_data(file_path):
    """
    Load data from a CSV file into a DataFrame.
    """
    try:
        print(f"Attempting to load data from: {file_path}")
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path)
        print("Data loaded successfully.")
        return df
    except FileNotFoundError:
        # Handle case where the file does not exist
        print(f"Error: Data file not found at {file_path}")
        raise
    except Exception as e:
        # Handle other errors during file loading
        print(f"Error loading data from {file_path}: {e}")
        raise