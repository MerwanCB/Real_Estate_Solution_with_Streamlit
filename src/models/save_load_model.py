import pickle
import os

def save_model(model, filepath):
    """
    Save a trained model to a file using pickle.

    Args:
        model: The trained model to save.
        filepath: Path where the model should be saved.
    """
    try:
        # Ensure the directory exists
        dir_name = os.path.dirname(filepath)
        if dir_name:  # Create directory only if filepath includes a directory part
            os.makedirs(dir_name, exist_ok=True)

        print(f"Saving model to {filepath}...")
        # Save the model to the specified file
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        print("Model successfully saved.")
    except Exception as e:
        # Handle errors during saving
        print(f"Error saving model to {filepath}: {e}")
        raise

def load_model(filepath):
    """
    Load a model from a pickle file.

    Args:
        filepath: Path to the saved model file.

    Returns:
        The loaded model.
    """
    try:
        print(f"Loading model from {filepath}...")
        # Check if the file exists
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found at {filepath}")
        # Load the model from the specified file
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        print("Model successfully loaded.")
        return model
    except FileNotFoundError:
        # Handle file not found error
        raise
    except Exception as e:
        # Handle other errors during loading
        print(f"Error loading model from {filepath}: {e}")
        raise