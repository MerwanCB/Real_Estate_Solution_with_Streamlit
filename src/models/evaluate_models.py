import pandas as pd
from sklearn.metrics import mean_absolute_error

def calculate_mae(model, X, y_true):
    """
    Calculate the Mean Absolute Error (MAE) for model predictions.

    Args:
        model: Trained model with a predict method.
        X: Features for predictions.
        y_true: True target values.

    Returns:
        The calculated MAE.
    """
    print(f"Evaluating model ({type(model).__name__}) using MAE...")
    y_pred = model.predict(X)
    mae = mean_absolute_error(y_true, y_pred)
    print(f"MAE calculated: {mae:.4f}")
    return mae