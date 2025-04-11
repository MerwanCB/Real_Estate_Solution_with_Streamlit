import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(X, y, test_size, stratify_feature_series=None, random_state=None):
    """
    Split data into training and testing sets.
    """
    # Log the parameters used for splitting
    print(
        f"Splitting data with test_size={test_size}, random_state={random_state}, stratify={'Yes' if stratify_feature_series is not None else 'No'}"
    )

    # Validate the stratify series length if provided
    if stratify_feature_series is not None and len(stratify_feature_series) != len(y):
        raise ValueError("Length of stratify series must match length of X and y.")

    # Perform the train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_feature_series,  # Use stratification if provided
    )
    print("Data splitting complete.")
    return X_train, X_test, y_train, y_test
