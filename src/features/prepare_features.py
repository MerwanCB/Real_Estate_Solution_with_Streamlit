import pandas as pd

def prepare_features(df, target_column):
    """
    Separate features and target from a DataFrame.
    """
    # Check if the target column exists in the DataFrame
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame columns: {df.columns.tolist()}")

    print(f"Separating features and target ('{target_column}')...")
    # Drop the target column to create the features DataFrame
    X = df.drop(target_column, axis=1)
    # Extract the target column as a Series
    y = df[target_column]
    print("Features and target separated.")
    return X, y