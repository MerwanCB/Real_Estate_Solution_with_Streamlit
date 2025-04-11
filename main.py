import os
import pickle
import pandas as pd
import numpy as np


# Custom module imports
from src.data.load_data import load_data
from src.features.prepare_features import prepare_features
from src.models.split_data import split_data
from src.models.train_models import (
    train_linear_regression,
    train_decision_tree,
    train_random_forest,
)
from src.models.evaluate_models import calculate_mae
from src.models.save_load_model import save_model, load_model
from src.visualization.visualize import plot_and_save_tree

# Define constants for paths and parameters
RAW_DATA_PATH = "data/raw/final.csv"
MODEL_SAVE_DIR = "models"
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, "RE_Model.pkl")
TREE_PLOT_PATH = "reports/figures/decision_tree.png"
PROCESSED_DATA_DIR = "data/processed"

# Create directories if they don't exist
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(os.path.dirname(TREE_PLOT_PATH), exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)


if __name__ == "__main__":
    # 1. Load Data
    print("Loading data...")
    df = load_data(RAW_DATA_PATH)
    print(f"Data loaded with shape: {df.shape}")

    # 2. Prepare Features and Target
    print("Preparing features and target...")
    target_column = "price"
    X, y = prepare_features(df, target_column)
    print(f"Features shape: {X.shape}, Target shape: {y.shape}")

    # 3. Split Data
    print("Splitting data into train and test sets...")
    stratify_column_name = "property_type_Bunglow"
    if stratify_column_name not in X.columns:
        raise ValueError(
            f"Stratify column '{stratify_column_name}' not found in features."
        )

    X_train, X_test, y_train, y_test = split_data(
        X,
        y,
        test_size=0.2,
        stratify_feature_series=X[stratify_column_name],
        random_state=None,
    )
    print(f"Train set shapes: X={X_train.shape}, y={y_train.shape}")
    print(f"Test set shapes: X={X_test.shape}, y={y_test.shape}")

    # 4. Train and Evaluate Linear Regression
    print("\nTraining Linear Regression...")
    lr_model = train_linear_regression(X_train, y_train)
    print(f"LR Coefficients: {lr_model.coef_}")
    print(f"LR Intercept: {lr_model.intercept_}")

    lr_train_pred = lr_model.predict(X_train)
    lr_train_mae = calculate_mae(lr_model, X_train, y_train)
    print(f"Linear Regression Train MAE: {lr_train_mae:.2f}")

    lr_test_mae = calculate_mae(lr_model, X_test, y_test)
    print(f"Linear Regression Test MAE: {lr_test_mae:.2f}")

    # 5. Train and Evaluate Decision Tree
    print("\nTraining Decision Tree...")
    dt_model = train_decision_tree(
        X_train, y_train, max_depth=3, max_features=10, random_state=567
    )

    dt_train_mae = calculate_mae(dt_model, X_train, y_train)
    print(f"Decision Tree Train MAE: {dt_train_mae:.2f}")

    dt_test_mae = calculate_mae(dt_model, X_test, y_test)
    print(f"Decision Tree Test MAE: {dt_test_mae:.2f}")

    # 6. Visualize Decision Tree
    print(f"Plotting and saving Decision Tree to {TREE_PLOT_PATH}...")
    plot_and_save_tree(
        dt_model, feature_names=X_train.columns.tolist(), output_path=TREE_PLOT_PATH
    )

    # 7. Train and Evaluate Random Forest
    print("\nTraining Random Forest...")
    rf_model = train_random_forest(
        X_train,
        y_train,
        n_estimators=200,
        criterion="absolute_error",
        random_state=None,
    )

    rf_train_mae = calculate_mae(rf_model, X_train, y_train)
    print(        f"Random Forest Train MAE: {rf_train_mae:.2f}")

    rf_test_mae = calculate_mae(rf_model, X_test, y_test)
    print(f"Random Forest Test MAE: {rf_test_mae:.2f}")

    # 8. Save the Decision Tree Model
    print(f"\nSaving Decision Tree model to {MODEL_SAVE_PATH}...")
    save_model(dt_model, MODEL_SAVE_PATH)
    print("Model saved.")

    # 9. Load and Test Saved Model (Example)
    print(f"Loading model from {MODEL_SAVE_PATH} for a test prediction...")
    loaded_dt_model = load_model(MODEL_SAVE_PATH)
    print("Model loaded.")

    # Example
    if not X_test.empty:
        sample_features = X_test.iloc[[5]]
        prediction = loaded_dt_model.predict(sample_features)
        actual = y_test.iloc[0]
        print(
            f"Example prediction on first test sample: {prediction[0]:.2f} (Actual: {actual:.2f})"
        )
    else:
        print("Test set is empty, skipping example prediction.")

    print("\nPipeline finished.")
