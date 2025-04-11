import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

def train_linear_regression(X_train, y_train):
    """
    Train a Linear Regression model.
    """
    print("Initializing and training Linear Regression model...")
    # Create and train the Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("Linear Regression training complete.")
    return model

def train_decision_tree(X_train, y_train, max_depth, max_features, random_state):
    """
    Train a Decision Tree Regressor model.
    """
    print(f"Initializing and training Decision Tree model (max_depth={max_depth}, max_features={max_features}, random_state={random_state})...")
    # Create and train the Decision Tree Regressor model
    model = DecisionTreeRegressor(
        max_depth=max_depth,
        max_features=max_features,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    print("Decision Tree training complete.")
    return model

def train_random_forest(X_train, y_train, n_estimators, criterion, random_state=None):
    """
    Train a Random Forest Regressor model.
    """
    print(f"Initializing and training Random Forest model (n_estimators={n_estimators}, criterion='{criterion}', random_state={random_state})...")
    # Create and train the Random Forest Regressor model
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        criterion=criterion,
        random_state=random_state,
        n_jobs=-1  # Use all available CPU cores
    )
    model.fit(X_train, y_train)
    print("Random Forest training complete.")
    return model