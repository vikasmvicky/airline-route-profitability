import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

from preprocess import run_preprocessing

# ---------------------------------
# Configuration
# ---------------------------------

RANDOM_STATE = 42

TEST_SIZE = 0.20

RF_PARAMS = {
    "n_estimators": 100,
    "max_depth": 12,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "random_state": RANDOM_STATE,
    "n_jobs": -1
}

MODEL_PATH = "../models/rf_model.pkl"

LR_MODEL_PATH = "../models/lr_model.pkl"

FEATURE_PATH = "../models/feature_names.pkl"

FIGURE_PATH = "../reports/figures/"

# ---------------------------------
# Split Dataset
# ---------------------------------

def split_dataset(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    print("\nDataset Split Completed")

    print(f"Training Rows : {len(X_train)}")

    print(f"Testing Rows  : {len(X_test)}")

    return X_train, X_test, y_train, y_test

# ---------------------------------
# Evaluation Function
# ---------------------------------

def evaluate_model(model, X_test, y_test, model_name):

    predictions = model.predict(X_test)

    rmse = np.sqrt(
        mean_squared_error(y_test, predictions)
    )

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    print(f"\n{model_name} Results")

    print("-" * 40)

    print(f"RMSE : {rmse:.2f}")

    print(f"MAE  : {mae:.2f}")

    print(f"R²   : {r2:.4f}")

    return predictions, rmse, mae, r2

# ---------------------------------
# Train Linear Regression
# ---------------------------------

def train_linear_regression(
    X_train,
    X_test,
    y_train,
    y_test
):

    print("\nTraining Linear Regression...")

    lr_model = LinearRegression()

    lr_model.fit(X_train, y_train)

    print("Linear Regression Training Completed")

    results = evaluate_model(
        lr_model,
        X_test,
        y_test,
        "Linear Regression"
    )

    # Save model
    joblib.dump(
        lr_model,
        LR_MODEL_PATH
    )

    print("Linear Regression Model Saved")

    return lr_model, results

# ---------------------------------
# Train Random Forest
# ---------------------------------

def train_random_forest(
    X_train,
    X_test,
    y_train,
    y_test
):

    print("\nTraining Random Forest...")

    rf_model = RandomForestRegressor(
        **RF_PARAMS
    )

    rf_model.fit(X_train, y_train)

    print("Random Forest Training Completed")

    results = evaluate_model(
        rf_model,
        X_test,
        y_test,
        "Random Forest"
    )

    # Overfitting Check
    train_score = rf_model.score(
        X_train,
        y_train
    )

    test_score = rf_model.score(
        X_test,
        y_test
    )

    print("\nOverfitting Check")

    print("-" * 40)

    print(f"Train R² : {train_score:.4f}")

    print(f"Test R²  : {test_score:.4f}")

    if train_score - test_score > 0.10:

        print("Warning: Possible Overfitting")

    else:

        print("Model Generalization Looks Good")

    # Save model
    joblib.dump(
        rf_model,
        MODEL_PATH
    )

    print("Random Forest Model Saved")

    return rf_model, results

# ---------------------------------
# Compare Models
# ---------------------------------

def compare_models(
    lr_results,
    rf_results
):

    print("\nModel Comparison")

    print("=" * 50)

    comparison = pd.DataFrame({

        "Model": [
            "Linear Regression",
            "Random Forest"
        ],

        "RMSE": [
            lr_results[1],
            rf_results[1]
        ],

        "MAE": [
            lr_results[2],
            rf_results[2]
        ],

        "R2 Score": [
            lr_results[3],
            rf_results[3]
        ]
    })

    print(comparison)

    return comparison

# ---------------------------------
# Feature Importance Plot
# ---------------------------------

def plot_feature_importance(
    model,
    feature_names
):

    importances = pd.Series(
        model.feature_importances_,
        index=feature_names
    )

    importances = importances.sort_values(
        ascending=False
    ).head(15)

    plt.figure(figsize=(10, 6))

    sns.barplot(
        x=importances.values,
        y=importances.index
    )

    plt.title("Top 15 Important Features")

    plt.xlabel("Importance Score")

    plt.ylabel("Features")

    plt.tight_layout()

    os.makedirs(
        FIGURE_PATH,
        exist_ok=True
    )

    plt.savefig(
        f"{FIGURE_PATH}/feature_importance.png"
    )

    plt.show()

# ---------------------------------
# Actual vs Predicted Plot
# ---------------------------------

def plot_predictions(
    y_test,
    predictions
):

    plt.figure(figsize=(8, 6))

    plt.scatter(
        y_test,
        predictions,
        alpha=0.6
    )

    plt.xlabel("Actual Profit")

    plt.ylabel("Predicted Profit")

    plt.title("Actual vs Predicted Profit")

    plt.tight_layout()

    plt.savefig(
        f"{FIGURE_PATH}/actual_vs_predicted.png"
    )

    plt.show()

# ---------------------------------
# Main Training Pipeline
# ---------------------------------

def run_training_pipeline():

    print("\nStarting Training Pipeline")

    # Load processed data
    X, y, encoders = run_preprocessing()

    # Save feature names
    joblib.dump(
        X.columns.tolist(),
        FEATURE_PATH
    )

    print("Feature names saved")

    # Split dataset
    X_train, X_test, y_train, y_test = split_dataset(
        X,
        y
    )

    # Train models
    lr_model, lr_results = train_linear_regression(
        X_train,
        X_test,
        y_train,
        y_test
    )

    rf_model, rf_results = train_random_forest(
        X_train,
        X_test,
        y_train,
        y_test
    )

    # Compare models
    compare_models(
        lr_results,
        rf_results
    )

    # Feature importance
    plot_feature_importance(
        rf_model,
        X.columns
    )

    # Prediction plot
    plot_predictions(
        y_test,
        rf_results[0]
    )

    print("\nTraining Pipeline Completed")

# ---------------------------------
# Run File
# ---------------------------------

if __name__ == "__main__":

    run_training_pipeline()