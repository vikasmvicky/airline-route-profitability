import pandas as pd
import numpy as np
import joblib
import os

from sklearn.preprocessing import LabelEncoder

# ---------------------------------
# Configuration
# ---------------------------------

RAW_DATA_PATH = "../data/raw/airline_route_profitability.csv"

PROCESSED_DATA_PATH = (
    "../data/processed/cleaned_airline_data.csv"
)

ENCODER_PATH = "../models/label_encoders.pkl"

TARGET_COLUMN = "Profit"

# Categorical columns
CATEGORICAL_COLUMNS = [
    "Destination",
    "Aircraft_Type",
    "Season",
    "Route_Category",
    "Demand_Level"
]

# Columns not used for training
DROP_COLUMNS = [
    "Flight_Number",
    "Flight_Date",
    "Route",
    "Origin",
    "Profit_Margin",
    "Total_Revenue",
    "Total_Cost",
    "Profit"
]

# ---------------------------------
# Load Dataset
# ---------------------------------

def load_data(filepath=RAW_DATA_PATH):

    df = pd.read_csv(filepath)

    print("\nDataset Loaded Successfully")

    print(f"Rows    : {df.shape[0]}")

    print(f"Columns : {df.shape[1]}")

    return df

# ---------------------------------
# Handle Missing Values
# ---------------------------------

def clean_data(df):

    print("\nCleaning Data...")

    missing_cols = [
        "Ancillary_Revenue",
        "Catering_Cost",
        "Handling_Cost"
    ]

    for col in missing_cols:

        median_value = df[col].median()

        df[col] = df[col].fillna(median_value)

        print(f"{col} -> filled with median")

    # Remove duplicates
    df.drop_duplicates(inplace=True)

    print("Duplicate rows removed")

    return df

# ---------------------------------
# Feature Engineering
# ---------------------------------

def engineer_features(df):

    print("\nEngineering Features...")

    # Convert date column
    df["Flight_Date"] = pd.to_datetime(
        df["Flight_Date"]
    )

    # Revenue per passenger
    df["Revenue_Per_Passenger"] = (
        df["Ticket_Revenue"] /
        df["Passengers"].replace(0, np.nan)
    )

    # Cost per passenger
    df["Cost_Per_Passenger"] = (
        (
            df["Fuel_Cost"] +
            df["Maintenance_Cost"] +
            df["Crew_Cost"] +
            df["Airport_Fees"]
        ) /
        df["Passengers"].replace(0, np.nan)
    )

    # Fuel cost ratio
    df["Fuel_Cost_Ratio"] = (
        df["Fuel_Cost"] /
        (
            df["Fuel_Cost"] +
            df["Maintenance_Cost"] +
            df["Crew_Cost"] +
            df["Airport_Fees"]
        ).replace(0, np.nan)
    )

    # Crew cost ratio
    df["Crew_Cost_Ratio"] = (
        df["Crew_Cost"] /
        (
            df["Fuel_Cost"] +
            df["Maintenance_Cost"] +
            df["Crew_Cost"] +
            df["Airport_Fees"]
        ).replace(0, np.nan)
    )

    # Ancillary revenue ratio
    df["Ancillary_Revenue_Ratio"] = (
        df["Ancillary_Revenue"] /
        df["Ticket_Revenue"].replace(0, np.nan)
    )

    # Revenue efficiency
    df["Revenue_Per_Flight_Hour"] = (
        df["Ticket_Revenue"] /
        df["Flight_Hours"].replace(0, np.nan)
    )

    # Time-based features
    df["Month"] = (
        df["Flight_Date"].dt.month
    )

    df["Quarter"] = (
        df["Flight_Date"].dt.quarter
    )

    # Replace infinite values
    df.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True
    )

    # Fill remaining null values
    df.fillna(
        df.median(numeric_only=True),
        inplace=True
    )

    print("Feature engineering completed")

    return df

# ---------------------------------
# Encode Categorical Features
# ---------------------------------

def encode_categorical_columns(df):

    print("\nEncoding Categorical Features...")

    encoders = {}

    for col in CATEGORICAL_COLUMNS:

        le = LabelEncoder()

        df[col] = le.fit_transform(
            df[col].astype(str)
        )

        encoders[col] = le

        print(f"{col} encoded")

    # Save encoders
    os.makedirs("../models", exist_ok=True)

    joblib.dump(encoders, ENCODER_PATH)

    print("\nEncoders saved")

    return df, encoders

# ---------------------------------
# Prepare Final Dataset
# ---------------------------------

def prepare_dataset(df):

    print("\nPreparing Final Dataset...")

    X = df.drop(columns=DROP_COLUMNS)

    y = df[TARGET_COLUMN]

    print(f"Features shape : {X.shape}")

    print(f"Target shape   : {y.shape}")

    return X, y

# ---------------------------------
# Save Processed Dataset
# ---------------------------------

def save_processed_data(df):

    os.makedirs(
        "../data/processed",
        exist_ok=True
    )

    df.to_csv(
        PROCESSED_DATA_PATH,
        index=False
    )

    print("\nProcessed dataset saved")

# ---------------------------------
# Main Pipeline
# ---------------------------------

def run_preprocessing():

    df = load_data()

    df = clean_data(df)

    df = engineer_features(df)

    df, encoders = encode_categorical_columns(df)

    save_processed_data(df)

    X, y = prepare_dataset(df)

    return X, y, encoders

# ---------------------------------
# Run File
# ---------------------------------

if __name__ == "__main__":

    X, y, encoders = run_preprocessing()

    print("\nPreprocessing Pipeline Completed")