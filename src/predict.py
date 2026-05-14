import os
import pandas as pd
import numpy as np
import joblib

# ---------------------------------
# Dynamic Model Paths
# ---------------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODELS_PATH = os.path.join(
    BASE_DIR,
    "models"
)

MODEL_PATH = os.path.join(
    MODELS_PATH,
    "rf_model.pkl"
)

ENCODER_PATH = os.path.join(
    MODELS_PATH,
    "label_encoders.pkl"
)

FEATURE_PATH = os.path.join(
    MODELS_PATH,
    "feature_names.pkl"
)

# ---------------------------------
# Load Model and Encoders
# ---------------------------------

model = joblib.load(MODEL_PATH)

encoders = joblib.load(ENCODER_PATH)

feature_names = joblib.load(FEATURE_PATH)

# ---------------------------------
# Aircraft Specifications
# ---------------------------------

AIRCRAFT_SPECS = {

    "Airbus A320": {
        "capacity": 180
    },

    "Boeing 737-800": {
        "capacity": 162
    },

    "Boeing 787-9": {
        "capacity": 296
    },

    "Airbus A350-900": {
        "capacity": 369
    },

    "Boeing 777-300ER": {
        "capacity": 396
    },

    "Airbus A380": {
        "capacity": 555
    }
}

# ---------------------------------
# Route Category
# ---------------------------------

def get_route_category(flight_hours):

    if flight_hours < 3:
        return "Short Haul"

    elif flight_hours < 7:
        return "Medium Haul"

    else:
        return "Long Haul"

# ---------------------------------
# Prepare Input Data
# ---------------------------------

def prepare_input_data(
    destination,
    aircraft_type,
    passengers,
    flight_hours,
    season,
    demand_level,
    ticket_revenue,
    ancillary_revenue,
    fuel_cost,
    maintenance_cost,
    crew_cost,
    depreciation_cost,
    insurance_cost,
    airport_fees,
    catering_cost,
    handling_cost,
    navigation_fees,
    sales_distribution_cost,
    passenger_service_cost,
    overhead_cost,
    marketing_cost,
    it_systems_cost
):

    aircraft_capacity = (
        AIRCRAFT_SPECS[aircraft_type]["capacity"]
    )

    load_factor = (
        passengers / aircraft_capacity
    )

    route_category = get_route_category(
        flight_hours
    )

    # Engineered Features

    revenue_per_passenger = (
        ticket_revenue / max(passengers, 1)
    )

    cost_per_passenger = (
        (
            fuel_cost +
            maintenance_cost +
            crew_cost +
            airport_fees
        ) / max(passengers, 1)
    )

    fuel_cost_ratio = (
        fuel_cost /
        max(
            (
                fuel_cost +
                maintenance_cost +
                crew_cost +
                airport_fees
            ),
            1
        )
    )

    crew_cost_ratio = (
        crew_cost /
        max(
            (
                fuel_cost +
                maintenance_cost +
                crew_cost +
                airport_fees
            ),
            1
        )
    )

    ancillary_revenue_ratio = (
        ancillary_revenue /
        max(ticket_revenue, 1)
    )

    revenue_per_flight_hour = (
        ticket_revenue /
        max(flight_hours, 1)
    )

    input_df = pd.DataFrame([{

        "Destination": destination,
        "Aircraft_Type": aircraft_type,
        "Passengers": passengers,
        "Aircraft_Capacity": aircraft_capacity,
        "Load_Factor": load_factor,
        "Flight_Hours": flight_hours,
        "Season": season,
        "Route_Category": route_category,
        "Demand_Level": demand_level,

        "Ticket_Revenue": ticket_revenue,
        "Ancillary_Revenue": ancillary_revenue,

        "Fuel_Cost": fuel_cost,
        "Maintenance_Cost": maintenance_cost,
        "Crew_Cost": crew_cost,
        "Depreciation_Cost": depreciation_cost,
        "Insurance_Cost": insurance_cost,
        "Airport_Fees": airport_fees,
        "Catering_Cost": catering_cost,
        "Handling_Cost": handling_cost,
        "Navigation_Fees": navigation_fees,
        "Sales_Distribution_Cost": sales_distribution_cost,
        "Passenger_Service_Cost": passenger_service_cost,
        "Overhead_Cost": overhead_cost,
        "Marketing_Cost": marketing_cost,
        "IT_Systems_Cost": it_systems_cost,

        "Revenue_Per_Passenger":
            revenue_per_passenger,

        "Cost_Per_Passenger":
            cost_per_passenger,

        "Fuel_Cost_Ratio":
            fuel_cost_ratio,

        "Crew_Cost_Ratio":
            crew_cost_ratio,

        "Ancillary_Revenue_Ratio":
            ancillary_revenue_ratio,

        "Revenue_Per_Flight_Hour":
            revenue_per_flight_hour,

        "Month": 6,
        "Quarter": 2
    }])

    # Encode categorical columns

    categorical_cols = [
        "Destination",
        "Aircraft_Type",
        "Season",
        "Route_Category",
        "Demand_Level"
    ]

    for col in categorical_cols:

        input_df[col] = encoders[col].transform(
            input_df[col]
        )

    # Match training feature order

    input_df = input_df[feature_names]

    return input_df

# ---------------------------------
# Prediction Function
# ---------------------------------

def predict_profit(
    destination,
    aircraft_type,
    passengers,
    flight_hours,
    season,
    demand_level,
    ticket_revenue,
    ancillary_revenue,
    fuel_cost,
    maintenance_cost,
    crew_cost,
    depreciation_cost,
    insurance_cost,
    airport_fees,
    catering_cost,
    handling_cost,
    navigation_fees,
    sales_distribution_cost,
    passenger_service_cost,
    overhead_cost,
    marketing_cost,
    it_systems_cost,
    verbose=True
):

    input_df = prepare_input_data(
        destination,
        aircraft_type,
        passengers,
        flight_hours,
        season,
        demand_level,
        ticket_revenue,
        ancillary_revenue,
        fuel_cost,
        maintenance_cost,
        crew_cost,
        depreciation_cost,
        insurance_cost,
        airport_fees,
        catering_cost,
        handling_cost,
        navigation_fees,
        sales_distribution_cost,
        passenger_service_cost,
        overhead_cost,
        marketing_cost,
        it_systems_cost
    )

    predicted_profit = model.predict(
        input_df
    )[0]

    total_revenue = (
        ticket_revenue +
        ancillary_revenue
    )

    profit_margin = (
        predicted_profit /
        max(total_revenue, 1)
    ) * 100

    if predicted_profit > 250000:
        status = "Highly Profitable"

    elif predicted_profit > 100000:
        status = "Moderately Profitable"

    elif predicted_profit > 0:
        status = "Low Profit"

    else:
        status = "Loss Making"

    result = {

        "Predicted_Profit":
            round(predicted_profit, 2),

        "Profit_Margin_Pct":
            round(profit_margin, 2),

        "Status":
            status
    }

    if verbose:

        print("\nProfit Prediction")

        print("=" * 40)

        print(f"Aircraft Type : {aircraft_type}")

        print(f"Destination   : {destination}")

        print(
            f"Predicted Profit : "
            f"{predicted_profit:,.2f}"
        )

        print(
            f"Profit Margin    : "
            f"{profit_margin:.2f}%"
        )

        print(f"Business Status  : {status}")

    return result

# ---------------------------------
# Example Prediction
# ---------------------------------

if __name__ == "__main__":

    predict_profit(

        destination="LHR",

        aircraft_type="Airbus A380",

        passengers=480,

        flight_hours=7.5,

        season="Peak",

        demand_level="High",

        ticket_revenue=420000,

        ancillary_revenue=55000,

        fuel_cost=130000,

        maintenance_cost=42000,

        crew_cost=22000,

        depreciation_cost=18000,

        insurance_cost=6000,

        airport_fees=15000,

        catering_cost=12000,

        handling_cost=9000,

        navigation_fees=7000,

        sales_distribution_cost=11000,

        passenger_service_cost=10000,

        overhead_cost=15000,

        marketing_cost=8000,

        it_systems_cost=4000
    )