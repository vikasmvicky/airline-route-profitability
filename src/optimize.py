import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from predict import predict_profit

# ---------------------------------
# Aircraft Specifications
# ---------------------------------

AIRCRAFT_OPTIONS = {

    "Airbus A320": {
        "capacity": 180,
        "fuel_cost": 42000
    },

    "Boeing 737-800": {
        "capacity": 162,
        "fuel_cost": 39000
    },

    "Boeing 787-9": {
        "capacity": 296,
        "fuel_cost": 76000
    },

    "Airbus A350-900": {
        "capacity": 369,
        "fuel_cost": 88000
    },

    "Boeing 777-300ER": {
        "capacity": 396,
        "fuel_cost": 98000
    },

    "Airbus A380": {
        "capacity": 555,
        "fuel_cost": 130000
    }
}

# ---------------------------------
# Generate Aircraft Economics
# ---------------------------------

def generate_aircraft_inputs(
    aircraft_type,
    destination,
    flight_hours,
    season
):

    specs = AIRCRAFT_OPTIONS[aircraft_type]

    capacity = specs["capacity"]

    # Demand estimation

    if season == "Peak":
        load_factor = 0.88

    elif season == "Low":
        load_factor = 0.65

    else:
        load_factor = 0.78

    passengers = int(
        capacity * load_factor
    )

    # Revenue estimation

    avg_ticket_price = 850

    ticket_revenue = (
        passengers * avg_ticket_price
    )

    ancillary_revenue = (
        ticket_revenue * 0.12
    )

    # Cost estimation

    fuel_cost = specs["fuel_cost"]

    maintenance_cost = fuel_cost * 0.28

    crew_cost = (
        flight_hours * 2400
    )

    depreciation_cost = (
        fuel_cost * 0.15
    )

    insurance_cost = (
        fuel_cost * 0.05
    )

    airport_fees = (
        capacity * 32
    )

    catering_cost = (
        passengers * 18
    )

    handling_cost = (
        passengers * 14
    )

    navigation_fees = (
        flight_hours * 900
    )

    sales_distribution_cost = (
        ticket_revenue * 0.06
    )

    passenger_service_cost = (
        passengers * 12
    )

    overhead_cost = (
        fuel_cost * 0.10
    )

    marketing_cost = (
        ticket_revenue * 0.03
    )

    it_systems_cost = (
        passengers * 4
    )

    return {

        "destination": destination,

        "aircraft_type": aircraft_type,

        "passengers": passengers,

        "flight_hours": flight_hours,

        "season": season,

        "demand_level": "High",

        "ticket_revenue": ticket_revenue,

        "ancillary_revenue": ancillary_revenue,

        "fuel_cost": fuel_cost,

        "maintenance_cost": maintenance_cost,

        "crew_cost": crew_cost,

        "depreciation_cost": depreciation_cost,

        "insurance_cost": insurance_cost,

        "airport_fees": airport_fees,

        "catering_cost": catering_cost,

        "handling_cost": handling_cost,

        "navigation_fees": navigation_fees,

        "sales_distribution_cost":
            sales_distribution_cost,

        "passenger_service_cost":
            passenger_service_cost,

        "overhead_cost":
            overhead_cost,

        "marketing_cost":
            marketing_cost,

        "it_systems_cost":
            it_systems_cost
    }

# ---------------------------------
# Recommendation Engine
# ---------------------------------

def recommend_aircraft(
    destination,
    flight_hours,
    season="Normal"
):

    print("\nAircraft Optimization Engine")

    print("=" * 60)

    results = []

    for aircraft in AIRCRAFT_OPTIONS.keys():

        inputs = generate_aircraft_inputs(
            aircraft,
            destination,
            flight_hours,
            season
        )

        prediction = predict_profit(
            **inputs,
            verbose=False
        )

        results.append({

            "Aircraft":
                aircraft,

            "Passengers":
                inputs["passengers"],

            "Predicted_Profit":
                prediction["Predicted_Profit"],

            "Profit_Margin":
                prediction["Profit_Margin_Pct"],

            "Business_Status":
                prediction["Status"]
        })

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="Predicted_Profit",
        ascending=False
    )

    # Best aircraft

    best_aircraft = results_df.iloc[0]

    print(
        f"\nRecommended Aircraft : "
        f"{best_aircraft['Aircraft']}"
    )

    print(
        f"Predicted Profit     : "
        f"{best_aircraft['Predicted_Profit']:,.2f}"
    )

    print(
        f"Profit Margin        : "
        f"{best_aircraft['Profit_Margin']:.2f}%"
    )

    print(
        f"Business Status      : "
        f"{best_aircraft['Business_Status']}"
    )

    print("\nAircraft Comparison")

    print("-" * 60)

    print(results_df)

    return results_df

# ---------------------------------
# Visualization
# ---------------------------------

def plot_aircraft_comparison(results_df):

    plt.figure(figsize=(10, 6))

    sns.barplot(
        data=results_df,
        x="Aircraft",
        y="Predicted_Profit"
    )

    plt.xticks(rotation=20)

    plt.title(
        "Aircraft Profitability Comparison"
    )

    plt.ylabel("Predicted Profit")

    plt.tight_layout()

    plt.show()

# ---------------------------------
# Example Execution
# ---------------------------------

if __name__ == "__main__":

    results = recommend_aircraft(

        destination="LHR",

        flight_hours=7.5,

        season="Peak"
    )

    plot_aircraft_comparison(results)