import os
import sys

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------
# Dynamic Path Fix for Render
# ---------------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

SRC_PATH = os.path.join(
    BASE_DIR,
    "src"
)

sys.path.append(SRC_PATH)

# ---------------------------------
# Imports
# ---------------------------------

from predict import predict_profit
from optimize import recommend_aircraft

# ---------------------------------
# Page Configuration
# ---------------------------------

st.set_page_config(
    page_title="Airline Profitability System",
    layout="wide"
)

# ---------------------------------
# Title
# ---------------------------------

st.title(
    "AI-Driven Airline Route Profitability & Aircraft Optimization System"
)

st.markdown(
    """
    This dashboard predicts airline route profitability
    and recommends the most profitable aircraft
    using machine learning and optimization techniques.
    """
)

# ---------------------------------
# Sidebar Inputs
# ---------------------------------

st.sidebar.header("Flight Inputs")

destination = st.sidebar.selectbox(
    "Destination",
    ["LHR", "JFK", "SIN", "CDG", "FRA"]
)

aircraft_type = st.sidebar.selectbox(
    "Aircraft Type",
    [
        "Airbus A320",
        "Boeing 737-800",
        "Boeing 787-9",
        "Airbus A350-900",
        "Boeing 777-300ER",
        "Airbus A380"
    ]
)

season = st.sidebar.selectbox(
    "Season",
    ["Peak", "Normal", "Low"]
)

flight_hours = st.sidebar.slider(
    "Flight Hours",
    1.0,
    15.0,
    7.5
)

passengers = st.sidebar.slider(
    "Passengers",
    50,
    550,
    300
)

ticket_revenue = st.sidebar.number_input(
    "Ticket Revenue",
    value=350000
)

ancillary_revenue = st.sidebar.number_input(
    "Ancillary Revenue",
    value=40000
)

fuel_cost = st.sidebar.number_input(
    "Fuel Cost",
    value=90000
)

maintenance_cost = st.sidebar.number_input(
    "Maintenance Cost",
    value=25000
)

crew_cost = st.sidebar.number_input(
    "Crew Cost",
    value=18000
)

# ---------------------------------
# Prediction Section
# ---------------------------------

st.header("Profit Prediction")

if st.button("Predict Profit"):

    result = predict_profit(

        destination=destination,

        aircraft_type=aircraft_type,

        passengers=passengers,

        flight_hours=flight_hours,

        season=season,

        demand_level="High",

        ticket_revenue=ticket_revenue,

        ancillary_revenue=ancillary_revenue,

        fuel_cost=fuel_cost,

        maintenance_cost=maintenance_cost,

        crew_cost=crew_cost,

        depreciation_cost=12000,

        insurance_cost=5000,

        airport_fees=10000,

        catering_cost=7000,

        handling_cost=6000,

        navigation_fees=5000,

        sales_distribution_cost=9000,

        passenger_service_cost=6000,

        overhead_cost=10000,

        marketing_cost=5000,

        it_systems_cost=3000,

        verbose=False
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Predicted Profit",
        f"${result['Predicted_Profit']:,.0f}"
    )

    col2.metric(
        "Profit Margin",
        f"{result['Profit_Margin_Pct']}%"
    )

    col3.metric(
        "Business Status",
        result["Status"]
    )

# ---------------------------------
# Optimization Section
# ---------------------------------

st.header("Aircraft Optimization")

if st.button("Recommend Best Aircraft"):

    optimization_results = recommend_aircraft(

        destination=destination,

        flight_hours=flight_hours,

        season=season
    )

    st.dataframe(optimization_results)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.barplot(
        data=optimization_results,
        x="Aircraft",
        y="Predicted_Profit",
        ax=ax
    )

    plt.xticks(rotation=20)

    plt.title(
        "Aircraft Profitability Comparison"
    )

    st.pyplot(fig)