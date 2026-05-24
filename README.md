

## AI-Driven Airline Route Profitability & Aircraft Optimization System

### Project Overview

The aviation industry operates in a highly competitive environment where operational efficiency and profitability play an important role in airline success. This project focuses on predicting airline route profitability and recommending the most profitable aircraft using Machine Learning and predictive analytics techniques.

The system analyzes operational and financial airline data such as passenger demand, flight duration, revenue, fuel cost, maintenance expenses, and aircraft capacity to support intelligent airline decision-making.

The project combines:

* Machine Learning prediction
* Aircraft optimization
* Visualization and analytics
* Interactive dashboard
* Cloud deployment

into a single intelligent aviation analytics platform.

---

# Objectives

* Predict airline route profitability using Machine Learning
* Recommend the most profitable aircraft
* Analyze operational and financial airline data
* Improve airline operational efficiency
* Provide real-time operational insights using visualization
* Develop an interactive dashboard for airline analytics

---

# Features

* Airline Profitability Prediction
* Aircraft Recommendation Engine
* Machine Learning-Based Analysis
* Data Preprocessing and Feature Engineering
* Feature Importance Analysis
* Interactive Streamlit Dashboard
* Profitability Visualization Graphs
* Real-Time Operational Prediction
* Cloud Deployment using Render
* Modular Project Architecture

---

# Dataset Description

The project uses an airline operational and financial dataset containing:

* 7974 records
* 33 features

Important dataset attributes include:

* Aircraft Type
* Passenger Count
* Flight Hours
* Ticket Revenue
* Fuel Cost
* Maintenance Cost
* Operational Expenses
* Profit Margin
* Seasonal Demand
* Route Category

---

# Machine Learning Models Used

## 1. Linear Regression

Used as a baseline regression model for profitability prediction.

## 2. Random Forest Regression

Selected as the final prediction model because of:

* high accuracy,
* nonlinear relationship handling,
* operational reliability,
* feature importance analysis capability.

---

# Model Evaluation Metrics

The models were evaluated using:

* RMSE (Root Mean Square Error)
* MAE (Mean Absolute Error)
* R² Score

### Final Model Performance

| Model                    | R² Score |
| ------------------------ | -------- |
| Random Forest Regression | 0.9981   |

---

# Project Workflow

```text id="j4x9kp"
Dataset Collection
        ↓
Data Preprocessing
        ↓
Feature Engineering
        ↓
Model Training
        ↓
Profitability Prediction
        ↓
Aircraft Recommendation
        ↓
Visualization Dashboard
        ↓
Cloud Deployment
```

---

# Project Structure

```text id="m7p2rw"
airline-route-profitability/
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   └── raw/
│       └── airline_route_profitability.csv
│
├── models/
│   ├── rf_model.pkl
│   ├── label_encoders.pkl
│   └── feature_names.pkl
│
├── notebooks/
│   └── 01_EDA_and_Modeling.ipynb
│
├── reports/
│
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── predict.py
│   └── optimize.py
│
├── requirements.txt
└── README.md
```

---

# Technologies Used

| Category             | Technologies        |
| -------------------- | ------------------- |
| Programming Language | Python              |
| Data Processing      | Pandas, NumPy       |
| Machine Learning     | Scikit-learn        |
| Visualization        | Matplotlib, Seaborn |
| Dashboard            | Streamlit           |
| Deployment           | Render              |
| Version Control      | Git & GitHub        |

---

# How to Run the Project

## 1. Clone Repository

```bash id="x5q8ka"
git clone https://github.com/vikasmvicky/airline-route-profitability.git
```

---

## 2. Install Dependencies

```bash id="t2m7vs"
pip install -r requirements.txt
```

---

## 3. Run Preprocessing

```bash id="c8p4rz"
cd src
python preprocess.py
```

---

## 4. Train Machine Learning Models

```bash id="y6n1tw"
python train.py
```

---

## 5. Run Profit Prediction

```bash id="q3m8lx"
python predict.py
```

---

## 6. Run Aircraft Optimization

```bash id="v4k9ps"
python optimize.py
```

---

## 7. Launch Streamlit Dashboard

```bash id="w1r7mn"
cd ..
streamlit run app/streamlit_app.py
```

---

# Visualization Graphs

The project includes:

* Feature Importance Graph
* Actual vs Predicted Profit Graph
* Profit Distribution Graph
* Aircraft Profitability Comparison Graph

---

# Deployment

The project is deployed using:

* GitHub
* Render Cloud Platform

Deployment improves:

* online accessibility,
* real-time usability,
* practical implementation capability.

---

# Future Scope

Future enhancements may include:

* Real-time aviation API integration
* Dynamic ticket pricing systems
* Deep Learning-based forecasting
* Multi-route optimization
* Weather and fuel-price integration
* Enterprise-level cloud deployment

---

# Conclusion

The project successfully demonstrates the application of Machine Learning and predictive analytics in airline operational optimization. The developed system predicts airline profitability accurately, recommends profitable aircraft, and provides real-time operational insights through visualization and cloud deployment.

---

# Author

Vikas

---

# License

This project is developed for academic and educational purposes.
