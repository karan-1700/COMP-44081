# Technical Test for Applied Scientist (Data Scientist) 
- Advertisement Number: 44081
- Manitoba Public Service Commission
- Government of Manitoba

# Toronto Island Ferry Demand Forecasting

## Problem Overview

> “We have built a simple forecasting model... it does not perform as well as we would like and it does not have any way to account for uncertainty in the forecast.”

This project tackles that limitation by:
- Building a robust forecasting pipeline using **LSTM deep learning models**
- Incorporating **time-based features** (e.g., day of week, weekend flag, holidays)
- Replacing the simplistic seasonal baseline model with a model that generalizes better across changing seasonal trends

---


## Highlights

### Improved Redemption Forecasting (Task 1)
- Base model (`seasonal_decompose`) MAPE ~86%
- LSTM model MAPE reduced to **~45%**
- Handles both short-term and long-term temporal dependencies

### New Sales Forecasting Model (Task 2)
- Identical architecture and pipeline as redemptions
- Easily extensible to other time series in the same domain

### Scientific Process (Task 3)
- Time-series cross-validation (4-fold, 365-day test windows)
- Diagnostics on input scaling, index alignment, and evaluation metrics
- Reproducible, modular codebase

### Assumptions
- Demand is seasonally and calendar-driven (weather not available)
- Next-day forecasting granularity
- No missing data post-aggregation

---

## Key Features Engineered

| Feature        | Description                            |
|----------------|----------------------------------------|
| `day_of_week`  | Integer: 0 (Monday) to 6 (Sunday)      |
| `is_weekend`   | Boolean: 1 if Saturday/Sunday          |
| `month`        | Integer: 1–12                          |
| `is_holiday`   | Boolean: 1 if public holiday in Ontario|

---

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run the main modeling pipeline
python main.py
