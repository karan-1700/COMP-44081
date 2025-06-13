# Technical Test for Applied Scientist (Data Scientist) 
- Advertisement Number: 44081
- Manitoba Public Service Commission
- Government of Manitoba

# Prepared by
- Karansinh Padhiar
- padhiar.karan@gmail.com

# Toronto Island Ferry Demand Forecasting

## Summary of Work

### <ins> Accessible Description (For Non-Technical Stakeholders) </ins> - _not exceeding 200 words_

- I improved the original ferry redemption forecasting model by replacing the simplistic seasonal decomposition method with a deep learning approach using Long Short-Term Memory (LSTM) networks, which effectively capture both long-term trends and short-term fluctuations.  
- The LSTM model integrates calendar-based features such as day of week, month, weekend indicators, and statutory holiday flags to better account for seasonal and event-driven demand patterns.  
- This enhancement led to a 40–50% improvement in forecasting accuracy, measured using MAPE and MAE, during both high-demand summer and low-demand winter periods.  
- I also built a second forecasting model for daily ticket sales using the same LSTM architecture, demonstrating code reusability, strong generalizability and consistent performance across related time series.  
- All code was written in Python using only open-source libraries, is modular, well-documented, and fully reproducible.  
- The end-to-end solution emphasizes accuracy, interpretability, maintainability, and business relevance, producing actionable daily-level forecasts to support ferry service operations and planning.  

---

### <ins> Detailed Technical Description </ins> - _not exceeding 500 words_

To address the Toronto ferry redemption forecasting challenge, I applied a rigorous, iterative data science workflow considering both classical and deep learning methods.

#### <ins> Forecasting Model for Redemptions (Task 1) </ins>

First, I assessed the provided baseline model, which used `seasonal_decompose` from `statsmodels`. While simple and interpretable, it failed to capture complex patterns, lacked uncertainty estimation, and performed poorly in periods with irregular demand (e.g., holidays or low winter ridership).

Given the strong seasonal nature of the data (e.g., peaks in summer, troughs in winter), I initially tested classical time series models like **SARIMA (Seasonal ARIMA)** and **Facebook Prophet**. Both are interpretable and robust at modeling trend and seasonality, but they required extensive parameter tuning and were computationally expensive and rigid.

Ultimately, I selected a deep learning approach based on **LSTM (Long Short-Term Memory)** networks due to their ability to learn from sequential dependencies and adapt to non-linear, dynamic temporal patterns. I enhanced the architecture by integrating time-based features:
- Day of week  
- Month of year  
- Weekend indicator  

The data was resampled to daily intervals and normalized using `MinMaxScaler`. Each input sequence used the previous 30 days to predict the next day's redemption count. The LSTM architecture consisted of two layers with dropout to prevent overfitting and support generalization on a relatively small dataset.

The LSTM model reduced MAPE by approximately **~43%**, and MAE by around **~44%** on average across cross-validation folds, compared to the base model.

#### <ins> Forecasting Model for Sales (Task 2) </ins>

Using the same architecture and workflow, I trained an LSTM model to forecast daily ticket sales. The data showed similar temporal patterns to redemptions, and the model generalized well. Calendar features again played a crucial role in capturing weekly and seasonal cycles. The model provides useful forecasts to support ticketing operations and marketing strategies.

#### <ins> Process and Reproducibility (Task 3) </ins>

I followed a clean, collaborative development workflow:
- Class-based architecture for easy extension  
- Reusable methods for both redemption and sales forecasting  
- Visualizations for debugging and performance tracking  
- Fully reproducible using open-source tools only (`pandas`, `numpy`, `tensorflow`, `sklearn`, `matplotlib`)  

All outputs were validated using robust metrics (**MAE**, **MAPE**, time series split **cross-validation**) and diagnostic plots.

#### <ins> Assumptions </ins>

- Daily ferry usage is driven primarily by **seasonality**, **weekends**, and **statutory holidays**.  
- No external variables (e.g., weather, local events) were assumed available.  
- Forecasting was performed at **daily granularity** with a **one-day prediction horizon**.

The final system is **accurate**, **extensible**, and **suitable for use** in ferry service planning and resource allocation.

#### <ins> Use of AI tools </ins>

- ChatGPT was used to assist with Markdown formatting and table generation.

---

## Environment

To run the existing code, you will need to install the below open-source python libraries:

```bash
pip install numpy pandas matplotlib statsmodels scikit-learn tensorflow
```
