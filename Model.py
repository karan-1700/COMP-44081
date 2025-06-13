# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 19:18:33 2025

@author: Karan
"""

# importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense, Input

class RedemptionModel:
    """
    Time-series forecasting model for ticket redemption/sales counts.
    Supports both a seasonal-decomposition baseline and LSTM model.
    """
    
    def __init__(self, X, target_col):
        """
        Initialize the model with input data and target column.

        Args:
            X (pd.DataFrame): Time-series input data with timestamp index.
            target_col (str): Target variable column name.
        """
        self._predictions = {}
        self.X = X
        self.target_col = target_col
        self.results = {}

    ###############################################################################################

    def mae(self, truth, preds):
        """Compute Mean Absolute Error."""
        return mean_absolute_error(truth, preds)
    
    def stable_mape(self, y_true, y_pred, epsilon=1e-5):
        """Compute MAPE with stability on small denominators."""
        return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100

    def masked_mape(self, y_true, y_pred, min_value=100):
        """Compute MAPE ignoring small true values below min_value."""
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        mask = y_true >= min_value
        if np.sum(mask) == 0:
            return np.inf
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    ###############################################################################################

    def plot(self, preds_base, preds_lstm):
        """
        Plot observed target values against predictions from both the base and LSTM models.
    
        Args:
            preds_base (pd.Series): Predictions from the base model.
            preds_lstm (pd.Series): Predictions from the LSTM model.
        """
        # Create two vertically-stacked subplots
        fig, ax = plt.subplots(2, 1, figsize=(15, 8))
        # fig.suptitle(self.target_col, fontsize='large')
        
        for i, (preds, title) in enumerate(zip([preds_base, preds_lstm], ["Base Model", "LSTM Model"])):
            ax[i].scatter(self.X.index, self.X[self.target_col], s=0.4, color='grey', label='Observed')
            ax[i].plot(preds, label=title, color='red')
            ax[i].set_title(title)
            ax[i].set_ylabel(self.target_col)
            # ax[i].set_xlabel("Time")
            ax[i].legend()
        
        plt.tight_layout() # Adjust layout to prevent overlapping text
        

    def plot_training_graph(self, history, epochs):
        """
        Plot training and validation loss over epochs.
    
        Args:
            history (History): Keras model training history object.
            epochs (int): Total number of epochs used in training.
        """
        plt.figure(figsize=(15,10))
        plt.plot(np.arange(0, epochs), history.history["loss"], label="Train Loss" )
        plt.plot(np.arange(0, epochs), history.history["val_loss"], label="Val Loss" )
        plt.title("Training loss over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.show()

    ###############################################################################################

    def _base_model(self, train, test):
        '''
        Our base, too-simple model.
        Your model needs to take the training and test datasets (dataframes)
        and output a prediction based on the test data.

        Please leave this method as-is.

        '''
        res = sm.tsa.seasonal_decompose(train[self.target_col], period=365)
        res_clip = res.seasonal.apply(lambda x: max(0,x))
        res_clip.index = res_clip.index.dayofyear
        res_clip = res_clip.groupby(res_clip.index).mean()
        res_dict = res_clip.to_dict()
        return pd.Series(index = test.index, 
                         data = map(lambda x: res_dict[x], test.index.dayofyear))

    ###############################################################################################
    
    def _lstm_model(self, train_df, test_df, sequence_length=30, epochs=10, batch_size=16):

        """
        Train and predict using an LSTM model on scaled multivariate time-series sequences.
    
        Args:
            train_df (pd.DataFrame): Training portion of the time-series data.
            test_df (pd.DataFrame): Testing portion of the time-series data.
            sequence_length (int): Number of time steps to include in each input sequence.
            epochs (int): Number of training epochs.
            batch_size (int): Size of each training batch.
    
        Returns:
            pd.Series: Predicted values aligned to the test set index.
        """
        
        # Proprocessing the dataset.
        
        # Define features to be included in the LSTM input
        features = [self.target_col, 'day_of_week', 'is_weekend', 'monthly', 'quarter']


        # 1. Scale features using MinMaxScaler. Scale using only train_df, then apply to test_df
        scaler = MinMaxScaler()
        train_scaled = scaler.fit_transform(train_df[features])
        test_scaled = scaler.transform(test_df[features])
    
    
        # 2. Combine scaled train and test for sequence generation
        full_scaled = np.concatenate([train_scaled, test_scaled], axis=0)
        
        
        # 3. Creating sequences from scaled data (X: multivariate, y: single target)
        # Convert time-series into overlapping sequences of fixed length
        X, y = [], []
        for i in range(sequence_length, len(full_scaled)):
            X.append(full_scaled[i-sequence_length:i])   # shape: (sequence_length, num_features)
            y.append(full_scaled[i][0])                  # target is the first column (scaled)
        X, y = np.array(X), np.array(y)
        
        
        # 4. Train/test split
        split_point = len(train_scaled) - sequence_length
        X_train, X_test = X[:split_point], X[split_point:]
        y_train = y[:split_point]    


        # 5. Define and train LSTM model
        model = Sequential([
            Input(shape=(sequence_length, X.shape[2])),  # multivariate input # Input shape: (timesteps, features)
            LSTM(128, return_sequences=True),
            Dropout(0.2),
            LSTM(64),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mae')
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, 
                      validation_split=0.1, verbose=0)
        # self.plot_training_graph(history, epochs)
    

        # 6. Predict and inverse scale the Redemption Count column
        y_pred_scaled = model.predict(X_test, verbose=0)
        
        
        # To inverse transform properly, we need to reinsert dummy values for other features (set to zero)
        dummy_input = np.zeros((len(y_pred_scaled), len(features)))
        dummy_input[:, 0] = y_pred_scaled.flatten()
        y_pred = scaler.inverse_transform(dummy_input)[:, 0]
        
        # 7. Align index with last part of test_df
        # Only take the matching portion of test_df index
        # Align prediction index
        aligned_index = test_df.index[-len(y_pred):] # ensure alignment with test set
        return pd.Series(y_pred, index=aligned_index)

    ###############################################################################################

    def run_models(self, n_splits=4, test_size=365):
        """
        Run baseline and LSTM models using time series cross-validation.
    
        Stores performance metrics (MAE, MAPE) in self.results.
        """        
        # Time series split
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
        
        cnt = 0 # keep track of splits
        for train, test in tscv.split(self.X):
            
            print(f"[INFO] Running CV split {cnt} {'#' * 30}")
            
            # Slice train and test sets for this split
            X_train = self.X.iloc[train]
            X_test = self.X.iloc[test]
            
            
            # Base model - please leave this here
            preds_base = self._base_model(X_train, X_test)

            # Initialize metrics dictionary if not already
            if 'Base' not in self.results:
                self.results['Base'] = {}
            if "MAPE" not in self.results['Base']:
                self.results['Base']["MAPE"] = {}
            if "MAE" not in self.results['Base']:
                self.results['Base']["MAE"] = {}
                
            self.results['Base']["MAPE"][cnt] = self.masked_mape(X_test[self.target_col], preds_base)
            self.results['Base']["MAE"][cnt] = self.mae(X_test[self.target_col], preds_base)
            
            
            # -------------------------------- LSTM Model ------------------------------- #
            # Train LSTM and generate predictions
            preds_lstm = self._lstm_model(X_train, X_test, sequence_length=30, epochs=5, batch_size=16)
            
            # Align true values with prediction indices (in case of dropped rows)
            true_values = X_test.loc[preds_lstm.index, self.target_col]                

            # Initialize metrics dictionary if not already
            if 'LSTM' not in self.results:
                self.results['LSTM'] = {}
            if "MAPE" not in self.results['LSTM']:
                self.results['LSTM']["MAPE"] = {}
            if "MAE" not in self.results['LSTM']:
                self.results['LSTM']["MAE"] = {}

            
            # Compute metrics for LSTM predictions
            self.results['LSTM']["MAPE"][cnt] = self.masked_mape(true_values, preds_lstm)
            self.results['LSTM']["MAE"][cnt] = self.mae(true_values, preds_lstm)
            
            # Plot predictions for visual inspection
            self.plot(preds_base=preds_base, preds_lstm=preds_lstm)
                        
            cnt += 1






