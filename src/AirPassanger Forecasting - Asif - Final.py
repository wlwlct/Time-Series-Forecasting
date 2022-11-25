# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 18:52:22 2022

@author: asifuzzaman
"""

# import required packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

import time
from datetime import datetime
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults

import tensorflow as tf

from sklearn.metrics import mean_squared_error
from math import sqrt

""" Data Processing """
# importing dataset 
df = pd.read_csv(r"C:\Git\Time-Series-Forecasting\Data\raw\AirPassengers.csv")

#df.head(5)
#df.info()

df['Date'] = pd.to_datetime(df['Month']) # converting to date format
df = df.drop(columns = 'Month') # dropping month column
df.set_index('Date', inplace=True) # setting the Date as index of the data frame
df = df.rename(columns = {'#Passengers':'Passengers'}) # renaming column
df = df.asfreq('m') # setting index freq as monthly



df['Date'] = pd.to_datetime(df['Date']) # convert the Date column type
df.set_index('Date', inplace=True) # setting the Date as index of the data frame
df = df.asfreq('d') # setting index freq as daily



# filling missing value by interpolating between nearest 2 points
for i in df.columns:
    df[i] = df[i].interpolate(option='linear')    
    
df.isna().any()

df.describe()

# Time Series Visualization
plt.figure(figsize=(20,5))
plt.plot(df['Adj Close'])

# trim dataset
#df = df.loc['2012-10-16':'2018-12-31']
df = df.loc['2016-01-01':'2017-01-31']
df.info()

df.index

plt.figure(figsize=(20,5))
plt.plot(df['Adj Close'])

# scaling data 
# converting to log > differncing 
df["log_adj_close"] = np.log(df["Adj Close"])
df["diff_log_adj_close"] = df["log_adj_close"] - df["log_adj_close"].shift()

# scaling data 
scaler = MinMaxScaler()
df["scaled_adj_close"] = scaler.fit_transform(np.expand_dims(df["Adj Close"].values, axis=1))

df.describe()

# future steps to predict
n_predict_days = 30 

# split train and test
df_train = df[:-n_predict_days]
df_test = df[-n_predict_days:]

min_train_date = df_train.index.min().strftime("%Y-%m-%d")
max_train_date = df_train.index.max().strftime("%Y-%m-%d")

min_test_date = df_test.index.min().strftime("%Y-%m-%d")
max_test_date = df_test.index.max().strftime("%Y-%m-%d")


""" Seasonal ARIMA (SARIMA) """
def forecast_SARIMA(train_data, test_data, p, d, q, s):

    # defining SARIMA
    t1 = time.time()
    model = ARIMA(train_data, order=(p, d, q), seasonal_order=(p, d, q, s)).fit()  # order=(p, d, q)
    train_time = time.time()-t1
    #print(model.summary())
    
    #residuals = pd.DataFrame(model.resid)
    #print(residuals.describe())
    
    # extracting min and max date 
    min_test_date = df_test.index.min().strftime("%Y-%m-%d")
    max_test_date = df_test.index.max().strftime("%Y-%m-%d")
    
    # predicting future steps
    t1 = time.time()
    forecast = model.predict(start=min_test_date, end=max_test_date)
    test_time = time.time()-t1
    
    return np.exp(forecast), train_time, test_time

# spliting a univariate sequence into samples
def split_sequence(sequence, n_steps_in, n_steps_out):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps_in
		out_end_ix = end_ix + n_steps_out
		# check if we are beyond the sequence
		if out_end_ix > len(sequence):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
		X.append(seq_x)
		y.append(seq_y)
        
	return np.array(X), np.array(y)

""" RNN """
def forecast_RNN(train_data, n_steps_in, n_steps_out, n_features, units, epochs, activation, patience_level):
    
    # converting to numpy array
    index = np.array(train_data)
    
    # split into samples
    X, y = split_sequence(index, n_steps_in, n_steps_out)
    
    # reshaping input from [samples, timesteps] into [samples, timesteps, features]
    X = X.reshape((X.shape[0], X.shape[1], n_features))
    
    # defining RNN model
    model = tf.keras.models.Sequential()
    #model.add(tf.keras.layers.LSTM(LSTM_units, activation='relu', input_shape=(n_steps_in, n_features)))
    model.add(tf.keras.layers.SimpleRNN(units, activation='tanh', input_shape=(n_steps_in, n_features)))
    model.add(tf.keras.layers.Dense(n_steps_out))
    model.compile(optimizer='adam', loss='mse')
    
    # defining early stopping 
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=patience_level)
    
    # fit model
    t1 = time.time()
    model.fit(X, y, epochs=epochs, verbose=0, callbacks=(callback))
    train_time = time.time()-t1
    
    # reshaping last input sequence for prediction
    x_input = index[-n_steps_in:]
    x_input = x_input.reshape((1, n_steps_in, n_features))  
    
    # predicting future steps
    t1 = time.time()
    forecasted = model.predict(x_input, verbose=0)
    test_time = time.time()-t1
    
    return scaler.inverse_transform(np.expand_dims(forecasted.flatten(), axis=1)), train_time, test_time

""" Vanilla LSTM """
def forecast_Vanilla_LSTM(train_data, n_steps_in, n_steps_out, n_features, units, epochs, activation, patience_level):
    
    # converting to numpy array
    index = np.array(train_data)
    
    # split into samples
    X, y = split_sequence(index, n_steps_in, n_steps_out)
    
    # reshaping input from [samples, timesteps] into [samples, timesteps, features]
    X = X.reshape((X.shape[0], X.shape[1], n_features))
    
    # defining Vanilla_LSTM model    
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(units, activation=activation, input_shape=(n_steps_in, n_features)))
    model.add(tf.keras.layers.Dense(n_steps_out))
    model.compile(optimizer='adam', loss='mse')
    
    # defining early stopping 
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=patience_level)
    
    # fit model
    t1 = time.time()
    model.fit(X, y, epochs=epochs, verbose=0, callbacks=(callback))
    train_time = time.time()-t1
    
    # reshaping last input sequence for prediction
    x_input = index[-n_steps_in:]
    x_input = x_input.reshape((1, n_steps_in, n_features))  
    
    # predicting future steps
    t1 = time.time()
    forecasted = model.predict(x_input, verbose=0)
    test_time = time.time()-t1
    
    return scaler.inverse_transform(np.expand_dims(forecasted.flatten(), axis=1)), train_time, test_time 

""" Stacked_LSTM """
def forecast_Stacked_LSTM(train_data, n_steps_in, n_steps_out, n_features, units, epochs, activation, patience_level):
    
    # converting to numpy array
    index = np.array(train_data)
    
    # split into samples
    X, y = split_sequence(index, n_steps_in, n_steps_out)
    
    # reshaping input from [samples, timesteps] into [samples, timesteps, features]
    X = X.reshape((X.shape[0], X.shape[1], n_features))
    
    # defining Stacked_LSTM model    
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(units, activation=activation, return_sequences=True, input_shape=(n_steps_in, n_features)))
    model.add(tf.keras.layers.LSTM(units, activation=activation))
    model.add(tf.keras.layers.Dense(n_steps_out))
    model.compile(optimizer='adam', loss='mse')
    
    # defining early stopping 
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=patience_level)
    
    # fit model
    t1 = time.time()
    model.fit(X, y, epochs=epochs, verbose=0, callbacks=(callback))
    train_time = time.time()-t1
    
    # reshaping last input sequence for prediction
    x_input = index[-n_steps_in:]
    x_input = x_input.reshape((1, n_steps_in, n_features))  
    
    # predicting future steps
    t1 = time.time()
    forecasted = model.predict(x_input, verbose=0)
    test_time = time.time()-t1
    
    return scaler.inverse_transform(np.expand_dims(forecasted.flatten(), axis=1)), train_time, test_time 

""" Bidirect_LSTM """
def forecast_Bidirect_LSTM(train_data, n_steps_in, n_steps_out, n_features, units, epochs, activation, patience_level):
    
    # converting to numpy array
    index = np.array(train_data)
    
    # split into samples
    X, y = split_sequence(index, n_steps_in, n_steps_out)
    
    # reshaping input from [samples, timesteps] into [samples, timesteps, features]
    X = X.reshape((X.shape[0], X.shape[1], n_features))   
    
    # defining Bidirect_LSTM model 
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units, activation=activation, input_shape=(n_steps_in, n_features))))
    #model.add(tf.keras.layers.LSTM(100, activation='relu'))
    model.add(tf.keras.layers.Dense(n_steps_out))
    model.compile(optimizer='adam', loss='mse')
    
    # defining early stopping 
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=patience_level)
    
    # fit model
    t1 = time.time()
    model.fit(X, y, epochs=epochs, verbose=0, callbacks=(callback))
    train_time = time.time()-t1
    
    # reshaping last input sequence for prediction
    x_input = index[-n_steps_in:]
    x_input = x_input.reshape((1, n_steps_in, n_features))  
    
    # predicting future steps
    t1 = time.time()
    forecasted = model.predict(x_input, verbose=0)
    test_time = time.time()-t1
    
    return scaler.inverse_transform(np.expand_dims(forecasted.flatten(), axis=1)), train_time, test_time 

""" CNN_LSTM """
def forecast_CNN_LSTM(train_data, n_steps_in, n_steps_out, n_features, n_seq, units, epochs, activation, patience_level):
    
    # converting to numpy array
    index = np.array(train_data)
    
    # split into samples
    X, y = split_sequence(index, n_steps_in, n_steps_out)
    
    # reshaping from [samples, timesteps] into [samples, subsequences, timesteps, features]
    
    n_seq_steps = int(n_steps_in/n_seq)
    X = X.reshape((X.shape[0], n_seq, n_seq_steps, n_features))
    
    # defining CNN_LSTM model    
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Conv1D(filters=64, kernel_size=1, activation=activation), input_shape=(None, n_seq_steps, n_features)))
    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling1D(pool_size=2)))
    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten()))
    model.add(tf.keras.layers.LSTM(units, activation=activation))
    model.add(tf.keras.layers.Dense(n_steps_out))
    model.compile(optimizer='adam', loss='mse')
    
    # defining early stopping 
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=patience_level)
    
    # fit model
    t1 = time.time()
    model.fit(X, y, epochs=epochs, verbose=0, callbacks=(callback))
    train_time = time.time()-t1
    
    # reshaping last input sequence for prediction
    x_input = index[-n_steps_in:]
    x_input = x_input.reshape((1, n_seq, n_seq_steps, n_features)) 
    
    # predicting future steps
    t1 = time.time()
    forecasted = model.predict(x_input, verbose=0)
    test_time = time.time()-t1
    
    return scaler.inverse_transform(np.expand_dims(forecasted.flatten(), axis=1)), train_time, test_time 

""" Encode_De_LSTM """
def forecast_Encode_De_LSTM(train_data, n_steps_in, n_steps_out, n_features, units, epochs, activation, patience_level):
    
    # converting to numpy array
    index = np.array(train_data)
    
    # split into samples
    X, y = split_sequence(index, n_steps_in, n_steps_out)
    
    # reshaping from [samples, timesteps] into [samples, timesteps, features]
    X = X.reshape((X.shape[0], X.shape[1], n_features))
    y = y.reshape((y.shape[0], y.shape[1], n_features))
    
    # defining Encode_De_LSTM model   
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(units, activation=activation, return_sequences=True, input_shape=(n_steps_in, n_features)))
    model.add(tf.keras.layers.LSTM(units, activation=activation))
    model.add(tf.keras.layers.RepeatVector(n_steps_out))
    model.add(tf.keras.layers.LSTM(units, activation=activation, return_sequences=True))
    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1)))
    model.compile(optimizer='adam', loss='mse')
    
    # defining early stopping 
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=patience_level)
    
    # fit model
    t1 = time.time()
    model.fit(X, y, epochs=epochs, verbose=0, callbacks=(callback))
    train_time = time.time()-t1
    
    # reshaping last input sequence for prediction
    x_input = index[-n_steps_in:]
    x_input = x_input.reshape((1, n_steps_in, n_features))  
    
    # predicting future steps
    t1 = time.time()
    forecasted = model.predict(x_input, verbose=0)
    test_time = time.time()-t1
    
    return scaler.inverse_transform(np.expand_dims(forecasted.flatten(), axis=1)), train_time, test_time 

""" executing SARIMA """
df_test['SARIMA'], train_time, test_time = forecast_SARIMA(df_train["log_adj_close"], df_test["log_adj_close"], 11, 1, 1, 30)

# calculating RMSE
RMSE = sqrt(mean_squared_error(df_test['Adj Close'], df_test['SARIMA']))
# storing model results 
Model_results = [['SARIMA', RMSE, train_time, test_time]]
#Model_results = [['SARIMA', 111.19373835479426, 2167.012885570526, 0.15523123741149902]]

""" executing RNN """ 
df_test['RNN'], train_time, test_time = forecast_RNN(train_data = df_train["scaled_adj_close"], n_steps_in=50, n_steps_out=30, n_features=1, units=50, epochs=100, activation='tanh', patience_level=5)

# calculating RMSE
RMSE = sqrt(mean_squared_error(df_test['Adj Close'], df_test['RNN']))
# storing model results 
Model_results.append(['RNN', RMSE, train_time, test_time])

""" executing Vanilla_LSTM """
df_test['Vanilla_LSTM'], train_time, test_time = forecast_Vanilla_LSTM(train_data = df_train["scaled_adj_close"], n_steps_in=50, n_steps_out=30, n_features=1, units=50, epochs=100, activation='relu', patience_level=5)

# calculating RMSE
RMSE = sqrt(mean_squared_error(df_test['Adj Close'], df_test['Vanilla_LSTM']))
# storing model results 
Model_results.append(['Vanilla_LSTM', RMSE, train_time, test_time])

""" executing Stacked_LSTM """ 
df_test['Stacked_LSTM'], train_time, test_time = forecast_Stacked_LSTM(train_data = df_train["scaled_adj_close"], n_steps_in=50, n_steps_out=30, n_features=1, units=50, epochs=100, activation='relu', patience_level=5)

# calculating RMSE
RMSE = sqrt(mean_squared_error(df_test['Adj Close'], df_test['Stacked_LSTM']))
# storing model results 
Model_results.append(['Stacked_LSTM', RMSE, train_time, test_time])

""" executing Bidirect_LSTM """ 
df_test['Bidirect_LSTM'], train_time, test_time = forecast_Bidirect_LSTM(train_data = df_train["scaled_adj_close"], n_steps_in=50, n_steps_out=30, n_features=1, units=50, epochs=100, activation='relu', patience_level=5)

# calculating RMSE
RMSE = sqrt(mean_squared_error(df_test['Adj Close'], df_test['Bidirect_LSTM']))
# storing model results 
Model_results.append(['Bidirect_LSTM', RMSE, train_time, test_time])

""" executing CNN_LSTM """ 
df_test['CNN_LSTM'], train_time, test_time = forecast_CNN_LSTM(train_data = df_train["scaled_adj_close"], n_steps_in=100, n_steps_out=30, n_seq=4, n_features=1, units=50, epochs=100, activation='relu', patience_level=5)

# calculating RMSE
RMSE = sqrt(mean_squared_error(df_test['Adj Close'], df_test['CNN_LSTM']))
# storing model results 
Model_results.append(['CNN_LSTM', RMSE, train_time, test_time])

""" executing Encode_De_LSTM """ 
df_test['Encode_De_LSTM'], train_time, test_time = forecast_Encode_De_LSTM(train_data = df_train["scaled_adj_close"], n_steps_in=50, n_steps_out=30, n_features=1, units=50, epochs=100, activation='relu', patience_level=5)

# calculating RMSE
RMSE = sqrt(mean_squared_error(df_test['Adj Close'], df_test['Encode_De_LSTM']))
# storing model results 
Model_results.append(['Encode_De_LSTM', RMSE, train_time, test_time])

""" Evaluation """

# plotting the Forecasts
plt.figure(figsize=(14,5))
plt.plot(df_test["Adj Close"], label="Test Data")
plt.plot(df_test["SARIMA"], color='cyan', label="SARIMA Forecast")
plt.plot(df_test["RNN"], color='lawngreen', label="RNN Forecast")
plt.plot(df_test["Vanilla_LSTM"], color='pink', label="Vanilla LSTM Forecast")
plt.plot(df_test["Stacked_LSTM"], color='green', label="Stacked LSTM Forecast")
plt.plot(df_test["Bidirect_LSTM"], color='violet', label="Bidirectional LSTM Forecast")
plt.plot(df_test["CNN_LSTM"], color='brown', label="CNN LSTM Forecast")
plt.plot(df_test["Encode_De_LSTM"], color='teal', label="Encoder Decoder LSTM Forecast")
plt.legend(loc='best')
plt.tight_layout()

df_results = pd.DataFrame(Model_results, columns=["Model","RMSE","Train_time","Test_time"])

# plotting the RMSE 
plt.figure(figsize=(8,5))
plt.bar(df_results['Model'],df_results['RMSE'])
plt.legend(loc='best')
plt.tight_layout()

print(df_results)

""" 
*** Observations ***
1. "LSTM units: dimensionality of the output space" plays a crucial role in forecating. 
More units give better RMSE measures

2. Stacked LSTM model is more stable than Vanilla or Bidirectional LSTM in this case 
(Stable RMSE measures for different runs and hyperparameter tuning)  

3. Learning rate increased with epochs  
"""