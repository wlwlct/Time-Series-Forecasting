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
# importing dataset with date as index
df = pd.read_csv(r"C:\Git\Time-Series-Forecasting\Data\raw\SNPTSX.csv")

df.head(5)

df['Date'] = pd.to_datetime(df['Date']) # convert the Date column type
df.set_index('Date', inplace=True) # setting the Date as index of the data frame
df = df.asfreq('d') # setting index freq as daily

df.info()

# filling missing value by interpolating between nearest 2 nearest points
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

# choosing a number of time steps
n_steps_in, n_steps_out = 50, 30

# split train and test
df_train = df[:-n_steps_out]
df_test = df[-n_steps_out:]

min_train_date = df_train.index.min().strftime("%Y-%m-%d")
max_train_date = df_train.index.max().strftime("%Y-%m-%d")

min_test_date = df_test.index.min().strftime("%Y-%m-%d")
max_test_date = df_test.index.max().strftime("%Y-%m-%d")

""" Seasonal ARIMA (SARIMA) """

# defining SARIMA
model = ARIMA(df_train["log_adj_close"], order=(11, 1, 1), seasonal_order=(11,1,1,30)).fit()  # order=(p, d, q)
print(model.summary())

residuals = pd.DataFrame(model.resid)
print(residuals.describe())

# predicting future steps
forecast = model.predict(start=min_test_date, end=max_test_date)

df_test['SARIMA'] = np.exp(forecast)


 
# converting to numpy array
index = np.array(df_train["scaled_adj_close"])
#index = np.array(df_log)
#index = np.array(df_log_diff)

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
 
# split into samples
X, y = split_sequence(index, n_steps_in, n_steps_out)

#X.shape

# reshaping input from [samples, timesteps] into [samples, timesteps, features]
n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))

# hyperparameter 
LSTM_units = 100
epochs_no = 100 
patience_level = 5

""" Vanilla LSTM """
# defining LSTM model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.LSTM(LSTM_units, activation='relu', input_shape=(n_steps_in, n_features)))
model.add(tf.keras.layers.Dense(n_steps_out))
model.compile(optimizer='adam', loss='mse')

# defining early stopping 
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=patience_level)

# fit model
model.fit(X, y, epochs=epochs_no, verbose=1, callbacks=(callback))


# predicting future steps
x_input = index[-n_steps_in:]
x_input = x_input.reshape((1, n_steps_in, n_features))
forecasted = model.predict(x_input, verbose=0)
#print(forecasted)

#df_test['scaled_forecast'] = forecasted.flatten()
#df_test['forecast'] = scaler.inverse_transform(np.expand_dims(df_test["scaled_forecast"].values, axis=1))
df_test['vanilla_LSTM_fcst'] = scaler.inverse_transform(np.expand_dims(forecasted.flatten(), axis=1))

""" Stacked LSTM """
# defining Stacked LSTM model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.LSTM(LSTM_units, activation='relu', return_sequences=True, input_shape=(n_steps_in, n_features)))
model.add(tf.keras.layers.LSTM(LSTM_units, activation='relu'))
model.add(tf.keras.layers.Dense(n_steps_out))
model.compile(optimizer='adam', loss='mse')

# defining early stopping 
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=patience_level)

# fit model
model.fit(X, y, epochs=epochs_no, verbose=1, callbacks=(callback))


# predicting future steps
x_input = index[-n_steps_in:]
x_input = x_input.reshape((1, n_steps_in, n_features))
forecasted = model.predict(x_input, verbose=0)
#print(forecasted)

#df_test['scaled_forecast'] = forecasted.flatten()
#df_test['forecast'] = scaler.inverse_transform(np.expand_dims(df_test["scaled_forecast"].values, axis=1))
df_test['stacked_LSTM_fcst'] = scaler.inverse_transform(np.expand_dims(forecasted.flatten(), axis=1))

""" Bidirectional LSTM """
# defining Stacked LSTM model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(LSTM_units, activation='relu', input_shape=(n_steps_in, n_features))))
#model.add(tf.keras.layers.LSTM(100, activation='relu'))
model.add(tf.keras.layers.Dense(n_steps_out))
model.compile(optimizer='adam', loss='mse')

# defining early stopping 
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=patience_level)

# fit model
model.fit(X, y, epochs=epochs_no, verbose=1, callbacks=(callback))


# predicting future steps
x_input = index[-n_steps_in:]
x_input = x_input.reshape((1, n_steps_in, n_features))
forecasted = model.predict(x_input, verbose=0)
#print(forecasted)

#df_test['scaled_forecast'] = forecasted.flatten()
#df_test['forecast'] = scaler.inverse_transform(np.expand_dims(df_test["scaled_forecast"].values, axis=1))
df_test['bidirect_LSTM_fcst'] = scaler.inverse_transform(np.expand_dims(forecasted.flatten(), axis=1))

""" Evaluation """

# plotting the Forecasts
plt.figure(figsize=(14,5))
plt.plot(df_test["Adj Close"], label="Test Data")
plt.plot(df_test["SARIMA"], color='cyan', label="SARIMA Forecast")
plt.plot(df_test["vanilla_LSTM_fcst"], color='pink', label="Vanilla LSTM Forecast")
plt.plot(df_test["stacked_LSTM_fcst"], color='green', label="Stacked LSTM Forecast")
plt.plot(df_test["bidirect_LSTM_fcst"], color='violet', label="Bidirectional LSTM Forecast")
plt.legend(loc='best')
plt.tight_layout()

# calculating RMSE
list_RMSE = {
    "Method": ["SARIMA",
               "Vanilla LSTM", 
               "Stacked LSTM",
               "Bidirectional LSTM"
               ],
    "RMSE": [sqrt(mean_squared_error(df_test['Adj Close'], df_test['SARIMA'])),
             sqrt(mean_squared_error(df_test['Adj Close'], df_test['vanilla_LSTM_fcst'])),
             sqrt(mean_squared_error(df_test['Adj Close'], df_test['stacked_LSTM_fcst'])),
             sqrt(mean_squared_error(df_test['Adj Close'], df_test['bidirect_LSTM_fcst']))
             ]
    }

df_RMSE = pd.DataFrame(list_RMSE)

# plotting the RMSE 
plt.figure(figsize=(8,5))
plt.bar(df_RMSE['Method'],df_RMSE['RMSE'], color = ['cyan', 'pink', 'green', 'violet' ])
plt.legend(loc='best')
plt.tight_layout()

df_RMSE.head(10)

""" 
*** Observations ***
1. "LSTM units: dimensionality of the output space" plays a crucial role in forecating. 
More units give better RMSE measures

2. Stacked LSTM model is more stable than Vanilla or Bidirectional LSTM in this case 
(Stable RMSE measures for different runs and hyperparameter tuning)  

3. Learning rate increased with epochs  
"""