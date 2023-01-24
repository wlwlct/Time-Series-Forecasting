# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 22:14:49 2022

@author: asifuzzaman, fatema, liwei

This is the compiled code blocks for project "Comparison of Traditional (SARIMA) and Deep Learning methods (LSTM) in Forecasting Time Series Data"

The file contains below sections:
1. Required Libraries (Based on the context, libraries are imported later as well) 
2. Data Loading & Preprocessing   
3. Exploratory data analysis and Statistical tests 
4. Functions: ARIMA & LSTM Models
5. ARIMA & LSTM Forecast, Results & Plots
6. Compare models with seasonal data (Air passenger data)
"""

""" 1. Required Libraries """
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

import time
import datetime
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults

import tensorflow as tf

from sklearn.metrics import mean_squared_error
from math import sqrt

""" 2. Data Loading & Preprocessing """
# importing dataset 
df = pd.read_csv(r"C:\Git\Time-Series-Forecasting\Data\raw\SNPTSX.csv")

df.head(5)

df['Date'] = pd.to_datetime(df['Date']) # convert the Date column type
df.set_index('Date', inplace=True) # setting the Date as index of the data frame
df = df.asfreq('d') # setting index freq as daily

df.info()

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
# converting to log for differncing 
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

""" 3. Exploratory data analysis and Statistical tests """

" 3.1 Check Stationarity with Visual Trends "
# creating function that takes a series and window size to plot rolling statistics of TS
def plot_rolling_stats(ts, window=30):
    
    mean = ts.rolling(window=window).mean()  # moving average time series
    std = ts.rolling(window=window).std()    # moving standard deviation time series
    
    # Visualize the original time series, moving average, and moving standard deviation as a plot.
    plt.figure(figsize=(10,5))
    orig = plt.plot(ts, color='blue',label='Original')    
    mean = plt.plot(mean, color='red', label='Rolling Mean')
    std = plt.plot(std, color='green', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')   
    #plt.show(block=False)

# check rolling statistics for original data   
plot_rolling_stats(df['Adj Close'], window=30)

"observation: Although the variation in rolling standard deviation is small, The rolling mean is moving with time. Hence, it is not a statinary data"

" 3.2 Check Stationarity with Dickey-Fuller Test "
# import required package
from statsmodels.tsa.stattools import adfuller

# creating function that takes a series. The output is the test statistics of Dickey-Fuller Test
def test_adf(series):
    print('Augmented Dickey-Fuller Test:')
    result = adfuller(series.dropna(),autolag='AIC') # .dropna() handles differenced data
    
    labels = ['ADF test statistic','p-value','# lags used','# observations']
    out = pd.Series(result[0:4],index=labels)
    
    critical_value = []
    for key,val in result[4].items():
        out['critical value ({})'.format(key)]=val
        critical_value.append(val)
        
        
    print(out.to_string())          # .to_string() removes the line "dtype: float64"
    
    # Checking conditions for output
    if result[1] <= 0.05:
        print("Reject the null hypothesis")
        print("Data has no unit root and is stationary")
    else:
        print("Fail to reject the null hypothesis")
        print("Data has a unit root and is non-stationary")
        
    if result[0]<critical_value[0]:
        print("ADF test statistic is smaller than 1% critical value")
    elif result[0]<critical_value[1]:
        print("ADF test statistic is smaller than 5% critical value")
    elif result[0]<critical_value[2]:
        print("ADF test statistic is smaller than 10% critical value")
    else:
        print("ADF test statistic is larger than all critical values!!")

# run test
test_adf(df['Adj Close'])

"observation: 'ADF test statistics' is larger than all 'critical values' of different confidence level. Hence, the series is not stationary."

" 3.3 Eliminating Trend & Seasonality "
"""remarks: Since, the stock data is volatile, regular trend elimination methods like aggregation, smoothing, polynomial fitting won't work in this data. 
Differencing or Decomposition is used in such cases. 
Differencing takes the difference with a particular time lag (first, second, third order and so on)
Decomposition is the process of modelling both trend and seasonality and eleminate them from the model."""

" 3.3.1 Differencing "
# applying first order differencing
df_log_diff = df["log_adj_close"] - df["log_adj_close"].shift() #tuning
plt.figure(figsize=(10,5))
plt.plot(df_log_diff)

# Check Stationarity of Differencing
# plot visual trends
plot_rolling_stats(df_log_diff, window=30)

# Dickey-Fuller Test of differenced data
test_adf(df_log_diff)

# using pdmarima check differencing
from pmdarima.arima.utils import ndiffs

ndiffs(df["log_adj_close"], test='adf')

"observation: for first order differencing, the data is stationary. We can use 'd'=1. Also ndiffs test returns the same"

" 3.3.2 Decomposing "
# import required package
from statsmodels.tsa.seasonal import seasonal_decompose

# apply decomposition on TS
df_log_dcom = seasonal_decompose(df_log_diff.dropna(), model='addictive', period = 90) #tuning

trend = df_log_dcom.trend
seasonal = df_log_dcom.seasonal
residual = df_log_dcom.resid

# plotting the trends
plt.figure(figsize=(10,5))
plt.subplot(411)
plt.plot(df["log_adj_close"], label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal,label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend(loc='best')
plt.tight_layout()

"observation: The trend ans seasonality are separated. The residuals seems more stationary."

# Check Stationarity of Residuals
df_log_dcom_resd = residual
df_log_dcom_resd.dropna(inplace=True)

# plot visual trends
plot_rolling_stats(df_log_dcom_resd, window=12)

# Dickey-Fuller Test on residuals
test_adf(df_log_dcom_resd)

"observation: The test statistics is lower than the 1% critical value. Hence, we can say that residuals are stationary."

" 3.4 ACF, PACF plots for ARIMA "
# plotting ACF and PACF to determine "p" and "q" term of ARIMA

" 3.4.1 Determining 'p' with PACF plot "
# import required packages
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plot_pacf(df_log_diff.dropna(), method='ywm'); # PACF : Partial Autocorrelation 
plt.show()

" 3.4.2 Determining 'q' with ACF plot "
plot_acf(df_log_diff.dropna())   # ACF : Autocorrelation 
plt.show()

"""observation:
Integrated/Differencing: d=1 (We checked the ADF test earlier on 1st order differncing)
Autoregression: p=11 (PACF chart cuts confidence interval at 11) 
Moving Average: q=1 (ACF chart cuts confidence interval at 1)
"""

""" 4. Functions: ARIMA & LSTM Models """

" 4.1 Seasonal ARIMA (SARIMA) "
def forecast_SARIMA(train_data, n_steps_out, p, d, q, s):

    # defining SARIMA
    t1 = time.time()
    model = ARIMA(train_data, order=(p, d, q), seasonal_order=(p, d, q, s)).fit()  # order=(p, d, q)
    train_time = time.time()-t1
    #print(model.summary())
    
    #residuals = pd.DataFrame(model.resid)
    #print(residuals.describe())
    
    # extracting min and max date 
    min_test_date = train_data.index.max()+datetime.timedelta(days=1)
    max_test_date = train_data.index.max()+datetime.timedelta(days=n_steps_out)
    
    min_test_date = min_test_date.strftime("%Y-%m-%d")
    max_test_date = max_test_date.strftime("%Y-%m-%d")
    
    # predicting future steps
    t1 = time.time()
    forecast = model.predict(start=min_test_date, end=max_test_date)
    test_time = time.time()-t1
    
    return np.exp(forecast), train_time, test_time

" 4.2 Splitting function for DL methods input-output sequence "
# spliting a univariate sequence into samples
def split_steps_in_out(sequence, n_steps_in, n_steps_out):
	X, y = list(), list()
	for i in range(len(sequence)):
		# finding the end of this pattern
		end_ix = i + n_steps_in
		out_end_ix = end_ix + n_steps_out
		# checking if we are beyond the sequence
		if out_end_ix > len(sequence):
			break
		# gathering input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
		X.append(seq_x)
		y.append(seq_y)
        
	return np.array(X), np.array(y)

" 4.3 RNN "
def forecast_RNN(train_data, n_steps_in, n_steps_out, n_features, units, epochs, activation, patience_level):
    
    # converting to numpy array
    index = np.array(train_data)
    
    # split into samples
    X, y = split_steps_in_out(index, n_steps_in, n_steps_out)
    
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

" 4.4 Vanilla LSTM "
def forecast_Vanilla_LSTM(train_data, n_steps_in, n_steps_out, n_features, units, epochs, activation, patience_level):
    
    # converting to numpy array
    index = np.array(train_data)
    
    # split into samples
    X, y = split_steps_in_out(index, n_steps_in, n_steps_out)
    
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

" 4.5 Stacked LSTM "
def forecast_Stacked_LSTM(train_data, n_steps_in, n_steps_out, n_features, units, epochs, activation, patience_level):
    
    # converting to numpy array
    index = np.array(train_data)
    
    # split into samples
    X, y = split_steps_in_out(index, n_steps_in, n_steps_out)
    
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

" 4.6 Bidirectional LSTM "
def forecast_Bidirect_LSTM(train_data, n_steps_in, n_steps_out, n_features, units, epochs, activation, patience_level):
    
    # converting to numpy array
    index = np.array(train_data)
    
    # split into samples
    X, y = split_steps_in_out(index, n_steps_in, n_steps_out)
    
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

" 4.7 CNN LSTM "
def forecast_CNN_LSTM(train_data, n_steps_in, n_steps_out, n_features, n_seq, units, epochs, activation, patience_level):
    
    # converting to numpy array
    index = np.array(train_data)
    
    # split into samples
    X, y = split_steps_in_out(index, n_steps_in, n_steps_out)
    
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

" 4.8 Encoder Decoder LSTM "
def forecast_Encode_De_LSTM(train_data, n_steps_in, n_steps_out, n_features, units, epochs, activation, patience_level):
    
    # converting to numpy array
    index = np.array(train_data)
    
    # split into samples
    X, y = split_steps_in_out(index, n_steps_in, n_steps_out)
    
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

""" 5. ARIMA & LSTM Forecast, Results & Plots """

" 5.1 executing models "

""" executing SARIMA """
df_test['SARIMA'], train_time, test_time = forecast_SARIMA(train_data=df_train["log_adj_close"], n_steps_out=30, p=11, d=1, q=1, s=30)

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

# creating dataframe
df_results = pd.DataFrame(Model_results, columns=["Model","RMSE","Train_time","Test_time"])


" 5.2 Save results "
df_results.to_csv(r"C:\Git\Time-Series-Forecasting\Data\raw\SNPTSX_Output_01.csv", index=False)

df_test.to_csv(r"C:\Git\Time-Series-Forecasting\Data\raw\SNPTSX_Predictions_01.csv", index=False)

" 5.3 Result & Plots"
# Forecasted Subplots
df_pred = pd.read_csv(r"C:\Git\Time-Series-Forecasting\Data\raw\SNPTSX_Predictions_01.csv")

# fitting and scaling the Actual value
min_max = MinMaxScaler()
df_pred_sc= pd.DataFrame(min_max.fit_transform(df_pred[["Adj Close"]]), columns=["Adj Close"])

# transform predicted value based on the fitted scaler 
df_pred_sc[["SARIMA"]] = min_max.transform(df_pred[["SARIMA"]])
df_pred_sc[["RNN"]] = min_max.transform(df_pred[["RNN"]])
df_pred_sc[["Vanilla_LSTM"]] = min_max.transform(df_pred[["Vanilla_LSTM"]])
df_pred_sc[["Stacked_LSTM"]] = min_max.transform(df_pred[["Stacked_LSTM"]])
df_pred_sc[["Bidirect_LSTM"]] = min_max.transform(df_pred[["Bidirect_LSTM"]])
df_pred_sc[["CNN_LSTM"]] = min_max.transform(df_pred[["CNN_LSTM"]])
df_pred_sc[["Encode_De_LSTM"]] = min_max.transform(df_pred[["Encode_De_LSTM"]])

# start index from 1
df_pred_sc.index += 1


fig, axs = plt.subplots(2, 4, sharey=True, figsize=(10,7))

axs[0, 0].plot(df_pred_sc["Adj Close"], linestyle='dotted', color='blue')
axs[0, 0].plot(df_pred_sc["SARIMA"], color='black')
axs[0, 0].set_title('SARIMA')

axs[0, 1].plot(df_pred_sc["Adj Close"], linestyle='dotted', color='blue')
axs[0, 1].plot(df_pred_sc["RNN"], color='black')
axs[0, 1].set_title('RNN')

axs[0, 2].plot(df_pred_sc["Adj Close"], linestyle='dotted', color='blue', label='Actual')
axs[0, 2].plot(df_pred_sc["Vanilla_LSTM"], color='black', label='Forecasted')
axs[0, 2].set_title('Vanilla LSTM')

axs[0, 3].axis('off')

axs[1, 0].plot(df_pred_sc["Adj Close"], linestyle='dotted', color='blue')
axs[1, 0].plot(df_pred_sc["Stacked_LSTM"], color='black')
axs[1, 0].set_title('Stacked LSTM')

axs[1, 1].plot(df_pred_sc["Adj Close"], linestyle='dotted', color='blue')
axs[1, 1].plot(df_pred_sc["Bidirect_LSTM"], color='black')
axs[1, 1].set_title('Bidirect LSTM')

axs[1, 2].plot(df_pred_sc["Adj Close"], linestyle='dotted', color='blue')
axs[1, 2].plot(df_pred_sc["CNN_LSTM"], color='black')
axs[1, 2].set_title('CNN LSTM')

axs[1, 3].plot(df_pred_sc["Adj Close"], linestyle='dotted', color='blue')
axs[1, 3].plot(df_pred_sc["Encode_De_LSTM"], color='black')
axs[1, 3].set_title('Encoder De LSTM')

axs[0, 2].legend(bbox_to_anchor=(1.9, 1)) # legend

for ax in axs.flat:
    #ax.xticks(np.arange(0, 30, 5))
    ax.set(xlabel='No of Days', ylabel='Adjusted Closing (Scaled)')
    

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()

# Stability RMSE
# importing csv file that contains RMSE measure for 10 iterations 
rs = pd.read_csv(r"C:\Git\Time-Series-Forecasting\Data\raw\SNPTSX_Output_10_iterations.csv")

rs_RMSE = rs.drop(['Train_time', 'Test_time'], axis=1) 
rs_RMSE = rs_RMSE.pivot(index='Run', columns='Model', values='RMSE')
rs_RMSE = rs_RMSE[["RNN","Vanilla_LSTM","Stacked_LSTM","Bidirect_LSTM","CNN_LSTM","Encode_De_LSTM"]]

# box plot 
plt.figure(figsize=(8,5))
plt.boxplot(rs_RMSE)
plt.xticks([1, 2, 3, 4, 5, 6], rs_RMSE.columns)
plt.xticks(rotation = 45)
plt.rcParams.update({'font.size': 14})
plt.xlabel('Methods', fontsize=14)
plt.ylabel('RMSE', fontsize=14)
plt.ylim(0,600)
plt.show()

# epochs
Model_results = [[]]
for i in range(1, 100):
    print(i)
    """ executing Stacked_LSTM """ 
    df_test['Stacked_LSTM'], train_time, test_time = forecast_Stacked_LSTM(train_data = df_train["scaled_adj_close"], n_steps_in=50, n_steps_out=30, n_features=1, units=50, epochs=i, activation='relu', patience_level=100)
    # calculating RMSE
    RMSE = sqrt(mean_squared_error(df_test['Adj Close'], df_test['Stacked_LSTM']))
    # storing model results 
    Model_results.append([i, RMSE])
    
df_epochs = pd.DataFrame(Model_results, columns=["Epoch","RMSE"])
df_epochs = df_epochs.drop(index=0)
  
    
plt.figure(figsize=(8,5))
plt.plot(df_epochs['RMSE'], color='black')
plt.xticks(rotation = 45)
plt.rcParams.update({'font.size': 14})
plt.xlabel('No of Epochs', fontsize=14)
plt.ylabel('RMSE', fontsize=14)
plt.show()     

# LSTM units - dimension of the hidden state
Model_results = [[]]
for i in range(1, 100):
    print(i)
    """ executing Stacked_LSTM """ 
    df_test['Stacked_LSTM'], train_time, test_time = forecast_Stacked_LSTM(train_data = df_train["scaled_adj_close"], n_steps_in=50, n_steps_out=30, n_features=1, units=i, epochs=50, activation='relu', patience_level=3)
    # calculating RMSE
    RMSE = sqrt(mean_squared_error(df_test['Adj Close'], df_test['Stacked_LSTM']))
    # storing model results 
    Model_results.append([i, RMSE])
    
df_units = pd.DataFrame(Model_results, columns=["Epoch","RMSE"])
df_units = df_units.drop(index=0)

plt.figure(figsize=(8,5))
plt.plot(df_units['RMSE'], color='black')
plt.xticks(rotation = 45)
plt.rcParams.update({'font.size': 14})
plt.xlabel('No of Units', fontsize=14)
plt.ylabel('RMSE', fontsize=14)
plt.show()  

"""observation:
1. "LSTM units: dimensionality of the output space" plays a crucial role in forecating. 
More units give better RMSE measures

2. Stacked LSTM model is more stable than Vanilla or Bidirectional LSTM in this case 
(Stable RMSE measures for different runs and hyperparameter tuning)  

3. Learning rate increased with epochs  
"""

""" 6. Compare models with seasonal data (Air passenger data) """

" 6.1 Air Passenger Data Processing """
# importing dataset 
df = pd.read_csv(r"C:\Git\Time-Series-Forecasting\Data\raw\AirPassengers.csv")

#df.head(5)
#df.info()

df['Date'] = pd.to_datetime(df['Month'], infer_datetime_format=True) # converting to date format
df = df.drop(columns = 'Month') # dropping month column
df = df.rename(columns = {'#Passengers':'Passengers'}) # renaming column
df.set_index('Date', inplace=True) # setting the Date as index of the data frame
#df = df.asfreq('m') # setting index freq as monthly

#df.head(5)
#df.info()

df.isna().any()
df.describe()

# Time Series Visualization
plt.figure(figsize=(20,5))
plt.plot(df['Passengers'])


# scaling data 
# converting to log > differncing 
df["log_Passengers"] = np.log(df["Passengers"])
df["diff_log_Passengers"] = df["log_Passengers"] - df["log_Passengers"].shift()

# scaling data 
scaler = MinMaxScaler()
df["scaled_Passengers"] = scaler.fit_transform(np.expand_dims(df["Passengers"].values, axis=1))

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

" 6.2 Executing Models"
""" executing SARIMA """
df_test['SARIMA'], train_time, test_time = forecast_SARIMA(train_data=df_train["log_Passengers"], n_steps_out=30, p=11, d=1, q=1, s=12)

# calculating RMSE
RMSE = sqrt(mean_squared_error(df_test['Passengers'], df_test['SARIMA']))
# storing model results 
Model_results = [['SARIMA', RMSE, train_time, test_time]]
#Model_results = [['SARIMA', 111.19373835479426, 2167.012885570526, 0.15523123741149902]]

# plotting the Forecasts
plt.figure(figsize=(14,5))
plt.plot(df_test["Passengers"], label="Test Data")
plt.plot(df_test["SARIMA"], color='cyan', label="SARIMA Forecast")
plt.legend(loc='best')
plt.tight_layout()

""" executing RNN """ 
df_test['RNN'], train_time, test_time = forecast_RNN(train_data = df_train["scaled_Passengers"], n_steps_in=50, n_steps_out=30, n_features=1, units=50, epochs=100, activation='tanh', patience_level=5)

# calculating RMSE
RMSE = sqrt(mean_squared_error(df_test['Passengers'], df_test['RNN']))
# storing model results 
Model_results.append(['RNN', RMSE, train_time, test_time])

""" executing Vanilla_LSTM """
df_test['Vanilla_LSTM'], train_time, test_time = forecast_Vanilla_LSTM(train_data = df_train["scaled_Passengers"], n_steps_in=50, n_steps_out=30, n_features=1, units=50, epochs=100, activation='relu', patience_level=5)

# calculating RMSE
RMSE = sqrt(mean_squared_error(df_test['Passengers'], df_test['Vanilla_LSTM']))
# storing model results 
Model_results.append(['Vanilla_LSTM', RMSE, train_time, test_time])

""" executing Stacked_LSTM """ 
df_test['Stacked_LSTM'], train_time, test_time = forecast_Stacked_LSTM(train_data = df_train["scaled_Passengers"], n_steps_in=50, n_steps_out=30, n_features=1, units=50, epochs=100, activation='relu', patience_level=5)

# calculating RMSE
RMSE = sqrt(mean_squared_error(df_test['Passengers'], df_test['Stacked_LSTM']))
# storing model results 
Model_results.append(['Stacked_LSTM', RMSE, train_time, test_time])

""" executing Bidirect_LSTM """ 
df_test['Bidirect_LSTM'], train_time, test_time = forecast_Bidirect_LSTM(train_data = df_train["scaled_Passengers"], n_steps_in=50, n_steps_out=30, n_features=1, units=50, epochs=100, activation='relu', patience_level=5)

# calculating RMSE
RMSE = sqrt(mean_squared_error(df_test['Passengers'], df_test['Bidirect_LSTM']))
# storing model results 
Model_results.append(['Bidirect_LSTM', RMSE, train_time, test_time])

""" executing CNN_LSTM """ 
df_test['CNN_LSTM'], train_time, test_time = forecast_CNN_LSTM(train_data = df_train["scaled_Passengers"], n_steps_in=50, n_steps_out=30, n_seq=2, n_features=1, units=50, epochs=100, activation='relu', patience_level=5)

# calculating RMSE
RMSE = sqrt(mean_squared_error(df_test['Passengers'], df_test['CNN_LSTM']))
# storing model results 
Model_results.append(['CNN_LSTM', RMSE, train_time, test_time])

""" executing Encode_De_LSTM """ 
df_test['Encode_De_LSTM'], train_time, test_time = forecast_Encode_De_LSTM(train_data = df_train["scaled_Passengers"], n_steps_in=50, n_steps_out=30, n_features=1, units=50, epochs=100, activation='relu', patience_level=5)

# calculating RMSE
RMSE = sqrt(mean_squared_error(df_test['Passengers'], df_test['Encode_De_LSTM']))
# storing model results 
Model_results.append(['Encode_De_LSTM', RMSE, train_time, test_time])

" 6.3 Save results "
df_results = pd.DataFrame(Model_results, columns=["Model","RMSE","Train_time","Test_time"])

df_results.to_csv(r"C:\Git\Time-Series-Forecasting\Data\raw\AirPassengers_Output_01.csv", index=False)

df_test.to_csv(r"C:\Git\Time-Series-Forecasting\Data\raw\AirPassengers_Predictions_01.csv", index=False)

" 6.4 Evaluation"

# Forecasted Subplots
df_pred = pd.read_csv(r"C:\Git\Time-Series-Forecasting\Data\raw\AirPassengers_Predictions_01.csv")

# fitting and scaling the Actual value
min_max = MinMaxScaler()
df_pred_sc= pd.DataFrame(min_max.fit_transform(df_pred[["Passengers"]]), columns=["Passengers"])

# transform predicted value based on the fitted scaler 
df_pred_sc[["SARIMA"]] = min_max.transform(df_pred[["SARIMA"]])
df_pred_sc[["Stacked_LSTM"]] = min_max.transform(df_pred[["Stacked_LSTM"]])

# start index from 1
df_pred_sc.index += 1

fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(10,6))

ax1.plot(df_pred_sc["Passengers"], color='blue', linestyle='dotted')
ax1.plot(df_pred_sc["SARIMA"], color='black')
ax1.set_title('SARIMA', fontsize=16)

ax2.plot(df_pred_sc["Passengers"], color='blue', linestyle='dotted', label='Forecasted')
ax2.plot(df_pred_sc["Stacked_LSTM"], color='black', label='Forecasted')
ax2.set_title('Stacked LSTM', fontsize=16)

ax2.legend(bbox_to_anchor=(1, 1), fontsize=14) # legend

ax1.set_xlabel('No of Months', fontsize=14)
ax1.set_ylabel('No of Passengers (Scaled)', fontsize=14)
ax2.set_xlabel('No of Months', fontsize=14)

