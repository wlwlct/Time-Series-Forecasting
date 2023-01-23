# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 22:14:49 2022

@author: asifuzzaman, fatema, liwei

This is the compiled code blocks for project "Comparison of Traditional (ARIMA) and Deep Learning methods (LSTM, Transformers) in Forecasting Time Series Data"

The file contains below sections:
1. Required Libraries (Based on the context, libraries are imported later as well) 
2. Data Loading & Preprocessing   
3. Exploratory data analysis and Statistical tests 
4. Functions: ARIMA & LSTM Models
5. ARIMA & LSTM Forecast, Results & Plots
6. Compare models with seasonal data (Air passenger data)
7. Transformer Forecast, Results
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

""" 7. Transformer Forecast , Results """
# %%
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd  
import numpy as np
from  sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn
import torch.nn.functional as F


from sklearn.metrics import mean_squared_error

torch.cuda.is_available()


seaborn.set_context(context="talk")

# %%
# ## Generate Train and Test

# %%
SNP = pd.read_csv(r"C:\Git\Time-Series-Forecasting\Data\raw\SNPTSX.csv", parse_dates=['Date'])
SNP.drop(columns=['Open', 'High', 'Low', 'Close', 'Volume'], inplace=True)

# %%
SNP.set_index('Date').plot()

# %%
dataset = SNP.set_index('Date').loc['2016-01-01':'2017-01-31','Adj Close']
dataset = np.array(dataset.astype('float32')).reshape(-1,1)

# %%
plt.plot(dataset)

# %% 
# ## Train Model

# %%
def norm(x):
    """min-max normalization"""
    return (x-np.min(x))/(np.max(x)-np.min(x))

# normalize dataset and split data into train and test
dataset=norm(dataset)

look_back=8
np.random.seed(7)
train_size = int(len(dataset) * 0.9)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
print(len(train), len(test))

def create_dataset(dataset, look_back=look_back):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return np.array(dataX), np.array(dataY)

trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)


X0=trainX[0:-2]
Y0=trainX[1:-1]

X0=X0.reshape(X0.shape[0],X0.shape[1],1).astype(np.float32)
Y0=Y0.reshape(Y0.shape[0],Y0.shape[1],1).astype(np.float32)

# %%
# Define class and functions for building transformer. 

class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        
    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask,
                            tgt, tgt_mask)
    
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, 8)

    def forward(self, x):
        return F.relu(self.proj(x))
        
        
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

class Decoder(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
 
    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('float32')
    return torch.from_numpy(subsequent_mask) == 0


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class Embeddings1(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings1, self).__init__()
        self.d_model = d_model
 
    def forward(self, x):
        return torch.cat(4*[x]).reshape(-1,8,self.d_model)  * math.sqrt(self.d_model)
 
class Embeddings2(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings2, self).__init__()
        self.d_model = d_model
 
    def forward(self, x):
        return torch.cat(4*[x]).reshape(-1,7,self.d_model)  * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x


def make_model(src_vocab, tgt_vocab, N=2, 
               d_model=4, d_ff=32, h=4, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), 
                             c(ff), dropout), N),
        nn.Sequential(Embeddings1(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings2(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab))
    
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model


class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, src, trg=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = \
                self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()
    
    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask

def run_epoch(data_iter, model, loss_compute):
    "Standard Training and Logging Function"
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src, batch.trg, 
                            batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                    (i, loss / batch.ntokens, tokens / elapsed))
            start = time.time()
            tokens = 0
    return total_loss #/ total_tokens

global max_src_in_batch, max_tgt_in_batch
def batch_size_fn(new, count, sofar):
    "Keep augmenting batch and calculate total number of tokens + padding."
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch,  len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch,  len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)

class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        for p in self.optimizer.param_groups:
            p['lr'] = learning
        self._rate = learning
        self.optimizer.step()
        
        
def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
            torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.98), eps=1e-9))


def data_gen(V, batch, nbatches):
    "Generate random data for a src-tgt copy task."
    for i in range(nbatches):
        data1 = torch.from_numpy(X0.reshape(X0.shape[0]
,8))#.long()
        data1[:, 0] = 1
        data2 = torch.from_numpy(Y0.reshape(X0.shape[0]
,8))#.long()
        data2[:, 0] = 1
        src = Variable(data1, requires_grad=False)
        tgt = Variable(data2, requires_grad=False)
        yield Batch(src, tgt, 0)


class SimpleLossCompute:
    "A simple loss compute and train function."
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion 
        self.opt = opt
        
    def __call__(self, x, y, norm):
        x = self.generator(x)
        x=torch.sum(x.reshape(X0.shape[0]
,7,-1), (2))
        loss = self.criterion(torch.sum(x,(0)), 
                              torch.sum(y,(0))) #/ norm
        if loss<0.01:
            learning=learning/3

        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss.data #* norm

# %%
# start to train the model
print(torch.__version__)

V = 200
criterion = nn.MSELoss()
learning=0.001
model = make_model(V, V, N=2)

# %%
model_opt = NoamOpt(model.src_embed[0].d_model, 10, 400,
        torch.optim.Adam(model.parameters(), lr=learning, betas=(0.9, 0.98), eps=1e-9))

# %%
error_cov ={}
for epoch in range(60):
    #model.train()
    loss = run_epoch(data_gen(V, 30, 20), model, 
              SimpleLossCompute(model.generator, criterion, model_opt))
    error_cov[epoch]=loss.detach()


# %%
# generate the error vs. epochs
plt.plot(list(error_cov.keys())[-20:], list(error_cov.values())[-20:])

# %%
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 7).fill_(start_symbol).type_as(src.data)
    for i in range(8-1):
        out = model.decode(memory, src_mask, 
                           Variable(ys), 
                           Variable(subsequent_mask(ys.size(1))
                                    .type_as(src.data)))
        prob = torch.sum(src*torch.sum(model.generator(out),(1)))
    return prob

# %%
<<<<<<< Updated upstream
# save model
PATH = './pytorch_time_series_model_loss_OK_norm.pth'
=======
PATH = r"C:\Git\Time-Series-Forecasting\Data\raw\pytorch_time_series_model_loss_OK_norm.pth"
>>>>>>> Stashed changes

torch.save(model.state_dict(), PATH)

X0=trainX[0:-2]
Y0=trainX[1:-1]

X0=X0.reshape(X0.shape[0],X0.shape[1],1).astype(np.float32)
Y0=Y0.reshape(Y0.shape[0],Y0.shape[1],1).astype(np.float32)

# %%
# input a sequence and test a single value
X0.shape, Y0.shape

# %%
place=100

X0=X0[place]
Y0=[Y0[place][0:7].reshape(1,-1)]

# %%
np.size(X0), np.size(Y0)

# %%
model.load_state_dict(torch.load(PATH))

src = Variable(torch.Tensor(X0.reshape(1,-1)) )
src_mask = Variable(torch.ones(1,1,8))

# %%
pred=greedy_decode(model, src, src_mask, max_len=8, start_symbol=1)

print("actual:", Y0[0][0][-1],"prediction:",pred.detach().numpy())

# %% [markdown]
# ## Inference

# %%
# 
plt.plot(range(trainX.shape[0]),dataset[:trainX.shape[0]])
plt.plot(range(trainX.shape[0],trainX.shape[0]+dataset[trainX.shape[0]:].shape[0],1),dataset[trainX.shape[0]:])
plt.show()

# %%
(trainX.shape, trainY.shape, testX.shape, testY.shape) #[len_dataset-look_back,look_back]

# %% [markdown]
# ### Predict from test data
# input a sequence and test a single value, repeat the process using the input from the test data.

# %%
model.load_state_dict(torch.load(PATH))

# %%
X0_begin=testX[0:-2]
Y0_begin=testX[1:-1]

X0_begin=X0_begin.reshape(X0_begin.shape[0],X0_begin.shape[1],1).astype(np.float32)
Y0_begin=Y0_begin.reshape(Y0_begin.shape[0],Y0_begin.shape[1],1).astype(np.float32)

# %%
X0_begin.shape, Y0_begin.shape, testY.shape

# %%

pred = []
for place in range(X0_begin.shape[0]):
  X0=X0_begin[place]
  Y0=[Y0_begin[place][0:7].reshape(1,-1)]

  src = Variable(torch.Tensor(X0.reshape(1,-1)) )
  src_mask = Variable(torch.ones(1,1,8))
  current_pred = greedy_decode(model, src, src_mask, max_len=8, start_symbol=1)

  print('** Predicted Value: ', current_pred)
  print('-- Actual Value: ', testY[place])
  pred.append( current_pred.detach().numpy() )

# %%
plt.plot(testY)


# %%
plt.plot(pred)

# %% [markdown]
# ### Predict from the last point
# use the last point of test data and predicted data as input to predict 1 following value.

# %%
model.load_state_dict(torch.load(PATH))

# %%
X0=trainX[0:-2]
Y0=trainX[1:-1]

X0=X0.reshape(X0.shape[0],X0.shape[1],1).astype(np.float32)
Y0=Y0.reshape(Y0.shape[0],Y0.shape[1],1).astype(np.float32)

# %%
X0.shape, Y0.shape

# %%
# predict from last point then forward
place=-1

pred = []
X0=X0[place]
Y0=[Y0[place][0:7].reshape(1,-1)]

for forward_step in range(testX.shape[0]):
  src = Variable(torch.Tensor(X0.reshape(1,-1)) )
  src_mask = Variable(torch.ones(1,1,8))
  current_pred = greedy_decode(model, src, src_mask, max_len=8, start_symbol=1)

  print('** Predicted Value: ', current_pred)
  print('-- Actual Value: ', dataset[trainX.shape[0]+forward_step])
  pred.append( current_pred.detach().numpy() )

  X0=np.append(X0[1:],[[current_pred.detach().numpy()]], axis=0)
  Y0=[np.append(Y0,[[[current_pred.detach().numpy()]]]).reshape(1,-1)]

# %%
plt.plot(dataset[trainX.shape[0]:trainX.shape[0]+testX.shape[0]])

# %%
plt.plot(pred)

# %% [markdown]
# ## Metrics

# %%
# RMSE
mean_squared_error(y_true= dataset[trainX.shape[0]:trainX.shape[0]+testX.shape[0]], y_pred = pred)


# %%




