# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 22:42:00 2022

@author: asifuzzaman
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

""" RMSE """
df_results = pd.read_csv(r"C:\Git\Time-Series-Forecasting\Data\raw\SNPTSX_Output_Final.csv")


# plotting the RMSE 
plt.figure(figsize=(10,7))
plt.bar(df_results['Model'],df_results['RMSE'], color=['blue', 'blue','blue','limegreen','blue','blue','blue'])
plt.legend(loc='best')
plt.title("Model Accuracy: RMSE", fontsize=30)
plt.xticks(rotation = 60)
plt.rcParams.update({'font.size': 24})
plt.xlabel('Methods', fontsize=24)
plt.ylabel('RMSE', fontsize=24)
plt.ylim(np.arrange(0,400,50))
plt.tight_layout()

""" Stability RMSE """
rs = pd.read_csv(r"C:\Git\Time-Series-Forecasting\Data\raw\SNPTSX_Output_10_iterations.csv")

rs_RMSE = rs.drop(['Train_time', 'Test_time'], axis=1) 
rs_RMSE = rs_RMSE.pivot(index='Run', columns='Model', values='RMSE')
rs_RMSE = rs_RMSE[["RNN","Vanilla_LSTM","Stacked_LSTM","Bidirect_LSTM","CNN_LSTM","Encode_De_LSTM"]]

# box plot 
plt.figure(figsize=(10,7))
box = plt.boxplot(rs_RMSE, patch_artist=True)
plt.xticks([1, 2, 3, 4, 5, 6], rs_RMSE.columns)
plt.title("Model Stability: RMSE Boxplot of 10 Iterations", fontsize=30)
plt.xticks(rotation = 60)
plt.rcParams.update({'font.size': 24})
plt.xlabel('Methods', fontsize=24)
plt.ylabel('RMSE', fontsize=24)
#plt.ylim(np.arrange(0,600,50))

colors = ['blue','blue','limegreen','blue','blue','blue']
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
                        
plt.show()

""" Actual vs Predicted (Stock) """ 
df_pred = pd.read_csv(r"C:\Git\Time-Series-Forecasting\Data\raw\parts\SNPTSX_Predictions_10.csv")

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

#fig, axs = plt.subplots(2, 4, sharey=True, figsize=(35,7))
fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7) = plt.subplots(1, 7, sharey=True, figsize=(65,14))

ax1.plot(df_pred_sc["Adj Close"], color='darkorange', linestyle='dotted', linewidth=6.0, label='Actual')
ax1.plot(df_pred_sc["SARIMA"], color='limegreen', linewidth=3.0, label='Forecasted')
ax1.set_title('SARIMA', fontsize=24)

ax2.plot(df_pred_sc["Adj Close"], color='darkorange', linestyle='dotted', linewidth=6.0)
ax2.plot(df_pred_sc["RNN"], color='blue', linewidth=3.0)
ax2.set_title('RNN', fontsize=24)

ax3.plot(df_pred_sc["Adj Close"], color='darkorange', linestyle='dotted', linewidth=6.0)
ax3.plot(df_pred_sc["Vanilla_LSTM"], color='blue', linewidth=3.0)
ax3.set_title('Vanilla LSTM', fontsize=24)

ax4.plot(df_pred_sc["Adj Close"], color='darkorange', linestyle='dotted', linewidth=6.0)
ax4.plot(df_pred_sc["Stacked_LSTM"], color='limegreen', linewidth=3.0)
ax4.set_title('Stacked LSTM', fontsize=24)

ax5.plot(df_pred_sc["Adj Close"], color='darkorange', linestyle='dotted', linewidth=6.0)
ax5.plot(df_pred_sc["Bidirect_LSTM"], color='blue', linewidth=3.0)
ax5.set_title('Bidirect LSTM', fontsize=24)

ax6.plot(df_pred_sc["Adj Close"], color='darkorange', linestyle='dotted', linewidth=6.0)
ax6.plot(df_pred_sc["CNN_LSTM"], color='blue', linewidth=3.0)
ax6.set_title('CNN LSTM', fontsize=24)

ax7.plot(df_pred_sc["Adj Close"], color='darkorange', linestyle='dotted', linewidth=6.0)
ax7.plot(df_pred_sc["Encode_De_LSTM"], color='blue', linewidth=3.0)
ax7.set_title('Encoder De LSTM', fontsize=24)

ax1.set_xlabel('No of Days', fontsize=24)
ax1.set_ylabel('Adj Close (Scaled)', fontsize=24)
ax2.set_xlabel('No of Days', fontsize=24)
ax3.set_xlabel('No of Days', fontsize=24)
ax4.set_xlabel('No of Days', fontsize=24)
ax5.set_xlabel('No of Days', fontsize=24)
ax6.set_xlabel('No of Days', fontsize=24)
ax7.set_xlabel('No of Days', fontsize=24)

#ax2.legend(bbox_to_anchor=(1, 1), fontsize=14) # legend
ax1.legend(loc='upper left', fontsize=20)
fig.suptitle('Stock Index: Actual vs Forecasted with scaled Y-axis', fontsize=30)

fig.tight_layout()
plt.show()



""" Actual vs Predicted (Air Passengers) """ 
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

fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(10,7))

ax1.plot(df_pred_sc["Passengers"], color='darkorange', linestyle='dotted', linewidth=6.0, label='Actual')
ax1.plot(df_pred_sc["SARIMA"], color='limegreen', linewidth=3.0, label='Forecasted')
ax1.set_title('SARIMA', fontsize=24)

ax2.plot(df_pred_sc["Passengers"], color='darkorange', linestyle='dotted', linewidth=6.0,)
ax2.plot(df_pred_sc["Stacked_LSTM"], color='blue', linewidth=3.0)
ax2.set_title('Stacked LSTM', fontsize=24)

ax1.set_xlabel('No of Months', fontsize=24)
ax1.set_ylabel('No of Passengers (Scaled)', fontsize=24)
ax2.set_xlabel('No of Months', fontsize=24)

#ax2.legend(bbox_to_anchor=(1, 1), fontsize=14) # legend
ax1.legend(loc='upper left', fontsize=20)

# Use the pyplot interface to change just one subplot...
plt.sca(ax1)
plt.xticks( [0, 10, 20, 30])
plt.sca(ax2)
plt.xticks( [0, 10, 20, 30])

fig.suptitle('Air Passengers Forecasted (scaled Y-axis)', fontsize=30)

fig.tight_layout()
plt.show()