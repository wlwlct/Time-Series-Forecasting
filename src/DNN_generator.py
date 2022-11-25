# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 01:04:35 2022

@author: asifuzzaman
"""

# univariate one step problem
from numpy import array
from keras.preprocessing.sequence import TimeseriesGenerator
# define dataset
series = array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# define generator
n_input = 3
generator = TimeseriesGenerator(series, series, length=n_input, batch_size=1)
# number of samples
print('Samples: %d' % len(generator))
# print each sample
for i in range(len(generator)):
	x, y = generator[i]
	print('%s => %s' % (x, y))



test = [['SARIMA', 1, 1]]


t1 = time.time()
train_time = time.time()-t1

test_time = time.time()-t1

# executing SARIMA 
df_test['SARIMA'], train_time, test_time = 

# calculating RMSE
RMSE = sqrt(mean_squared_error(df_test['Adj Close'], df_test['SARIMA']))

# storing model results 
Model_results = [['SARIMA', RMSE, train_time, test_time]]


""" ? """
def forecast_?(train_data, n_steps_in, n_steps_out, n_features, units, epochs, activation, patience_level):
    
    # converting to numpy array
    index = np.array(train_data)
    
    # split into samples
    X, y = split_sequence(index, n_steps_in, n_steps_out)
    
    # defining ? model    
    
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

# executing ?  
df_test['?'], train_time, test_time = forecast_?(train_data = df_train["scaled_adj_close"], n_steps_in=50, n_steps_out=30, n_features=1, units=50, epochs=100, activation='relu', patience_level=5)

# calculating RMSE
RMSE = sqrt(mean_squared_error(df_test['Adj Close'], df_test['?']))

# storing model results 
Model_results.append(['?', RMSE, train_time, test_time])


int(100/4)