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





