# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 19:46:50 2022

@author: asifu
"""
import numpy as np
import time
import tensorflow as tf
tf.__version__
tf.config.list_physical_devices('GPU')

s = 3000

a = np.random.randint(10, size=(s, s))
b = np.random.randint(10, size=(s, s))

t1 = time.time()
c = np.matmul(a,b)
print("Numpy:")
print(time.time()-t1)

t1 = time.time()
c = tf.matmul(a,b)
print("Tensorflow: GPU")
print(time.time()-t1)

with tf.device('/CPU:0'):
    t1 = time.time()
    c = tf.matmul(a,b)
    print("Tensorflow: CPU")
    print(time.time()-t1)