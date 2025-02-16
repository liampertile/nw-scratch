import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


x = np.array([200.0, 16.0]) # x is the vector with the inputs, so, it's a0 actually
x = x.reshape(1, -1) # that array above is 1d, so with this, i create a 2d. Because keras with layers.Dense only admit 2d vectors
layer_1 = layers.Dense(units=3, activation='sigmoid') # units means neurons, and the layer is the rectangle that we know from the lecture
a1= layer_1(x) #a1 is the result of activation, think about it in terms of probability! it's a vector of probabilities 

layer_2 = layers.Dense(units=1, activation='sigmoid')
a2= layer_2(a1)
