import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# x = np.array([[200.0, 16.0]]) # x is the vector with the inputs, so, it's a0 actually  (NOTE, USE DOUBLE BRACKETS, IF YOU USE ONLY 1 PAIR OF THEM THEN, FOR PY, IS A 1D VECTOR AND TENSORFLOW DOESN'T LIKE IT, HE LIKE A MATRIX INTEAD.)
# layer_1 = layers.Dense(units=3, activation='sigmoid') # units means neurons, and the layer is the rectangle that we saw from the lecture
# a1= layer_1(x) #a1 is the result of activation, think about it in terms of probability! it's a vector of probabilities 

# layer_2 = layers.Dense(units=1, activation='sigmoid')
# a2= layer_2(a1)

# ^^^^ ^^^^ ^^^^ this is so MONKEY. USE Sequential INSTEAD. 

x = np.array([[200.0, 16.0]])

y = ... # some array

model = Sequential([
    layers.Dense(units=3,activation='relu'), #relu for hidden layers always, faster learning.
    layers.Dense(units=1,activation='sigmoid')
])

model.compile(loss=...)  # this is where you select your cost function

model.fit(X,y,epochs=n) # that's backpropagation, the epochs are the number of times of iterations of the gradient descent. Choose a good learning rate and follow the principles.

model.predict(x_new)  # that's forward pass.
