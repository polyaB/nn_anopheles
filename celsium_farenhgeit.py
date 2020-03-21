import pandas as pd
import numpy as np
import tensorflow as tf
from one_hot_encode import sequences_to_onehot
from tensorflow.keras import layers
from tensorflow.keras import backend as K
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)


celsius_q    = np.array([-40, -10,  0,  8, 15, 22,  38],  dtype=float)
fahrenheit_a = np.array([-40,  14, 32, 46, 59, 72, 100],  dtype=float)

l0 = tf.keras.layers.Dense(units=4, input_shape=[1])
l1 = tf.keras.layers.Dense(units=1, input_shape=[1])
model = tf.keras.Sequential([l0])
model.compile(loss='mean_squared_error',
              optimizer=tf.keras.optimizers.Adam(0.1))
history = model.fit(celsius_q, fahrenheit_a, epochs=500, verbose=False)

inp = model.input                                           # input placeholder
outputs = [layer.output for layer in model.layers]          # all layer outputs
functors = [K.function([inp], [out]) for out in outputs]    # evaluation functions
# Testing
test = np.random.random([1])[np.newaxis,...]
layer_outs = [func([test]) for func in functors]
print (layer_outs)
print("Finished training the model")
print(model.predict([100.0]))
print("These are the layer variables: {}".format(l0.get_weights()))
