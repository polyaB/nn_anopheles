import tensorflow as tf
from tensorflow import keras
import layers

seq_len = 700000 #TODO create classs for sequences
sequence = keras.Input(shape=(seq_len, 4), name = "sequence")
current = sequence
# augmentation
current, reverse_bool = layers.StochasticReverseComplement()(current)
current = layers.StochasticShift(11)(current) #augment shift=11
for i in range(10):
    current = keras.layers.ReLU()(current)
    current = keras.layers.Conv1D(filters=96, kernel_size=11, padding='same', strides=1)(current)
    current = keras.layers.BatchNormalization(momentum = 0.9265,gamma_initializer='zeros',
      fused=None)(current)
    current = keras.layers.MaxPool1D(pool_size=2, padding='same')(current)
for i in range(2):
    current = keras.layers.ReLU()(current)
    current = keras.layers.Conv1D(filters=96, kernel_size=11, padding='same', strides=1)(current)
    current = keras.layers.BatchNormalization(momentum=0.9265, gamma_initializer='zeros',
                                              fused=None)(current)
current = keras.layers.Dropout()
# make model trunk
trunk_output = current
model_trunk = tf.keras.Model(inputs=sequence, outputs=trunk_output)
