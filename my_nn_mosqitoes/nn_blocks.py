import tensorflow as tf
import numpy as np
from cooltools.lib.numutils import set_diag

def conv_block(inputs, filters=None, kernel_size=1, activation='relu', activation_end=None,
    strides=1, dilation_rate=1, l2_scale=0, dropout=0, conv_type='standard', residual=False,
    pool_size=1, batch_norm=False, bn_momentum=0.99, bn_gamma=None, bn_type='standard',
    kernel_initializer='he_normal', padding='same'):
  # Construct dilated convolution block.
  current = inputs
  # activation
  current = tf.keras.layers.ReLU()(current)
  #convolution
  # print("input shape", current.output_shape)
  current = tf.keras.layers.Conv1D(
    filters=filters,
    kernel_size=kernel_size,
    strides=strides,
    padding=padding,
    use_bias=False,
    dilation_rate=dilation_rate,
    kernel_initializer=tf.keras.initializers.he_normal(seed=38),
    kernel_regularizer=tf.keras.regularizers.l2(l2_scale))(current)
  # batch norm
  if batch_norm:
      current = tf.keras.layers.BatchNormalization(momentum=bn_momentum, gamma_initializer=bn_gamma)(current)
  # dropout
  if dropout > 0:
    current = tf.keras.layers.Dropout(rate=dropout, seed=38)(current)

  # # residual add
  # if residual:
  #   current = tf.keras.layers.Add()([inputs, current])
  #
  # # end activation
  # if activation_end is not None:
  #   current = layers.activate(current, activation_end)

  # Pool
  # print("before pool", current.shape)
  if pool_size > 1:
    current = tf.keras.layers.MaxPool1D(
      pool_size=pool_size,
      padding='same')(current)

  return current

def from_upper_triu(vector_repr, matrix_len, num_diags):
  z = np.zeros((matrix_len, matrix_len))
  triu_tup = np.triu_indices(matrix_len, num_diags)
  z[triu_tup] = vector_repr
  for i in range(-num_diags + 1, num_diags):
    set_diag(z, np.nan, i)
  return z + z.T