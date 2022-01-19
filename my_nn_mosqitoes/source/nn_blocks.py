import sys
import os
source_path = os.path.dirname(os.path.abspath(sys.argv[0])) + "/../../"
sys.path.append(source_path)
import tensorflow as tf
import numpy as np
from cooltools.lib.numutils import set_diag
from basenji.basenji import layers
from my_nn_mosqitoes.source import my_nn_layers
def conv_block(inputs, filters=None, kernel_size=1, activation='relu', activation_end=None,
    strides=1, dilation_rate=1, l2_scale=0, dropout=0, conv_type='standard', residual=False,
    pool_size=1, batch_norm=False, bn_momentum=0.99, bn_gamma=None, bn_type='standard',
    kernel_initializer='he_normal', padding='same'):
  # Construct dilated convolution block.
  current = inputs
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
    kernel_regularizer=tf.keras.regularizers.l2(l2_scale),
    activation=activation)(current)

  # activation
  # current = tf.keras.layers.ReLU()(current)
  # current = tf.keras.activations.sigmoid(current)
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

def conv_block_3x(inputs, filters=None, kernel_size=1, activation='relu', activation_end=None,
    strides=1, dilation_rate=1, l2_scale=0, dropout=0, conv_type='standard', residual=False,
    pool_size=1, batch_norm=False, bn_momentum=0.99, bn_gamma=None, bn_type='standard',
    kernel_initializer='he_normal', padding='same', repeat_conv=3):
  current = inputs
    #convolution
  # print("input shape", current.output_shape)
  for i in range(repeat_conv):
    current = tf.keras.layers.Conv1D(
      filters=filters,
      kernel_size=kernel_size,
      strides=strides,
      padding=padding,
      use_bias=False,
      dilation_rate=dilation_rate,
      kernel_initializer=tf.keras.initializers.he_normal(seed=38),
      activation=activation)(current)
      # batch norm
    if batch_norm:
        current = tf.keras.layers.BatchNormalization(momentum=bn_momentum, gamma_initializer=bn_gamma)(current)
  # dropout
  if dropout > 0:
    current = tf.keras.layers.Dropout(rate=dropout, seed=38)(current)
  # Pool
  # print("before pool", current.shape)
  if pool_size > 1:
    current = tf.keras.layers.AveragePooling1D(
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

def one_to_two(inputs, operation='mean', **kwargs):
  current = layers.OneToTwo(operation)(inputs)
  return current

def concat_dist_2d(inputs, **kwargs):
  current = layers.ConcatDist2D()(inputs)
  return current

def cropping_2d(inputs, cropping, **kwargs):
  current = tf.keras.layers.Cropping2D(cropping)(inputs)
  return current

def upper_tri(inputs, diagonal_offset=2, **kwargs):
  current = layers.UpperTri(diagonal_offset)(inputs)
  return current

def substraction(input_left, input_right,**kwargs):
  current = tf.math.subtract(input_left,input_right)
  return current
