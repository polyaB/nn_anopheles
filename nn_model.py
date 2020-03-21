import tensorflow as tf
from tensorflow.keras import layers

_ALLOWED_BASES = 'ACGT'
class BaseHparams(object):
  """Default hyperparameters."""

  def __init__(self,
               total_epochs=100,
               learning_rate=0.004,
               l2=0.001,
               batch_size=256,
               window_size=21,
               ref_path='hs37d5.fa.gz',
               vcf_path='NA12878_calls.vcf.gz',
               bam_path='NA12878_sliced.bam',
               out_dir='examples',
               model_dir='ngs_model',
               log_dir='logs'):

    self.total_epochs = total_epochs
    self.learning_rate = learning_rate
    self.l2 = l2
    self.batch_size = batch_size
    self.window_size = window_size
    self.ref_path = ref_path
    self.vcf_path = vcf_path
    self.bam_path = bam_path
    self.out_dir = out_dir
    self.model_dir = model_dir
    self.log_dir = log_dir

def build_model(hparams):
  """Convolutional neural network architecture."""

  l2_reg = tf.keras.regularizers.l2

  return tf.keras.models.Sequential([

      # Two convolution + maxpooling blocks
      layers.Conv1D(
          filters=1,
          kernel_size=10,
          activation=tf.nn.relu,
          kernel_regularizer=l2_reg(hparams.l2)),
      layers.MaxPool1D(pool_size=3, strides=1),
      layers.Conv1D(
          filters=1,
          kernel_size=3,
          activation=tf.nn.relu,
          kernel_regularizer=l2_reg(hparams.l2)),
      layers.MaxPool1D(pool_size=3, strides=1),

      # Flatten the input volume
      layers.Flatten(),

      # Two fully connected layers, each followed by a dropout layer
      layers.Dense(
          units=16,
          activation=tf.nn.relu,
          kernel_regularizer=l2_reg(hparams.l2)),
      layers.Dropout(rate=0.3),
      layers.Dense(
          units=16,
          activation=tf.nn.relu,
          kernel_regularizer=l2_reg(hparams.l2)),
      layers.Dropout(rate=0.3),

      # Output layer with softmax activation
      layers.Dense(units=1, activation='softmax')
  ])

def build_simple_model(hparams):
    inputs = tf.keras.Input(shape=(10,4))
    print(inputs)
    x = tf.keras.layers.Dense(20, activation=tf.nn.relu)(inputs)
    outputs = tf.keras.layers.Dense(1, activation=tf.nn.softmax)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def build_simple_model2(hparams):
    return tf.keras.models.Sequential([
        layers.Flatten(input_shape=(10, 4)),
        layers.Dense(units=128, activation=tf.nn.relu),
        # layers.Flatten(),
        # layers.Dense(20, activation=tf.nn.relu),
        layers.Dense(units=1)
    ])