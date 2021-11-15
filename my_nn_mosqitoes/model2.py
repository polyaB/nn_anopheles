import tensorflow as tf
from typing import Tuple
from tensorflow.keras import initializers
from random import randint
from nn_blocks import conv_block


# import tensorflow_addons as tfa

def build_feature_extractor2(
        input_seq_len: int,
        batch_size: int,
) -> Tuple[tf.keras.Model, int]:
    seq_input = tf.keras.layers.Input(shape=(input_seq_len, 4), batch_size=batch_size,
                                      dtype='float32', name='input_sequence')
    current = seq_input
    for i in range(11):
        current = conv_block(current, filters = 96,kernel_size=3, dilation_rate=3, pool_size=2)
    current = tf.keras.layers.Flatten()(current)
    united_embedding_size = 512
    united_emb_layer = tf.keras.layers.Dense(
        units=united_embedding_size,
        activation='relu',
        kernel_initializer=initializers.he_normal(seed=38),
        name='UnitedEmbeddingLayer')

    final_output = united_emb_layer(current)
    fe_model = tf.keras.Model(
        inputs=seq_input,
        outputs=final_output,
        name='FeatureExtractionModel'
    )

    fe_model.build(input_shape=(batch_size, input_seq_len, 4))

    return fe_model, united_embedding_size