import tensorflow as tf
from typing import Tuple
from tensorflow.keras import initializers
from random import randint
from nn_blocks import conv_block


# import tensorflow_addons as tfa

def build_feature_extractor(
        input_seq_len: int,
        batch_size: int,
) -> Tuple[tf.keras.Model, int]:
    seq_input = tf.keras.layers.Input(shape=(input_seq_len, 4), batch_size=batch_size,
                                      dtype='float32', name='input_sequence')
    current = seq_input
    for i in range(11):
        current = conv_block(current, filters = 96,kernel_size=11, dilation_rate=1, pool_size=2, batch_norm=False)
    current = tf.keras.layers.Conv1D(filters=1, kernel_size=11, dilation_rate=1, padding='same')(current)
    current=tf.squeeze(current)

    united_embedding_size = 512
    united_emb_layer1 = tf.keras.layers.Dense(
        units=united_embedding_size,
        kernel_initializer=initializers.he_normal(seed=38),
        name='UnitedEmbeddingLayer1')

    united_emb_layer2 = tf.keras.layers.Dense(
        units=united_embedding_size,
        kernel_initializer=initializers.he_normal(seed=38),
        name='UnitedEmbeddingLayer2')

    united_emb_layer3 = tf.keras.layers.Dense(
        units=united_embedding_size,
        kernel_initializer=initializers.he_normal(seed=38),
        name='UnitedEmbeddingLayer3')

    current = united_emb_layer1(current)
    current = tf.keras.activations.relu(current)
    current = united_emb_layer2(current)

    current = one_to_two(current, operation='mean')
    #add information about distance
    current = concat_dist_2d(current)
    #crooping layer, cause we cropped matrix on edges during train preparing
    current = cropping_2d(current, cropping=cropping)
    #get the upper triu from the matrix
    current = upper_tri(current, diagonal_offset=diagonal_offset)

    fe_model = tf.keras.Model(
        inputs=seq_input,
        outputs=final_output,
        name='FeatureExtractionModel'
    )

    fe_model.build(input_shape=(batch_size, input_seq_len, 4))

    return fe_model, united_embedding_size


def build_twin_regressor(united_embedding_size: int,
                         batch_size: int,
                         target_size: int) -> tf.keras.Model:
    left_input = tf.keras.layers.Input(shape=(united_embedding_size,), dtype=tf.float32,
                                       batch_size=batch_size, name='features_left')
    right_input = tf.keras.layers.Input(shape=(united_embedding_size,), dtype=tf.float32,
                                        batch_size=batch_size, name='features_right')
    diff = left_input - right_input
    twin_regression_model = tf.keras.Model(
        inputs=[left_input, right_input],
        outputs=diff,
        name='TwinRegressionModel'
    )
    twin_regression_model.build(input_shape=[(batch_size, united_embedding_size),
                                             (batch_size, united_embedding_size)])
    return twin_regression_model


def build_neural_network(
        seq_len: int,
        batch_size: int,
        target_size: int) -> Tuple[tf.keras.Model, tf.keras.Model,
                                   tf.keras.Model]:
    fe_layer, emb_units = build_feature_extractor(input_seq_len=seq_len, batch_size=batch_size)
    left_input = tf.keras.layers.Input(shape=(seq_len, 4), batch_size=batch_size,
                                       dtype='float32', name='left_sequence')

    right_input = tf.keras.layers.Input(shape=(seq_len, 4), batch_size=batch_size,
                                        dtype='float32', name='right_sequence')

    left_output = fe_layer(left_input)
    right_output = fe_layer(right_input)
    regression_model = build_twin_regressor(emb_units, batch_size, target_size)
    regression_layer = regression_model([left_output, right_output])
    siamese_model = tf.keras.Model(
        inputs=[left_input,
                right_input],
        outputs=regression_layer,
        name='SiameseModel'
    )
    # radam = tf.optimizers.RectifiedAdam(learning_rate=1e-5)
    # ranger = tf.optimizers.Lookahead(radam, sync_period=6, slow_step_size=0.5)
    siamese_model.compile(optimizer='Adam', loss=tf.keras.losses.MeanSquaredError())
    return siamese_model, fe_layer, regression_model