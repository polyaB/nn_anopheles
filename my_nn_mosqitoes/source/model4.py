import tensorflow as tf
from typing import Tuple
from tensorflow.keras import initializers
from random import randint
from nn_blocks import conv_block, one_to_two, concat_dist_2d, cropping_2d, upper_tri, substraction, conv_block_3x


# import tensorflow_addons as tfa

def build_feature_extractor(
        input_seq_len: int,
        batch_size: int,
) -> Tuple[tf.keras.Model, int]:
    seq_input = tf.keras.layers.Input(shape=(input_seq_len, 4), batch_size=batch_size,
                                      dtype='float32', name='input_sequence')
    current = seq_input
    #
    current = conv_block_3x(current, filters = 64,kernel_size=3, dilation_rate=1, pool_size=2, batch_norm=True, activation='elu')
    current = conv_block_3x(current, filters=64, kernel_size=3, dilation_rate=1, pool_size=2, batch_norm=True,
                            activation='elu')
    current = conv_block_3x(current, filters=128, kernel_size=3, dilation_rate=1, pool_size=2, batch_norm=True,activation='elu')
    current = conv_block_3x(current, filters=128, kernel_size=3, dilation_rate=1, pool_size=2, batch_norm=True,
                            activation='elu')
    current = conv_block_3x(current, filters=256, kernel_size=3, dilation_rate=1, pool_size=2, batch_norm=True,
                            activation='elu')
    current = conv_block_3x(current, filters=256, kernel_size=3, dilation_rate=1, pool_size=2, batch_norm=True,
                            activation='elu')
    current = conv_block_3x(current, filters=512, kernel_size=3, dilation_rate=1, pool_size=2, batch_norm=True,
                            activation='elu')
    current = conv_block_3x(current, filters=1024, kernel_size=3, dilation_rate=1, pool_size=2, batch_norm=True,
                            activation='elu')
    final_output = tf.keras.layers.GlobalAveragePooling1D()(current)

    fe_model = tf.keras.Model(
        inputs=seq_input,
        outputs=final_output,
        name='FeatureExtractionModel'
    )
    united_embedding_size = final_output.shape[-1]
    fe_model.build(input_shape=(batch_size, input_seq_len, 4))

    return fe_model, united_embedding_size


def build_twin_regressor(united_embedding_size: int,
                         batch_size: int,
                         target_size: int) -> tf.keras.Model:
    left_input = tf.keras.layers.Input(shape=(united_embedding_size,), dtype=tf.float32,
                                       batch_size=batch_size, name='features_left')
    right_input = tf.keras.layers.Input(shape=(united_embedding_size,), dtype=tf.float32,
                                        batch_size=batch_size, name='features_right')
    concatenated_features = tf.keras.layers.Concatenate(
        name='ConcatFeatures'
    )([left_input, right_input])
    regression_layer = tf.keras.layers.Dense(
        units=target_size, input_dim=united_embedding_size * 2, activation=None,
        kernel_initializer=tf.keras.initializers.GlorotNormal(seed=42),
        bias_initializer='zeros',
        name='RegressionLayer'
    )(concatenated_features)
    twin_regression_model = tf.keras.Model(
        inputs=[left_input, right_input],
        outputs=regression_layer,
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
    # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    #     initial_learning_rate=1e-2, decay_steps=2, decay_rate=0.9)
    # radam = tf.optimizers.RectifiedAdam(learning_rate=1e-5)
    # ranger = tf.optimizers.Lookahead(radam, sync_period=6, slow_step_size=0.5)
    siamese_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss=tf.keras.losses.MeanSquaredError())
    return siamese_model, fe_layer, regression_model