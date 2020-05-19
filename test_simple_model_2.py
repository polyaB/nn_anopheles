import numpy as np
import tensorflow as tf
from tensorflow import keras
from one_hot_encode import sequences_to_onehot
from tensorflow.keras import layers
from plot_history import plot_history
from tensorflow.keras import backend as K
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)
import datetime
import pickle
import pandas as pd

data = pd.read_csv("sequences_train_test_10.txt", sep="\t")
with open("/mnt/scratch/ws/psbelokopytova/202004261138polina_data/nn_anopheles/output/test.pickle", 'rb') as f:
    data = pickle.load(f)
# data["gc"] = data["gc"].apply(lambda x:x/100)
# data["seq"] = data["seq"].apply(lambda x: x.upper())
train_data = data[data["label"]=="train"]
test_data = data[data["label"]=="test"]
print("train", len(train_data))
print("test", len(test_data))
# train_encode_tensor_seqs = sequences_to_onehot(train_data["seq"]).to_tensor()
# test_encode_tensor_seqs = sequences_to_onehot(test_data["seq"]).to_tensor()
# input_shape =

l0 = layers.Flatten(input_shape=(700000, 4))
# l1 = layers.Dense(units=32, activation=tf.nn.relu)
# l4 = layers.Dense(units=16, activation=tf.nn.relu)
l2 = layers.Dense(units=30, activation=tf.nn.sigmoid)
l3 = layers.Dense(units=1)

epochs = 2000

# model4 = tf.keras.models.Sequential([l0,l3])
# model4.compile(
#     optimizer=tf.keras.optimizers.RMSprop(),
#     loss='mean_absolute_error',
#     metrics=['mean_absolute_percentage_error']
# )
# hist_one_neuron = model4.fit(x=train_encode_tensor_seqs,
#                     y=np.array(train_data["gc"]),
#                     validation_split=0.1, epochs=epochs,
#                     verbose=2, use_multiprocessing=True,
#                     )


model2 = tf.keras.models.Sequential([l0,l2,l3])
model2.compile(
    optimizer=tf.keras.optimizers.RMSprop(),
    loss='mean_squared_error',
    metrics=['mean_absolute_percentage_error']
)
# model3 = tf.keras.models.Sequential([l0,l2,l3])
# model3.compile(
#     optimizer=tf.keras.optimizers.RMSprop(),
#     loss='mean_squared_error',
#     metrics=['mean_absolute_percentage_error'])
# model1 = tf.keras.models.Sequential([l0,l2,l3])
# model1.compile(
#     optimizer=tf.keras.optimizers.RMSprop(),
#     loss='mean_squared_error',
#     metrics=['mean_absolute_percentage_error']
# )
print(model2.summary())
hist_base = model1.fit(x=train_encode_tensor_seqs,
                    y=np.array(train_data["gc"]),
                    validation_split=0.1, epochs=epochs,
                    verbose=2, use_multiprocessing=True,
                    )

#                     )
hist_mult_100 = model2.fit(x=train_encode_tensor_seqs,
                    y=np.array(train_data["gc"]*100),
                    validation_split=0.1,
                    verbose=2, epochs=epochs,
                    )
# print(model2.summary())
hist_div_100 = model3.fit(x=train_encode_tensor_seqs,
                    y=np.array(train_data["gc"]/100),
                    validation_split=0.1, epochs=epochs,
                    verbose=2
                    )
# model8 = tf.keras.models.Sequential([l0,l1,l4,l2,l3])
# model8.compile(
#     optimizer=tf.keras.optimizers.RMSprop(),
#     loss='mean_absolute_error',
#     metrics=['mean_absolute_percentage_error']
# )
# print(model8.summary())
# hist_more_layers = model8.fit(x=train_encode_tensor_seqs,
#                     y=np.array(train_data["gc"]),
#                     validation_split=0.1, epochs=epochs,
#                     verbose=2, use_multiprocessing=True,

plot_history([('hist_base', hist_base), ('hist_mult_100', hist_mult_100), ('hist_div_100', hist_div_100)], key='mean_absolute_percentage_error',
             file_name=str(datetime.datetime.now())+"dif_input_MAPE_sigmoid05")
# plot_history([('hist_more_layers', hist_more_layers)], key='mean_absolute_percentage_error',
#              file_name=str(datetime.datetime.now())+"dif_input_MAPE_more_layers")
# keras.losses.mean_squared_error(y_true, y_pred)

# for epoch in range(2000):
#     if epoch % 50 == 0:
#         print("\nEpoch " + str(epoch))
#         hist = model.fit(x=train_encode_tensor_seqs,
#                         y=np.array(train_data["gc"]),
#                         validation_split=0.1,
#                         verbose=2
#                         )
#     else:
#         hist = model.fit(x=train_encode_tensor_seqs,
#                         y=np.array(train_data["gc"]),
#                         validation_split=0.1,
#                         verbose=0
#                         )
# gc_predict = model.predict(test_encode_tensor_seqs, verbose=1)

# truth = list(test_data["gc"])
# print("deltas")
# for i in range(len(gc_predict)):
#     delta = truth[i] - gc_predict[i]
#     print(delta)
#     if(abs(delta) > 10):
#         print("delta > 10 ", test_encode_tensor_seqs[i])
#         print()
