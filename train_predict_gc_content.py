import tensorflow as tf
from tensorflow.keras import layers
from nn_model import BaseHparams, build_model, build_simple_model, build_simple_model2
import pickle
import numpy as np
import pandas as pd
from one_hot_encode import sequences_to_onehot


# with open("encode_tensor_seqs_test_data.pickle", 'rb') as f:
#     data = pickle.load(f)
data = pd.read_csv("sequences_train_test_10", sep="\t")
data["gc"] = data["gc"].apply(lambda x:x*100)
train_data = data[data["label"]=="train"]
print("____________________________-------------------------------___________________________-------------------")
print("train", len(train_data))
test_data = data[data["label"]=="test"]
print("test", len(test_data))
train_encode_tensor_seqs = sequences_to_onehot(train_data["seq"]).to_tensor()
test_encode_tensor_seqs = sequences_to_onehot(test_data["seq"]).to_tensor()

hparams = BaseHparams()

optimizer = tf.keras.optimizers.RMSprop(0.001)
# tensorboard_callback = tf.keras.callbacks.TensorBoard(
#       hparams.log_dir, histogram_freq=1)
print("____________________________-------------------------------___________________________-------------------")
print("input_shape  ", train_encode_tensor_seqs.shape)
print(train_encode_tensor_seqs)
print("outputshape",np.array(train_data["gc"]).shape)
print(train_data["gc"])
# print(np.array(train_data["gc"]).shape)
# model = build_model(hparams)
model = build_simple_model2(hparams)
print(model.summary())
model.compile(
      optimizer=optimizer,
      loss=tf.keras.losses.sparse_categorical_crossentropy,
      metrics=['accuracy'])
model.fit(train_encode_tensor_seqs, np.array(train_data["gc"]), epochs=2,
      validation_split=0.1,
      verbose=1)
gc_predict = model.predict(test_encode_tensor_seqs, verbose=1)
print("real gc")
print(test_data["gc"])
print("predicted")
print(gc_predict)



