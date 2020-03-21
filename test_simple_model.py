import pandas as pd
import numpy as np
import tensorflow as tf
from one_hot_encode import sequences_to_onehot
from tensorflow.keras import layers
from tensorflow.keras import backend as K
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)



data = pd.read_csv("sequences_train_test_10", sep="\t")
data["gc"] = data["gc"].apply(lambda x:x*100)
train_data = data[data["label"]=="train"]
test_data = data[data["label"]=="test"]
print("train", len(train_data))
print("test", len(test_data))
train_encode_tensor_seqs = sequences_to_onehot(train_data["seq"]).to_tensor()
test_encode_tensor_seqs = sequences_to_onehot(test_data["seq"]).to_tensor()

l0 = layers.Flatten(input_shape=(10, 4))
l1 = layers.Dense(units=128, activation=tf.nn.relu)
l2 = layers.Dense(units=1)
model = tf.keras.models.Sequential([l0,l1,l2])
model.compile(
      optimizer="RMSprop",
      loss=tf.keras.losses.sparse_categorical_crossentropy,
      metrics=['accuracy'])

print(model.summary)

hist = model.fit(train_encode_tensor_seqs, np.array(train_data["gc"]), epochs=2,
      validation_split=0.1,
      verbose=1)
gc_predict = model.predict(test_encode_tensor_seqs, verbose=1)
print("real gc")
print(test_data["gc"])
print("predicted")
print(gc_predict)

print("These are the layer variables: {}".format(layers.Dense(units=128, activation=tf.nn.relu).get_weights()))