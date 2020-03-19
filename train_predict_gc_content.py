import tensorflow as tf
from tensorflow.keras import layers
from nn_model import BaseHparams, build_model
import pickle
import numpy as np

with open("encode_tensor_seqs_test_data.pickle", 'rb') as f:
    data = pickle.load(f)
data["encoded_seq"] = data["encoded_seq"].apply(lambda x: x.to_tensor())
train_data = data[data["label"]=="train"]
print(len(train_data))
test_data = data[data["label"]=="test"]
print("test", len(test_data))
hparams = BaseHparams()

optimizer = tf.keras.optimizers.Adam(lr=hparams.learning_rate)
tensorboard_callback = tf.keras.callbacks.TensorBoard(
      hparams.log_dir, histogram_freq=1)
model = build_model(hparams)
model.compile(
      optimizer=optimizer,
      loss=tf.keras.losses.sparse_categorical_crossentropy,
      metrics=['accuracy'])
model.fit(np.array(train_data["encoded_seq"]), train_data["gc"], epochs=hparams.total_epochs,
      validation_split=0.1,
      callbacks=[tensorboard_callback],
      verbose=1)
gc_predict = model.predict(test_data["encoded_seq"], verbose=1)
print("real gc")
print(test_data["gc"])
print("predicted")
print(gc_predict)



