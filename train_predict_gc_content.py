import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from plot_history import plot_history
from tensorflow.keras import backend as K
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)
import datetime
import pickle
import pandas as pd
import sys
import os
import tensorflow as tf
import random
source_path = os.path.dirname(os.path.abspath(sys.argv[0])) + "/3Dpredictor/source"
source_path2 = os.path.dirname(os.path.abspath(sys.argv[0])) + "/3Dpredictor/nn/source"
sys.path.append(source_path)
sys.path.append(source_path2)
from GapReader import GapReader
from fastaFileReader import fastaReader, GC_content
from shared import Interval
import logging
import pandas as pd
import pickle
import numpy as np
logging.basicConfig(format='%(asctime)s %(name)s: %(message)s', datefmt='%I:%M:%S', level=logging.DEBUG)
with open("/mnt/scratch/ws/psbelokopytova/202004261138polina_data/nn_anopheles/output/test.pickle", 'rb') as f:
    data = pickle.load(f)
print(data["train_seq"].shape)
print(data["train_seq"][0].shape)
input_shape =(700000, 4)

l0 = layers.Flatten(input_shape=input_shape)
# l1 = layers.Dense(units=32, activation=tf.nn.relu)
# l4 = layers.Dense(units=16, activation=tf.nn.relu)
l2 = layers.Dense(units=30, activation=tf.nn.sigmoid)
l3 = layers.Dense(units=1)

epochs = 2700
model = tf.keras.models.Sequential([l0,l2,l3])
model.compile(
    optimizer=tf.keras.optimizers.RMSprop(),
    loss='mean_squared_error',
    metrics=['mean_absolute_percentage_error']
)
print(model.summary())
hist_base = model.fit(x=data["train_seq"],
                    y=data["train_gc"],
                    validation_split=0.1, epochs=epochs,
                    verbose=2, use_multiprocessing=True,
                    )

plot_history([('hist_base', hist_base)], key='mean_absolute_percentage_error',
             file_name="700kb_test"+"MAPE_6000epochs")
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
gc_predict = model.predict(data["test_seq"], verbose=1)
#
truth = list(data["test_gc"])
print("deltas")
for i in range(len(gc_predict)):
    delta = truth[i] - gc_predict[i]
    print(delta)
    if(abs(delta) > 10):
        print("delta > 10 ", data["test_seq"][i])
        print()
