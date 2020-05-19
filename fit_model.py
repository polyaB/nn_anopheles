import pickle
import tensorflow as tf

with open("/mnt/scratch/ws/psbelokopytova/202006201036data/nn_anopheles/output/test_train_dataset.pickle", 'rb') as f:
    data = pickle.load(f)
inputs = data["inputs"]
targets = data["tergets"]
inputs = tf.convert_to_tensor(inputs, dtype=tf.int64)
print(inputs.shape)
print(data.keys())
print(data["intervals"])