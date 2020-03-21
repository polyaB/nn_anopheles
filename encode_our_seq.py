import tensorflow as tf
import pandas as pd
from one_hot_encode import sequences_to_onehot
import pickle
import numpy as np
# from sklearn.preprocessing import LabelEncoder
# from sklearn.preprocessing import OneHotEncoder
# import tensorflow_io as tfio
data = pd.read_csv("sequences_train_test_10", sep="\t")
#print(data["seq"])
seqs = data.iloc[0:10, 1]
print("seqs")
print()
print(seqs)
#seqs = ["AAGTC", "GGTAA"]
#gc_content = [0.2, 0,4]

# def one_hot_encode(seq):
#     seq_array = array(list(seq))
#
# tensor_seqs = []
# for seq in data["seq"]:
#     tensor_seq =tf.strings.as_string(seq, precision=-1, scientific=False, shortest=False, width=-1, fill='',name=None)
#     tensor_seqs.append(tensor_seq)
# print(tensor_seqs)
# print(data)
#encode_tensor_seqs = sequences_to_onehot(seqs)
encode_tensor_seqs = sequences_to_onehot(data["seq"])
print(encode_tensor_seqs.shape)
encode_tensor_seqs.to_tensor()
print(encode_tensor_seqs.to_tensor().shape)
data["encoded_seq"] = encode_tensor_seqs.to_tensor()
print(data["encoded_seq"].shape)
# print(encode_tensor_seqs)
with open("encode_tensor_seqs_test_data.pickle", 'wb') as f:
    pickle.dump(data, f)



