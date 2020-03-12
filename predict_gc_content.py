import tensorflow as tf
import pandas as pd
import tensorflow_io as tfio

data = pd.read_csv("Z:/nn_anopheles/sequences_train_test", sep="\t")
# tensor_seqs = []
# for seq in data["seq"]:
#     tensor_seq =tf.strings.as_string(seq, precision=-1, scientific=False, shortest=False, width=-1, fill='',name=None)
#     tensor_seqs.append(tensor_seq)
# print(tensor_seqs)
print(data)
encode_tensor_seqs = tfio.genome.sequences_to_onehot(data["seq"])
print(encode_tensor_seqs)