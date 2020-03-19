from Bio import SeqIO
from Bio.SeqUtils import GC
import pandas as pd
import numpy as np
import random

path_to_fasta = "E:/RNA-seq_anopheles/FASTA/AcolNg_V3.fa"
records = list(SeqIO.parse(path_to_fasta, "fasta"))
print(records)
seq_slices = []
GC_content = []
len_seq=10
# Now it generates data only for one chromosome
for record in records:
    i = 0
    for start in range(100000,len(record.seq)-300000+1, len_seq//2):
        slice = record.seq[start:start+len_seq]
        seq_slices.append(str(slice))
        GC_content.append(GC(slice))
        i+=1
        if i ==500:
            break
    break
    # print(record.id)
    # print(len(record.seq))
data = pd.DataFrame(data={'seq':seq_slices, 'gc':GC_content})
# data = pd.DataFrame(np.array([seq_slices, GC_content]), columns=["seq", "gc"])
data["label"] = ["train"]*len(data)
#data = data.sample(500)
ids_test = random.sample(range(0,len(data)), len(data)*20//100)
data.iloc[ids_test, data.columns.get_loc("label")] = "test"
data.to_csv("Z:/nn_anopheles/sequences_train_test_10", sep="\t", index=False)
print(data)
print(GC_content)

