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
# Now it generates data only for one chromosome
for record in records:
    for start in range(100000,len(record.seq)-300000+1, 100000):
        slice = record.seq[start:start+200000]
        seq_slices.append(str(slice))
        GC_content.append(GC(slice))
    break
    # print(record.id)
    # print(len(record.seq))
data = pd.DataFrame(data={'seq':seq_slices, 'gc':GC_content})
# data = pd.DataFrame(np.array([seq_slices, GC_content]), columns=["seq", "gc"])
data["label"] = ["train"]*len(data)
ids_test = random.sample(range(0,len(data)), len(data)*20//100)
data.iloc[ids_test, data.columns.get_loc("label")] = "test"
data.to_csv("Z:/nn_anopheles/sequences_train_test", sep="\t", index=False)
print(data)
print(GC_content)

