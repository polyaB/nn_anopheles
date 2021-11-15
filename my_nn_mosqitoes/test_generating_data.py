import pandas as pd
from generate_data import generate_pair_index_table, generate_big_array
import numpy as np

data_folder = "/mnt/scratch/ws/psbelokopytova/202112281307data_Polya/nn_anopheles/dataset_like_Akita/data/test_new/"
seq_bed_data = pd.read_csv("/mnt/scratch/ws/psbelokopytova/202112281307data_Polya/nn_anopheles/dataset_like_Akita/data/test_new/sequences.bed", sep="\t",
names=["chr", "start", "end", "mode"])
combs = generate_pair_index_table(seq_bed_data)
print("len of combinations is ", len(combs))

generate_big_array(data_folder+"seqs_cov/0.h5", data_folder+"sequences.bed", 
                                    "/mnt/scratch/ws/psbelokopytova/202112281307data_Polya/nn_anopheles/input/genomes/AatrE3_V4.fa", data_folder)
newfp_inp = np.memmap(data_folder+"inputs", dtype='float16', mode='r', shape=(len(seq_bed_data), 1048576, 4))
print("sample from input", newfp_inp[0,:][0])
newfp = np.memmap(data_folder+"targets", dtype='float16', mode='r', shape=(len(seq_bed_data), 99681))
print("sample from target", newfp[0])

