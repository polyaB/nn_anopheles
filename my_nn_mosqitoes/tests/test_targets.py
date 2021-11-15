import numpy as np
import pandas as pd
import h5py

data_folder = "/mnt/scratch/ws/psbelokopytova/202112281307data_Polya/nn_anopheles/dataset_like_Akita/data/test_new/"
seq_bed_data = pd.read_csv(data_folder+"sequences.bed", sep="\t",
names=["chr", "start", "end", "mode"])
target_data = np.memmap(data_folder+"targets_5", dtype='float16', mode='r', shape=(len(seq_bed_data), 99681))
input_data = np.memmap(data_folder+"inputs", dtype='float16', mode='r', shape=(len(seq_bed_data), 1048576, 4))
print(len(target_data))
for index,target in enumerate(target_data):
    # print(np.count_nonzero(target))
    if np.count_nonzero(target)==0:
        print("index", index)
        break


# for index,input in enumerate(input_data):
#     print(np.count_nonzero(input))
#     if np.count_nonzero(input)==0:
#         print("index", index)
#         break

# seq_pool_len_hic = h5py.File(data_folder+"seqs_cov/1.h5", 'r')['targets'].shape[1]
# num_targets = len(seq_bed_data)
# targets = np.zeros((num_targets, seq_pool_len_hic), dtype='float16')
# # read target file
# seqs_cov_open = h5py.File(data_folder+"seqs_cov/0.h5", 'r')
# for index,target in enumerate(seqs_cov_open['targets']):
#     print(np.count_nonzero(target))
#     if np.count_nonzero(target)==0:
#         print("index", index)
#         break
# targets[:,:] = seqs_cov_open['targets'][:,:]
# seqs_cov_open.close()