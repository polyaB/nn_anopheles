# import os
# import sys
# source_path = os.path.dirname(os.path.abspath(sys.argv[0])) + "/basenji/source"
# source_path2 = os.path.dirname(os.path.abspath(sys.argv[0])) + "/basenji/basenji"
# source_path3 = os.path.dirname(os.path.abspath(sys.argv[0])) + "/3Dpredictor/source"
# source_path4 = os.path.dirname(os.path.abspath(sys.argv[0])) + "/source"
# sys.path.append(source_path)
# sys.path.append(source_path2)
# sys.path.append(source_path3)
# sys.path.append(source_path4)
#
# from shared import Interval
# import json
# import dataset, dna_io, seqnn
# from Predictions_interpeter import from_upper_triu, predict_big_region_from_seq
import numpy as np
import pandas as pd
# import cooler
# import matplotlib.pyplot as plt
#
# genome_hic_cool = cooler.Cooler("/mnt/scratch/ws/psbelokopytova/202103211631polina/nn_anopheles/input/coolers/Aste_2048.cool")
# seq_hic_raw = genome_hic_cool.matrix(balance=True).fetch(('2L', 20000000, 21998848))
# print(seq_hic_raw.shape)
# im = plt.matshow(mean_array, fignum=False, cmap='RdBu_r')  # , vmax=2, vmin=-2)
# plt.colorbar(im, fraction=.04, pad=0.05)  # , ticks=[-2,-1, 0, 1,2])
# plt.savefig("/mnt/scratch/ws/psbelokopytova/202103211631polina/nn_anopheles/test")
# plt.clf()
# breakpoint()
n = 6
m = 6
k = 3
arr = np.empty((k,m,n))
arr[:]=np.nan
# print(arr)
print(arr.shape)
x = np.array([[2, 2, 2,2,2,2], [2, 2]], np.int32)
# print(x)
assert x.shape[0]==x.shape[1]
assert x.shape[0]==arr.shape[1]
# print(len(x))
stride = 1
arr_stride = 0
for k_matrix in range(0, k):
    # print(k_matrix)
    predicted_array = x
    for i in range(len(predicted_array)):
        # print(k_matrix, i, 0+arr_stride, len(predicted_array)+arr_stride)
        arr[k_matrix][i][0+arr_stride:len(predicted_array)+arr_stride] = predicted_array[i]
    arr_stride+=1
    # print(x[1])
    # print(arr[0][:][0:3])
    # arr[0][:][0:3] = x
# print(arr)
new_arr = np.nanmean(arr, axis=0)
# print(new_arr)
print(new_arr.shape)
print(np.triu_indices(new_arr, 2))
# bisize=1
# starts = []
# ends = []
# values = []
# for i in range(new_arr.shape[0]):
#     print(new_arr[i])
#     for j in range(new_arr.shape[1]):
#         print(np.isnan(new_arr[i][j]))
#         if not np.isnan(new_arr[i][j]):
#             starts.append(i*bisize)
#             ends.append(j*bisize)
#             values.append(new_arr[i][j])
# print(starts)
# print(ends)
# print(values)
# data = {'chr': ['2L']*len(starts), 'contact_st': starts, 'contact_en':ends, 'contact_count':values}
# df = pd.DataFrame(data=data)
# print(df)
# mp = MatrixPlotter()
# mp.set_data(predicted_data)
# mp.set_control(validation_data)