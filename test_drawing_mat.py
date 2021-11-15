import os
import sys

source_path = os.path.dirname(os.path.abspath(sys.argv[0])) + "/basenji/source"
source_path2 = os.path.dirname(os.path.abspath(sys.argv[0])) + "/basenji/basenji"
source_path3 = os.path.dirname(os.path.abspath(sys.argv[0])) + "/3Dpredictor/source"
source_path4 = os.path.dirname(os.path.abspath(sys.argv[0])) + "/source"
sys.path.append(source_path)
sys.path.append(source_path2)
sys.path.append(source_path3)
sys.path.append(source_path4)
import json
import subprocess
os.environ["CUDA_VISIBLE_DEVICES"] = '-1' ### run on CPU

import tensorflow as tf
print(tf.__version__)
if tf.__version__[0] == '1':
    tf.compat.v1.enable_eager_execution()

import numpy as np
import pandas as pd
import pysam
import matplotlib.pyplot as plt
from cooltools.lib.numutils import set_diag
import dataset, dna_io, seqnn
from Predictions_interpeter import from_oe_to_contacts, from_upper_triu
from shared import Interval

### names of targets ###
# data_dir ='/mnt/scratch/ws/psbelokopytova/202103211631polina/nn_anopheles/dataset_like_Akita/data/Aalb_test_1sample'
# data_dir ='/mnt/scratch/ws/psbelokopytova/202109061534Polya/nn_anopheles/dataset_like_Akita/data/Atrop_man_gaps5'
data_dir ='/mnt/scratch/ws/psbelokopytova/202109061534Polya/nn_anopheles/dataset_like_Akita/data/Atrop_moresamples/'
pic_dir = '/mnt/scratch/ws/psbelokopytova/202109061534Polya/nn_anopheles/dataset_like_Akita/data/Atrop_moresamples/pictures/'
hic_targets = pd.read_csv(data_dir+'/targets.txt',sep='\t')
hic_file_dict_num = dict(zip(hic_targets['index'].values, hic_targets['file'].values) )
hic_file_dict     = dict(zip(hic_targets['identifier'].values, hic_targets['file'].values) )
hic_num_to_name_dict = dict(zip(hic_targets['index'].values, hic_targets['identifier'].values) )
print(hic_num_to_name_dict)

# read data parameters
data_stats_file = '%s/statistics.json' % data_dir
with open(data_stats_file) as data_stats_open:
    data_stats = json.load(data_stats_open)
seq_length = data_stats['seq_length']
target_length = data_stats['target_length']
hic_diags =  data_stats['diagonal_offset']
target_crop = data_stats['crop_bp'] // data_stats['pool_width']
target_length1 = data_stats['seq_length'] // data_stats['pool_width']

### load data ###
sequences = pd.read_csv(data_dir+'/sequences.bed', sep='\t', names=['chr','start','stop','type'])
sequences_test = sequences.iloc[sequences['type'].values=='valid']
sequences_test.reset_index(inplace=True, drop=True)
print("going to load test dataset")
test_data = dataset.SeqDataset(data_dir, 'valid', batch_size=8)

# test_targets is a float array with shape
# [#regions, #pixels, target #target datasets]
# representing log(obs/exp)data, where #pixels
# corresponds to the number of entries in the flattened
# upper-triangular representation of the matrix

# test_inputs are 1-hot encoded arrays with shape
# [#regions, 2^20 bp, 4 nucleotides datasets]

test_inputs, test_targets = test_data.numpy(return_inputs=True, return_outputs=True)
print(test_inputs)
# print(test_targets)

# ### for converting from flattened upper-triangluar vector to symmetric matrix  ###
# def from_upper_triu(vector_repr, matrix_len, num_diags):
#     z = np.zeros((matrix_len,matrix_len))
#     triu_tup = np.triu_indices(matrix_len,num_diags)
#     z[triu_tup] = vector_repr
#     for i in range(-num_diags+1,num_diags):
#         set_diag(z, np.nan, i)
#     return z + z.T

target_length1_cropped = target_length1 - 2*target_crop
print('flattened representation length:', target_length)
print('symmetrix matrix size:', '('+str(target_length1_cropped)+','+str(target_length1_cropped)+')')

fig2_examples = [
                    #Aalb
                    # '2L:12992512-14041088',
                    # '2L:24403968-25452544',
                    # '2L:12992512-14041088',
                    # '2R:40747008-41795584',
                    # '3R:26742784-27791360',
                    # '3R:25497600-26546176',
                    # '3R:25530368-26578944',
                    # '3R:25563136-26611712',
                    # '3R:25595904-6644480',
                    # '3R:25628672-26677248',
                    #Atrop
                    # "2L:39407616-40456192",
                    # "2L:37703680-38752256",
                    # "2L:37998592-39047168",
                    # "X:753664-1802240",
                    # "2L:39735296-40783872",
                    # "2L:39964672-41013248",
    #small region
                    # "2L:38031360-38555648",
                    # "2R:21700608-22224896",
                    # "2R:26419200-26943488",
                    # "2R:39421952-39946240",
                    # "3R:54353920-54878208"
                    "3R:59293696-59817984",
                    "3R:57868288-58392576"
                    #merge all
                    # "Amer_3R:44619776-45668352",
                    # "Aatr_3R:25380864-26429440",
                    # "Amer_2L:30466048-31514624",
                    # "Acol_3R:14903296-15951872",
                    # "Aatr_3R:27379712-28428288",
                    # "Acol_2R:33259520-34308096",
                    # "Aatr_3R:50499584-51548160",
                    # "Aatr_3R:25544704-26593280",
                    # "Acol_3R:21915648-22964224",
                    # "ASteph_2L:31787008-32835584",
                    # "Aatr_X:17547264-18595840",
                    # "Acol_X:2850816-3899392",
                    #Dros
                    #valid
                    # "chrX:4931584-5980160",
                    # "chrX:5357568-6406144",
                    # "chrX:5193728-6242304",
                    # "chrX:5586944-6635520",
                    # "chr4:163840-1212416",
                    # "chrX:4898816-5947392",
                    # "chrX:5128192-6176768",
                    # "chrX:5652480-6701056",
                    # "chrX:5095424-6144000",
                    # "chrX:5750784-6799360",
                    # "chrX:5718016-6766592",
                    # "chrX:5488640-6537216",
                    # "chr4:229376-1277952"
                    #train
                    # "chrX:8921088-9969664",
                    # "chr2L:9437184-10485760",
                    # "chr2L:16252928-17301504",
                    # "chrX:20905984-21954560",
                    # "chr3L:21016576-22065152",
                    # "chr3R:786432-1835008",
                    # "chr3R:9961472-11010048",
                    # "chrX:20643840-21692416",
                    # "chr3R:8388608-9437184",
                    # "chr3L:2142208-3190784",
                    # "chr3R:524288-1572864",
                    # "chrX:7086080-8134656",
                    # "chr3R:19136512-20185088",
                    # "chr3R:16252928-17301504",
                    # "chr2L:2621440-3670016",
                    # "chr3R:6291456-7340032",
                    # "chr3L:20754432-21803008",
                    # "chr3R:7864320-8912896",
                    # "chr2R:11796480-12845056"
                    ]
                    # 'chr11:75429888-76478464',
                    # 'chr15:63281152-64329728'

fig2_inds = []
for seq in fig2_examples:
    # print(seq)
    # print(np.unique(sequences_test['chr'].values))
    chrm,start,stop = seq.split(':')[0], seq.split(':')[1].split('-')[0], seq.split(':')[1].split('-')[1]
    # print(np.where(sequences_test['chr'].values== chrm))
    # print(np.where(sequences_test['start'].values== int(start)))
    # print(np.where(sequences_test['stop'].values==  int(stop )))
    test_ind = np.where( (sequences_test['chr'].values== chrm) *
                         (sequences_test['start'].values== int(start))*
                         (sequences_test['stop'].values==  int(stop ))  )[0][0]
    fig2_inds.append(test_ind)
# print(fig2_inds)

    target_index = 0
    for test_index in fig2_inds:
        chrm, seq_start, seq_end = sequences_test.iloc[test_index][0:3]
        myseq_str = chrm + ':' + str(seq_start) + '-' + str(seq_end)
        print(myseq_str)
    #     print(myseq_str)
        test_target = test_targets[test_index:test_index + 1, :, :]
        # print("mean", np.mean(mat), "max", np.max(mat), "min", np.min(mat))
        # plot target
        # plt.subplot(122)
        mat = from_upper_triu(test_target[:, :, target_index], target_length1_cropped, hic_diags)
        # print(mat)
        # print("mean",np.mean(mat), "max", np.max(mat), "min", np.min(mat))
        #draw matrix before returning from oe to contacts
        im = plt.matshow(mat, fignum=False, cmap='RdBu_r')#, vmax=vmax, vmin=vmin)
        plt.colorbar(im, fraction=.04, pad=0.05)#, ticks=[-2, -1, 0, 1, 2])
        plt.title('target-' + str(hic_num_to_name_dict[target_index]+myseq_str), y=1.15)
        plt.tight_layout()
        plt.savefig(pic_dir+"/atrop_"+str(chrm)+"_"+str(seq_start)+"_"+str(seq_end)+".png")
        plt.clf()
        #draw_after
        # returned_mat = from_oe_to_contacts(seq_hic_obsexp=mat, genome_hic_expected_file='/mnt/scratch/ws/psbelokopytova/202109061534Polya/nn_anopheles/input/coolers/Aatr_2048.expected',
        #                                    interval=Interval('2R', 32083968,33132544), seq_len_pool=target_length1_cropped)
        # im = plt.matshow(returned_mat, fignum=False, cmap='OrRd')  # , vmax=vmax, vmin=vmin)
        # plt.colorbar(im, fraction=.04, pad=0.05)  # , ticks=[-2, -1, 0, 1, 2])
        # plt.title('target-' + str(hic_num_to_name_dict[target_index] + myseq_str), y=1.15)
        # plt.tight_layout()
        # plt.savefig(data_dir + "/test_after_" + str(chrm) + "_" + str(seq_start) + "_" + str(
        #     seq_end) + ".png")
        plt.clf()