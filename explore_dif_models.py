import os
import sys
import pickle
source_path = os.path.dirname(os.path.abspath(sys.argv[0])) + "/basenji/source"
source_path2 = os.path.dirname(os.path.abspath(sys.argv[0])) + "/basenji/basenji"
sys.path.append(source_path)
sys.path.append(source_path2)
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
from source.common_functions import str2hash
import dataset, dna_io, seqnn

model_dirs = []
# model_dir = '/mnt/scratch/ws/psbelokopytova/202107130921Polina/nn_anopheles/dataset_like_Akita/data/Atrop_man_gaps5/train_out2/'
for i in range(1,3):
    model_dirs.append("/mnt/scratch/ws/psbelokopytova/202109061534Polya/nn_anopheles/dataset_like_Akita/data/Atrop_man_gaps5/train_out"+str(i))
for i in range(5,7):
    model_dirs.append("/mnt/scratch/ws/psbelokopytova/202109061534Polya/nn_anopheles/dataset_like_Akita/data/Atrop_man_gaps5/train_out_repeat" + str(i))
# model_dirs.append("/mnt/scratch/ws/psbelokopytova/202109061534Polya/nn_anopheles/dataset_like_Akita/data/Atrop_man_gaps5/train_out11")
# for i in range(1,5):
#     model_dirs.append("/mnt/scratch/ws/psbelokopytova/202109061534Polya/nn_anopheles/dataset_like_Akita/data/Aalb_2048bp_repeat/train_out_test"+str(i)+"_fix_random3")
# model_dirs.append("/mnt/scratch/ws/psbelokopytova/202109061534Polya/nn_anopheles/dataset_like_Akita/data/merge_all/train_out1")
# model_dirs.append("/mnt/scratch/ws/psbelokopytova/202109061534Polya/nn_anopheles/dataset_like_Akita/data/merge_Aalb_Aatr/train_out12")
# model_dirs.append("/mnt/scratch/ws/psbelokopytova/202109061534Polya/nn_anopheles/dataset_like_Akita/data/merge_Aalb_Aatr/train_out1")

fasta_file = "/mnt/scratch/ws/psbelokopytova/202109061534Polya/nn_anopheles/input/genomes/AatrE3_V4.fa"
# fasta_file = "/mnt/scratch/ws/psbelokopytova/202109061534Polya/nn_anopheles/input/genomes/AalbS2_V4.fa"

data_dir = '/mnt/scratch/ws/psbelokopytova/202109061534Polya/nn_anopheles/dataset_like_Akita/data/Atrop_man_gaps5/'
# data_dir = '/mnt/scratch/ws/psbelokopytova/202109061534Polya/nn_anopheles/dataset_like_Akita/data/Aalb_2048bp_repeat/'
readme = open("/mnt/scratch/ws/psbelokopytova/202109061534Polya/nn_anopheles/dataset_like_Akita/data/explore_dif_models/readme.txt", "a")
prediction_dir = "/mnt/scratch/ws/psbelokopytova/202109061534Polya/nn_anopheles/dataset_like_Akita/data/explore_dif_models/"

### for converting from flattened upper-triangluar vector to symmetric matrix  ###
def from_upper_triu(vector_repr, matrix_len, num_diags):
    z = np.zeros((matrix_len,matrix_len))
    triu_tup = np.triu_indices(matrix_len,num_diags)
    z[triu_tup] = vector_repr
    for i in range(-num_diags+1,num_diags):
        set_diag(z, np.nan, i)
    return z + z.T
### names of targets ###
# data_dir ='/mnt/scratch/ws/psbelokopytova/202107130921Polina/nn_anopheles/dataset_like_Akita/data/Atrop_man_gaps5'
hic_targets = pd.read_csv(data_dir+'/targets.txt',sep='\t')
hic_file_dict_num = dict(zip(hic_targets['index'].values, hic_targets['file'].values) )
hic_file_dict     = dict(zip(hic_targets['identifier'].values, hic_targets['file'].values) )
hic_num_to_name_dict = dict(zip(hic_targets['index'].values, hic_targets['identifier'].values) )

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
test_inputs, test_targets = test_data.numpy(return_inputs=True, return_outputs=True)
# print(test_targets)

target_length1_cropped = target_length1 - 2*target_crop
print('flattened representation length:', target_length)
print('symmetrix matrix size:', '('+str(target_length1_cropped)+','+str(target_length1_cropped)+')')
print("opening fasta and coding fasta...")
fasta_open = pysam.Fastafile(fasta_file)
for model_dir in model_dirs:
    print(model_dir)
    anoph = model_dir.split("/")[-2].split("_")[0]
    model_fold = model_dir.split("/")[-1].split("_")[-1]
    params_file = model_dir+'/params.json'
    model_file  = model_dir+"/model_best.h5"
    # model_file  = model_dir+'model_best.h5'
    with open(params_file) as params_open:
        params = json.load(params_open)
        params_model = params['model']
        params_train = params['train']
        params_model['out_model_summary_dir'] = model_dir
    seqnn_model = seqnn.SeqNN(params_model)
    ### restore model ###
    seqnn_model.restore(model_file)
    print('successfully loaded')
    fig2_examples = [
                        #Atrop
                        # "X:1736704-2785280",
                        "3R:58605568-59654144"
                        #Aalb
                        # '2L:12992512-14041088',
                        # merge_all
                        # "Aatr_X:17547264-18595840",
                        # "Aalb__2R:10616832-11665408"
                        ]
    fig2_inds = []
    for seq in fig2_examples:
        print(seq)
        # print(np.unique(sequences_test['chr'].values))
        chrm, start, stop = seq.split(':')[0], seq.split(':')[1].split('-')[0], seq.split(':')[1].split('-')[1]
        # print(np.where(sequences_test['chr'].values== chrm))
        # print(np.where(sequences_test['start'].values== int(start)))
        # print(np.where(sequences_test['stop'].values==  int(stop )))
        test_ind = np.where((sequences_test['chr'].values == chrm) *
                            (sequences_test['start'].values == int(start)) *
                            (sequences_test['stop'].values == int(stop)))[0][0]
        fig2_inds.append(test_ind)
    ### make predictions and plot examples above ###
    target_index = 0  #Aalb
    for test_index in fig2_inds:
        chrm, seq_start, seq_end = sequences_test.iloc[test_index][0:3]
        myseq_str = chrm + ':' + str(seq_start) + '-' + str(seq_end)
        print(myseq_str)

        test_target = test_targets[test_index:test_index + 1, :, :]

        seq = fasta_open.fetch(chrm, seq_start, seq_end).upper()
        # print(seq)
        seq_1hot = dna_io.dna_1hot(seq)
        print("predict")
        test_pred  = seqnn_model.model.predict(np.expand_dims(seq_1hot, 0))
        # print("inputs")
        # print(seq_1hot)
        plt.figure(figsize=(8, 4))
        target_index = 0
        vmin = -2
        vmax = 2
        # plot pred
        mat = from_upper_triu(test_pred[:, :, target_index], target_length1_cropped, hic_diags)
        print(target_index, target_length1_cropped, hic_diags)
        print(mat)
        plt.subplot(121)
        im = plt.matshow(mat, fignum=False, cmap='RdBu_r')#, vmax=vmax, vmin=vmin)
        plt.colorbar(im, fraction=.04, pad=0.05)#, ticks=[-2, -1, 0, 1, 2])
        plt.title('pred-' + str(hic_num_to_name_dict[target_index]+myseq_str), y=1.15)
        plt.ylabel(myseq_str)
        # plot target
        plt.subplot(122)
        mat = from_upper_triu(test_target[:, :, target_index], target_length1_cropped, hic_diags)
        im = plt.matshow(mat, fignum=False, cmap='RdBu_r')#, vmax=vmax, vmin=vmin)
        plt.colorbar(im, fraction=.04, pad=0.05)#, ticks=[-2, -1, 0, 1, 2])
        plt.title('target-' + str(hic_num_to_name_dict[target_index]+myseq_str), y=1.15)
        plt.tight_layout()
        # plt.suptitle("epoch " + str(i))
        plt.savefig(prediction_dir +str(chrm)+"_"+str(seq_start)+"_"+str(seq_end)+"_"+anoph+"_"+model_fold+"_"+str2hash(model_file)+".png")
        plt.clf()
        readme.write(str2hash(model_file)+"\t"+model_file+"\n")
readme.close()