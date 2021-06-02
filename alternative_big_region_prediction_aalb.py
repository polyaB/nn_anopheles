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

from shared import Interval
import json
import dataset, dna_io, seqnn
from Predictions_interpeter import from_upper_triu, predict_big_region_from_seq
import sys

chr = str(sys.argv[1])
start = int(sys.argv[2])
end = int(sys.argv[3])
print(chr, start, end)
# model_dir = './explore_best_model/'
# model_dir = '/mnt/scratch/ws/psbelokopytova/202105171236data_Polina/nn_anopheles/dataset_like_Akita/data/Aste_2048/train_out/'
model_dir = '/mnt/scratch/ws/psbelokopytova/202105171236data_Polina/nn_anopheles/dataset_like_Akita/data/Aalb_2048bp_repeat/train_out_test5_fix_random3/'
params_file = model_dir+'params.json'
# model_file  = model_dir+'model_check.h5'
model_file  = model_dir+'model_best.h5'
with open(params_file) as params_open:
    params = json.load(params_open)
    params_model = params['model']
    params_train = params['train']
seqnn_model = seqnn.SeqNN(params_model)
### restore model ###
seqnn_model.restore(model_file)
print('successfully loaded')

# read data parameters
data_dir ='/mnt/scratch/ws/psbelokopytova/202105171236data_Polina/nn_anopheles/dataset_like_Akita/data/Aalb_2048'
data_stats_file = '%s/statistics.json' % data_dir
with open(data_stats_file) as data_stats_open:
    data_stats = json.load(data_stats_open)
seq_length = data_stats['seq_length']
target_length = data_stats['target_length']
hic_diags =  data_stats['diagonal_offset']
target_crop = data_stats['crop_bp'] // data_stats['pool_width']
target_length1 = data_stats['seq_length'] // data_stats['pool_width']
target_length1_cropped = target_length1 - 2*target_crop

predict_big_region_from_seq(Interval(chr, start, end), binsize=data_stats['pool_width'], seq_len=seq_length, stride = 300*data_stats['pool_width'],
                            fasta_file="/mnt/scratch/ws/psbelokopytova/202105171236data_Polina/nn_anopheles/input/genomes/AalbS2_V4.fa", seqnn_model = seqnn_model,
                            crop_bp = data_stats['crop_bp'],target_length_cropped=target_length1_cropped, hic_diags = hic_diags,
                            prediction_folder=model_dir,
                            genome_hic_expected_file='/mnt/scratch/ws/psbelokopytova/202105171236data_Polina/nn_anopheles/input/coolers/Aalb_2048.expected',
                            use_control=True, genome_cool_file = '/mnt/scratch/ws/psbelokopytova/202105171236data_Polina/nn_anopheles/input/coolers/Aalb_2048.cool')