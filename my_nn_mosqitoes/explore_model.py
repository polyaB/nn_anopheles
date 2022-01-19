import tensorlow as tf
import json
import sys
import os
import datetime

sys.path.append("/mnt/storage/home/psbelokopytova/nn_anopheles/my_nn_mosqitoes/source/")
sys.path.append("/mnt/storage/home/psbelokopytova/nn_anopheles/")
from model2 import build_neural_network

prefix = 'model3_1sample'
data_folder = "/mnt/scratch/ws/psbelokopytova/202112281307data_Polya/nn_anopheles/dataset_like_Akita/data/test_new/"
# data_folder = "/home/polina/nn_anopheles/test_new/"
log_dir = data_folder+"tensor_board_logs/tb_log_"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+prefix
checkpoint_filepath = os.path.dirname(os.path.abspath(sys.argv[0]))+"/"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+prefix+'_checkpoint_model.h5'
model_summary_path = os.path.dirname(os.path.abspath(sys.argv[0]))+"/"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+prefix+'_modelsummary.txt'
batch_size=2
with open(data_folder + "statistics.json") as data_stats_open:
    data_stats = json.load(data_stats_open)
    seq_length = data_stats['seq_length']
    target_length = data_stats['target_length']
    diagonal_offset = data_stats['diagonal_offset']
    crop = data_stats['crop_bp'] // data_stats['pool_width']

siamese_model, fe_layer, regression_model = build_neural_network( seq_len=seq_length, target_size=target_length,batch_size=batch_size,
                                                                  cropping=crop, diagonal_offset=diagonal_offset)
model = siamese_model.load_weights(checkpoint_filepath)
