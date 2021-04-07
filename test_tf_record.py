import tensorflow as tf
import json
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

import dataset
import seqnn

# read model parameters
# data_direct = "/mnt/scratch/ws/psbelokopytova/202105171236data_Polina/nn_anopheles/dataset_like_Akita/data/Aalb_2048bp_repeat/"
data_direct = "/mnt/scratch/ws/psbelokopytova/202105171236data_Polina/nn_anopheles/dataset_like_Akita/data/Aalb_test_1sample/"
# params_file = "/mnt/scratch/ws/psbelokopytova/202105171236data_Polina/nn_anopheles/dataset_like_Akita/params.json"
params_file = data_direct + "params.json"
with open(params_file) as params_open:
    params = json.load(params_open)
    params_model = params['model']
    params_train = params['train']

# read datasets
train_data = []
eval_data = []
data_dirs = [data_direct]
for data_dir in data_dirs:
    # load train data
    train_data.append(dataset.SeqDataset(data_dir,
    split_label='train',
    batch_size=params_train['batch_size'],
    mode=tf.estimator.ModeKeys.TRAIN,
    tfr_pattern=None))
print(train_data[0].dataset)
# print(len(list(train_data[0].dataset)))


# print(ds_size = sum(1 for _ in train_data[0].dataset))
# load eval data
# eval_data.append(dataset.SeqDataset(data_dir,
#                                     split_label='valid',
#     batch_size=params_train['batch_size'],
#     mode=tf.estimator.ModeKeys.EVAL,
#     tfr_pattern=None))
# print(eval_data[0].dataset)
# num_elements = 0
# for element in eval_data[0].dataset:
#     print(element)
#     num_elements += 1
#     print(num_elements)