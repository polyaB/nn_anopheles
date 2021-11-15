import json
import os
import sys
import pickle
source_path = os.path.dirname(os.path.abspath(sys.argv[0])) + "/basenji/source"
source_path2 = os.path.dirname(os.path.abspath(sys.argv[0])) + "/basenji/basenji"
sys.path.append(source_path)
sys.path.append(source_path2)
import dataset, dna_io, seqnn
from tensorflow import keras
from random import randint

# model_dir = '/mnt/scratch/ws/psbelokopytova/202107130921Polina/nn_anopheles/dataset_like_Akita/data/Atrop_man_gaps5/train_out2/'
transfer_dir = '/mnt/scratch/ws/psbelokopytova/202107130921Polina/nn_anopheles/dataset_like_Akita/data/transfer_learning/'
# model_file  = model_dir+'model_best.h5'
model_file = transfer_dir + 'model_best_from_basenji_github.h5'
params_file = transfer_dir+'params.json'
with open(params_file) as params_open:
    params = json.load(params_open)
    params_model = params['model']
    params_train = params['train']
    params_model['out_model_summary_dir'] = transfer_dir
seqnn_model = seqnn.SeqNN(params_model)
with open(transfer_dir +"model_summary_test", "w") as f:
    seqnn_model.model.summary(print_fn=lambda x: f.write(x + '\n'))
seqnn_model.save('/mnt/scratch/ws/psbelokopytova/202107130921Polina/nn_anopheles/dataset_like_Akita/data/transfer_learning/model_test.h5')
### restore model ###
# seqnn_model.restore(model_file)
seqnn_model.restore('/mnt/scratch/ws/psbelokopytova/202107130921Polina/nn_anopheles/dataset_like_Akita/data/transfer_learning/model_test.h5')
print('successfully loaded')
with open(transfer_dir +"model_summary_test_restore", "w") as f:
    seqnn_model.model.summary(print_fn=lambda x: f.write(x + '\n'))
model2= keras.models.Model(inputs=seqnn_model.model.input, outputs=seqnn_model.model.layers[-3].output)

with open(transfer_dir +"model_summary2", "w") as f:
    model2.summary(print_fn=lambda x: f.write(x + '\n'))
# current = tf.keras.layers.Dense(
#     units=units,
#     use_bias=True,
#     activation=activation,
#     kernel_initializer=c),
#     kernel_regularizer=tf.keras.regularizers.l1_l2(l1_scale, l2_scale)
# )(inputs)
my_layer = keras.layers.Dense(units = 1,activation='linear', kernel_initializer=keras.initializers.he_normal(seed=randint(0, 100000)),kernel_regularizer=keras.regularizers.l1_l2(0, 0))(model2.output)
model3 = keras.models.Model(inputs=model2.input, outputs=my_layer)
with open(transfer_dir +"model_summary3", "w") as f:
    model3.summary(print_fn=lambda x: f.write(x + '\n'))