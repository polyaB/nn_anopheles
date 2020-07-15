import json
import sys
import os
import tensorflow as tf
import pickle
import numpy as np
import logging
from termcolor import colored, cprint
logging.basicConfig(format='%(asctime)s %(name)s: %(message)s', datefmt='%I:%M:%S', level=logging.INFO)
source_path = os.path.dirname(os.path.abspath(sys.argv[0])) + "/../3Dpredictor/source"
source_path2 = os.path.dirname(os.path.abspath(sys.argv[0])) + "/../3Dpredictor/nn/source"
source_path3 = os.path.dirname(os.path.abspath(sys.argv[0])) + "/../"
sys.path.append(source_path)
sys.path.append(source_path2)
sys.path.append(source_path3)
from basenji.basenji import seqnn
from basenji.basenji import metrics
from draw_hic_map import from_upper_triu
import matplotlib.pyplot as plt



out_dir = "/mnt/scratch/ws/psbelokopytova/202008151203data_Polya/nn_anopheles/output/test_model/"
# pickle_train = "test_train_dataset_25.05.pickle"
# pickle_train = "train_X_15.07"
# pickle_test = "test_X_15.07"
pickle_train = "train_dataset_14.07.pickle"

logging.info(colored("Going to pickle load train dataset for " + pickle_train, 'green'))
with open("/mnt/scratch/ws/psbelokopytova/202008151203data_Polya/nn_anopheles/output/"+pickle_train, 'rb') as f:
    data_train = pickle.load(f)

# with open("/mnt/scratch/ws/psbelokopytova/202008151203data_Polya/nn_anopheles/output/"+pickle_test, 'rb') as f:
#     data_test = pickle.load(f)
logging.info(colored("Loaded ", 'green'))
params_file = "params.json"

print("------------------------------------------------------")
print()
inputs = tf.convert_to_tensor(data_train["inputs"][5:])
# inputs = tf.expand_dims(inputs, -1)
print("inputs shape", inputs.shape)
print(inputs)
targets =  tf.convert_to_tensor(data_train["targets"][5:])
targets = tf.expand_dims(targets, -1)
print("targets shape", targets.shape)
test_inputs = tf.convert_to_tensor(data_train["inputs"][0:5])
print("test inputs shape", test_inputs.shape)
test_targets = data_train["targets"][0:5]
test_targets = tf.expand_dims(test_targets, -1)
print("test targets shape", test_targets.shape)



# read model parameters
with open(params_file) as params_open:
    params = json.load(params_open)
    params_model = params['model']
    params_train = params['train']

# test_index = 0
# test_target = test_targets[test_index:test_index+1,:,:]
# print("test_target")
# print(test_target)
# target_crop = params_model['target_crop']
# hic_diags = 2
# target_length = 128
# target_length_cropped = target_length - 2 * target_crop
# tlen = (target_length_cropped - hic_diags) * (target_length_cropped - hic_diags + 1) // 2
# mat = from_upper_triu(test_target[:,:,test_index], target_length_cropped, hic_diags)
# im = plt.matshow(mat, fignum=False, cmap='RdBu_r', vmax=2, vmin=-2)
# plt.colorbar(im, fraction=.04, pad=0.05, ticks=[-2, -1, 0, 1, 2])
# plt.ylabel(str(data_train["intervals"][test_index]))
# plt.show()

# initialize model
seqnn_model = seqnn.SeqNN(params_model)
#compile model
for model in seqnn_model.models:
    num_targets = model.output_shape[-1]
    model.compile(loss="mse",
                  optimizer="sgd",
                  metrics=[metrics.PearsonR(num_targets), metrics.R2(num_targets)])

num_seqs = len(data_train)
batch_size = 2
steps_per_epoch = num_seqs // batch_size
tf.config.experimental.list_physical_devices('GPU')

callbacks = [
      tf.keras.callbacks.EarlyStopping(monitor='val_pearsonr', mode='max', verbose=1,
                       patience=20),
      tf.keras.callbacks.TensorBoard(out_dir),
      tf.keras.callbacks.ModelCheckpoint('%s/model_check.h5'%out_dir),
      tf.keras.callbacks.ModelCheckpoint('%s/model_best.h5'%out_dir, save_best_only=True,
                                         monitor='val_pearsonr', mode='max', verbose=1)]

seqnn_model.model.fit(x = inputs,
                      y = targets,
                      epochs=20,
                    # steps_per_epoch=steps_per_epoch,
                      batch_size=batch_size,
                    callbacks=callbacks,
                    # validation_data=self.eval_data[0].dataset,
                    # validation_steps=self.eval_epoch_batches[0],
                    use_multiprocessing=True)
#check prediction and plot matrix
test_index = 0
# test_target = test_targets[test_index:test_index+1,:,:]
# print("test_target")
# print(test_target)
# test_pred = seqnn_model.model.predict(test_inputs[test_index:test_index+1,:,:])
test_pred = seqnn_model.model.predict(test_inputs)
with open("/mnt/scratch/ws/psbelokopytova/202008151203data_Polya/nn_anopheles/output/pred_15.07", 'wb') as f:
    pickle.dump(test_pred, f)
# print("test_predict")
# print(test_pred)
target_crop = params_model['target_crop']
hic_diags = 2
target_length = 128
target_length_cropped = target_length - 2 * target_crop
tlen = (target_length_cropped - hic_diags) * (target_length_cropped - hic_diags + 1) // 2

plt.subplot(121)
mat = from_upper_triu(test_pred[:,:,0][test_index], target_length_cropped, hic_diags)
im = plt.matshow(mat, fignum=False, cmap='RdBu_r', vmax=2, vmin=-2)
plt.colorbar(im, fraction=.04, pad=0.05, ticks=[-2, -1, 0, 1, 2])
plt.title('pred-'+str(data_train["intervals"][test_index]),y=1.15 )
plt.ylabel(str(data_train["intervals"][test_index]))

plt.subplot(122)
mat = from_upper_triu(test_targets[:,:,0][test_index], target_length_cropped, hic_diags)
im = plt.matshow(mat, fignum=False, cmap='RdBu_r', vmax=2, vmin=-2)
plt.colorbar(im, fraction=.04, pad=0.05, ticks=[-2, -1, 0, 1, 2])
plt.title('real-'+str(data_train["intervals"][test_index]),y=1.15 )
plt.ylabel(str(data_train["intervals"][test_index]))
plt.show()





