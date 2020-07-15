from source.draw_hic_map import from_upper_triu
import pickle
import logging
import json
from termcolor import colored, cprint
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
logging.basicConfig(format='%(asctime)s %(name)s: %(message)s', datefmt='%I:%M:%S', level=logging.INFO)

pickle_model = "test_train_dataset_25.05.pickle"
params_file = "source/params.json"
logging.info(colored("Going to pickle load train dataset for " + pickle_model, 'green'))
with open(params_file) as params_open:
    params = json.load(params_open)
    params_model = params['model']
    params_train = params['train']
with open("/mnt/scratch/ws/psbelokopytova/202008151203data_Polya/nn_anopheles/output/"+pickle_model, 'rb') as f:
    data_train = pickle.load(f)

print(data_train.keys())
print(len(data_train))
print(data_train["intervals"])

data_train = pd.DataFrame.from_dict(data_train)
test = data_train.iloc[0:5,:]
train = data_train.iloc[5::,:]
test.to_dict()
train.to_dict()
with open("/mnt/scratch/ws/psbelokopytova/202008151203data_Polya/nn_anopheles/output/train_X_15.07", 'wb') as f:
    pickle.dump(train, f)
with open("/mnt/scratch/ws/psbelokopytova/202008151203data_Polya/nn_anopheles/output/test_X_15.07", 'wb') as f:
    pickle.dump(test, f)

targets =  tf.convert_to_tensor(data_train["targets"])
targets = tf.expand_dims(targets, -1)
target_index = 3
print("targets shape", targets.shape)
# print(targets[:,:,0][target_index])
#these values we should save somewhere while generate train and test dataset!!!
target_crop = params_model['target_crop']
hic_diags = 2
target_length = 128
target_length_cropped = target_length - 2*target_crop
tlen = (target_length_cropped - hic_diags) * (target_length_cropped - hic_diags+1) // 2

mat = from_upper_triu(targets[:,:,0][target_index], target_length_cropped, hic_diags)
im = plt.matshow(mat, fignum=False, cmap= 'RdBu_r', vmax=2, vmin=-2)
plt.colorbar(im, fraction=.04, pad = 0.05, ticks=[-2,-1, 0, 1,2])
plt.ylabel(str(data_train["intervals"][target_index]))

plt.show()

# plt.title('pred-'+str(hic_num_to_name_dict[target_index]),y=1.15 )
