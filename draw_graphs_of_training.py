import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# train1 = pd.read_csv("Z:/scratch/202107130921Polina/nn_anopheles/dataset_like_Akita/data/Aalb_test_1sample/train_out11/model_stat.txt", sep="\t")
train1 = pd.read_csv("Z:/scratch/202107130921Polina/nn_anopheles/dataset_like_Akita/data/train_1sample_try2/train_out2/model_stat.txt", sep="\t")
train2 = pd.read_csv("Z:/scratch/202107130921Polina/nn_anopheles/dataset_like_Akita/data/train_1sample_try2/train_out3/model_stat.txt", sep="\t")
# train3 = pd.read_csv("Z:/scratch/202107130921Polina/nn_anopheles/dataset_like_Akita/data/Aalb_2048bp_repeat/train_out_test3_fix_random3/model33.txt", sep=" ", names=range(24))
# global_oe = pd.read_csv("./model_graphs/aalb_globaloe_without_x.txt", sep=" ", names=range(24))
# print(best[10])
#plt.plot(train1["epoch"], np.log(train1["train_loss_epoch"]), linestyle='solid', label = 'train_loss')
plt.plot(train1["epoch"], np.log(train1["valid_loss_epoch"]), linestyle='dashed', label = 'valid1_valid_loss')
plt.plot(train2["epoch"], np.log(train2["valid_loss_epoch"]), label = 'valid2_valid_loss')
# plt.plot(train3[1], np.log(train3[15]), label = 'train3_valid_loss')
plt.legend()
plt.title("valid_loss_epoch")
plt.show()
# print(best)