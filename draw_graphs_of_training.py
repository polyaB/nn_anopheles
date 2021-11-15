import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
model_dirs = []
# model_dir = 'Z:/scratch/202107130921Polina/nn_anopheles/dataset_like_Akita/data/Atrop_man_gaps5/train_out2/'
# for i in range(1,3):
    # model_dirs.append("Z:/scratch/202109061534Polya/nn_anopheles/dataset_like_Akita/data/Atrop_man_gaps5/train_out"+str(i))
for i in range(7,8):
    model_dirs.append("Z:/scratch/202109061534Polya/nn_anopheles/dataset_like_Akita/data/Atrop_man_gaps5/train_out_repeat" + str(i))
# model_dirs.append("Z:/scratch/202109061534Polya/nn_anopheles/dataset_like_Akita/data/Atrop_man_gaps5/train_out11")
# for i in range(1,5):
#     model_dirs.append("Z:/scratch/202109061534Polya/nn_anopheles/dataset_like_Akita/data/Aalb_2048bp_repeat/train_out_test"+str(i)+"_fix_random3")
# model_dirs.append("Z:/scratch/202109061534Polya/nn_anopheles/dataset_like_Akita/data/merge_all/train_out1")
# model_dirs.append("Z:/scratch/202109061534Polya/nn_anopheles/dataset_like_Akita/data/merge_Aalb_Aatr/train_out12")
# model_dirs.append("Z:/scratch/202109061534Polya/nn_anopheles/dataset_like_Akita/data/merge_Aalb_Aatr/train_out1")
# train1 = pd.read_csv("Z:/scratch/202107130921Polina/nn_anopheles/dataset_like_Akita/data/Aalb_test_1sample/train_out11/model_stat.txt", sep="\t")
# train1 = pd.read_csv("Z:/scratch_link/nn_anopheles/dataset_like_Akita/data/Atrop_man_gaps5/train_out_repeat1/model_stat.txt", sep="\t")
# train2 = pd.read_csv("Z:/scratch_link/nn_anopheles/dataset_like_Akita/data/Atrop_man_gaps5/train_out_repeat2/model_stat.txt", sep="\t")
# train3 = pd.read_csv("Z:/scratch_link/nn_anopheles/dataset_like_Akita/data/Atrop_man_gaps5/train_out_repeat3/model_stat.txt", sep="\t")
# train4 = pd.read_csv("Z:/scratch_link/nn_anopheles/dataset_like_Akita/data/Atrop_man_gaps5/train_out_repeat4/model_stat.txt", sep="\t")
# model_dirs.append("Z:/scratch/202109061534Polya/nn_anopheles/dataset_like_Akita/data/Dros_2048/train_out")
# model_dirs.append("Z:/scratch/202109061534Polya/nn_anopheles/dataset_like_Akita/data/Dros_2048/train_out2")
# model_dirs.append("Z:/scratch/202109061534Polya/nn_anopheles/dataset_like_Akita/data/Dros_2048/train_out3")
# model_dirs.append("Z:/scratch/202109061534Polya/nn_anopheles/dataset_like_Akita/data/Atrop_man_gaps5/train_out_transfer_mergeall1")
# model_dirs.append("Z:/scratch/202109061534Polya/nn_anopheles/dataset_like_Akita/data/Atrop_man_gaps5/train_out_transfer_mergeAalbAatr1")
# train1 = pd.read_csv("Z:/scratch_link/nn_anopheles/dataset_like_Akita/data/Dros_2048/train_out2/model_stat.txt", sep="\t")
# train2 = pd.read_csv("Z:/scratch_link/nn_anopheles/dataset_like_Akita/data/merge_Aalb_Aatr/train_out12/model_stat.txt", sep="\t")
# train3 = pd.read_csv("Z:/scratch/202107130921Polina/nn_anopheles/dataset_like_Akita/data/Aalb_2048bp_repeat/train_out_test3_fix_random3/model33.txt", sep=" ", names=range(24))
# global_oe = pd.read_csv("./model_graphs/aalb_globaloe_without_x.txt", sep=" ", names=range(24))
# print(best[10])
#plt.plot(train1["epoch"], np.log(train1["train_loss_epoch"]), linestyle='solid', label = 'train_loss')
# train1 = train1[train1["epoch"]<=1600]
for model_dir in  model_dirs:
    print(model_dir)
    anoph = model_dir.split("/")[-2].split("_")[0]
    model_fold = model_dir.split("/")[-1].split("_")[-1]
    train1 = pd.read_csv(model_dir+"/model_stat.txt", sep="\t")
    print(train1.keys())
    print(anoph, model_fold)
    plt.plot(train1["epoch"], np.log(train1["train_loss_epoch"]), label = anoph+"_"+model_fold+'_train_loss')
    plt.plot(train1["epoch"], np.log(train1["valid_loss_epoch"]), label = anoph+"_"+model_fold+'_valid_loss', color="blue")
    # plt.plot(train3["epoch"], np.log(train1["valid_loss_epoch"]), label = 'valid3_loss', color="green")
    # plt.plot(train4["epoch"], np.log(train1["valid_loss_epoch"]), label = 'valid4_loss', color="orange")
    # plt.plot(train1["epoch"], np.log(train1["valid_loss_epoch"]), linestyle='dashed', label = 'valid1_loss', color="blue")
    # plt.plot(train1["epoch"], np.log(train1["train_r2_epoch"]), label = 'train_r2', color="blue")
    # plt.plot(train1["epoch"], np.log(train1["valid_r2_epoch"]), linestyle='dashed', label = 'valid_r2', color="red")
    # plt.plot(train3[1], np.log(train3[15]), label = 'train3_valid_loss')
plt.legend()
    # plt.title("valid_loss_epoch")
plt.show()
    # print(best)