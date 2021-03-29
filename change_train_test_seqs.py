import pandas as pd

seqs = pd.read_csv("Z:/scratch/202103211631polina/nn_anopheles/dataset_like_Akita/data/Aaalb_2048_new2/original_sequences.bed", sep="\t",
                   names=["chr", "start", "ensd", "test_train"])

for i in range(len(seqs)):
    if seqs.iloc[i,0]=="X":
        seqs.iloc[i,3] = "test"
    elif seqs.iloc[i,3] == "test":
        seqs.iloc[i, 3] = "train"
seqs.to_csv("Z:/scratch/202103211631polina/nn_anopheles/dataset_like_Akita/data/Aaalb_2048_new2/sequences.bed", sep="\t", index=False, header=False)