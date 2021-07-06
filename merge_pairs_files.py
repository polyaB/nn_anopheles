import pandas as pd

merge_df = pd.DataFrame(columns=["0", "1", "2", "3"])
for anoph in ["Aalb", "Aatr"]:
    pairs_data = pd.read_csv("/mnt/scratch/ws/psbelokopytova/202107130921Polina/nn_anopheles/input/coolers/"+anoph+".pairs.txt", names=["0", "1", "2", "3"], sep="\t")
    print(pairs_data.keys())
    pairs_data["0"]= pairs_data["0"].apply(lambda  x: anoph+"_"+x)
    pairs_data["2"] = pairs_data["2"].apply(lambda x: anoph + "_" + x)
    print(len(pairs_data))
    merge_df = pd.concat([merge_df, pairs_data])
    len(merge_df)
merge_df.to_csv("/mnt/scratch/ws/psbelokopytova/202107130921Polina/nn_anopheles/input/coolers/merge_AAlb_Aatr.pairs.txt", sep="\t",
                index=False, header=False)