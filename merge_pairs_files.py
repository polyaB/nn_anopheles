import pandas as pd
import numpy as np

merge_df = pd.DataFrame(columns=["0", "1", "2", "3"])
# print("reading")
# pairs_data = pd.read_csv("~/scratch_link/nn_anopheles/input/coolers/Aalb_modified.pairs.txt", names=["0","1","2","3","4"], sep="\t")
# print("search for nan")
# import numpy as np
# index = pairs_data['1'].index[pairs_data['1'].apply(np.isnan)]
# b = pd.isna(pairs_data["1"])
# print("convert to array")
# b=np.array(b)
# print(pd.isna(pairs_data["0"]))
# print(np.where(b)[0])
# print(pairs_data.iloc[pd.isna(pairs_data["0"])])
# rows_with_nan = []
# for index, row in pairs_data.iterrows():
#     is_nan_series = row.isnull()
#     if is_nan_series.any():
#         rows_with_nan.append(index)
# print(rows_with_nan)
# print(pairs_data.iloc[rows_with_nan[0],0])
for anoph in ["Aalb", "Aatr", "ASteph", "Amer", "Acol"]:
    pairs_data = pd.read_csv("/mnt/scratch/ws/psbelokopytova/202109061534Polya/nn_anopheles/input/coolers/"+anoph+".pairs.txt", names=["0", "1", "2", "3"], sep="\t")
    print(pairs_data.keys())
    pairs_data["0"]= pairs_data["0"].apply(lambda  x: anoph+"_"+x)
    pairs_data["2"] = pairs_data["2"].apply(lambda x: anoph + "_" + x)
    pairs_data.to_csv("/mnt/scratch/ws/psbelokopytova/202109061534Polya/nn_anopheles/input/coolers/"+anoph+".mod.pairs.txt", sep="\t",
                index=False, header=False)
    print(len(pairs_data))
    # merge_df = pd.concat([merge_df, pairs_data])
    # len(merge_df)
# merge_df.to_csv("/mnt/scratch/ws/psbelokopytova/202107130921Polina/nn_anopheles/input/coolers/all_anoph.pairs.txt", sep="\t",
#                 index=False, header=False)