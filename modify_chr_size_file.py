import pandas as pd
import numpy as np
print("read pairs file")
pairs_data = pd.read_csv("/mnt/scratch/ws/psbelokopytova/202101241522data/nn_anopheles/input/coolers/Acol.pairs.txt",
                         sep="\t", names=["chr1", "pos1", "chr2", "pos2"])
print("read chrsize file")
chr_size_data = pd.read_csv("/mnt/scratch/ws/psbelokopytova/202101241522data/nn_anopheles/input/genomes/AcolNg_V4.chr.sizes",
                            sep="\t", names=["chr", "size"])
chrmosomes = ["2L", "2R", "3L", "3R", "X"]
maximums = []
for chr in chrmosomes:
    print("chr", chr)
    pos_list = list(pairs_data[pairs_data.chr1 == chr]["pos1"].values) + list(pairs_data[pairs_data.chr2 == chr]["pos2"].values)
    maximums.append(np.max(pos_list))
    print(chr, np.max(pos_list))