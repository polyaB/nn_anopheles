import pandas as pd
import numpy as np
print("read pairs file")
pairs_data = pd.read_csv("/mnt/scratch/ws/psbelokopytova/202105171236data_Polina/nn_anopheles/input/coolers/Drosophila.pairs.txt",
                         sep="\t", names=["chr1", "pos1", "chr2", "pos2"])
print("read chrsize file")
chr_size_data = pd.read_csv("/mnt/scratch/ws/psbelokopytova//202105171236data_Polina/nn_anopheles/input/genomes/dm3.chr.sizes",
                            sep="\t", names=["chr", "size"])
# chrmosomes = ["chr2L", "chr2R", "chr3L", "chr3R", "chrX", "chr4", "chrY"]
chrmosomes = ["chr2L", "chr2LHet", "chr2R", "chr2RHet", "chr3L", "chr3LHet", "chr3R", "chr3RHet", "chr4", "chrM", "chrX", "chrXHet", "chrYHet"]
maximums = []
for chr in chrmosomes:
    print("chr", chr)
    pos_list = list(pairs_data[pairs_data.chr1 == chr]["pos1"].values) + list(pairs_data[pairs_data.chr2 == chr]["pos2"].values)
    maximums.append(np.max(pos_list))
    print(chr, np.max(pos_list))