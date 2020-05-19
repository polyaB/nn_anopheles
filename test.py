from find_gaps import generate_gaps
from seq_dataset import bed_seq_data
from process_hic_data import normalize_hic_map, get_norm_matrix_for_region
import cooler
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")
import os
from cooler.cli.balance import balance
cool_file = "/mnt/scratch/ws/psbelokopytova/202006201036data/nn_anopheles/input/hi-c_data/AAcol/Acol_5Kb.cool"
chr_size_file = "/mnt/scratch/ws/psbelokopytova/202006201036data/nn_anopheles/input/genomes/Acol_V4.chr.sizes"
output_gap_file = "/mnt/scratch/ws/psbelokopytova/202006201036data/nn_anopheles/input/gaps_files/2gaps_ACol_90_processed_24.04.bed"
out_train_test = "/mnt/scratch/ws/psbelokopytova/202006201036data/nn_anopheles/input/train/train_test_Acol_V4_2804.bed"
chr_list=["X"]
balance(cool_file)



#get normalized hi-c matrix
# chr_norm_hic_data = normalize_hic_map(cool_file=cool_file, chrs=["X"], gap_chr_data=gaps_chr_data)
genome_cool =cooler.Cooler(cool_file)
norm_hic_data = get_norm_matrix_for_region(chr="X", genome_cool=genome_cool, gap_chr_data=gaps_chr_data, chr_region = "X:3600000-4200000")
print(norm_hic_data.shape)
test_region = norm_hic_data[0:30, 0:30]
print(test_region.shape)
picture_dir = "/mnt/scratch/ws/psbelokopytova/202006201036data/nn_anopheles/input/hi-c_data/pictures/"
plt.imshow(test_region, cmap="OrRd")
plt.colorbar()
plt.savefig(os.path.join(picture_dir, "Acol_X_test.png"))
plt.clf()

print(norm_hic_data)

