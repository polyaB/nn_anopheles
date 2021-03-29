import os
import sys
source_path = os.path.dirname(os.path.abspath(sys.argv[0])) + "/3Dpredictor/source"
source_path2 = os.path.dirname(os.path.abspath(sys.argv[0])) + "/3Dpredictor/nn/source"
source_path3 = os.path.dirname(os.path.abspath(sys.argv[0])) + "/source"
sys.path.append(source_path)
sys.path.append(source_path2)
sys.path.append(source_path3)
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
output_gap_folder = "/mnt/scratch/ws/psbelokopytova/202006201036data/nn_anopheles/input/gaps_files/"
out_train_test = "/mnt/scratch/ws/psbelokopytova/202006201036data/nn_anopheles/input/train/train_test_Acol_V4_2804.bed"
out_norm_hic_dump_file = "/mnt/scratch/ws/psbelokopytova/202006201036data/nn_anopheles/input/hi-c_data/AAcol/test_2L_30300000-30800000"
chr_list=["X"]

#get normalized hi-c matrix
# chr_norm_hic_data = normalize_hic_map(cool_file=cool_file, chrs=["X"], gap_chr_data=gaps_chr_data)
genome_cool =cooler.Cooler(cool_file)
chr_region = "2L:30300000-30800000"
gaps_chr_data= generate_gaps(chr_list=chr_list, cool_file=cool_file, output_gap_folder=output_gap_folder, zero_proc_in_line=95)
norm_hic_data = get_norm_matrix_for_region(chr="2L", genome_cool=genome_cool, gap_chr_data=gaps_chr_data, chr_region = None,
                                           out_dump_file=out_norm_hic_dump_file, obs_exp=True
                                           )
# print(norm_hic_data.shape)
# test_region = norm_hic_data[0:30, 0:30]
# print(test_region.shape)
# picture_dir = "/mnt/scratch/ws/psbelokopytova/202006201036data/nn_anopheles/input/hi-c_data/pictures/"
# plt.imshow(test_region, cmap="OrRd")
# plt.colorbar()
# plt.savefig(os.path.join(picture_dir, "Acol_X_test.png"))
# plt.clf()

print(norm_hic_data)

