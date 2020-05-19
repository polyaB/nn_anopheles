from find_gaps import generate_gaps
from seq_dataset import bed_seq_data, generate_train_dataset
from process_hic_data import normalize_hic_map, get_norm_matrix_for_region
import sys,os
source_path = os.path.dirname(os.path.abspath(sys.argv[0])) + "/3Dpredictor/source"
source_path2 = os.path.dirname(os.path.abspath(sys.argv[0])) + "/3Dpredictor/nn/source"
sys.path.append(source_path)
sys.path.append(source_path2)
from fastaFileReader import fastaReader
cool_file = "/mnt/scratch/ws/psbelokopytova/202006201036data/nn_anopheles/input/hi-c_data/AAcol/Acol_5Kb.cool"
chr_size_file = "/mnt/scratch/ws/psbelokopytova/202006201036data/nn_anopheles/input/genomes/Acol_V4.chr.sizes"
output_gap_file = "/mnt/scratch/ws/psbelokopytova/202006201036data/nn_anopheles/input/gaps_files/2gaps_ACol_90_processed_24.04.bed"
out_train_test = "/mnt/scratch/ws/psbelokopytova/202006201036data/nn_anopheles/input/train/train_test_Acol_V4_2804.bed"
fasta_file = "/mnt/scratch/ws/psbelokopytova/202006201036data/nn_anopheles/input/genomes/AcolNg_V4.fa"
chr_list=["X"]

#get gaps
gaps_chr_data = generate_gaps(chr_list=chr_list, cool_file=cool_file, output_gap_file=output_gap_file, zero_proc_in_line=90)
# print(gaps_chr_data)

#get intervals labeled for train and test
seq_chr_data = bed_seq_data(gap_chr_data=gaps_chr_data, chr_size_file=chr_size_file, out_bed_file=out_train_test, chr_list="all")
# print(seq_chr_data)

#get normalized hi-c matrix
chr_norm_hic_data = normalize_hic_map(cool_file=cool_file, chrs=["X"], gap_chr_data=gaps_chr_data)
print(chr_norm_hic_data)

#write in train file
genome = fastaReader(fasta_file, name="ACol",useOnlyChromosomes=["X"])
genome = genome.read_data()
generate_train_dataset(seq_chr_data, genome, chr_norm_hic_data, chrms=set("X"))
#for interval in train intervals
# get one-hot encoded sequence
# get target hic matrix
#write in test file
