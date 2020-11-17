from find_gaps import generate_gaps
from seq_dataset import bed_seq_data, generate_train_dataset
from process_hic_data import normalize_hic_map, get_norm_matrix_for_region
import sys,os
import numpy as np
import logging
from termcolor import colored, cprint
logging.basicConfig(format='%(asctime)s %(name)s: %(message)s', datefmt='%I:%M:%S', level=logging.INFO)

source_path = os.path.dirname(os.path.abspath(sys.argv[0])) + "/../3Dpredictor/source"
source_path2 = os.path.dirname(os.path.abspath(sys.argv[0])) + "/../3Dpredictor/nn/source"
sys.path.append(source_path)
sys.path.append(source_path2)
from fastaFileReader import fastaReader

out_folder = "/mnt/scratch/ws/psbelokopytova/202011291709Polya_data/nn_anopheles/"
cool_file = "/mnt/scratch/ws/psbelokopytova/202011291709Polya_data/nn_anopheles/input/hi-c_data/AAcol/Acol_4096bp.cool"
chr_size_file = "/mnt/scratch/ws/psbelokopytova/202011291709Polya_data/nn_anopheles/input/genomes/AcolNg_modified.chr.sizes"
output_gap_folder = "/mnt/scratch/ws/psbelokopytova/202011291709Polya_data/nn_anopheles/input/gaps_files/"
out_train_test_folder = "/mnt/scratch/ws/psbelokopytova/202011291709Polya_data/nn_anopheles/input/train/"
fasta_file = "/mnt/scratch/ws/psbelokopytova/202011291709Polya_data/nn_anopheles/input/genomes/AcolNg_V4.fa"
out_norm_hic_dump_file = "/mnt/scratch/ws/psbelokopytova/202011291709Polya_data/nn_anopheles/input/hi-c_data/AAcol/norm_hic_data_all_chrs_contacts"

chr_list=["2L", "2R", "3L", "3R", "X"]
# chr_list = ["3L"]

#get gaps
logging.info(colored("Going to find gaps in " + cool_file, 'green'))
gaps_chr_data= generate_gaps(chr_list=chr_list, cool_file=cool_file, output_gap_folder=output_gap_folder, zero_proc_in_line=95)

# print(gaps_chr_data)

#get intervals labeled for train and test
logging.info(colored("Going to generate bed intervals for train and test from " + cool_file, 'green'))
seq_chr_data = bed_seq_data(gap_chr_data=gaps_chr_data, chr_size_file=chr_size_file, out_bed_folder=out_train_test_folder,
                            chr_list=chr_list, seq_len = 524288, shift = 524288//4)

#get normalized hi-c matrix
logging.info(colored("going to normalize hi-c data", 'green'))
chr_norm_hic_data = normalize_hic_map(cool_file=cool_file, chrs=chr_list, gap_chr_data=gaps_chr_data,
                                      out_dump_file=out_folder+"input/hi-c_data/AAcol/norm_hic_data_"+str(chr_list)+".pickle", obs_exp=False)
logging.info(colored("succesfully normalize hi-c data", 'green'))
# exit()
# print(chr_norm_hic_data["X"])
# print(chr_norm_hic_data["X"].shape)
# print("nonzeroes",np.count_nonzero(chr_norm_hic_data["X"]))

#for interval in train or test intervals
# get one-hot encoded sequence
# get target hic matrix
#write in test file
logging.info(colored("going to generate encoded sequences and targets dataset", 'green'))
genome = fastaReader(fasta_file, name="ACol")
genome = genome.read_data()
generate_train_dataset(seq_chr_data, genome, chr_norm_hic_data, chrms=set(chr_list), out_file=out_folder+"output/train_dataset_"+str(chr_list)+".pickle", train_test="train", target_crop_bp=0)
# generate_train_dataset(seq_chr_data, genome, chr_norm_hic_data, chrms=set(chr_list), out_file=out_folder+"output/test_dataset_"+str(chr_list)+".pickle", train_test="test", target_crop_bp=0)


