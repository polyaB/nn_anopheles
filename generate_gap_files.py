import os
import sys
source_path = os.path.dirname(os.path.abspath(sys.argv[0])) + "/3Dpredictor/source"
source_path2 = os.path.dirname(os.path.abspath(sys.argv[0])) + "/3Dpredictor/nn/source"
source_path3 = os.path.dirname(os.path.abspath(sys.argv[0])) + "/source"
sys.path.append(source_path)
sys.path.append(source_path2)
sys.path.append(source_path3)

from find_gaps import generate_gaps
import logging
logging.basicConfig(format='%(asctime)s %(name)s: %(message)s', datefmt='%I:%M:%S', level=logging.INFO)

# chr_list = ['2L', '2R', '3L', '3R', 'X']
chr_list = ["chr2L", "chr2LHet", "chr2R", "chr2RHet", "chr3L", "chr3LHet", "chr3R", "chr3RHet", "chr4", "chrM", "chrX", "chrXHet", "chrYHet"]

generate_gaps(chr_list=chr_list, cool_file = "/mnt/scratch/ws/psbelokopytova/202105171236data_Polina/nn_anopheles/input/coolers/Drosophila_4096.cool",
              output_gap_folder="/mnt/scratch/ws/psbelokopytova/202105171236data_Polina/nn_anopheles/input/genomes/Dros_4096",
              zero_proc_in_line=97)