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

chr_list = ['2L', '2R', '3L', '3R', 'X']
generate_gaps(chr_list=chr_list, cool_file = "/mnt/scratch/ws/psbelokopytova/202101241522data/nn_anopheles/input/coolers/Acol_4096.cool",
              output_gap_folder="/mnt/scratch/ws/psbelokopytova/202101241522data/nn_anopheles/input/genomes/Acol_4096_")