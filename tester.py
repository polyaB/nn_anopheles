import os
import sys
source_path = os.path.dirname(os.path.abspath(sys.argv[0])) + "/basenji/source"
print(source_path)
source_path2 = os.path.dirname(os.path.abspath(sys.argv[0])) + "/basenji/basenji"
source_path3 = os.path.dirname(os.path.abspath(sys.argv[0])) + "/3Dpredictor/source"
source_path4 = os.path.dirname(os.path.abspath(sys.argv[0])) + "/source"
sys.path.append(source_path)
sys.path.append(source_path2)
sys.path.append(source_path3)
sys.path.append(source_path4)
from termcolor import colored

from shared import Interval
import json
import dataset, dna_io, seqnn
from intervals_to_hic import from_upper_triu, predict_big_region_from_seq
import sys
import errno

# ask intervals
print(colored('This program asks to type interval parameters per interval and (y/n) depending whether you have another interval you want to predict.', 'yellow'))
print(colored('Type everything in one line, separate by spaces.', 'yellow'))
print(colored('First parameter is name of the chromosome harboring this interval.', 'yellow'))
print(colored('Second parameter is the number of first nucleotide of the interval.', 'yellow'))
print(colored('Third parameter is the number of last nucleotide of the interval.', 'yellow'))
print(colored('Example of input: 2L 10000000 13000000 n', 'yellow'))
choice = 'y'
interval_list = []
while choice == 'y':  # check that it is y or n? unnecessary right
    print(colored('Type parameters of one interval.', 'yellow'))
    input_list = list(map(str, input().split()))
    chr = input_list[0]
    start = int(input_list[1])
    end = int(input_list[2])
    choice = input_list[3]
    interval = Interval(chr, start, end)
    interval_list.append(interval)

# create folders
print(colored('Type path for directory to be created for sending output to.', 'yellow'))
print(colored('If such folder already exists its contents will be overwritten.', 'yellow'))
print(colored('Example: path to directory named "pleasework". It is shown below.', 'yellow')) # add example for Windows too
print(colored('/home/konstantin/konstantin/2/nn_anopheles/pleasework/', 'yellow'))
prediction_folder = input()
try:
    os.makedirs(prediction_folder)
    os.makedirs(prediction_folder + 'pre')
    os.makedirs(prediction_folder + 'hic')
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

# sort intervals by chromosomes
interval_list = sorted(interval_list, key=lambda intervalI: intervalI.chr)

model_dir = '/home/konstantin/konstantin/2/nn_anopheles/pleasework/'
params_file = model_dir+'params.json'
model_file  = model_dir+'model_best.h5'
with open(params_file) as params_open:
    params = json.load(params_open)
    params_model = params['model']
    params_train = params['train']
seqnn_model = seqnn.SeqNN(params_model)
### restore model ###
seqnn_model.restore(model_file)
print('successfully loaded')

# read data parameters
data_dir ='/home/konstantin/konstantin/2/nn_anopheles/pleasework/'
data_stats_file = '%s/statistics.json' % data_dir
with open(data_stats_file) as data_stats_open:
    data_stats = json.load(data_stats_open)
seq_length = data_stats['seq_length']
target_length = data_stats['target_length']
hic_diags =  data_stats['diagonal_offset']
target_crop = data_stats['crop_bp'] // data_stats['pool_width']
target_length1 = data_stats['seq_length'] // data_stats['pool_width']
target_length1_cropped = target_length1 - 2*target_crop

predict_big_region_from_seq(interval_list, binsize=data_stats['pool_width'], seq_len=seq_length, stride = 300*data_stats['pool_width'],
                            fasta_file="/home/konstantin/konstantin/2/nn_anopheles/pleasework/AalbS2_V4.fa", seqnn_model = seqnn_model,
                            crop_bp = data_stats['crop_bp'],target_length_cropped=target_length1_cropped, hic_diags = hic_diags,
                            prediction_folder=prediction_folder,
                            genome_hic_expected_file='/home/konstantin/konstantin/2/nn_anopheles/pleasework/Aalb_2048.expected',
                            use_control=True, genome_cool_file = '/home/konstantin/konstantin/2/nn_anopheles/pleasework/Aalb_2048.cool')
