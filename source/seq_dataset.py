import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # or any {'0', '1', '2'}
import tensorflow as tf
import random
source_path = os.path.dirname(os.path.abspath(sys.argv[0])) + "/../3Dpredictor/source"
source_path2 = os.path.dirname(os.path.abspath(sys.argv[0])) + "/../3Dpredictor/nn/source"
sys.path.append(source_path)
sys.path.append(source_path2)
from fastaFileReader import fastaReader, GC_content
from shared import Interval
import logging
import pandas as pd
import pickle
import numpy as np
from BedReader import BedReader
logging.basicConfig(format='%(asctime)s %(name)s: %(message)s', datefmt='%I:%M:%S', level=logging.INFO)

fasta_file = "/mnt/scratch/ws/psbelokopytova/202004261138polina_data/nn_anopheles/input/genomes/AcolNg_V3.fa"
out_file_path = "/mnt/scratch/ws/psbelokopytova/202004261138polina_data/nn_anopheles/output/ACol/ACol_train_700kb" #TODO generate name of file using characteristics
# fasta_file = "/mnt/scratch/ws/psbelokopytova/202004261138polina_data/nn_anopheles/input/genomes/AcolNg_V3.fa"
gaps_folder = "/mnt/scratch/ws/psbelokopytova/202004261138polina_data/nn_anopheles/input/gaps_files/"
anopheles_name = "ACol"
seq_len = 524288
shift = seq_len//2
test_fraction = 0.2
#this function generate sequences for train and test
# seq_len is length of train sequence fragment
# shift is shift for start of next sequence in train and test dataset
# cutting_chr_edges is number of nucleotides from ends of chromosome (because chr ends have low read coverage)
# chr_list is list of chromosomes using for train and test
def bed_seq_data(gap_chr_data, chr_size_file, out_bed_folder, seq_len=524288, shift=524288//2, cutting_chr_edges = 100000, test_fraction=0.2, chr_list="all"):

    if os.path.exists(out_bed_folder+"train_test_shift"+str(seq_len//shift)+"_"+str(chr_list)+".bed"):
        logging.info("Found dump for seq bed file " + out_bed_folder+"train_test_shift"+str(seq_len//shift)+"_"+str(chr_list)+".bed")
        bed_reader = BedReader(out_bed_folder+"train_test_shift"+str(seq_len//shift)+"_"+str(chr_list)+".bed")
        bed_reader.read_file(renamer = {"0":"chr","1":"start","2":"end", "3":"train_test"})
        return bed_reader.chr_data
    else:
        chr_size_data = pd.read_csv(chr_size_file, header=None, names = ["chr", "size"], sep="\t")
        chr_data = dict()
        if chr_list=="all":
            chr_list = chr_size_data["chr"]
        else:
            chr_list = chr_list
        # print("chr_list", chr_list)
        for chr in chr_list:
            chr_size = chr_size_data[chr_size_data["chr"]==chr]
            assert len(chr_size)==1
            chr_len = chr_size.iloc[0][1]
            if chr not in gap_chr_data.keys():
                logging.getLogger(__name__).warning("There are no gaps on " + chr + " chromosome")
                gaps = [(chr_len - cutting_chr_edges , chr_len - cutting_chr_edges)]
            else:
                gaps = list(zip( gap_chr_data[chr]["start"], gap_chr_data[chr]["end"]))
            chrs, starts, ends = [], [], []
            start_seq = cutting_chr_edges
            for count,gap in enumerate(sorted(gaps)):
                end_seq = gap[0] - seq_len #start of last seq in region between gaps
                if end_seq-start_seq < seq_len:
                    start_seq = gap[1]
                    continue
                else:
                    for start in range(start_seq, end_seq+1,shift):
                        chrs.append(chr)
                        starts.append(start)
                        ends.append(start+seq_len)
                    start_seq = gap[1]
                if count == len(gaps) - 1:
                    start_seq = gap[1]
                    end_seq = chr_len - cutting_chr_edges -seq_len
                    if end_seq - start_seq > seq_len:
                        for start in range(start_seq, end_seq,shift):
                            chrs.append(chr)
                            starts.append(start)
                            ends.append(start+seq_len)
            data = pd.DataFrame({"chr":chrs, "start":starts, "end":ends})
            data["train_test"] = ["train"] * len(data)
            # print(data)
            random.seed()
            rand_int = random.randint(0, len(data) - round(len(data)*test_fraction))
            # print("rand_int", rand_int)
            data.iloc[rand_int:rand_int + round(len(data)*test_fraction),3] = "test"
            chr_data[chr] = data
        conc_data = pd.concat([chr_data[chr] for chr in chr_data.keys()])
        conc_data.to_csv(out_bed_folder+"train_test_shift"+str(seq_len//shift)+"_"+str(chr_list)+".bed", sep="\t", header = False, index=False)
        conc_data[conc_data["train_test"]=="train"][["chr", "start", "end"]].to_csv(out_bed_folder+"train_shift"+str(seq_len//shift)+"_"+str(chr_list)+".bed",  header=False, index=False, sep="\t")
        conc_data[conc_data["train_test"] == "test"][["chr", "start", "end"]].to_csv(out_bed_folder + "test_shift"+str(seq_len//shift)+"_"+str(chr_list)+".bed",header=False, index=False, sep="\t")
        return chr_data



    # data = {}
    # indices=np.array(range(0, len(seqs)))
    # np.random.shuffle(indices)
    # seq_arr = np.stack(seqs)
    # gc_arr  = np.array(GC_list)*100
    # assert len(gc_arr == len(seq_arr))
    # test_indices = indices[0: round(len(seq_arr)*test_fraction)]
    # train_indices = indices[round(len(seq_arr)*test_fraction):len(seq_arr)]
    # data["test_seq"] = seq_arr[test_indices]
    # data["test_gc"] = gc_arr[test_indices]
    # data["train_seq"] = seq_arr[train_indices]
    # data["train_gc"] = gc_arr[train_indices]

    # data = pd.DataFrame(data={'seq':np.stack(seqs), 'gc':GC_list})
    # data["label"] = ["train"]*len(data)
    # ids_test = random.sample(range(0,len(data)), len(data)*20//100)
    # data.iloc[ids_test, data.columns.get_loc("label")] = "test"

# def encode_seq():
#     seq = genome.get_interval(Interval(chr, start, start + seq_len))
#     gc = GC_content(seq)
#     encoded_seq = tf.constant(tf.one_hot(seq, depth=4))

def generate_train_dataset(seq_chr_data, fasta_genome, chr_norm_hic_data, out_file, train_test = "train", chrms = "all",
                           target_crop_bp=0, diagonal_offset=2):
    intervals, inputs, targets = [], [], []
    print("train_test", train_test)
    for chr in seq_chr_data.keys():
        print(chr)
        if chrms == "all" or chr in chrms:
            data = seq_chr_data[chr]
            seq_chr_data[chr] = seq_chr_data[chr][seq_chr_data[chr]["train_test"] == train_test]
            print(seq_chr_data[chr])
            for seq in list(zip(seq_chr_data[chr]["start"], seq_chr_data[chr]["end"])):
                seq_region = fasta_genome.get_interval(Interval(chr, seq[0], seq[1]))
                # print("seq_region")
                # print(seq_region)
                encoded_seq = tf.constant(tf.one_hot(seq_region, depth=4))
                inputs.append(encoded_seq)
                binsize = 4096 #TODO create class hic_data and its method binsize
                # compute dimensions
                seq_len_nt = seq[1] - seq[0]
                seq_len_pool = seq_len_nt // binsize

                if target_crop_bp == 0:
                    seq_len_crop = seq_len_pool
                else:
                    crop_start = target_crop_bp // binsize
                    crop_end = seq_len_pool - crop_start
                    seq_len_crop = seq_len_pool - 2 * crop_start

                # unroll upper triangular
                target = chr_norm_hic_data[chr][seq[0]//binsize:seq[1]//binsize, seq[0]//binsize:seq[1]//binsize]
                assert target.shape[0] == target.shape[1]
                assert target.shape[0] * binsize == len(seq_region)
                # compute upper triangular indexes
                triu_tup = np.triu_indices(seq_len_crop, diagonal_offset)
                target = target[triu_tup]
                targets.append(target)
                intervals.append((chr, seq[0], seq[1]))
                # print(len(seq_region), encoded_seq.shape)
                # print(target.shape)

    data = dict()
    print(len(intervals), len(inputs), len(targets))
    data["intervals"] = intervals
    data["inputs"] = inputs
    data["targets"] = targets
    with open(out_file, 'wb') as f:
        pickle.dump(data, f)

    # pickle.dump()
    # print("here")
    # inputs = np.asarray(inputs)
    # print("here")
    # intervals = np.asarray(intervals)
    # targets = np.asarray(targets)
    # print(intervals.shape, inputs.shape, targets.shape)



