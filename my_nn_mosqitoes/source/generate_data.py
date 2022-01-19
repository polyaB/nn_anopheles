import itertools 
import h5py
import pysam
import numpy as np
import sys
import os
source_path = os.path.dirname(os.path.abspath(sys.argv[0])) + "/../../"
sys.path.append(source_path)
from basenji.basenji.dna_io import dna_1hot
from basenji.bin.basenji_data import ModelSeq


def generate_pair_index_table(seqs_bed_data, mode="train"):
  mode_indices = seqs_bed_data[seqs_bed_data["mode"] == mode].index.tolist()
  combinations = []
  print(len(mode_indices))
  for i in mode_indices:
    for j in mode_indices:
      combinations.append((i,j))
  #deprecated
  # combinations = list(itertools.combinations(mode_indices, 2))
  return combinations

def generate_big_array(genome_cov_file, seqs_bed_file, fasta_file, out_folder):
  # read model sequences
  model_seqs = []
  for line in open(seqs_bed_file):
    a = line.split()
    model_seqs.append(ModelSeq(a[0],int(a[1]),int(a[2]),a[3]))
    
  # initialize targets
  seq_pool_len_hic = h5py.File(genome_cov_file, 'r')['targets'].shape[1]
  num_targets = len(model_seqs)
  targets = np.zeros((num_targets, seq_pool_len_hic), dtype='float16')
  # read target file
  seqs_cov_open = h5py.File(genome_cov_file, 'r')
  targets[:,:] = seqs_cov_open['targets'][:,:]
  seqs_cov_open.close()
  targets_mem_map = np.memmap(out_folder+"targets", dtype='float16', mode='w+', shape=(num_targets, seq_pool_len_hic))
  targets_mem_map[:] = targets[:]
  
  #generate encoded fasta for every seq
  # open FASTA
  fasta_open = pysam.Fastafile(fasta_file)
  seq_len = model_seqs[0].end - model_seqs[0].start
  print(seq_len)
  inputs = np.zeros((num_targets, seq_len, 4), dtype='float16')
  print(len(model_seqs))
  for si in range(len(model_seqs)):
    if si%10==0:
      print(si)
    mseq = model_seqs[si]
    # read FASTA
    seq_dna = fasta_open.fetch(mseq.chr, mseq.start, mseq.end)
    # one hot code
    seq_1hot = dna_1hot(seq_dna)
    # print(seq_1hot[0])
    inputs[si,:] = seq_1hot
    # print(inputs[si,:][0])
  fasta_open.close()
  inputs_mem_map = np.memmap(out_folder+"inputs", dtype='float16', mode='w+', shape=(num_targets, seq_len, 4))
  inputs_mem_map[:] = inputs[:]
