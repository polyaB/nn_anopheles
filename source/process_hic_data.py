import sys
import os
import cooltools
import cooler
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")
import numpy as np
import logging
from cooltools.lib.numutils import adaptive_coarsegrain, observed_over_expected
from cooltools.lib.numutils import interpolate_bad_singletons, set_diag, interp_nan
from astropy.convolution import Gaussian2DKernel
from astropy.convolution import convolve
import pickle

source_path = os.path.dirname(os.path.abspath(sys.argv[0])) + "/3Dpredictor/source"
source_path2 = os.path.dirname(os.path.abspath(sys.argv[0])) + "/3Dpredictor/nn/source"
sys.path.append(source_path)
sys.path.append(source_path2)
from termcolor import colored
import pandas as pd
from shared import FileReader
from termcolor import colored
fname = "/mnt/scratch/ws/psbelokopytova/202008151203data_Polya/nn_anopheles/input/hi-c_data/AAcol/Acol_5Kb.cool"
picture_dir =  "/mnt/scratch/ws/psbelokopytova/202008151203data_Polya/nn_anopheles/input/hi-c_data/pictures/"
chrs = ["X", "2R", "2L", "3R", "3L"]
# sample region of about 6000x6000
def get_norm_matrix_for_region(chr, genome_cool, gap_chr_data, diagonal_offset=2, log=True, kernel_stddev = 0, clip=None, chr_region=None, obs_exp=True, **kwargs):
    if chr_region == None:
        hic_mat = genome_cool.matrix(balance=True).fetch(chr)
        hic_mat_raw = genome_cool.matrix(balance=False).fetch(chr)
        logging.info("going to normalize hi-c matrix for " + chr + " chr")
    else:
        hic_mat = genome_cool.matrix(balance=True).fetch(chr_region)
        hic_mat_raw = genome_cool.matrix(balance=False).fetch(chr_region)
    # print(colored("balanced mat", 'blue'))
    # print(hic_mat)
    # print(hic_mat.shape)


    # plt.imshow(np.log(hic_mat), cmap="RdBu_r")
    # plt.colorbar()
    # plt.savefig(os.path.join(picture_dir, "Acol_2L_balance_mat_0906.png"))
    # plt.clf()
    #
    # plt.imshow(np.log(hic_mat_raw), cmap="RdBu_r")
    # plt.colorbar()
    # plt.savefig(os.path.join(picture_dir, "Acol_2L_raw_mat.png"))
    # plt.clf()
    # set blacklist (gaps from hi-c maps) to NaNs
    if chr in gap_chr_data:
        gaps = list(zip( gap_chr_data[chr]["start"], gap_chr_data[chr]["end"]))
        for gap in gaps:
            # adjust for sequence indexes
            black_seq_start =gap[0]//genome_cool.binsize
            black_seq_end = gap[1]//genome_cool.binsize

            hic_mat[:, black_seq_start:black_seq_end] = np.nan
            hic_mat[black_seq_start:black_seq_end, :] = np.nan
    seq_hic_nan = np.isnan(hic_mat)
    # print(len(seq_hic_nan))
    # print(hic_mat)
    # print(np.count_nonzero(~np.isnan(hic_mat)))
    logging.info("clip first diagonals and high values")
    # clip first diagonals and high values
    clipval = np.nanmedian(np.diag(hic_mat, diagonal_offset))
    for i in range(-diagonal_offset + 1, diagonal_offset):
        set_diag(hic_mat, clipval, i)
    hic_mat = np.clip(hic_mat, 0, clipval)
    hic_mat[seq_hic_nan] = np.nan
    # print(hic_mat)
    plt.imshow(np.log(hic_mat), cmap="RdBu_r")
    plt.colorbar()
    plt.savefig(os.path.join(picture_dir, "Acol_2L_clipdiags_mat.png"))
    plt.clf()

    #adaptively coarse-grain
    logging.info("adaptively coarse-grain")
    mat_cg = adaptive_coarsegrain(hic_mat, hic_mat_raw)
    plt.imshow(np.log(mat_cg), cmap="RdBu_r")
    plt.colorbar()
    plt.savefig(os.path.join(picture_dir, "Acol_"+str(chr)+"2L_coarsegrained_mat.png"))
    plt.clf()
    print(colored("coarse-grained mat", 'blue'))
    print(mat_cg)
    if obs_exp:
        #normalize for the distance-dependent decrease in contact frequency
        logging.info("normalize for the distance-dependent decrease in contact frequency")
        seq_hic_obsexp = observed_over_expected(mat_cg, ~seq_hic_nan)[0]
    else:
        seq_hic_obsexp = mat_cg
    print(colored("oe mat", 'blue'))
    print(seq_hic_obsexp)
    plt.imshow(np.log(seq_hic_obsexp), cmap="RdBu_r")
    plt.colorbar()
    plt.savefig(os.path.join(picture_dir, "Acol_2L_oe_mat.png"))
    plt.clf()
    #log and clip values in range interval
    logging.info("log and clip values in range interval, linearly interpolate")
    if log == True:
        seq_hic_obsexp = np.log(seq_hic_obsexp)
        if clip is not None:
            seq_hic_obsexp = np.clip(seq_hic_obsexp, -clip, clip)
        interp_seq_hic_obsexp = interp_nan(seq_hic_obsexp)
        for i in range(-diagonal_offset + 1, diagonal_offset): set_diag(interp_seq_hic_obsexp, 0, i)
    else:
        if clip is not None:
            seq_hic_obsexp = np.clip(seq_hic_obsexp, 0, clip)
        interp_seq_hic_obsexp = interp_nan(seq_hic_obsexp)
        for i in range(-diagonal_offset + 1, diagonal_offset): set_diag(interp_seq_hic_obsexp, 1, i)
    print(colored("log and interpolated mat", 'blue'))
    print(interp_seq_hic_obsexp)
    plt.imshow(interp_seq_hic_obsexp, cmap="RdBu_r")
    plt.colorbar()
    plt.savefig(os.path.join(picture_dir, "Acol_2L_log_oe_interp_mat.png"))
    plt.clf()

    if kernel_stddev > 0:
        # initialize Gaussian kernel
        kernel = Gaussian2DKernel(x_stddev=kernel_stddev)
    else:
        kernel = None
    # apply kernel
    if kernel is not None:
        seq_hic = convolve(interp_seq_hic_obsexp, kernel)
    else:
        seq_hic = interp_seq_hic_obsexp
    print(colored("final mat", 'blue'))
    print(seq_hic)
    # print(hic_mat)
    plt.imshow(seq_hic, cmap="RdBu_r")
    plt.colorbar()
    plt.savefig(os.path.join(picture_dir, "Acol_chr2L_"+chr+".png"))
    plt.clf()
    return seq_hic

def normalize_hic_map(cool_file, chrs, gap_chr_data, out_dump_file, diagonal_offset=2, log=True, kernel_stddev = 0, clip=None, obs_exp=True):
    if os.path.exists(out_dump_file):
        logging.info("Found dump for normalized hi-c map")
        with open(out_dump_file, "rb") as f:
            chr_hic_norm_data = pickle.load(f)
        return chr_hic_norm_data
    genome_cool = cooler.Cooler(cool_file)
    chr_hic_norm_data = dict()
    for chr in chrs:
        chr_hic_matrix = get_norm_matrix_for_region(chr = chr, genome_cool = genome_cool, gap_chr_data = gap_chr_data,
                                                    diagonal_offset=diagonal_offset, log=log, kernel_stddev = kernel_stddev, clip=clip, obs_exp=obs_exp)
        chr_hic_norm_data[chr] = chr_hic_matrix
    with open(out_dump_file, "wb") as f:
        pickle.dump(chr_hic_norm_data, f)
    return chr_hic_norm_data
