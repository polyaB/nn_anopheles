import sys
import os
import cooltools
import cooler
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")
import numpy as np
from cooltools.lib.numutils import adaptive_coarsegrain, observed_over_expected
from cooltools.lib.numutils import interpolate_bad_singletons, set_diag, interp_nan
from astropy.convolution import Gaussian2DKernel
from astropy.convolution import convolve

source_path = os.path.dirname(os.path.abspath(sys.argv[0])) + "/3Dpredictor/source"
source_path2 = os.path.dirname(os.path.abspath(sys.argv[0])) + "/3Dpredictor/nn/source"
sys.path.append(source_path)
sys.path.append(source_path2)
import pandas as pd
from shared import FileReader
fname = "/mnt/scratch/ws/psbelokopytova/202006201036data/nn_anopheles/input/hi-c_data/AAcol/Acol_5Kb.cool"
picture_dir =  "/mnt/scratch/ws/psbelokopytova/202006201036data/nn_anopheles/input/hi-c_data/pictures/"
chrs = ["X", "2R", "2L", "3R", "3L"]
# sample region of about 6000x6000
region_str = "2L:9300000-9900000"
def get_norm_matrix_for_region(chr, genome_cool, gap_chr_data, diagonal_offset=2, log=True, kernel_stddev = 0, clip=None, chr_region=None, **kwargs):
    if chr_region == None:
        hic_mat = genome_cool.matrix(balance=True).fetch(chr)
        hic_mat_raw = genome_cool.matrix(balance=False).fetch(chr)
    else:
        hic_mat = genome_cool.matrix(balance=True).fetch(chr_region)
        hic_mat_raw = genome_cool.matrix(balance=False).fetch(chr_region)

    picture_dir = "/mnt/scratch/ws/psbelokopytova/202006201036data/nn_anopheles/input/hi-c_data/pictures/"
    plt.imshow(np.log(hic_mat), cmap="OrRd")
    plt.colorbar()
    plt.savefig(os.path.join(picture_dir, "Acol_X_3600-4200_balance_mat.png"))
    plt.clf()

    plt.imshow(np.log(hic_mat_raw), cmap="OrRd")
    plt.colorbar()
    plt.savefig(os.path.join(picture_dir, "Acol_X_3600-4200_raw_mat.png"))
    plt.clf()
    # set blacklist (gaps from hi-c maps) to NaNs
    if chr in gap_chr_data:
        gaps = list(zip( gap_chr_data[chr]["start"], gap_chr_data[chr]["end"]))
        for gap in gaps:
            # adjust for sequence indexes
            black_seq_start =gap[0]
            black_seq_end = gap[1]
            hic_mat[:, black_seq_start:black_seq_end] = np.nan
            hic_mat[black_seq_start:black_seq_end, :] = np.nan
    seq_hic_nan = np.isnan(hic_mat)
    # clip first diagonals and high values
    clipval = np.nanmedian(np.diag(hic_mat, diagonal_offset))
    for i in range(-diagonal_offset + 1, diagonal_offset):
        set_diag(hic_mat, clipval, i)
    hic_mat = np.clip(hic_mat, 0, clipval)
    hic_mat[seq_hic_nan] = np.nan
    plt.imshow(np.log(hic_mat), cmap="OrRd")
    plt.colorbar()
    plt.savefig(os.path.join(picture_dir, "Acol_X_3600-4200_clipdiags_mat.png"))
    plt.clf()

    #adaptively coarse-grain
    mat_cg = adaptive_coarsegrain(hic_mat, hic_mat_raw)
    plt.imshow(np.log(mat_cg), cmap="OrRd")
    plt.colorbar()
    plt.savefig(os.path.join(picture_dir, "Acol_X_3600-4200_coarsegrained_mat.png"))
    plt.clf()
    #normalize for the distance-dependent decrease in contact frequency
    seq_hic_obsexp = observed_over_expected(mat_cg, ~seq_hic_nan)[0]
    plt.imshow(np.log(seq_hic_obsexp), cmap="OrRd")
    plt.colorbar()
    plt.savefig(os.path.join(picture_dir, "Acol_X_3600-4200_oe_mat.png"))
    plt.clf()
    #log and clip values in range interval
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


    plt.imshow(interp_seq_hic_obsexp, cmap="OrRd")
    plt.colorbar()
    plt.savefig(os.path.join(picture_dir, "Acol_X_3600-4200_log_oe_interp_mat.png"))
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

    plt.imshow(seq_hic, cmap="OrRd")
    plt.colorbar()
    plt.savefig(os.path.join(picture_dir, "Acol_X_3600-4200_kernel_std_finalmat.png"))
    plt.clf()
    return seq_hic

def normalize_hic_map(cool_file, chrs, gap_chr_data, diagonal_offset=2, log=True, kernel_stddev = 0, clip=None):
    genome_cool = cooler.Cooler(cool_file)
    chr_hic_norm_data = dict()
    for chr in chrs:
        chr_hic_matrix = get_norm_matrix_for_region(chr = chr, genome_cool = genome_cool, gap_chr_data = gap_chr_data,
                                                    diagonal_offset=diagonal_offset, log=log, kernel_stddev = kernel_stddev, clip=clip)
        chr_hic_norm_data[chr] = chr_hic_matrix
    return chr_hic_norm_data
