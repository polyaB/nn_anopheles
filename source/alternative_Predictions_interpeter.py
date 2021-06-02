import os
import sys
source_path = os.path.dirname(os.path.abspath(sys.argv[0])) + "/../basenji/source"
source_path2 = os.path.dirname(os.path.abspath(sys.argv[0])) + "/../basenji/basenji"
source_path3 = os.path.dirname(os.path.abspath(sys.argv[0])) + "/../3Dpredictor/source"
sys.path.append(source_path)
sys.path.append(source_path2)
sys.path.append(source_path3)
import numpy as np
from cooltools.lib.numutils import set_diag
import pysam
import pandas as pd
import dataset, dna_io, seqnn
import datetime
import matplotlib.pyplot as plt
import math
import h5py
import pickle
from matrix_plotter import MatrixPlotter
from matplot2hic import MatPlot2HiC
import cooler
from cooltools.lib.numutils import adaptive_coarsegrain



### for converting from flattened upper-triangluar vector to symmetric matrix  ###
def from_oe_to_contacts(seq_hic_obsexp, genome_hic_expected_file, interval, seq_len_pool, no_log=False):
    if no_log==False:
        seq_hic_obsexp = np.exp(seq_hic_obsexp)
    genome_hic_expected = pd.read_csv(genome_hic_expected_file, sep='\t')
    exp_chr = genome_hic_expected.iloc[genome_hic_expected['chrom'].values == interval.chr][0:seq_len_pool]
    if len(exp_chr) == 0:
        raise ValueError('no expected values found for chr:' + interval.chr)
    exp_map = np.zeros((seq_len_pool, seq_len_pool))
    print(exp_map.shape)
    for i in range(seq_len_pool):
        set_diag(exp_map, exp_chr['balanced.avg'].values[i], i)
        set_diag(exp_map, exp_chr['balanced.avg'].values[i], -i)
    seq_hic_smoothed = exp_map * seq_hic_obsexp
    return seq_hic_smoothed

def from_upper_triu(vector_repr, matrix_len, num_diags):
    z = np.zeros((matrix_len,matrix_len))
    triu_tup = np.triu_indices(matrix_len,num_diags)
    z[triu_tup] = vector_repr
    for i in range(-num_diags+1,num_diags):
        set_diag(z, np.nan, i)
    return z + z.T
def mat_to_pandas_df(mat, binsize, interval):
    starts = []
    ends = []
    values = []
    for i in range(mat.shape[0]):
        #     print(new_arr[i])
        for j in range(mat.shape[1]):
            #         print(np.isnan(new_arr[i][j]))
            if not np.isnan(mat[i][j]):
                starts.append(i * binsize + interval.start)
                ends.append(j * binsize + interval.start)
                values.append(mat[i][j])
    # print(starts)
    # print(ends)
    # print(values)
    data = {'chr': [interval.chr] * len(starts), 'contact_st': starts, 'contact_en': ends, 'contact_count': values}
    data_df = pd.DataFrame(data=data)
    return data_df

def plot_juicebox_from_predicted_array(mat, binsize, interval, out_dir,diagonal_offset, use_control=False, **kwargs):
        predicted_data = mat_to_pandas_df(mat=mat, binsize=binsize, interval=interval)
        print(predicted_data.isna().sum())
        print(predicted_data)

        mp = MatrixPlotter()
        mp.set_data(predicted_data)
        if not use_control:
            mp.set_control(predicted_data)
        else:
            if 'genome_cool_file' not in kwargs:
                print("please add path to control cool file")
                raise Exception
            # process hic data
            print("open and process control cool file")
            genome_hic_cool = cooler.Cooler(kwargs['genome_cool_file'])
            mseq_str = '%s:%d-%d' % (interval.chr, interval.start, interval.end)
            seq_hic_raw = genome_hic_cool.matrix(balance=True).fetch(mseq_str)
            print("seq_hic from cool file shape:", seq_hic_raw.shape, "predicted matrix shape:",mat.shape)
            assert seq_hic_raw.shape == mat.shape
            clipval = np.nanmedian(np.diag(seq_hic_raw, diagonal_offset))
            for i in range(-diagonal_offset + 1, diagonal_offset):
                set_diag(seq_hic_raw, clipval, i)
            seq_hic_raw = np.clip(seq_hic_raw, 0, clipval)
            # adaptively coarsegrain based on raw counts
            seq_hic_smoothed = adaptive_coarsegrain(
                seq_hic_raw,
                genome_hic_cool.matrix(balance=False).fetch(mseq_str),
                cutoff=2, max_levels=8)
            control_data = mat_to_pandas_df(mat=seq_hic_smoothed, binsize=binsize, interval=interval)
            print(len(control_data))
            #choose only contacts <= seqlen
            control_data_merge = pd.merge(predicted_data, control_data, on=["chr", "contact_st", "contact_en"])
            control_data = control_data_merge[['chr', 'contact_st', 'contact_en','contact_count_y']]
            control_data.rename(columns={'contact_count_y':'contact_count'}, inplace=True)
            print(control_data)
            print(len(control_data))
            mp.set_control(control_data)
        # mp.set_apply_log(self.apply_log)
        MatPlot2HiC(mp, interval.chr + "_" + str(interval.start) + '_' + str(interval.end) , out_dir+"hic/")

def predict_big_region_from_seq(interval,
                                binsize,
                                seq_len,
                                stride,
                                fasta_file,
                                seqnn_model,
                                crop_bp,
                                target_length_cropped,
                                hic_diags,
                                prediction_folder,
                                returned_to_contacts = True,
                                save_as_hic=True,
                                use_control=False,
                                **kwargs):
    """
        Predict big region by stacking the predicted small region units. Write the prediction to .hic file if it need
        Parameters
        ----------
        interval : 3DPredictor.shared.Interval
            Interval object
        binsize : int
        seq_len : int
            len of one predicted unit
        stride : int
            stride in the interval for predicted units
        fasta_file : str
            path to fasta file with the genome
        Returns
        -------
        """
    if interval.len< seq_len:
        print("can't predict such short region")
    # define shape of predicted array
    n_end = math.ceil(interval.end/binsize)
    n_start = math.floor(interval.start/binsize)
    n = n_end-n_start
    #deprecated n = math.ceil((interval.end - interval.start)/binsize)+1
    len_predicted_mat = (seq_len - 2*crop_bp)//binsize
    m=n
    print("Stride is", stride, ",", stride//binsize, "bins" )
    mat_stride = stride//binsize
    k = (n - (len_predicted_mat - mat_stride))//mat_stride
    print(datetime.datetime.now())
    print("...allocating array...", k,m,n)
    arr = np.empty((k,m,n))
    arr[:]=np.nan
    print(datetime.datetime.now(), "DONE")
    # print(arr.shape)
    start = interval.start
    arr_stride = crop_bp//binsize
    fasta_open = pysam.Fastafile(fasta_file)

    # predict k units
    print("going to predict", k, "matrix units")
    for k_matrix in range(0, k):
        # predict matrix for one region
        if k_matrix%5 == 0:
            print("predict", k_matrix, "matrix unit")
        chrm, seq_start, seq_end = interval.chr, int(start), int(start + seq_len)
        seq = fasta_open.fetch(chrm, seq_start, seq_end).upper()
        # with open(prediction_folder+"preseq"+str(seq_start)+"-"+str(seq_end)+".pickle", 'wb') as f:
        #     pickle.dump(seq, f)
        seq_1hot = dna_io.dna_1hot(seq)
        # print (seq[21680:21685])
        # print(seq_1hot[21680:21685][:])
        # with open(prediction_folder+"prepred"+str(seq_start)+"-"+str(seq_end)+".pickle", 'wb') as f:
        #     pickle.dump(seq_1hot, f)
        test_pred_from_seq = seqnn_model.model.predict(np.expand_dims(seq_1hot, 0))


        predicted_mat = from_upper_triu(test_pred_from_seq[:, :, 0], target_length_cropped, hic_diags)
        with open(prediction_folder + "prred_mat" + str(seq_start) + "-" + str(seq_end) + ".pickle", 'wb') as f:
            pickle.dump(predicted_mat, f)
        # print(0, target_length_cropped, hic_diags)
        # im = plt.matshow(predicted_mat, fignum=False, cmap='RdBu_r')  # , vmax=2, vmin=-2)
        # plt.colorbar(im, fraction=.04, pad=0.05)  # , ticks=[-2,-1, 0, 1,2])
        # plt.savefig(prediction_folder+"testtest"+str(seq_start)+"-"+str(seq_end))
        # plt.clf()
        assert predicted_mat.shape[0] == predicted_mat.shape[1]
        # write predicted unit to array for big interval
        for i in range(len(predicted_mat)):
            arr[k_matrix][i+arr_stride][0 + arr_stride:len(predicted_mat) + arr_stride] = predicted_mat[i]
        arr_stride += stride//binsize
        start+=stride
    #get mean array from predictions
    mat = np.nanmean(arr,axis=0)
    # empty_mat = np.empty((mat.shape[0],1))
    # print(mat.shape)
    # im = plt.matshow(mat, fignum=False, cmap='RdBu_r')  # , vmax=2, vmin=-2)
    # plt.colorbar(im, fraction=.04, pad=0.05)  # , ticks=[-2,-1, 0, 1,2])
    # plt.savefig(prediction_folder +"prediction_"+
    #             interval.chr+"_"+str(interval.start)+"-"+str(interval.end))
    # plt.clf()

    # return predicted values from oe to contacts
    if returned_to_contacts:
        if 'genome_hic_expected_file' not in kwargs:
            print("Please add path to expected file")
        mat = from_oe_to_contacts(seq_hic_obsexp=mat, genome_hic_expected_file=kwargs['genome_hic_expected_file'],
                                           interval=interval, seq_len_pool=n)
        # im = plt.matshow(mat, fignum=False, cmap='OrRd')  # , vmax=2, vmin=-2)
        # plt.colorbar(im, fraction=.04, pad=0.05)  # , ticks=[-2,-1, 0, 1,2])
        # plt.savefig(prediction_folder + "prediction_returned_" +
        #             interval.chr + "_" + str(interval.start) + "-" + str(interval.end))
    if save_as_hic:
        print("going to save in hic format")
        plot_juicebox_from_predicted_array(mat=mat, binsize=binsize, interval = interval, out_dir=prediction_folder, diagonal_offset=hic_diags, use_control=use_control,
                                           genome_cool_file=kwargs["genome_cool_file"])

    # Write predicted regions to bed file
    bed_file = open(prediction_folder + "predictions.bed", "w")
    bed_file.write(str(0) + "\t" + interval.chr + "\t" + str(interval.start) + "\t" + str(interval.end) + "\n")