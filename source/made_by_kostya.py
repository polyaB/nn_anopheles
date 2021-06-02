# remember to import things

def made_by_kostya(interval: Interval,
                                    binsize,
                                    seq_len,
                                    stride,
                                    fasta_file,
                                    seqnn_model,
                                    crop_bp,
                                    target_length_cropped,
                                    hic_diags,
                                    prediction_folder,
                                    returned_to_contacts=True,
                                    save_as_hic=True,
                                    use_control=False,
                                    minimal_length=10000000,

                                    **kwargs):

    # open cool file, out of cycle - need to do only once, then just transmit as an argument
    # TODO add argument 'genome_hic_cool' to functions that should use it
    genome_hic_cool = cooler.Cooler(kwargs['genome_cool_file'])

    # make dirs, out of cycle - need to do only once
    try:
        os.makedirs(out_folder + '/' + fname)
        os.makedirs(out_folder + '/' + fname + '/pre')
        os.makedirs(out_folder + '/' + fname + '/hic')
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


    def predictor(subinterval: Interval):

        # define shape of predicted array
        n_end = math.ceil(subinterval.end / binsize)
        n_start = math.floor(subinterval.start / binsize)
        n = n_end - n_start
        # deprecated n = math.ceil((interval.end - interval.start)/binsize)+1
        len_predicted_mat = (seq_len - 2 * crop_bp) // binsize
        m = n
        print("Stride is", stride, ",", stride // binsize, "bins")
        mat_stride = stride // binsize
        k = (n - (len_predicted_mat - mat_stride)) // mat_stride
        print(datetime.datetime.now())
        print("...allocating array...", k, m, n)
        arr = np.empty((k, m, n))
        arr[:] = np.nan
        print(datetime.datetime.now(), "DONE")
        # print(arr.shape)
        start = subinterval.start
        arr_stride = crop_bp // binsize
        fasta_open = pysam.Fastafile(fasta_file)

        # predict k units
        print("going to predict", k, "matrix units")
        for k_matrix in range(0, k):
            # predict matrix for one region
            if k_matrix % 5 == 0:
                print("predict", k_matrix, "matrix unit")
            chrm, seq_start, seq_end = subinterval.chr, int(start), int(start + seq_len)
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
                arr[k_matrix][i + arr_stride][0 + arr_stride:len(predicted_mat) + arr_stride] = predicted_mat[i]
            arr_stride += stride // binsize
            start += stride
        # get mean array from predictions
        mat = np.nanmean(arr, axis=0)
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
            mat = from_oe_to_contacts(seq_hic_obsexp=mat,
                                      genome_hic_expected_file=kwargs['genome_hic_expected_file'],
                                      interval=subinterval, seq_len_pool=n)
            # im = plt.matshow(mat, fignum=False, cmap='OrRd')  # , vmax=2, vmin=-2)
            # plt.colorbar(im, fraction=.04, pad=0.05)  # , ticks=[-2,-1, 0, 1,2])
            # plt.savefig(prediction_folder + "prediction_returned_" +
            #             interval.chr + "_" + str(interval.start) + "-" + str(interval.end))

        # TODO rename 'plot_juicebox_from_predicted_array'
        if save_as_hic:
            print("going to save in hic format")
            plot_juicebox_from_predicted_array(mat=mat, binsize=binsize, interval=subinterval,
                                               out_dir=prediction_folder,
                                               diagonal_offset=hic_diags, use_control=use_control,
                                               genome_cool_file=kwargs["genome_cool_file"])

        # Write predicted regions to bed file
        bed_file = open(prediction_folder + "predictions.bed", "w")
        bed_file.write(
            str(0) + "\t" + subinterval.chr + "\t" + str(subinterval.start) + "\t" + str(subinterval.end) + "\n")

    assert minimal_length >= seq_len
    assert interval.len >= seq_len
    if interval.len <= minimal_length:
        predictor(interval)
    else:
        if interval.len // minimal_length == 1:
            predictor(interval)
        else:  # elif interval.len // minimal_length > 1:
            if interval.len % minimal_length == 0:
                i = 0  # how many bps we've predicted
                while i != interval.len:
                    predictor(subinterval=Interval(interval.chr,
                                                   interval.start + i,
                                                   i + minimal_length))
                    i += minimal_length
            else:  # elif interval.len % minimal_length > 0:
                residual_interval = Interval(interval.chr,
                                             interval.end - (minimal_length + interval.len % minimal_length),
                                             interval.end)
                predictor(subinterval=residual_interval)
                without_residue_interval = Interval(interval.chr,
                                                    interval.start,
                                                    interval.end - (minimal_length + interval.len % minimal_length))
                i = 0  # how many bps we've predicted
                while i != without_residue_interval.len:
                    predictor(subinterval=Interval(interval.chr, interval.start + i, i + minimal_length))
                    i += minimal_length


    def plot_juicebox_from_predicted_array(mat, binsize, interval, out_dir, diagonal_offset, use_control=False, **kwargs):
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
            mseq_str = '%s:%d-%d' % (interval.chr, interval.start, interval.end)
            seq_hic_raw = genome_hic_cool.matrix(balance=True).fetch(mseq_str)
            print("seq_hic from cool file shape:", seq_hic_raw.shape, "predicted matrix shape:", mat.shape)
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
            # choose only contacts <= seqlen
            control_data_merge = pd.merge(predicted_data, control_data, on=["chr", "contact_st", "contact_en"])
            control_data = control_data_merge[['chr', 'contact_st', 'contact_en', 'contact_count_y']]
            control_data.rename(columns={'contact_count_y': 'contact_count'}, inplace=True)
            print(control_data)
            print(len(control_data))
            mp.set_control(control_data)
        # mp.set_apply_log(self.apply_log)
        MatPlot2HiC(mp, interval.chr + "_" + str(interval.start) + '_' + str(interval.end), out_dir + "hic/")


    def MatPlot2HiC(matplot_obj, fname, out_folder, juicer_path=None):
        def Pandas2ChrSizes(chrsizes_filename,
                            pandas_df):  # This func takes all the chromosomes from pandas object, find out their sizes and write into file
            # chromosomes = pandas_df.ix[:, 0].unique()
            chromosomes = pandas_df["chr"].unique()
            chrsizes_table = pd.DataFrame(columns=chromosomes)

            for i in range(len(chromosomes)):
                buf = pandas_df.loc[pandas_df['chr'] == chromosomes[i]][['contact_st', 'contact_en']]
                max1 = buf.max().max()
                chrsizes_table.at[0, chromosomes[i]] = max1

                print('Completed: {}%'.format(i * 100 // len(chromosomes)), end='\r')

            chr_list = list(chrsizes_table)

            chrsizes_file = open(chrsizes_filename, 'w')

            for j in range(len(chr_list)):
                chrsizes_file.write(chr_list[j] + '\t' + str(chrsizes_table.iloc[0][chr_list[j]]) + '\n')

            chrsizes_file.close()

        def Pandas2Pre(pre_filename, pandas_df):  # This func makes pre-HiC file from the pandas object, control or data
            pre_file = open(pre_filename, 'rw')  # 'w' overwrites, 'rw' adds to existing file
            data_rows = pandas_df.shape[0]

            pandas_df.columns = ["chr1", "start", "end", "count"]
            pandas_df['str1'] = 0
            # assert len(pandas_df.loc[(pandas_df['count'] < 0.000001) & (pandas_df['count'] != 0)]) < (len(pandas_df['count']) / 10)
            pandas_df['exp'] = pandas_df['count'] * ( 1000000 )
            pandas_df['exp'] = round(pandas_df['exp']).astype(int)

            pandas_df.to_csv(pre_file, sep=" ",
                             columns=['str1', 'chr1', 'start', 'start', 'str1', 'chr1', 'end', 'end', 'exp'], header=False,
                             index=False,
                             line_terminator="\n")

            pre_file.close()

        # make filenames
        chromsizes_filename = out_folder + '/' + fname + '/pre/chrom.sizes'
        pre_data_filename = out_folder + '/' + fname + '/pre/pre_data.txt'
        hic_data_filename = out_folder + '/' + fname + '/hic/data.hic'
        pre_control_filename = out_folder + '/' + fname + '/pre/pre_control.txt'
        hic_control_filename = out_folder + '/' + fname + '/hic/control.hic'

        # make chrom.sizes, pre-Hic for data and control
        print('Make chromosome sizes file...')
        time1 = time.time()
        # print(matplot_obj.data)
        Pandas2ChrSizes(chromsizes_filename, matplot_obj.data)
        time2 = time.time()
        print('Time: ' + str(round(time2 - time1, 3)) + ' sec\n')
        print(colored("[SUCCESS]", 'green') + ' Chromosome sizes file created.\n')

        print('Make data pre-HiC file...')
        time1 =- time.time()
        Pandas2Pre(pre_data_filename, matplot_obj.data)
        time2 = time.time()
        print('Time: ' + str(round(time2 - time1, 3)) + ' sec\n')
        print(colored("[SUCCESS]", 'green') + ' DATA pre-HiC file created.\n')

        print('Make control pre-HiC file...')
        time1 = time.time()
        Pandas2Pre(pre_control_filename, matplot_obj.control)
        time2 = time.time()
        matplot_obj.columns = ["chr1", "start", "end", "count"]
        # binsize = str(get_bin_size(matplot_obj.control, fields=["start", "start"]))
        binsize=str(2048)
        print('Time: ' + str(round(time2 - time1, 3)) + ' sec\n')
        print(colored("[SUCCESS]", 'green') + ' CONTROL pre-HiC file created.\n')

    # everything below should be out of cycle
    # call juicer
    if juicer_path is None:
        juicer_path = os.path.join(os.path.dirname(os.path.abspath(os.path.abspath(__file__))),
                                   "juicer_tools.jar")
    cmd =  ['java', '-jar', juicer_path, 'pre', pre_data_filename, hic_data_filename, chromsizes_filename, '-n',
         '-r', binsize]
    print("Running command:")
    print (" ".join(map(str,cmd)))
    try:
        subprocess.check_output(cmd)
    except subprocess.CalledProcessError as e:
        print (e.output)
        raise Exception()
    print(colored("[SUCCESS]", 'green') + ' DATA HiC file created.\n')

    cmd = ['java', '-jar', juicer_path, 'pre', pre_control_filename, hic_control_filename, chromsizes_filename,
         '-n', '-r', binsize]
    print("Running command:")
    print (" ".join(map(str,cmd)))
    try:
        subprocess.check_output(cmd)
    except subprocess.CalledProcessError as e:
        print (e.output)
        raise Exception()

    print(colored("[SUCCESS]", 'green') + ' CONTROL HiC file created.\n')