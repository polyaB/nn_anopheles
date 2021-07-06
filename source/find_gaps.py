import pandas as pd
import numpy as np
import cooler
import os
import logging
from BedReader import BedReader
logging.basicConfig(format='%(asctime)s %(name)s: %(message)s', datefmt='%I:%M:%S', level=logging.INFO)


#This function merge bin gaps in big gaps.
# minimum_gap_between - minimum distance between gaps to consider them distinct gaps
# min_gap_length - minimum length of gap
def merge_gaps(gaps, minimum_gap_between, min_gap_length, bed_file):
    chr = []
    start = []
    end= []
    with open(bed_file, "w") as out_file:
        current_start = 0
        current_end = 0
        for gap in gaps:
            if gap[1] - current_end < minimum_gap_between:
                current_end = gap[2]
            else:
                if(current_end - current_start) >= min_gap_length:
                    out_file.write(gap[0]+"\t"+str(current_start)+"\t"+str(current_end)+"\n")
                    chr.append(gap[0])
                    start.append(current_start)
                    end.append(current_end)
                current_start = int(gap[1])
                current_end = int(gap[2])
    data = pd.DataFrame({"chr":chr, "start":start, "end":end})
    return data
# This function generate gaps bed file. Gaps are generated if there are a lot of zeroes in matrix column.
# zero_proc_in_line is percent of zeroes in column of hi-c matrix
# bins_min_gap_between is minimum gap in bins between
# bins_min_gap is a minimum gap in bins between non gap regions
def generate_gaps(chr_list, cool_file, output_gap_folder, zero_proc_in_line=97, bins_min_gap_between=3, bins_min_gap=3):
    output_gap_file = output_gap_folder+"gaps+chr"+str(chr_list)+"_proc"+str(zero_proc_in_line)+".bed"
    if os.path.exists(output_gap_file):
        logging.info("Found dump for gap file " + output_gap_file)
        bed_reader = BedReader(output_gap_file)
        bed_reader.read_file()
        return bed_reader.chr_data
    else:
        c= cooler.Cooler(cool_file)
        chr_data = dict()
        for chr in chr_list:
            data = c.matrix(balance=False).fetch(chr)
            bin_size = c.binsize
            minimum_gap_between_length = bin_size*bins_min_gap_between
            min_gap_length = bin_size*bins_min_gap
            zero_counts = []
            proc_zero_counts = []
            gaps = []
            for n,row in enumerate(data):
                #calculate procent of zeroes in every column
                zero_count = len(row) - np.count_nonzero(row)
                proc_zero = zero_count/len(row)*100
                zero_counts.append(zero_count)
                proc_zero_counts.append(proc_zero)
                if proc_zero>=zero_proc_in_line:
                    gaps.append((chr, (n+1)*bin_size, (n+1)*bin_size+bin_size))
            # print(gaps)
            data = merge_gaps(gaps, minimum_gap_between_length, min_gap_length, output_gap_file)
            # convert to chr-dict
            chr_data[chr] = data
            logging.info("Found " + str(len(chr_data[chr])) + " gaps on chr " + chr)
        conc_data = pd.concat([chr_data[chr] for chr in chr_data.keys()])
        conc_data.to_csv(output_gap_file, sep="\t", header=False, index=False)
        return chr_data

#with open(gap_bed, "w") as out_file:
   # for gap in gaps:
    #    print(int(gap[2]) - int(gap[1]))
      #  if (int(gap[2]) - int(gap[1])) >= minimum_gap_length:
     #   #    print(gap)
       #     out_file.write(gap[0]+"\t"+str(gap[1])+"\t"+str(gap[2])+"\n")

#plt.hist(zero_counts)
#plt.show()
#plt.hist(proc_zero_counts)
#plt.show()


