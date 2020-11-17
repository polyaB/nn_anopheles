import logging,os, sys
source_path = os.path.dirname(os.path.abspath(sys.argv[0])) + "/../3Dpredictor/source"
source_path2 = os.path.dirname(os.path.abspath(sys.argv[0])) + "/../3Dpredictor/nn/source"
sys.path.append(source_path)
sys.path.append(source_path2)
import pandas as pd
from shared import FileReader

class BedReader(FileReader): #Class process files with different data, for example gap files or train and test
    def __init__(self,fname):
        self.data = None
        self.fname = fname

        # check duplicates, nested intervals, set mids, and split by chromosomes and sort

    def process_data(self, data):
        # check duplicates
        duplicated = data.duplicated(subset=["chr", "start", "end"])
        if sum(duplicated) > 0:
            logging.getLogger(__name__).warning(
                "Duplicates by genomic positions found in file " + self.fname)
        data.drop_duplicates(
            inplace=True)
        del duplicated

        # convert to chr-dict
        chr_data = dict([(chr, data[data["chr"] == chr]) \
                         for chr in pd.unique(data["chr"])])
        # sort
        sorted_data = {}
        for chr, data in chr_data.items():
            sorted_data[chr] = data.sort_values(by=["chr", "start"])
        del chr_data

        # # check for nested intervals
        # nested_intevals_count = 0
        # print_example = True
        # for data in sorted_data.values():
        #     data_shifted = data.shift()
        #     a = (data["start"][1:] - data_shifted["start"][1:] > 0)
        #     b = (data["end"][1:] - data_shifted["end"][1:] < 0)
        #     c = (a & b)
        #     nested = [False] + list(c.values)
        #
        #     # nested = [False] + (data["start"][1:] - data_shifted["start"][1:] > 0) & \
        #     #    (data["end"][1:] - data_shifted["end"][1:] < 0)
        #
        #     nested_intevals_count += sum(nested)
        #     if print_example and sum(nested) > 0:
        #         logging.getLogger(__name__).debug("Nested intervals found. Examples: ")
        #         logging.getLogger(__name__).debug(data[nested].head(1))
        #         print_example = False
        # if nested_intevals_count > 0:
        #     logging.getLogger(__name__).warning("Number of nested intervals: " + \
        #                                         str(nested_intevals_count) + " in " + \
        #                                         str(sum([len(i) for i in sorted_data.values()])))

        return sorted_data

    def read_file(self,
                  renamer = {"0":"chr","1":"start","2":"end"}): # store CTCF peaks as sorted pandas dataframe
        logging.getLogger(__name__).info(msg="Reading Bed file "+self.fname)
        # set random temporary labels
        if self.fname.endswith(".gz"):  # check gzipped files
            import gzip
            temp_file = gzip.open(self.fname)
        else:
            temp_file = open(self.fname)
        Nfields = len(temp_file.readline().strip().split())
        temp_file.close()
        names = list(map(str, list(range(Nfields))))
        data = pd.read_csv(self.fname, sep="\t", header=None, names=names, comment='#')

        # subset and rename
        data_fields = list(map(int,renamer.keys()))
        data = data.iloc[:,data_fields]
        data.rename(columns=renamer,
                        inplace=True)

        chr_data = self.process_data(data)
        del data
        #save
        self.chr_data = chr_data
