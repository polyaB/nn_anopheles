import logging,os
import pandas as pd
import numpy as np
from shared import FileReader

class GapReader(FileReader): #Class process files with ChipSeq peaks
    def __init__(self,fname):
        self.data = None
        self.fname = fname
    def read_file(self): # store CTCF peaks as sorted pandas dataframe
        logging.getLogger(__name__).info(msg="Reading Gap file "+self.fname)
        data = pd.read_csv(self.fname, header=None, names = ["chr", "start", "end"], sep="\t")
        self.data = data


