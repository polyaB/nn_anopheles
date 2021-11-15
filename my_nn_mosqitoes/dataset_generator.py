from itertools import combinations
import tensorflow as tf
import numpy as np
import pandas as pd
from typing import Union
import json
import random
from generate_data import generate_pair_index_table
from nn_blocks import from_upper_triu
import matplotlib.pyplot as plt

class DatasetGen(tf.keras.utils.Sequence):
    def __init__(self, input_data_file: str, target_data_file: str,
                 data_stats_file: str,
                 apply_augmentation: bool,
                 seq_data: pd.DataFrame, 
                 mode : str,
                 batch_size: int, 
                 batches_per_epoch: Union[int, None] = None,
                 draw_target=False):
        
        with open(data_stats_file) as data_stats_open:
            data_stats = json.load(data_stats_open)
            self.seq_length = data_stats['seq_length']
            self.target_length = data_stats['target_length']
            self.diagonal_offset = data_stats["diagonal_offset"]
            self.target_length1 = data_stats['seq_length'] // data_stats['pool_width']
            self.target_crop = data_stats['crop_bp'] // data_stats['pool_width']
            self.target_length1_cropped = self.target_length1 - 2 * self.target_crop
        self.input_data = np.memmap(input_data_file, dtype='float16', mode='r', shape=(len(seq_data), self.seq_length, 4))
        self.target_data = np.memmap(target_data_file, dtype='float16', mode='r', shape=(len(seq_data), self.target_length))
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch
        self.apply_augmentation = apply_augmentation
        self.draw_target = draw_target
        self.combinations = generate_pair_index_table(seq_data, mode)
        random.shuffle(self.combinations)
        self.n_samples = len(self.combinations)
        self.epoch_number = 0
        self.count = 0
    
    def __len__(self):
        if self.batches_per_epoch is None:
            return int(np.ceil(self.n_samples / float(self.batch_size)))
        return self.batches_per_epoch

    def __getitem__(self, idx):
        #Choose random samples from
        batch_input_left = np.zeros(
            shape=(self.batch_size, self.input_data.shape[1], self.input_data.shape[2]),
            dtype='float16'
        )
        batch_input_right = np.zeros(
            shape=(self.batch_size, self.input_data.shape[1], self.input_data.shape[2]),
            dtype='float16'
        )
        batch_target = np.zeros(
            (self.batch_size, self.target_data.shape[1]),
            dtype=np.float32
        )
        if self.batches_per_epoch is None:
            batch_start = idx * self.batch_size
            batch_end = min(len(self.combinations), batch_start + self.batch_size)
            # print("batch start", batch_start, "batch end", batch_end)
            for sample_idx in range(batch_end - batch_start):
                left_key, right_key = self.combinations[sample_idx + batch_start]
                batch_input_left[sample_idx] = self.input_data[left_key]
                left_target = self.target_data[left_key]
                batch_input_right[sample_idx] = self.input_data[right_key]
                right_target = self.target_data[right_key]
                batch_target[sample_idx] = left_target-right_target
            #What should to do with last incomplete batch
            n_pad = self.batch_size - (batch_end - batch_start)
            if n_pad > 0:
                for sample_idx in range(batch_end - batch_start, self.batch_size):
                    batch_input_left[sample_idx] = batch_input_left[sample_idx - 1]
                    batch_input_right[sample_idx] = batch_input_right[sample_idx - 1]
                    batch_target[sample_idx] = batch_target[sample_idx - 1]
        else:
            for sample_idx in range(self.batch_size):
                left_key, right_key = random.choice(self.combinations)
                if self.apply_augmentation:
                   raise NotImplementedError
                else:
                    batch_input_left[sample_idx] = self.input_data[left_key]
                    left_target = self.target_data[left_key]
                    batch_input_right[sample_idx] = self.input_data[right_key]
                    right_target = self.target_data[right_key]
                    batch_target[sample_idx] = left_target-right_target
                if self.draw_target:
                    left_mat = from_upper_triu(left_target, self.target_length1_cropped, self.diagonal_offset)
                    print(left_mat)
                    right_mat = from_upper_triu(right_target, self.target_length1_cropped, self.diagonal_offset)
                    print(right_mat)
                    mat = from_upper_triu(left_target-right_target, self.target_length1_cropped, self.diagonal_offset)
                    #plot left target
                    plt.subplot(131)
                    im = plt.matshow(left_mat, fignum=False, cmap='RdBu_r')  # , vmax=vmax, vmin=vmin)
                    plt.colorbar(im, fraction=.04, pad=0.05)  # , ticks=[-2, -1, 0, 1, 2])
                    plt.title('left_mat')
                    # plot right target
                    plt.subplot(132)
                    im = plt.matshow(right_mat, fignum=False, cmap='RdBu_r')  # , vmax=vmax, vmin=vmin)
                    plt.colorbar(im, fraction=.04, pad=0.05)  # , ticks=[-2, -1, 0, 1, 2])
                    plt.title('right_mat')
                    plt.subplot(133)
                    im = plt.matshow(mat, fignum=False, cmap='RdBu_r')  # , vmax=vmax, vmin=vmin)
                    plt.colorbar(im, fraction=.04, pad=0.05)  # , ticks=[-2, -1, 0, 1, 2])
                    plt.title('left-right_mat')
                    # plt.suptitle("epoch " + str(i))
                    plt.savefig("/mnt/scratch/ws/psbelokopytova/202112281307data_Polya/nn_anopheles/dataset_like_Akita/data/test_new/"+str(sample_idx)+".png")
                    plt.clf()
        batch_input = [batch_input_left, batch_input_right]
        del batch_input_left, batch_input_right
        return batch_input, batch_target, None
            # if per_epoch:
            #     self.count += 1
            #     if self.count > self.N_epoch:
            #         self.count = 0
            #         self.epoch_number += 1
            #         if self.epoch_number >= max_number_epoches:
            #             self.epoch_number = 0
            #             # shuffle????
            #     idx = idx + self.epoch_number
    def on_epoch_end(self):
        """Method called at the end of every epoch.
        """
        random.shuffle(combinations)

