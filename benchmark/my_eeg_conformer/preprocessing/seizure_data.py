import os
from epilepsy2bids.annotations import Annotations
from epilepsy2bids.eeg import Eeg
import numpy as np
import torch
import torch.nn as nn
from scipy.signal import butter, lfilter, iirnotch, filtfilt
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models


class SeizureData:
    def __init__(self, edf_file, label_file, window_size_sec=25, fs=256):
        super(SeizureData, self).__init__()
        self.edf_file = edf_file  # carry the edf file name with the entity
        self.label_file = label_file  # carry the label file name with the entity

        self.eeg = Eeg.loadEdfAutoDetectMontage(edfFile=edf_file)
        if self.eeg.montage is not Eeg.Montage.UNIPOLAR:
            raise Exception("Error: Only unipolar montages are supported.")
        if not os.path.exists(label_file):
            raise Exception("Error: Label file does not exist.")
        self.annotations = Annotations.loadTsv(label_file)
        self.mask = self.annotations.getMask(self.eeg.fs)

        # standardize
        data_mean = np.mean(self.eeg.data)
        data_std = np.std(self.eeg.data)
        self.data = (self.eeg.data - data_mean) / data_std

        self.channels = self.eeg.channels

        self.window_size = int(window_size_sec * fs)
        self.fs = fs
        self.recording_duration = int(self.data.shape[1] / fs)

        # params for preprocessing
        self.lowcut = 0.5
        self.highcut = 120
        notch_1_b, notch_1_a = iirnotch(1, Q=30, fs=fs)
        notch_60_b, notch_60_a = iirnotch(60, Q=30, fs=fs)
        self.notch_1_b = notch_1_b
        self.notch_1_a = notch_1_a
        self.notch_60_b = notch_60_b
        self.notch_60_a = notch_60_a

        window_idx = np.arange(0, self.recording_duration * self.fs, self.fs).astype(
            int
        )
        self.window_idx = window_idx[
            window_idx < self.recording_duration * self.fs - self.window_size
        ]

    def __len__(self):
        return len(self.window_idx)

    # if the mean of the mask is greater than threshold, return 1, else return 0
    def mask2label(self, mask_clip, threshold=0.5):
        label = np.mean(mask_clip) > threshold
        return label

    def butter_bandpass_filter(self, data, order=3):
        nyq = 0.5 * self.fs
        low = self.lowcut / nyq
        high = self.highcut / nyq
        b, a = butter(order, [low, high], btype="band")
        y = lfilter(b, a, data)
        return y

    def preprocess_clip(self, eeg_clip):
        bandpass_filtered_signal = self.butter_bandpass_filter(eeg_clip, order=3)
        filtered_1_signal = lfilter(
            self.notch_1_b, self.notch_1_a, bandpass_filtered_signal
        )
        filtered_60_signal = lfilter(
            self.notch_60_b, self.notch_60_a, filtered_1_signal
        )
        eeg_clip = filtered_60_signal
        return eeg_clip

    def get_a_clip(self, idx):
        eeg_clip = self.data[
            :, self.window_idx[idx] : self.window_idx[idx] + self.window_size
        ]
        mask_clip = self.mask[
            self.window_idx[idx] : self.window_idx[idx] + self.window_size
        ]
        return eeg_clip, mask_clip


if __name__ == "__main__":
    data_root = "/data/jjiang10/Data/EEG/BIDS_Siena"

    train_cases = [
        "sub-00",
        "sub-01",
        "sub-03",
        "sub-05",
        "sub-06",
        "sub-07",
        "sub-09",
        "sub-10",
        "sub-11",
        "sub-12",
    ]
    test_cases = ["sub-13", "sub-14", "sub-16", "sub-17"]

    for case in test_cases:
        data_dir = os.path.join(data_root, case)
        if not os.path.exists(data_dir):
            raise Exception("Error: Data directory does not exist.")
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith(".edf"):
                    print(file)
                    edf_file = os.path.join(root, file)
                    label_file = edf_file.replace("eeg.edf", "events.tsv")
                    eeg_dataset = SeizureData(edf_file, label_file)
                    print(len(eeg_dataset))
                    print(eeg_dataset[0][0].shape)  # torch.Size([19, 6400])

    print("Done")
