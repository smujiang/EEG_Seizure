import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset



class BalancedSeizureDataset(nn.Module):
    def __init__(self, eeg_npy_file, window_size_points=6400, sample_stride=256):
        super(BalancedSeizureDataset, self).__init__()
        data_dict = np.load(eeg_npy_file, allow_pickle=True).item()
        
        self.eeg_clip = data_dict['sampled_clips']
        self.mask_clip = data_dict['mask_clips']
        self.fs = data_dict['fs']
        #Default: window_size_sec = 25, fs = 256, so:  window_size_points=6400 (25*256), sample_stride=256
        # but we can use smaller sampling stride and window size, but the model layer dimensions may need to be modifed.
        self.window_size = window_size_points

        eeg_clip_size = self.eeg_clip.shape[1] 
        self.window_idx = np.arange(0, eeg_clip_size-window_size_points-1, sample_stride).astype(int)

    def __len__(self):
        return int(len(self.window_idx))

    # if the mean of the mask is greater than threshold, return 1, else return 0
    def mask2label(self, m_clip, threshold=0.5):
        label = np.mean(m_clip) > threshold
        return label

    def __getitem__(self, idx):
        e_clip = self.eeg_clip[:, self.window_idx[idx]:self.window_idx[idx]+self.window_size]
        m_clip = self.mask_clip[self.window_idx[idx]:self.window_idx[idx]+self.window_size]
        return e_clip, m_clip

if __name__ == "__main__":
    from utils import plot_eeg

    data_dir = "/Users/jjiang10/Data/EEG/BIDS_Siena_pred"
    fn_list = os.listdir(data_dir)

    PLOT = True

    total = 0
    for fn in fn_list:
        if not fn.endswith(".npy"):
            continue
        eeg_npy_fn = os.path.join(data_dir, fn)
        eeg_dataset = BalancedSeizureDataset(eeg_npy_fn, window_size_points=6400, sample_stride=256)

        total += len(eeg_dataset)
        print(len(eeg_dataset))
        if PLOT:
            for eeg, mask in eeg_dataset:
                plot_eeg(eeg, mask)
                PLOT = False 
                break

    print(total)





