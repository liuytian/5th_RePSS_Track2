"""Data loading for radar heart-rate estimation.

Loaders for the competition .mat / .npy files, plus the training Dataset that
yields a positive heartbeat IQ window, a negative (noise) IQ window, and the
ground-truth PPG waveform used as the contrast-loss positive target.
"""
import os
import numpy as np
import scipy.io as sio
from torch.utils.data import Dataset

import config
from signal_proc import find_range


# --------------------------------------------------------------------------- #
# File loaders
# --------------------------------------------------------------------------- #
def load_train_data_f(folder):
    """Training complex range matrix -> (frames, 256) complex."""
    m = sio.loadmat(os.path.join(folder, "tx1_rx1_complex_range_matrix.mat"))
    crm = np.asarray(m["save_data"][0, 0]["complex_range_matrix"])   # (256, 3599)
    return crm.T.astype(np.complex128)


def load_test_data_f(seg_folder):
    """Test radar segment -> (frames, 256) complex."""
    m = sio.loadmat(os.path.join(seg_folder, "radar_segment.mat"))
    rd = np.asarray(m["segment_data"][0, 0]["radar_data"])           # (256, 1200)
    return rd.T.astype(np.complex128)


def load_gt_ppg(folder):
    """Ground-truth PPG-like waveform (variable length, fs=30)."""
    return np.load(os.path.join(folder, "vital_dict.npy"),
                   allow_pickle=True).astype(np.float64)


# --------------------------------------------------------------------------- #
# Training dataset
# --------------------------------------------------------------------------- #
class RHBDataset(Dataset):
    """Per trial: (positive IQ window, negative IQ window, GT PPG).

    Positive = the `window_size` range bins around find_range (heartbeat).
    Negative = a random window far from the heartbeat bin (noise contrast).
    The GT PPG is the contrast-loss positive target (replaces the baseline's
    pseudo / traditional-phase target).
    """

    def __init__(self, datapaths, frame_length_ppg=900, sampling_ratio=4,
                 window_size=5, samples=config.ADC_SAMPLES,
                 samp_f=config.SAMP_F, freq_slope=config.FREQ_SLOPE):
        self.files = datapaths
        self.sampling_ratio = sampling_ratio
        self.window_size = window_size
        self.frame_length_ppg = frame_length_ppg

        self.rf_p, self.rf_n, self.gt = [], [], []
        for f in datapaths:
            folder = os.path.join(config.TRAIN_ROOT, f)
            data_f = load_train_data_f(folder)
            ri = find_range(data_f, samp_f, freq_slope, samples)
            w = window_size // 2

            raw_p = data_f[:, ri - w: ri + w + 1]
            left = data_f[:, 0: ri - w - 1]
            right = data_f[:, ri + w + 30:]
            if left.shape[1] > right.shape[1]:
                s = np.random.choice(left.shape[1] - window_size)
                raw_n = left[:, s:s + window_size]
            else:
                s = np.random.choice(right.shape[1] - window_size)
                raw_n = right[:, s:s + window_size]

            self.rf_p.append(np.transpose(np.array([raw_p.real, raw_p.imag]), (0, 2, 1)))
            self.rf_n.append(np.transpose(np.array([raw_n.real, raw_n.imag]), (0, 2, 1)))
            self.gt.append(load_gt_ppg(folder)[:frame_length_ppg].astype(np.float32))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        p, n = self.rf_p[i], self.rf_n[i]
        sr, L = self.sampling_ratio, self.frame_length_ppg
        # circular pad then crop to sr*L frames (matches baseline windowing)
        p = np.concatenate((p, p[:, :, :100]), axis=2)[:, :, : sr * L]
        n = np.concatenate((n, n[:, :, :100]), axis=2)[:, :, : sr * L]
        return p.astype(np.float32), n.astype(np.float32), self.gt[i]
