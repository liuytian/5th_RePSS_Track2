"""Adapter to load competition .mat files into the data_f format the
Radar-APLANC pipeline expects.

The baseline computes:
    frames = organizer.organize()          # raw ADC -> frames
    data_f = create_fast_slow_matrix(...)  # -> (num_frames, num_samples=256), complex

The competition instead ships the *already computed* range matrix:
    train: tx1_rx1_complex_range_matrix.mat -> save_data.complex_range_matrix (256, 3599)
    test : radar_segment.mat                -> segment_data.radar_data        (256, 1200)

Both are (range_bins, time_frames). data_f = matrix.T  -> (time_frames, range_bins).
"""
import os
import numpy as np
import scipy.io as sio


def load_train_data_f(folder):
    """Load training complex_range_matrix -> data_f (num_frames, 256) complex."""
    path = os.path.join(folder, "tx1_rx1_complex_range_matrix.mat")
    m = sio.loadmat(path)
    sd = m["save_data"][0, 0]
    crm = np.asarray(sd["complex_range_matrix"])  # (256, 3599)
    data_f = crm.T.astype(np.complex128)          # (3599, 256)
    return data_f


def load_test_data_f(seg_folder):
    """Load a test radar_segment.mat -> data_f (num_frames, 256) complex."""
    path = os.path.join(seg_folder, "radar_segment.mat")
    m = sio.loadmat(path)
    sd = m["segment_data"][0, 0]
    rd = np.asarray(sd["radar_data"])             # (256, 1200)
    data_f = rd.T.astype(np.complex128)           # (1200, 256)
    return data_f


def load_gt_ppg(folder):
    """Load training ground-truth PPG-like waveform (variable length, fs=30)."""
    path = os.path.join(folder, "vital_dict.npy")
    return np.load(path, allow_pickle=True).astype(np.float64)
