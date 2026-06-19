"""RHB dataset adapted to competition .mat inputs.

Mirrors Radar-APLANC/data/datasets.py::RHB but loads data_f from the
pre-computed complex_range_matrix instead of rf.pkl + organizer.
"""
import os
import sys
import numpy as np
import scipy.signal as sig
from torch.utils.data import Dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "baseline"))
from rf.proc import find_range
from rf.IQ_to_PhaseAngle import IQ_to_PhaseAngle

import mat_io

TRAIN_ROOT = os.path.join(os.path.dirname(__file__), "..", "RHB_train")


class RHB(Dataset):
    """Returns (pseudo, p_rf, n_rf, t_rf) for each trial, identical layout
    to the original RHB dataset so train.py logic is unchanged."""

    def __init__(self, datapaths, pseudo_root, frame_length_ppg=900,
                 sampling_ratio=4, window_size=5, samples=256,
                 samp_f=5e6, freq_slope=60.012e12, neg_mode="random",
                 guard=8, target="traditional"):
        # target: "traditional" -> stage1 loss target = fixed center bin [2].
        #         "pseudo"      -> stage2 loss target = Algorithm-1 pseudo.npy.
        # This single switch IS the native stage1/stage2 difference (the native
        # train.py never changed; the operator swapped t_rf for pseudo).
        self.target = target
        self.rf_file_list = datapaths
        self.sampling_ratio = sampling_ratio
        self.window_size = window_size
        self.samples = samples
        self.samp_f = samp_f
        self.freq_slope = freq_slope
        self.frame_length_ppg = frame_length_ppg
        # neg_mode: "random" (original) or "hard" (highest-energy non-target bin
        # outside a guard band so the heartbeat does not leak into the negative).
        self.neg_mode = neg_mode
        self.guard = guard

        # Load pseudo labels (generated after stage 1).
        self.Pseudo_list = []
        for folder in self.rf_file_list:
            fp = os.path.join(pseudo_root, folder, "pseudo.npy")
            if os.path.exists(fp):
                self.Pseudo_list.append(np.load(fp, allow_pickle=True)[0:900])
            else:
                raise FileNotFoundError(f"pseudo.npy not found for {folder}")

        self.rf_data_p_list = []
        self.rf_data_n_list = []
        self.traditional_list = []
        for rf_file in self.rf_file_list:
            data_f = mat_io.load_train_data_f(os.path.join(TRAIN_ROOT, rf_file))
            range_index = find_range(data_f, self.samp_f, self.freq_slope, self.samples)

            raw_data_p = data_f[:, range_index - self.window_size // 2:
                                   range_index + self.window_size // 2 + 1]
            raw_data_left = data_f[:, 0:range_index - self.window_size // 2 - 1]
            raw_data_right = data_f[:, range_index + self.window_size // 2 + 30:]
            Phase_data_p = IQ_to_PhaseAngle(raw_data_p, 0.8, 2.5, 120)
            Phase_data_p = sig.decimate(Phase_data_p, 4)
            if self.neg_mode == "hard":
                raw_data_n = self._hard_negative(data_f, range_index)
            else:
                if raw_data_left.shape[1] > raw_data_right.shape[1]:
                    n_start = np.random.choice(raw_data_left.shape[1] - self.window_size)
                    raw_data_n = raw_data_left[:, n_start:n_start + self.window_size]
                else:
                    n_start = np.random.choice(raw_data_right.shape[1] - self.window_size)
                    raw_data_n = raw_data_right[:, n_start:n_start + self.window_size]

            raw_data_p = np.array([np.real(raw_data_p), np.imag(raw_data_p)])
            raw_data_n = np.array([np.real(raw_data_n), np.imag(raw_data_n)])
            raw_data_p = np.transpose(raw_data_p, axes=(0, 2, 1))
            raw_data_n = np.transpose(raw_data_n, axes=(0, 2, 1))
            self.rf_data_p_list.append(raw_data_p)
            self.rf_data_n_list.append(raw_data_n)
            self.traditional_list.append(Phase_data_p)

    def _hard_negative(self, data_f, range_index):
        """Pick the highest-energy non-target window, outside a guard band.

        Rationale (#1): a purely random non-target bin is often weak noise, an
        unstable negative. The hardest *valid* negative is the strongest bin
        that is still clearly NOT the heartbeat bin. The guard band keeps it far
        enough from range_index that the heartbeat does not leak in.
        """
        w = self.window_size
        energy = np.sum(np.abs(data_f), axis=0)            # (256,) per-bin energy
        n_bins = energy.shape[0]
        best_e, best_c = -1.0, None
        for c in range(w // 2, n_bins - w // 2):
            if abs(c - range_index) <= self.guard:
                continue                                   # inside guard band -> skip
            e = energy[c - w // 2:c + w // 2 + 1].sum()
            if e > best_e:
                best_e, best_c = e, c
        if best_c is None:                                 # fallback: random right side
            best_c = min(range_index + self.guard + w, n_bins - w // 2 - 1)
        return data_f[:, best_c - w // 2:best_c + w // 2 + 1]

    def __len__(self):
        return len(self.rf_file_list)

    def __getitem__(self, idx):
        p_data_f = self.rf_data_p_list[idx]
        n_data_f = self.rf_data_n_list[idx]
        traditional_data_f = self.traditional_list[idx]
        some_data_p = p_data_f[:, :, 0:100]
        some_data_n = n_data_f[:, :, 0:100]
        some_data_t = traditional_data_f[:, 0:100]
        p_data_f = np.concatenate((p_data_f, some_data_p), axis=2)
        n_data_f = np.concatenate((n_data_f, some_data_n), axis=2)
        traditional_data_f = np.concatenate((traditional_data_f, some_data_t), axis=1)

        frame_start = 0
        sr, L = self.sampling_ratio, self.frame_length_ppg
        p_data_f = p_data_f[:, :, frame_start * sr:frame_start * sr + sr * L]
        n_data_f = n_data_f[:, :, frame_start * sr:frame_start * sr + sr * L]
        traditional_data_f = traditional_data_f[2, frame_start:frame_start + L]

        Pseudo = self.Pseudo_list[idx][frame_start:frame_start + L]
        assert len(Pseudo) == L, f"Expected {L}, got {len(Pseudo)}"
        # The trainer uses the 4th element as the loss target (t_rf). For stage2
        # we put the pseudo there so the contrast loss pulls toward Algorithm-1's
        # selected bin instead of the fixed center bin.
        t_target = Pseudo if self.target == "pseudo" else traditional_data_f
        return Pseudo, p_data_f, n_data_f, t_target
