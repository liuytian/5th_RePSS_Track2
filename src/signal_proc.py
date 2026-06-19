"""Signal-processing utilities for FMCW-radar vital-sign estimation.

Range localisation, IQ augmentation/normalisation, phase extraction, and
heart-rate readout from a PPG-like waveform's power spectral density.
"""
import numpy as np
import torch
from scipy import signal
from scipy.signal import butter, filtfilt


# --------------------------------------------------------------------------- #
# Range localisation
# --------------------------------------------------------------------------- #
def find_range(data_f, samp_f, freq_slope, samples,
               max_range_allowed=1.0, min_idx=5):
    """Index of the strongest range bin (L1 energy over slow time).

    Args:
        data_f: range slow-time matrix, shape (frames, range_bins), complex.
        samp_f, freq_slope, samples: FMCW radar params (ADC rate, chirp slope,
            ADC samples) used to bound the searched range.
        min_idx: skip the first few bins (hardware enclosure reflections).
    """
    max_idx = max_range_allowed / (samp_f * 2.98e8 / freq_slope / 2 / samples)
    d = np.abs(data_f[:, min_idx:int(max_idx)])
    d = np.sum(d, axis=0)
    return int(np.argmax(d)) + min_idx


# --------------------------------------------------------------------------- #
# IQ normalisation / augmentation
# --------------------------------------------------------------------------- #
def normalize(x):
    """Zero-mean unit-variance normalisation."""
    return (x - x.mean()) / x.std()


def rotateIQ(iq_array):
    """Random global IQ phase rotation (identity-preserving augmentation).

    iq_array: (B, 2, bins, T) tensor; rotates the (I, Q) pair by a random angle.
    """
    theta = 2 * np.pi * np.random.rand()
    rot = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]])
    rot = torch.tensor(rot).type(torch.float32)
    for i in range(iq_array.shape[2]):
        iq_array[:, :, i, :] = torch.matmul(rot, iq_array[:, :, i, :])
    return iq_array


# --------------------------------------------------------------------------- #
# Phase extraction
# --------------------------------------------------------------------------- #
def butter_bandpass(sig, lowcut, highcut, fs, order=2):
    sig = np.reshape(sig, -1)
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, sig)


def IQ_to_PhaseAngle(IQ_data, lowcut, highcut, fs):
    """Per-bin unwrapped, band-passed phase of a complex range window."""
    out = []
    ang = np.angle(IQ_data)
    for i in range(ang.shape[1]):
        ph = np.unwrap(ang[:, i])
        ph = butter_bandpass(ph, lowcut=lowcut, highcut=highcut, fs=fs)
        out.append(ph)
    return np.array(out)


# --------------------------------------------------------------------------- #
# Heart-rate readout
# --------------------------------------------------------------------------- #
def hr_from_psd(pleth_sig, fs, lo_bpm, up_bpm, butter_order=6, fres_bpm=0.1):
    """Pulse rate (bpm) from the PSD peak of a PPG-like signal.

    Butterworth band-pass [lo,up] bpm, then periodogram peak in band.
    (after McDuff & Blackford, 2019)
    """
    N = (60 * fs) / fres_bpm
    if butter_order:
        b, a = signal.butter(butter_order, [lo_bpm / 60, up_bpm / 60],
                             btype='bandpass', fs=fs)
        pleth_sig = signal.filtfilt(b, a, np.double(pleth_sig))
    F, Pxx = signal.periodogram(x=pleth_sig, nfft=N, fs=fs)
    mask = (F >= lo_bpm / 60) & (F <= up_bpm / 60)
    return float(F[np.argmax(Pxx * mask)] * 60)
