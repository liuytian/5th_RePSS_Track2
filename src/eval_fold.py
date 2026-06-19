"""Evaluate a trained Radar-APLANC model on a fold's val/test splits.

These splits are carved from the training set and DO have ground truth
(vital_dict.npy), so MAE / RMSE / Pearson R can be computed locally.
This mirrors Radar-APLANC/utils/eval.py::eval_RHB but loads data_f from the
competition .mat files and reports structured metrics to metrics.json.

NOTE: the competition test set (track2_testdata) has no ground truth, so its
true error cannot be computed locally — only the leaderboard can score it.
"""
import os
import sys
import json
import argparse
import pickle
import numpy as np
import torch
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "baseline"))
from utils_sig import normalize
from rf.model import RF_conv_decoder
from rf.proc import find_range
from utils.errors import getErrors
from utils.utils import pulse_rate_from_power_spectral_density as prpsd

import mat_io

TRAIN_ROOT = os.path.join(os.path.dirname(__file__), "..", "RHB_train")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict_ppg(model, data_f, sequence_length=900, sampling_ratio=4,
                rf_window_size=5, samp_f=5e6, freq_slope=60.012e12, adc_samples=256):
    """Replicates eval_RHB windowing: pad to >=3600 frames, run model, take 900."""
    range_index = find_range(data_f, samp_f, freq_slope, adc_samples)
    temp_window = np.blackman(rf_window_size)
    raw = data_f[:, range_index - len(temp_window) // 2:
                    range_index + len(temp_window) // 2 + 1]      # (T,5)
    circ = raw[0:800]
    raw = np.concatenate((raw, circ))                            # extend
    raw = np.array([np.real(raw), np.imag(raw)])                 # (2,T',5)
    raw = np.transpose(raw, axes=(0, 2, 1))                      # (2,5,T')
    rf_data = np.transpose(raw, axes=(2, 0, 1))                  # (T',2,5)

    need = sequence_length * sampling_ratio                      # 3600
    cur_est = None
    cat = None
    for n in range(rf_data.shape[0]):
        cur = torch.tensor(rf_data[n]).type(torch.float32)
        cur = normalize(cur).unsqueeze(0).to(DEVICE)
        if n % need == 0:
            cat = cur
        else:
            cat = torch.cat((cat, cur), 0)
        if cat.shape[0] == need:
            with torch.no_grad():
                c = cat.unsqueeze(0)
                c = torch.transpose(c, 1, 2)
                c = torch.transpose(c, 2, 3)
                iq = torch.reshape(c, (c.shape[0], -1, c.shape[3]))
                est, _ = model(iq)
                est = est.squeeze().cpu().numpy()
            cur_est = est if cur_est is None else np.concatenate((cur_est, est), -1)
    return cur_est[0:900]


def hr_windows(x, win=300, stride=300):
    x = (x - np.mean(x)) / np.std(x)
    hrs = []
    for s in range(0, len(x) - win, stride):
        w = x[s:s + win]
        w = (w - np.mean(w)) / np.std(w)
        hrs.append(prpsd(w, 30, 45, 150, BUTTER_ORDER=6, DETREND=False))
    return np.array(hrs)


def evaluate(model, files):
    mae_list, rmse_list = [], []
    all_est, all_gt = [], []
    for f in files:
        data_f = mat_io.load_train_data_f(os.path.join(TRAIN_ROOT, f))
        gt = mat_io.load_gt_ppg(os.path.join(TRAIN_ROOT, f))[0:900]
        est = predict_ppg(model, data_f)

        hr_est = hr_windows(est)
        hr_gt = hr_windows(gt)
        m = min(len(hr_est), len(hr_gt))
        hr_est, hr_gt = hr_est[:m], hr_gt[:m]

        RMSE, MAE, _, _ = getErrors(np.array([hr_est]), hr_gt)
        mae_list.append(float(MAE[0]))
        rmse_list.append(float(RMSE[0]))
        all_est.extend(hr_est.tolist())
        all_gt.extend(hr_gt.tolist())

    est = np.array(all_est)
    gt = np.array(all_gt)
    r = float(stats.pearsonr(gt, est)[0]) if len(est) > 1 else float("nan")
    return {
        "num_trials": len(files),
        "num_windows": len(all_est),
        "mae_mean": float(np.mean(mae_list)),
        "rmse_mean": float(np.mean(rmse_list)),
        "pearson_r": r,
        "mae_per_trial": dict(zip(files, [round(x, 3) for x in mae_list])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--folds-path", required=True)
    ap.add_argument("--out", default="metrics.json")
    ap.add_argument("--splits", nargs="+", default=["val_files", "test_files"])
    args = ap.parse_args()

    model = RF_conv_decoder().to(DEVICE)
    model.load_state_dict(torch.load(args.ckpt, map_location=DEVICE)["model_P_state_dict"])
    model.eval()

    with open(args.folds_path, "rb") as f:
        fold = pickle.load(f)

    results = {"ckpt": args.ckpt, "folds_path": args.folds_path}
    for split in args.splits:
        files = fold[split]
        print(f"Evaluating {split}: {len(files)} trials ...")
        res = evaluate(model, files)
        results[split] = res
        print(f"  {split}: MAE={res['mae_mean']:.3f}  "
              f"RMSE={res['rmse_mean']:.3f}  R={res['pearson_r']:.3f}")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved metrics -> {args.out}")


if __name__ == "__main__":
    main()
