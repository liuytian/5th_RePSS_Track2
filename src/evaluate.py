"""Held-out evaluation: per-trial HR MAE / RMSE / Pearson R.

  python evaluate.py --ckpt ckpt_seed11/best.pth \
      --folds-path folds/fold_heldtrial.pkl --splits val_files
"""
import os
import argparse
import pickle
import numpy as np
import torch
from scipy import stats

import config
from models import RF_conv_decoder
from signal_proc import find_range, normalize, hr_from_psd
from metrics import getErrors
from data import load_train_data_f, load_gt_ppg

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict_ppg(model, data_f, sequence_length=900, sampling_ratio=4, rf_window_size=5):
    """Pad to >=3600 frames, run the model, return a 900-pt PPG."""
    ri = find_range(data_f, config.SAMP_F, config.FREQ_SLOPE, config.ADC_SAMPLES)
    w = rf_window_size // 2
    raw = data_f[:, ri - w: ri + w + 1]                          # (T,5)
    raw = np.concatenate((raw, raw[0:800]))
    raw = np.transpose(np.array([raw.real, raw.imag]), axes=(0, 2, 1))   # (2,5,T')
    rf = np.transpose(raw, axes=(2, 0, 1))                       # (T',2,5)

    need = sequence_length * sampling_ratio                      # 3600
    est_all, cat = None, None
    for n in range(rf.shape[0]):
        cur = normalize(torch.tensor(rf[n]).type(torch.float32)).unsqueeze(0).to(DEVICE)
        cat = cur if n % need == 0 else torch.cat((cat, cur), 0)
        if cat.shape[0] == need:
            with torch.no_grad():
                c = torch.transpose(torch.transpose(cat.unsqueeze(0), 1, 2), 2, 3)
                iq = torch.reshape(c, (c.shape[0], -1, c.shape[3]))
                est = model(iq)[0].squeeze().cpu().numpy()
            est_all = est if est_all is None else np.concatenate((est_all, est), -1)
    return est_all[:900]


def hr_windows(x, win=300, stride=300):
    hrs = []
    for s in range(0, len(x) - win, stride):
        w = x[s:s + win]
        w = (w - w.mean()) / w.std()
        hrs.append(hr_from_psd(w, config.PPG_FS, config.HR_LO, config.HR_HI))
    return np.array(hrs)


def evaluate(model, files):
    maes, rmses, all_est, all_gt = [], [], [], []
    for f in files:
        folder = os.path.join(config.TRAIN_ROOT, f)
        est = predict_ppg(model, load_train_data_f(folder))
        gt = load_gt_ppg(folder)[:900]
        he, hg = hr_windows(est), hr_windows(gt)
        k = min(len(he), len(hg)); he, hg = he[:k], hg[:k]
        RMSE, MAE, _, _ = getErrors(np.array([he]), hg)
        maes.append(float(MAE[0])); rmses.append(float(RMSE[0]))
        all_est.extend(he.tolist()); all_gt.extend(hg.tolist())
    r = float(stats.pearsonr(all_gt, all_est)[0]) if len(all_est) > 1 else float("nan")
    return {"num_trials": len(files), "mae_mean": float(np.mean(maes)),
            "rmse_mean": float(np.mean(rmses)), "pearson_r": r}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--folds-path", required=True)
    ap.add_argument("--splits", nargs="+", default=["val_files"])
    args = ap.parse_args()

    model = RF_conv_decoder().to(DEVICE)
    model.load_state_dict(torch.load(args.ckpt, map_location=DEVICE)["model_P_state_dict"])
    model.eval()
    fold = pickle.load(open(args.folds_path, "rb"))
    for split in args.splits:
        res = evaluate(model, fold[split])
        print(f"{split}: MAE={res['mae_mean']:.3f} RMSE={res['rmse_mean']:.3f} R={res['pearson_r']:.3f}")


if __name__ == "__main__":
    main()
