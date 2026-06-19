"""Inference -> submission CSV.

  python inference.py --ckpts ckpt_seed11/best.pth ckpt_seed22/best.pth ... \
      --out submission_ensemble.csv --q-fallback 0.10
"""
import os
import csv
import glob
import argparse
import numpy as np
import torch
from scipy import signal as SIG

import config
from models import RF_conv_decoder
from signal_proc import find_range, normalize

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEARCH = 3


def pred_at_bin(model, data_f, cb, need=3600):
    """Run the model on the 5-bin window centred at `cb` -> 300-pt PPG."""
    raw = data_f[:, cb - 2:cb + 3]
    if raw.shape[0] < need:
        raw = np.concatenate([raw] * int(np.ceil(need / raw.shape[0])), axis=0)
    raw = raw[:need]
    rf = np.transpose(np.transpose(np.array([raw.real, raw.imag]), (0, 2, 1)), (2, 0, 1))
    cat = None
    for n in range(rf.shape[0]):
        c = normalize(torch.tensor(np.ascontiguousarray(rf[n])).type(torch.float32)).unsqueeze(0).to(DEVICE)
        cat = c if cat is None else torch.cat((cat, c), 0)
    with torch.no_grad():
        cc = torch.transpose(torch.transpose(cat.unsqueeze(0), 1, 2), 2, 3)
        iq = torch.reshape(cc, (cc.shape[0], -1, cc.shape[3]))
        est = model(iq)[0].squeeze().cpu().numpy()
    return est[:300]


def _band_psd(x):
    x = (x - x.mean()) / (x.std() + 1e-8)
    f, P = SIG.periodogram(x, fs=config.PPG_FS, nfft=1800)
    m = (f >= config.HR_LO / 60) & (f <= config.HR_HI / 60)
    return f[m] * 60, P[m]


def quality(x):
    _, P = _band_psd(x)
    return float(P.max() / (P.sum() + 1e-9))     # spectral concentration


def hr_of(x):
    fb, P = _band_psd(x)
    return float(fb[np.argmax(P)])


def seg_hr_binsearch(model, data_f, search=SEARCH):
    """Quality bin-search: pick the find_range +-search bin with sharpest PSD."""
    ri = find_range(data_f, config.SAMP_F, config.FREQ_SLOPE, config.ADC_SAMPLES)
    best_q, best_x = -1.0, None
    for cb in range(max(ri - search, 3), min(ri + search + 1, data_f.shape[1] - 3)):
        x = pred_at_bin(model, data_f, cb)
        q = quality(x)
        if q > best_q:
            best_q, best_x = q, x
    return hr_of(best_x), best_q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--search", type=int, default=SEARCH)
    ap.add_argument("--q-fallback", type=float, default=0.10)
    ap.add_argument("--reduce", choices=["median", "mean"], default="median")
    args = ap.parse_args()

    from data import load_test_data_f
    models = []
    for ck in args.ckpts:
        m = RF_conv_decoder().to(DEVICE)
        m.load_state_dict(torch.load(ck, map_location=DEVICE)["model_P_state_dict"])
        m.eval(); models.append(m)
    print(f"loaded {len(models)} model(s)")

    subjects = sorted([d for d in glob.glob(os.path.join(config.TEST_ROOT, "*")) if os.path.isdir(d)],
                      key=lambda p: tuple(int(x) for x in os.path.basename(p).split("_")))
    rows = [("ids", "HR")]
    n_fixed = 0
    for sd in subjects:
        sid = os.path.basename(sd)
        segs = sorted([d for d in glob.glob(os.path.join(sd, "*")) if os.path.isdir(d)],
                      key=lambda p: int(os.path.basename(p)))
        hrs, qs, names = [], [], []
        for seg in segs:
            df = load_test_data_f(seg)
            seed_hr, seed_q = [], []
            for m in models:
                hr, q = seg_hr_binsearch(m, df, args.search)
                seed_hr.append(hr); seed_q.append(q)
            hr_e = np.median(seed_hr) if args.reduce == "median" else np.mean(seed_hr)
            hrs.append(float(hr_e)); qs.append(float(np.max(seed_q)))
            names.append(os.path.basename(seg))
        hrs, qs = np.array(hrs), np.array(qs)

        # same-subject quality fallback
        if args.q_fallback > 0:
            good = qs >= args.q_fallback
            if 0 < good.sum() < len(hrs):
                ref = np.average(hrs[good], weights=qs[good])
                for i in range(len(hrs)):
                    if qs[i] < args.q_fallback:
                        hrs[i] = ref; n_fixed += 1
        for nm, hr in zip(names, hrs):
            rows.append((f"{sid}/{nm}", round(float(hr), 1)))
        print(f"{sid}: {len(segs)} segments")

    csv.writer(open(args.out, "w", newline="")).writerows(rows)
    print(f"wrote {len(rows)-1} predictions -> {args.out}  (fallback fixed {n_fixed})")


if __name__ == "__main__":
    main()
