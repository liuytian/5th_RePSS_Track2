"""Test-time bin search: find_range picks the wrong bin (held: 3/25 == oracle;
MAE 3.92 vs oracle 0.91). Pick the bin (within +/-search of find_range) whose
predicted PPG has the SHARPEST band-limited PSD peak (spectral concentration) --
an unsupervised quality proxy that recovered ~70% of the oracle gain (3.92->1.80)
with no GT and no retraining.

Generates a submission for the 10s test segments using this bin selector.
"""
import os
import sys
import csv
import glob
import argparse
import numpy as np
import torch
from scipy import signal as SIG

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "baseline"))
from utils_sig import normalize
from rf.model import RF_conv_decoder
from rf.proc import find_range
from utils.utils import pulse_rate_from_power_spectral_density as prpsd

import mat_io

TEST_ROOT = os.path.join(os.path.dirname(__file__), "..", "track2_testdata")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pred_at_bin(model, data_f, cb, need=3600):
    raw = data_f[:, cb - 2:cb + 3]
    # circular-pad up to >= need frames (test seg is 1200)
    if raw.shape[0] < need:
        reps = int(np.ceil(need / raw.shape[0]))
        raw = np.concatenate([raw] * reps, axis=0)
    raw = raw[:need]
    raw = np.array([np.real(raw), np.imag(raw)])
    raw = np.transpose(raw, (0, 2, 1))
    rf = np.transpose(raw, (2, 0, 1))
    cat = None
    for n in range(rf.shape[0]):
        c = normalize(torch.tensor(np.ascontiguousarray(rf[n])).type(torch.float32)).unsqueeze(0).to(DEVICE)
        cat = c if cat is None else torch.cat((cat, c), 0)
    with torch.no_grad():
        cc = cat.unsqueeze(0); cc = torch.transpose(cc, 1, 2); cc = torch.transpose(cc, 2, 3)
        iq = torch.reshape(cc, (cc.shape[0], -1, cc.shape[3]))
        e, _ = model(iq); e = e.squeeze().cpu().numpy()
    return e[:300]   # 10s -> 300pt


def quality(x):
    x = (x - x.mean()) / (x.std() + 1e-8)
    f, P = SIG.periodogram(x, fs=30, nfft=1800)
    mask = (f >= 45 / 60) & (f <= 150 / 60)
    P = P[mask]
    return P.max() / (P.sum() + 1e-9)


def hr_of(x):
    # periodogram peak (cleaner than the Butterworth-based prpsd on weak segments;
    # held RMSE 5.91 -> 4.52 just from this extractor swap). GT side still uses
    # the fixed reference; this only changes how we read HR off the PREDICTION.
    x = (x - x.mean()) / (x.std() + 1e-8)
    f, P = SIG.periodogram(x, fs=30, nfft=1800)
    mask = (f >= 45 / 60) & (f <= 150 / 60)
    return f[mask][np.argmax(P[mask])] * 60


def seg_hr_binsearch(model, data_f, search=3, center_only=False):
    ri = find_range(data_f, 5e6, 60.012e12, 256)
    if center_only:
        # v1: skip the quality bin search entirely -- use the find_range center
        # bin directly. (held 10s: center RMSE 3.51 vs quality-search 6.12.)
        # Still return the center bin's quality so same-subject fallback works.
        x = pred_at_bin(model, data_f, ri)
        return hr_of(x), quality(x)
    cand = range(max(ri - search, 3), min(ri + search + 1, data_f.shape[1] - 3))
    best_q, best_x = -1, None
    for cb in cand:
        x = pred_at_bin(model, data_f, cb)
        q = quality(x)
        if q > best_q:
            best_q, best_x = q, x
    return hr_of(best_x), best_q   # also return quality for same-subject fallback


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--search", type=int, default=3)
    ap.add_argument("--q-fallback", type=float, default=0.0,
                    help="If >0, replace segments with quality<thr by the quality-"
                         "weighted mean of the SAME SUBJECT's good segments.")
    ap.add_argument("--soft-q0", type=float, default=0.0,
                    help="If >0, SOFT fusion: hr = w*self + (1-w)*ref, "
                         "w=sigmoid((q-q0)/tau), ref=qw-mean of good segs (q>=q-ref-good).")
    ap.add_argument("--soft-tau", type=float, default=0.015)
    ap.add_argument("--soft-ref-good", type=float, default=0.08,
                    help="quality threshold defining 'good' segs that form the reference.")
    ap.add_argument("--center-only", action="store_true",
                    help="v1: use find_range center bin directly (no quality bin search).")
    args = ap.parse_args()
    m = RF_conv_decoder().to(DEVICE)
    m.load_state_dict(torch.load(args.ckpt, map_location=DEVICE)["model_P_state_dict"]); m.eval()

    sample_dirs = sorted([d for d in glob.glob(os.path.join(TEST_ROOT, "*")) if os.path.isdir(d)],
                         key=lambda p: tuple(int(x) for x in os.path.basename(p).split("_")))
    rows = [("ids", "HR")]
    n_fixed = 0
    for sd in sample_dirs:
        sid = os.path.basename(sd)
        segs = sorted([d for d in glob.glob(os.path.join(sd, "*")) if os.path.isdir(d)],
                      key=lambda p: int(os.path.basename(p)))
        hrs, qs, names = [], [], []
        for seg in segs:
            df = mat_io.load_test_data_f(seg)
            hr, q = seg_hr_binsearch(m, df, args.search, center_only=args.center_only)
            hrs.append(hr); qs.append(q); names.append(os.path.basename(seg))
        hrs = np.array(hrs); qs = np.array(qs)
        if args.soft_q0 > 0:
            # continuous fusion: every seg blended with same-subject reference by its quality
            good = qs >= args.soft_ref_good
            if good.sum() > 0:
                ref = np.average(hrs[good], weights=qs[good])
                w = 1.0 / (1.0 + np.exp(-(qs - args.soft_q0) / args.soft_tau))
                new = w * hrs + (1 - w) * ref
                n_fixed += int(np.sum(np.abs(new - hrs) > 0.05))
                hrs = new
        elif args.q_fallback > 0:
            good = qs >= args.q_fallback
            if 0 < good.sum() < len(hrs):
                ref = np.average(hrs[good], weights=qs[good])
                for i in range(len(hrs)):
                    if qs[i] < args.q_fallback:
                        hrs[i] = ref; n_fixed += 1
        for nm, hr in zip(names, hrs):
            rows.append((f"{sid}/{nm}", round(float(hr), 1)))
        print(f"{sid}: {len(segs)} segments")
    with open(args.out, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"Wrote {len(rows)-1} predictions -> {args.out}  (same-subject fallback fixed {n_fixed} segs)")


if __name__ == "__main__":
    main()
