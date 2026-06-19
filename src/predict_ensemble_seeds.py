"""Multi-seed ensemble on top of the 6.86 pipeline (quality bin-search + same-subject
fallback). RMSE is dominated by ~3% random disaster segments (signal-weak); different
seeds fail on DIFFERENT disaster segments, so a per-segment HR MEDIAN over seeds
cancels that random long tail -- directly attacking the RMSE metric.

Per segment: each seed model runs quality bin-search -> (HR_s, quality_s).
  ensemble HR      = median_s HR_s            (robust to per-seed disaster outliers)
  ensemble quality = max_s quality_s          (a seg is 'good' if ANY seed is confident)
Then the same same-subject q-fallback as 6.86.

Usage:
  python predict_ensemble_seeds.py --ckpts ckpt_gt_s1_full_ep200/best.pth \
      ckpt_seed11/best.pth ckpt_seed22/best.pth ... \
      --out track2_submission_ens.csv --q-fallback 0.10
"""
import os
import csv
import glob
import argparse
import numpy as np
import torch

from predict_binsearch import (TEST_ROOT, DEVICE, RF_conv_decoder,
                               seg_hr_binsearch)
import mat_io


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--search", type=int, default=3)
    ap.add_argument("--q-fallback", type=float, default=0.10)
    ap.add_argument("--reduce", choices=["median", "mean"], default="median")
    args = ap.parse_args()

    models = []
    for ck in args.ckpts:
        m = RF_conv_decoder().to(DEVICE)
        m.load_state_dict(torch.load(ck, map_location=DEVICE)["model_P_state_dict"])
        m.eval()
        models.append(m)
    print(f"loaded {len(models)} seed models")

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
            seed_hr, seed_q = [], []
            for m in models:
                hr, q = seg_hr_binsearch(m, df, args.search)
                seed_hr.append(hr); seed_q.append(q)
            hr_e = float(np.median(seed_hr)) if args.reduce == "median" else float(np.mean(seed_hr))
            q_e = float(np.max(seed_q))                 # confident if ANY seed is
            hrs.append(hr_e); qs.append(q_e); names.append(os.path.basename(seg))
        hrs = np.array(hrs); qs = np.array(qs)
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
    print(f"Wrote {len(rows)-1} preds -> {args.out}  (ensemble {len(models)} seeds, fallback {n_fixed})")


if __name__ == "__main__":
    main()
