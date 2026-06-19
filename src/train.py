"""Train a GT-stage1 model.

The model is trained from scratch with a contrastive loss whose positive target
is the real ground-truth PPG waveform (the noise negative term is kept). Run this
with several seeds to build the inference ensemble.

  python train.py --folds-path folds/fold_all.pkl --ckpt-dir ckpt_seed11 \
      --epochs 200 --lr 1e-4 --seed 11
"""
import os
import csv
import time
import random
import argparse
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader

import config
from models import RF_conv_decoder
from losses import ContrastLoss
from signal_proc import normalize, rotateIQ
from data import RHBDataset, load_gt_ppg
import evaluate


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds-path", required=True)
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--val-period", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def main():
    args = parse()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.ckpt_dir, exist_ok=True)

    fold = pickle.load(open(args.folds_path, "rb"))
    train_files, val_files = fold["train_files"], fold["val_files"]
    print(f"GT-stage1: train {len(train_files)} val {len(val_files)} "
          f"seed={args.seed} lr={args.lr}")

    ds = RHBDataset(train_files, frame_length_ppg=900)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        drop_last=True, num_workers=0)

    mP = RF_conv_decoder().to(device)
    mN = RF_conv_decoder().to(device)
    contrast = ContrastLoss(300, 4, 30, 45, 180)   # positive target = GT PPG
    opt = torch.optim.AdamW(list(mP.parameters()) + list(mN.parameters()),
                            lr=args.lr, weight_decay=args.wd)

    mpath = os.path.join(args.ckpt_dir, "metrics.csv")
    with open(mpath, "w", newline="") as fc:
        csv.writer(fc).writerow(["epoch", "loss", "val_mae", "val_rmse", "val_r"])

    best = float("inf")
    bpath = os.path.join(args.ckpt_dir, "best.pth")
    for ep in range(1, args.epochs + 1):
        mP.train(); mN.train()
        t0 = time.time(); tot = 0.0; nb = 0
        for p_rf, n_rf, gt in loader:
            p_rf = normalize(p_rf).type(torch.float32)
            n_rf = normalize(n_rf).type(torch.float32)
            gt_norm = normalize(gt.type(torch.float32)).to(device)

            p_rf = rotateIQ(p_rf); p_rf = torch.reshape(p_rf, (p_rf.shape[0], -1, p_rf.shape[3])).to(device)
            n_rf = rotateIQ(n_rf); n_rf = torch.reshape(n_rf, (n_rf.shape[0], -1, n_rf.shape[3])).to(device)

            pred_p, _ = mP(p_rf); pred_p = pred_p.squeeze(1)
            pred_n, _ = mN(n_rf); pred_n = pred_n.squeeze(1)

            # positive term pulls pred_p -> GT PPG; negative term -> noise window
            loss, _, _, _ = contrast(pred_p, pred_n, gt_norm)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1

        msg = f"Ep {ep}/{args.epochs} loss={tot/nb:.5f} ({time.time()-t0:.1f}s)"
        vm = vr = vrr = ""
        if ep % args.val_period == 0 or ep == args.epochs:
            mP.eval()
            res = evaluate.evaluate(mP, val_files)
            vm, vr, vrr = res["mae_mean"], res["rmse_mean"], res["pearson_r"]
            msg += f" | val MAE={vm:.3f} RMSE={vr:.3f} R={vrr:.3f}"
            if vr < best:
                best = vr
                torch.save({"model_P_state_dict": mP.state_dict(),
                            "model_N_state_dict": mN.state_dict()}, bpath)
                msg += "  [best]"
        print(msg)
        with open(mpath, "a", newline="") as fc:
            csv.writer(fc).writerow([ep, f"{tot/nb:.6f}", vm, vr, vrr])

    print(f"Best val RMSE: {best:.3f} -> {bpath}")


if __name__ == "__main__":
    main()
