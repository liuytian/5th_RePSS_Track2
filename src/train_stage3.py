"""Stage 3: GT-calibrated finetune (the third lever).

Built on the user's analysis:
 - Improvement 1 (REPLACE, not add): the ContrastLoss positive term pulls pred_P
   toward its 3rd arg. We feed GT PPG there instead of the traditional-phase
   pseudo. The noise NEGATIVE term is kept untouched (the paper's real anti-noise
   regularizer, label-free). So positive = clean GT, negative = noise contrast.
 - Improvement 2: add a differentiable HR loss (soft-argmax on the PSD) so we
   optimize the actual eval metric (HR), closing the PSD-shape vs peak-location gap.

L = ContrastLoss(pred_P, pred_N, GT)  +  lambda_hr * MSE(HR_softargmax(pred_P), HR_GT)

Continue-finetune from the converged stage2 model. Held-trial RMSE is the judge
(must beat stage2's 3.78, and ideally the semisup 3.24). Iron rule: win first.
"""
import os
import sys
import csv
import time
import random
import argparse
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "baseline"))
from utils_sig import normalize
from rf.model import RF_conv_decoder
from rf.proc import rotateIQ
from rf.loss import ContrastLoss, CalculateNormPSD

import dataset_rhb
import eval_fold
import mat_io

TRAIN_ROOT = os.path.join(os.path.dirname(__file__), "..", "RHB_train")


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


class SoftHRLoss(torch.nn.Module):
    """Differentiable HR via soft-argmax over the band-limited PSD (Fs=30).

    Returns MSE between pred-HR and GT-HR (bpm), both from soft-argmax so the
    target is on the SAME estimator as the prediction (no quantization bias).
    """
    def __init__(self, delta_t=300, K=4, Fs=30, high_pass=45, low_pass=180, temp=1.0):
        super().__init__()
        self.delta_t = delta_t; self.K = K; self.Fs = Fs
        self.high_pass = high_pass; self.low_pass = low_pass; self.temp = temp

    def _psd(self, x):  # x: (delta_t,) -> (freq,)
        x = x - x.mean()
        Xr = torch.view_as_real(torch.fft.rfft(x, norm='forward'))
        P = Xr[:, 0] ** 2 + Xr[:, 1] ** 2
        Fn = self.Fs / 2
        freqs = torch.linspace(0, Fn, P.shape[0], device=x.device)
        m = (freqs >= self.high_pass / 60) & (freqs <= self.low_pass / 60)
        return P[m], freqs[m]

    def _hr(self, x):
        P, freqs = self._psd(x)
        w = F.softmax(P / self.temp, dim=-1)
        return (w * freqs).sum() * 60.0   # bpm

    def forward(self, pred_p, gt):  # (B,T),(B,T)
        loss = 0.0; cnt = 0
        T = min(pred_p.shape[-1], gt.shape[-1])
        for b in range(pred_p.shape[0]):
            for _ in range(self.K):
                off = torch.randint(0, T - self.delta_t + 1, (1,)).item()
                hp = self._hr(pred_p[b, off:off + self.delta_t])
                with torch.no_grad():
                    hg = self._hr(gt[b, off:off + self.delta_t])
                loss = loss + (hp - hg) ** 2
                cnt += 1
        return loss / max(cnt, 1)


def load_gt(files):
    return {f: mat_io.load_gt_ppg(os.path.join(TRAIN_ROOT, f))[:900].astype(np.float32)
            for f in files}


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds-path", required=True)
    ap.add_argument("--pseudo-root", required=True)
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--init-ckpt", default=None,
                    help="Continue from this ckpt; omit to train GT-stage1 from scratch.")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--lambda-hr", type=float, default=0.001)
    ap.add_argument("--oversample-hr", type=float, default=0.0,
                    help="If >0, oversample high-HR trials: weight = 1 + k*relu(HR-90)/20.")
    ap.add_argument("--augment", type=float, default=0.0,
                    help="If >0, identity-preserving aug strength: Gaussian noise + "
                         "amplitude scale + time shift on the IQ input.")
    ap.add_argument("--val-period", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def main():
    args = parse()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.ckpt_dir, exist_ok=True)

    fold = pickle.load(open(args.folds_path, "rb"))
    train_files, val_files = fold["train_files"], fold["val_files"]
    print(f"Stage3 GT-replace: train {len(train_files)} val {len(val_files)} "
          f"lambda_hr={args.lambda_hr} lr={args.lr}")

    base = dataset_rhb.RHB(train_files, args.pseudo_root, frame_length_ppg=900,
                           target="pseudo")  # we ignore t_rf; GT replaces positive
    idx_to_file = list(base.rf_file_list)
    gt_dict = load_gt(train_files)

    class W(Dataset):
        def __len__(self): return len(base)
        def __getitem__(self, i):
            pseudo, p, n, t = base[i]
            return p, n, gt_dict[idx_to_file[i]]

    if args.oversample_hr > 0:
        # diagnosis: HR>110 trials are 0.7% of data and have held RMSE 27.9.
        # weight each trial by its GT HR so high-HR (rare, hard) segs appear more.
        from utils.utils import pulse_rate_from_power_spectral_density as _prpsd
        weights = []
        for f in idx_to_file:
            g = gt_dict[f][:300]; g = (g - g.mean()) / (g.std() + 1e-8)
            hr = _prpsd(g, 30, 45, 150, BUTTER_ORDER=6, DETREND=False)
            weights.append(1.0 + args.oversample_hr * max(0.0, hr - 90) / 20.0)
        weights = np.array(weights, dtype=np.float64)
        sampler = WeightedRandomSampler(weights, num_samples=len(base), replacement=True)
        loader = DataLoader(W(), batch_size=args.batch_size, sampler=sampler,
                            drop_last=True, num_workers=0)
        print(f"Oversample HR>90 (k={args.oversample_hr}): weight range "
              f"{weights.min():.2f}-{weights.max():.2f}, mean {weights.mean():.2f}")
    else:
        loader = DataLoader(W(), batch_size=args.batch_size, shuffle=True,
                            drop_last=True, num_workers=0)

    mP = RF_conv_decoder().to(device); mN = RF_conv_decoder().to(device)
    if args.init_ckpt:
        sd = torch.load(args.init_ckpt, map_location=device)
        mP.load_state_dict(sd["model_P_state_dict"]); mN.load_state_dict(sd["model_N_state_dict"])
        print(f"Init from {args.init_ckpt}")
    else:
        print("Training GT-stage1 FROM SCRATCH (random init)")

    contrast = ContrastLoss(300, 4, 30, 45, 180)   # positive arg = GT below
    hr_loss = SoftHRLoss(300, 4, 30, 45, 180)
    opt = torch.optim.AdamW(list(mP.parameters()) + list(mN.parameters()),
                            lr=args.lr, weight_decay=args.wd)

    mpath = os.path.join(args.ckpt_dir, "metrics.csv")
    with open(mpath, "w", newline="") as fc:
        csv.writer(fc).writerow(["epoch", "loss", "c_loss", "hr_loss", "val_mae", "val_rmse", "val_r"])

    best = float("inf")
    bpath = os.path.join(args.ckpt_dir, "best.pth")
    for ep in range(1, args.epochs + 1):
        mP.train(); mN.train()
        t0 = time.time(); tot = ct = ht = 0.0; nb = 0
        for p_rf, n_rf, gt in loader:
            if args.augment > 0:
                # identity-preserving: per-sample amplitude scale + additive
                # Gaussian noise (scaled to the IQ magnitude). No time shift -- it
                # would desync the IQ<->GT alignment the contrast loss depends on.
                a = args.augment
                for t in (p_rf, n_rf):
                    sc = 1.0 + a * 0.3 * (torch.rand(t.shape[0], 1, 1, 1) - 0.5) * 2
                    t.mul_(sc)
                    t.add_(a * 0.05 * t.abs().mean() * torch.randn_like(t))
            p_rf = normalize(p_rf).type(torch.float32)
            n_rf = normalize(n_rf).type(torch.float32)
            gt = gt.type(torch.float32).to(device)
            gt_norm = normalize(gt).type(torch.float32)        # same norm as t_rf path

            p_rf = rotateIQ(p_rf); p_rf = torch.reshape(p_rf, (p_rf.shape[0], -1, p_rf.shape[3])).to(device)
            n_rf = rotateIQ(n_rf); n_rf = torch.reshape(n_rf, (n_rf.shape[0], -1, n_rf.shape[3])).to(device)

            pred_p, _ = mP(p_rf); pred_p = pred_p.squeeze(1)
            pred_n, _ = mN(n_rf); pred_n = pred_n.squeeze(1)

            # positive term pulls pred_p -> GT (REPLACE); negative -> noise (KEEP)
            c_loss, _, _, _ = contrast(pred_p, pred_n, gt_norm)
            h_loss = hr_loss(pred_p, gt_norm)
            loss = c_loss + args.lambda_hr * h_loss

            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); ct += c_loss.item(); ht += h_loss.item(); nb += 1

        msg = f"Ep {ep}/{args.epochs} loss={tot/nb:.5f} c={ct/nb:.5f} hr={ht/nb:.3f} ({time.time()-t0:.1f}s)"
        vm = vr = vrr = ""
        if ep % args.val_period == 0 or ep == args.epochs:
            mP.eval()
            res = eval_fold.evaluate(mP, val_files)
            vm, vr, vrr = res["mae_mean"], res["rmse_mean"], res["pearson_r"]
            msg += f" | val MAE={vm:.3f} RMSE={vr:.3f} R={vrr:.3f}"
            if vr < best:
                best = vr
                torch.save({"model_P_state_dict": mP.state_dict(),
                            "model_N_state_dict": mN.state_dict()}, bpath)
                msg += "  [best]"
        print(msg)
        with open(mpath, "a", newline="") as fc:
            csv.writer(fc).writerow([ep, f"{tot/nb:.6f}", f"{ct/nb:.6f}", f"{ht/nb:.6f}", vm, vr, vrr])

    print(f"Best val RMSE: {best:.3f} -> {bpath}  (stage2=3.78, semisup=3.24)")


if __name__ == "__main__":
    main()
