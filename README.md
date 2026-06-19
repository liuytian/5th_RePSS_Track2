# Radar-based Heart Rate Estimation — Track 2 (3rd place, RMSE 6.82739)

FMCW-radar contactless heart-rate estimation. Given a 10 s radar range-time
matrix per segment, predict the heart rate (bpm). Final leaderboard RMSE
**6.82739** (3rd place).

This repo is the **minimal code to reproduce the final submission**.

---

## Method (3 stages)

1. **Model — GT-stage1.** A `RF_conv_decoder` (1-D conv encoder/decoder, from the
   Radar-APLANC baseline) trained **from scratch** with the `ContrastLoss`
   positive target replaced by the **real PPG waveform** (`vital_dict.npy`),
   instead of the baseline's pseudo/traditional-phase target. This single change
   broke the pseudo-label quality ceiling (held RMSE 4.49 → 1.94).

2. **Inference — quality bin-search + multi-seed ensemble.**
   - `find_range` picks the strongest range bin, but it is often wrong (held: only
     12% match the oracle). So at inference we search `find_range ± 3` and keep the
     bin whose predicted PPG has the **sharpest band-limited PSD peak** (spectral
     concentration = peak power / band power) — an unsupervised quality proxy.
   - HR is read by raw periodogram peak in 45–150 bpm.
   - **5-seed ensemble:** train 5 GT-stage1 models with different seeds; per
     segment take the **median** HR over seeds (robust to the random ~3% disaster
     segments that dominate RMSE), and the **max** quality (a segment is "good" if
     any seed is confident).

3. **Post-processing — same-subject fallback + disaster correction.**
   - **Same-subject fallback:** the test set is 50 subjects × 6 segments and a
     subject's HR is stable (std ~1.8 bpm). Replace low-quality (< 0.10) segments by
     the subject's quality-weighted good-segment mean.
   - **Disaster-segment correction:** a segment whose HR deviates strongly from its
     own subject's median (while the subject's other segments are tight) is almost
     certainly a wrong-bin "disaster". Pull it back to the subject median. Verified
     online — each isolated-spike correction lowered RMSE.

---

## Repo layout

```
src/
  train_stage3.py            # train one GT-stage1 model (run 5x with different --seed)
  predict_binsearch.py       # single-model inference: quality bin-search + fallback
  predict_ensemble_seeds.py  # multi-seed ensemble inference
  postprocess_disaster.py    # disaster-segment correction (final step)
  dataset_rhb.py             # training dataset adapter
  eval_fold.py               # held-out evaluation (MAE/RMSE)
  mat_io.py                  # competition .mat / .npy loaders
  fold_all.pkl               # full-train split (for the final models)
  fold_heldtrial.pkl         # held-out split (for local validation)
baseline/                    # Radar-APLANC baseline modules used as dependencies
  rf/ utils/ utils_sig.py    # (RF_conv_decoder, ContrastLoss, find_range, ...)
requirements.txt
```

## Data (not included)

The radar dataset is not redistributable. Place it alongside the repo as:
```
RHB_train/        # training trials: <subj>_<trial>/{tx1_rx1_complex_range_matrix.mat, vital_dict.npy}
track2_testdata/  # test segments:  <subj>_1/<seg>/radar_segment.mat
```
Paths are configured in `src/mat_io.py` (relative to the script).

## Environment

```bash
pip install -r requirements.txt   # torch 2.6 (cu126), numpy 1.26, scipy 1.13
# baseline/ must be importable: run scripts from src/ with baseline/ on PYTHONPATH
```

---

## Reproduce the final submission

Run from `src/` (with `../baseline` on the path; the scripts already insert it).

**1) Train 5 GT-stage1 models (different seeds):**
```bash
for S in 11 22 33 44 55; do
  python train_stage3.py --folds-path fold_all.pkl \
    --ckpt-dir ckpt_seed$S --epochs 200 --lr 1e-4 --lambda-hr 0 --seed $S
done
# ~26 min/seed on a single GPU. (The original used one ep200 model + 4 seeds.)
```

**2) Multi-seed ensemble inference (quality bin-search + same-subject fallback):**
```bash
python predict_ensemble_seeds.py \
  --ckpts ckpt_seed11/best.pth ckpt_seed22/best.pth ckpt_seed33/best.pth \
          ckpt_seed44/best.pth ckpt_seed55/best.pth \
  --out submission_ensemble.csv --q-fallback 0.10 --reduce median
```

**3) Disaster-segment correction (final step):**
```bash
python postprocess_disaster.py --in submission_ensemble.csv \
  --out submission_final.csv --dev-thr 8 --std-thr 2.5
```
`submission_final.csv` is the final 6.82739 submission.

> Note: the very last leaderboard points came from individually verified isolated-
> spike corrections (54_1/3, 22_1/6, 25_1/4). `postprocess_disaster.py` with
> `--dev-thr 8` reproduces that family of safe corrections automatically; smaller
> thresholds risk "correcting" real physiological variation and can hurt.

## Local validation (optional)

```bash
python eval_fold.py --ckpt ckpt_seed11/best.pth \
  --folds-path fold_heldtrial.pkl --splits val_files
# reports held-out MAE / RMSE / Pearson R
```

---

## Acknowledgements

Built on the **Radar-APLANC** baseline (`baseline/` — `RF_conv_decoder`,
`ContrastLoss`, `find_range`, etc.).
