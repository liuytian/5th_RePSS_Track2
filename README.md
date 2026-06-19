# Radar-based Heartbeat Signal Extraction

Contactless heartbeat signal extraction from FMCW radar signals through the analysis of subtle heartbeat-induced chest motions.

## Environment

```bash
pip install -r requirements.txt
```
- Python 3.10+, CUDA GPU recommended
- torch 2.6, numpy 1.26, scipy 1.13

## Data

Place the dataset next to the repo (or set `RHB_TRAIN_ROOT` / `TEST_ROOT`):
```
RHB_train/        # training: <subj>_<trial>/{tx1_rx1_complex_range_matrix.mat, vital_dict.npy}
track2_testdata/  # test:     <subj>_1/<seg>/radar_segment.mat
```

## Project layout

```
src/
  config.py        # paths and radar/signal constants
  models.py        # conv encoder/decoder (RF_conv_decoder)
  losses.py        # contrastive loss
  signal_proc.py   # range localisation, IQ aug/norm, phase, HR-from-PSD
  metrics.py       # RMSE / MAE / Pearson
  data.py          # .mat/.npy loaders + training dataset
  train.py         # train one model
  inference.py     # ensemble inference -> submission CSV
  postprocess.py   # same-subject disaster-segment correction
  evaluate.py      # held-out evaluation
  folds/           # train / held-out splits
```

## Reproduce

Run from `src/`.

**1 — Train models (one per seed):**
```bash
for S in 11 22 33 44 55; do
  python train.py --folds-path folds/fold_all.pkl --ckpt-dir ckpt_seed$S \
    --epochs 200 --lr 1e-4 --seed $S
done
```

**2 — Ensemble inference:**
```bash
python inference.py \
  --ckpts ckpt_seed11/best.pth ckpt_seed22/best.pth ckpt_seed33/best.pth \
          ckpt_seed44/best.pth ckpt_seed55/best.pth \
  --out submission_ensemble.csv --q-fallback 0.10 --reduce median
```

**3 — Post-process (final submission):**
```bash
python postprocess.py --in submission_ensemble.csv \
  --out submission.csv --dev-thr 8 --std-thr 2.5
```

**Evaluate on the held-out split (optional):**
```bash
python evaluate.py --ckpt ckpt_seed11/best.pth \
  --folds-path folds/fold_heldtrial.pkl --splits val_files
```

## Method summary

1. **Model** — a 1-D conv encoder/decoder maps a multi-bin radar IQ window to a
   PPG-like waveform, trained with a contrastive loss whose positive target is the
   ground-truth PPG and negative target is a noise window.
2. **Inference** — for each segment, search the range bins around the detected
   peak and keep the one whose predicted PPG has the sharpest in-band spectrum;
   read HR by periodogram peak; ensemble multiple seeds by median.
3. **Post-processing** — exploit the fact that a subject's heart rate is stable
   across its segments: pull any segment whose HR deviates strongly from its own
   subject's consensus back to that consensus.
