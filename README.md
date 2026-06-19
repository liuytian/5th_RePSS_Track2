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
```text
RHB_train/        # training: <subj>_<trial>/{tx1_rx1_complex_range_matrix.mat, vital_dict.npy}
track2_testdata/  # test:     <subj>_1/<seg>/radar_segment.mat
```

## Project layout

```text
src/
  config.py        # paths and radar/signal constants
  models.py        # conv encoder/decoder
  losses.py        # contrastive loss
  signal_proc.py   # range localisation, IQ processing, spectral readout
  metrics.py       # RMSE / MAE / Pearson
  data.py          # .mat/.npy loaders + training dataset
  train.py         # model training
  inference.py     # inference -> submission CSV
  postprocess.py   # subject-level refinement
  evaluate.py      # held-out evaluation
  folds/           # train / held-out splits
```

## Reproduce

Run from `src/`.

**1. Train**
```bash
python train.py --folds-path folds/fold_all.pkl --ckpt-dir . --epochs 200 --lr 1e-4
```

This produces `best.pth`.

**2. Inference**
```bash
python inference.py --ckpts best.pth --out submission_raw.csv
```

**Evaluate on the held-out split (optional)**
```bash
python evaluate.py --ckpt best.pth --folds-path folds/fold_heldtrial.pkl --splits val_files
```
