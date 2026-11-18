# NRSS-DRL

Implementation of **NRSS-DRL** for continuous TOC prediction from well logs:
noise-robust depth-context perception + DDPG sequential policy + few-shot action-mapping calibration.

This repository provides the training / inference / calibration scripts used in our study.

## Contents

```
NRSS-DRL/
├── config.json          # experiment hyperparameters
├── requirements.txt
├── train.py             # train on the source well (Well A)
├── infer.py             # zero-shot inference on a target well
├── calibrate.py         # few-shot calibration on Well C
├── data/                # put well CSVs here (not shipped)
└── nrss_drl/            # model and training utilities
```

## Environment

```bash
cd NRSS-DRL
pip install -r requirements.txt
```

## Data

Field well logs used in the paper are subject to confidentiality restrictions and are **not** included.
Place your own CSVs under `data/` (or change `data.data_dir` in `config.json`):

| File | Role |
|------|------|
| `Well_A.csv` | source well (train / validation) |
| `Well_B.csv` | zero-shot target |
| `Well_C.csv` | few-shot calibration target |

Required columns:

```
Depth, PE, DEN, AC, GR, RS, CNL, M2R1, TOC
```

`RS` corresponds to the shallow resistivity curve (R25) used in the experiments.

## Usage

```bash
# Train on Well A (saves checkpoints/best_nrss_drl.pt)
python train.py --config config.json

# Zero-shot prediction on Well B
python infer.py --config config.json --well B --ckpt checkpoints/best_nrss_drl.pt

# Few-shot calibration on Well C (e.g. 10% labeled depths)
python calibrate.py --config config.json --ratio 0.1
```

Optional:

```bash
python train.py --config config.json --episodes 50
python infer.py --config config.json --well C --out outputs/pred_WellC.csv
```

## Notes

- Preprocessing statistics (3-sigma clip + Z-score) are fit on the Well A training split only and then frozen.
- Action initialization uses the mean TOC of the Well A training subset.
- Depth gaps larger than `1.5 * sampling_interval` split the sequence; each segment resets the previous action.
- Default hyperparameters are in `config.json` (300 episodes, Huber thresholds, DDPG settings, etc.).
