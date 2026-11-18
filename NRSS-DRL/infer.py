#!/usr/bin/env python
"""Zero-shot inference with a trained NRSS-DRL model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from nrss_drl.data.dataset import prepare_well_sequence
from nrss_drl.data.preprocess import PreprocessStats
from nrss_drl.metrics import evaluate
from nrss_drl.rl.ddpg import NRDDPGAgent
from nrss_drl.utils import load_config, resolve_data_path, resolve_device, set_seed


def parse_args():
    p = argparse.ArgumentParser(description="Zero-shot inference with NRSS-DRL")
    p.add_argument("--config", type=str, default="config.json")
    p.add_argument("--well", type=str, choices=["A", "B", "C", "val"], default="B")
    p.add_argument("--ckpt", type=str, default="checkpoints/best_nrss_drl.pt")
    p.add_argument("--out", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    root = Path(__file__).resolve().parent
    cfg = load_config(root / args.config)
    set_seed(int(cfg["train"]["seed"]))
    device = resolve_device(cfg["train"]["device"])

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_absolute():
        ckpt_path = root / ckpt_path
    stats_path = ckpt_path.parent / "preprocess_stats.json"
    with open(stats_path, "r", encoding="utf-8") as f:
        stats = PreprocessStats.from_dict(json.load(f))

    dcfg = cfg["data"]
    well_map = {
        "A": dcfg["well_a"],
        "B": dcfg["well_b"],
        "C": dcfg["well_c"],
    }
    if args.well == "val":
        path = resolve_data_path(cfg, dcfg["well_a"])
        name = "WellA_full"
    else:
        path = resolve_data_path(cfg, well_map[args.well])
        name = f"Well{args.well}"

    state = torch.load(ckpt_path, map_location=device)
    y_init = float(state.get("y_init", 0.0))
    seq, _ = prepare_well_sequence(
        path,
        feature_cols=dcfg["feature_cols"],
        target_col=dcfg["target_col"],
        depth_col=dcfg["depth_col"],
        L=int(dcfg["window_length"]),
        stats=stats,
        name=name,
        y_mean_train=y_init,
        sampling_interval=float(dcfg.get("sampling_interval", 0.125)),
    )

    agent = NRDDPGAgent(cfg, device)
    agent.load_state_dict_train(state)
    agent.set_action_init(y_init)

    y_hat = agent.predict_sequence(seq)
    metrics = evaluate(seq.toc.astype(float), y_hat)
    print(f"[{name}] N={len(seq)} segments={seq.n_segments} metrics={metrics}")

    out = args.out or str(root / "outputs" / f"pred_Well{args.well}.csv")
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"TOC_true": seq.toc, "TOC_pred": y_hat})
    if seq.depths is not None:
        df.insert(0, "Depth", seq.depths)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
