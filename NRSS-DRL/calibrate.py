#!/usr/bin/env python
"""Few-shot action-mapping calibration on a target well (Well C)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from nrss_drl.calibration import calibrate_action_mapping, make_calibration_mask
from nrss_drl.data.dataset import prepare_well_sequence
from nrss_drl.data.preprocess import PreprocessStats
from nrss_drl.metrics import evaluate
from nrss_drl.rl.ddpg import NRDDPGAgent
from nrss_drl.utils import ensure_dir, load_config, resolve_data_path, resolve_device, set_seed


def parse_args():
    p = argparse.ArgumentParser(description="Few-shot calibration on Well C")
    p.add_argument("--config", type=str, default="config.json")
    p.add_argument("--ckpt", type=str, default="checkpoints/best_nrss_drl.pt")
    p.add_argument("--ratio", type=float, default=0.1, help="Labeled depth fraction")
    p.add_argument("--epochs", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    root = Path(__file__).resolve().parent
    cfg = load_config(root / args.config)
    set_seed(int(cfg["calibration"]["seed"]))
    device = resolve_device(cfg["train"]["device"])

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_absolute():
        ckpt_path = root / ckpt_path
    with open(ckpt_path.parent / "preprocess_stats.json", "r", encoding="utf-8") as f:
        stats = PreprocessStats.from_dict(json.load(f))

    dcfg = cfg["data"]
    path_c = resolve_data_path(cfg, dcfg["well_c"])
    state = torch.load(ckpt_path, map_location=device)
    y_init = float(state.get("y_init", 0.0))

    seq, _ = prepare_well_sequence(
        path_c,
        feature_cols=dcfg["feature_cols"],
        target_col=dcfg["target_col"],
        depth_col=dcfg["depth_col"],
        L=int(dcfg["window_length"]),
        stats=stats,
        name="WellC",
        y_mean_train=y_init,
        sampling_interval=float(dcfg.get("sampling_interval", 0.125)),
    )

    agent = NRDDPGAgent(cfg, device)
    agent.load_state_dict_train(state)
    agent.set_action_init(y_init)

    y0 = agent.predict_sequence(seq)
    zs = evaluate(seq.toc.astype(float), y0)
    print(f"[Well C zero-shot] {zs}")

    mask = make_calibration_mask(seq.T, args.ratio, seed=int(cfg["calibration"]["seed"]))
    blind = mask < 0.5
    print(f"calibration labels={int(mask.sum())}/{seq.T}, blind={int(blind.sum())}")

    ccfg = cfg["calibration"]
    epochs = args.epochs if args.epochs is not None else int(ccfg["epochs"])
    result = calibrate_action_mapping(
        agent,
        seq,
        mask,
        epochs=epochs,
        lr=float(ccfg["lr"]),
        delta_c=float(ccfg["delta_c"]),
        lambda_a=float(ccfg["lambda_a"]),
    )

    y1 = agent.predict_sequence(seq)
    metrics_blind = evaluate(seq.toc[blind].astype(float), y1[blind])
    print(f"[Well C few-shot {args.ratio:.0%} blind] {metrics_blind}")

    out_dir = ensure_dir(root / "outputs")
    torch.save(agent.state_dict_train(), out_dir / f"calibrated_r{args.ratio:.2f}.pt")
    with open(out_dir / f"calibration_r{args.ratio:.2f}.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "ratio": args.ratio,
                "zero_shot": zs,
                "few_shot_blind": metrics_blind,
                "history_tail": result.get("final", {}),
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
