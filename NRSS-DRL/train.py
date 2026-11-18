#!/usr/bin/env python
"""Train NRSS-DRL on the source well (Well A)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from nrss_drl.data.dataset import prepare_source_train_val
from nrss_drl.metrics import evaluate
from nrss_drl.rl.ddpg import NRDDPGAgent
from nrss_drl.utils import (
    ensure_dir,
    load_config,
    resolve_data_path,
    resolve_device,
    set_seed,
)


def parse_args():
    p = argparse.ArgumentParser(description="Train NRSS-DRL on Well A")
    p.add_argument("--config", type=str, default="config.json")
    p.add_argument("--episodes", type=int, default=None, help="Override episode count")
    return p.parse_args()


def main():
    args = parse_args()
    root = Path(__file__).resolve().parent
    cfg = load_config(root / args.config)
    set_seed(int(cfg["train"]["seed"]))
    device = resolve_device(cfg["train"]["device"])
    print(f"[NRSS-DRL] device={device}")

    dcfg = cfg["data"]
    path_a = resolve_data_path(cfg, dcfg["well_a"])

    train_seq, val_seq, stats, y_mean = prepare_source_train_val(
        path_a,
        feature_cols=dcfg["feature_cols"],
        target_col=dcfg["target_col"],
        depth_col=dcfg["depth_col"],
        L=int(dcfg["window_length"]),
        train_val_ratio=float(dcfg["train_val_ratio"]),
        sampling_interval=float(dcfg.get("sampling_interval", 0.125)),
    )
    print(
        f"Well A train windows={len(train_seq)} (segments={train_seq.n_segments}), "
        f"val windows={len(val_seq)} (segments={val_seq.n_segments}), "
        f"y_init(mean TOC)={y_mean:.4f}"
    )

    agent = NRDDPGAgent(cfg, device)
    agent.set_action_init(y_mean)

    ckpt_dir = ensure_dir(root / cfg["train"]["checkpoint_dir"])
    with open(ckpt_dir / "preprocess_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats.to_dict(), f, indent=2)

    episodes = int(cfg["train"]["episodes"])
    if args.episodes is not None:
        episodes = args.episodes

    patience = int(cfg["train"]["early_stop_patience"])
    best_r2 = -1e9
    best_path = ckpt_dir / "best_nrss_drl.pt"
    bad = 0
    history = []

    for ep in tqdm(range(1, episodes + 1), desc="episodes"):
        stats_ep = agent.collect_episode(train_seq)
        y_hat = agent.predict_sequence(val_seq)
        metrics = evaluate(val_seq.toc.astype(float), y_hat)
        row = {"episode": ep, **stats_ep, **{f"val_{k}": v for k, v in metrics.items()}}
        history.append(row)

        if ep % max(1, int(cfg["train"]["log_every"])) == 0:
            tqdm.write(
                f"ep={ep} reward={stats_ep.get('ep_reward', 0):.2f} "
                f"val_R2={metrics['R2']:.4f} val_RMSE={metrics['RMSE']:.4f}"
            )

        if metrics["R2"] > best_r2:
            best_r2 = metrics["R2"]
            bad = 0
            torch.save(agent.state_dict_train(), best_path)
        else:
            bad += 1
            if patience > 0 and bad >= patience:
                tqdm.write(f"Early stop at episode={ep}, best val R2={best_r2:.4f}")
                break

    torch.save(agent.state_dict_train(), ckpt_dir / "last_nrss_drl.pt")
    with open(ckpt_dir / "train_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"Best checkpoint: {best_path} (val R2={best_r2:.4f})")


if __name__ == "__main__":
    main()
