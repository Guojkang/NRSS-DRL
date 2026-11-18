from __future__ import annotations

import numpy as np
import torch

from .metrics import evaluate
from .rl.ddpg import NRDDPGAgent
from .utils import huber


def make_calibration_mask(n: int, ratio: float, seed: int = 42) -> np.ndarray:
    """Select a random subset of depths as few-shot labels."""
    rng = np.random.default_rng(seed)
    mask = np.zeros(n, dtype=np.float32)
    k = max(1, int(round(n * ratio))) if ratio > 0 else 0
    if k == 0:
        return mask
    idx = rng.choice(n, size=k, replace=False)
    mask[idx] = 1.0
    return mask


def calibrate_action_mapping(
    agent: NRDDPGAgent,
    seq,
    mask: np.ndarray,
    epochs: int = 50,
    lr: float = 1e-4,
    delta_c: float = 1.0,
    lambda_a: float = 1e-3,
) -> dict:
    """
    Freeze perception and actor backbone; fine-tune only the output head.
    Labeled depths contribute to the supervised loss; an L2 anchor keeps the
    head close to the source-well parameters.
    """
    agent.freeze_perception_and_backbone()
    agent.perception.eval()
    agent.actor.backbone.eval()

    psi_src = {
        k: v.detach().clone()
        for k, v in agent.actor.output_layer.state_dict().items()
    }

    opt = torch.optim.Adam(agent.actor.output_parameters(), lr=lr)
    history = []

    for ep in range(epochs):
        agent.actor.output_layer.train()
        a_prev = float(agent.y_init)
        labeled_losses = []

        for t in range(seq.T):
            if NRDDPGAgent._is_segment_start(seq, t):
                a_prev = float(agent.y_init)
            s_t = torch.as_tensor(
                seq.windows[t], dtype=torch.float32, device=agent.device
            ).unsqueeze(0)
            a_prev_t = torch.tensor([a_prev], dtype=torch.float32, device=agent.device)

            with torch.no_grad():
                h = agent.perception(s_t, add_noise=False)
            q = agent.build_q(h, a_prev_t)
            with torch.no_grad():
                feat = agent.actor.features(q)
            a = agent.actor.output_layer(feat).squeeze(-1)

            if mask[t] > 0.5:
                y = torch.tensor(
                    [float(seq.toc[t])], dtype=torch.float32, device=agent.device
                )
                labeled_losses.append(huber(a - y, delta_c))
            a_prev = float(a.detach().item())

        if not labeled_losses:
            break

        loss_sup = torch.stack(labeled_losses).mean()
        loss_anchor = 0.0
        for name, p in agent.actor.output_layer.named_parameters():
            loss_anchor = loss_anchor + torch.sum((p - psi_src[name]).pow(2))
        loss = loss_sup + lambda_a * loss_anchor

        opt.zero_grad()
        loss.backward()
        opt.step()

        with torch.no_grad():
            y_hat = agent.predict_sequence(seq)
            metrics = evaluate(seq.toc.astype(np.float64), y_hat)
        history.append({"epoch": ep, "loss": float(loss.item()), **metrics})

    return {"history": history, "final": history[-1] if history else {}}
