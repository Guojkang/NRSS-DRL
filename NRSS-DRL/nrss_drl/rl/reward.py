from __future__ import annotations

import torch

from ..utils import huber


def compute_reward(
    a_i: torch.Tensor,
    y_i: torch.Tensor,
    a_prev: torch.Tensor | None = None,
    y_prev: torch.Tensor | None = None,
    lambda_g: float = 0.5,
    delta_p: float = 0.5,
    delta_g: float = 0.1,
    is_first: bool = False,
) -> torch.Tensor:
    """Point-error reward plus optional depth-consistency term."""
    r = -huber(a_i - y_i, delta_p)
    if (not is_first) and a_prev is not None and y_prev is not None:
        r = r - lambda_g * huber((a_i - a_prev) - (y_i - y_prev), delta_g)
    return r
