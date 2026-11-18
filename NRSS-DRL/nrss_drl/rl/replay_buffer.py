from __future__ import annotations

from typing import Dict

import numpy as np
import torch


class ReplayBuffer:
    """Replay of raw windows and behavioral actions (avoids stale embeddings)."""

    def __init__(self, capacity: int, L: int, dx: int):
        self.capacity = int(capacity)
        self.L = L
        self.dx = dx
        self.ptr = 0
        self.size = 0

        self.s = np.zeros((capacity, L, dx), dtype=np.float32)
        self.s_next = np.zeros((capacity, L, dx), dtype=np.float32)
        self.a_prev = np.zeros((capacity,), dtype=np.float32)
        self.a = np.zeros((capacity,), dtype=np.float32)
        self.r = np.zeros((capacity,), dtype=np.float32)
        self.done = np.zeros((capacity,), dtype=np.float32)

    def __len__(self) -> int:
        return self.size

    def add(
        self,
        s: np.ndarray,
        a_prev: float,
        a: float,
        r: float,
        s_next: np.ndarray,
        done: float,
    ) -> None:
        i = self.ptr
        self.s[i] = s
        self.s_next[i] = s_next
        self.a_prev[i] = a_prev
        self.a[i] = a
        self.r[i] = r
        self.done[i] = done
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        idx = np.random.randint(0, self.size, size=batch_size)
        return {
            "s": torch.as_tensor(self.s[idx], device=device),
            "s_next": torch.as_tensor(self.s_next[idx], device=device),
            "a_prev": torch.as_tensor(self.a_prev[idx], device=device),
            "a": torch.as_tensor(self.a[idx], device=device),
            "r": torch.as_tensor(self.r[idx], device=device),
            "done": torch.as_tensor(self.done[idx], device=device),
        }
