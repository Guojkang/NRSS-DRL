from __future__ import annotations

import copy
from typing import Dict

import numpy as np
import torch
import torch.nn as nn

from ..models.actor_critic import Actor, Critic
from ..models.nrdcp import NRDCP
from ..utils import soft_update
from .replay_buffer import ReplayBuffer
from .reward import compute_reward


class NRDDPGAgent(nn.Module):
    """DDPG agent with depth-context perception for sequential TOC prediction."""

    def __init__(self, cfg: dict, device: torch.device):
        super().__init__()
        self.cfg = cfg
        self.device = device
        m = cfg["model"]
        t = cfg["train"]
        d = cfg["data"]

        self.L = int(d["window_length"])
        self.dx = int(m["dx"])
        self.dm = int(m["dm"])
        self.state_dim = self.dm + 1

        self.gamma = float(t["gamma"])
        self.tau = float(t["tau"])
        self.sigma_a = float(t["exploration_sigma"])
        self.beta_rec = float(t["beta_rec"])
        self.weight_decay = float(t["weight_decay"])
        self.lambda_g = float(t["lambda_g"])
        self.delta_p = float(t["delta_p"])
        self.delta_g = float(t["delta_g"])
        self.grad_clip = float(t.get("grad_clip", 5.0))
        self.batch_size = int(t["batch_size"])

        self.perception = NRDCP(
            dx=self.dx,
            dz=int(m["dz"]),
            dm=self.dm,
            n_layers=int(m["n_transformer_layers"]),
            n_heads=int(m["n_heads"]),
            ffn_dim=int(m["ffn_dim"]),
            dropout=float(m["dropout"]),
            input_noise_sigma=float(t["input_noise_sigma"]),
        ).to(device)

        self.actor = Actor(
            state_dim=self.state_dim,
            hidden=tuple(m["actor_hidden"]),
        ).to(device)
        self.critic = Critic(
            state_dim=self.state_dim,
            hidden=tuple(m["critic_hidden"]),
        ).to(device)

        self.perception_t = copy.deepcopy(self.perception).to(device)
        self.actor_t = copy.deepcopy(self.actor).to(device)
        self.critic_t = copy.deepcopy(self.critic).to(device)
        for p in (
            list(self.perception_t.parameters())
            + list(self.actor_t.parameters())
            + list(self.critic_t.parameters())
        ):
            p.requires_grad = False

        self.opt_critic = torch.optim.Adam(
            self.critic.parameters(), lr=float(t["lr_critic"])
        )
        # Joint Actor + perception update (Critic fixed)
        self.opt_actor_perc = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.perception.parameters()),
            lr=float(t["lr_actor"]),
        )

        self.buffer = ReplayBuffer(
            capacity=int(t["replay_capacity"]), L=self.L, dx=self.dx
        )
        self.y_init = 0.0

    def set_action_init(self, y_mean: float) -> None:
        self.y_init = float(y_mean)

    def build_q(self, h: torch.Tensor, a_prev: torch.Tensor) -> torch.Tensor:
        if a_prev.dim() == 1:
            a_prev = a_prev.unsqueeze(-1)
        return torch.cat([h, a_prev], dim=-1)

    @torch.no_grad()
    def act(
        self,
        s: torch.Tensor,
        a_prev: torch.Tensor,
        explore: bool = False,
    ) -> torch.Tensor:
        self.perception.eval()
        self.actor.eval()
        h = self.perception(s, add_noise=False)
        q = self.build_q(h, a_prev)
        a = self.actor(q)
        if explore and self.sigma_a > 0:
            a = a + torch.randn_like(a) * self.sigma_a
        return a

    def update(self) -> Dict[str, float]:
        if len(self.buffer) < self.batch_size:
            return {}

        batch = self.buffer.sample(self.batch_size, self.device)
        s = batch["s"]
        s_next = batch["s_next"]
        a_prev = batch["a_prev"]
        a_beh = batch["a"]
        r = batch["r"]
        done = batch["done"]

        # --- Critic update ---
        self.perception.train()
        self.actor.train()
        self.critic.train()

        with torch.no_grad():
            h = self.perception(s, add_noise=False)
            q = self.build_q(h, a_prev)

            h_next = self.perception_t(s_next, add_noise=False)
            q_next = self.build_q(h_next, a_beh)
            a_next = self.actor_t(q_next)
            q_tgt = self.critic_t(q_next, a_next)
            y_td = r + self.gamma * (1.0 - done) * q_tgt

        h_online = self.perception(s, add_noise=False)
        q_online = self.build_q(h_online, a_prev)

        q_pred = self.critic(q_online.detach(), a_beh)
        loss_q = torch.mean((q_pred - y_td).pow(2))

        self.opt_critic.zero_grad()
        loss_q.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
        self.opt_critic.step()

        # --- Actor + perception joint update (Critic fixed) ---
        for p in self.critic.parameters():
            p.requires_grad = False

        h_ap = self.perception(s, add_noise=True)
        q_ap = self.build_q(h_ap, a_prev)
        a_pi = self.actor(q_ap)
        q_val = self.critic(q_ap, a_pi)
        loss_rec = self.perception.reconstruction_loss(s, weight_decay=self.weight_decay)
        loss_ap = -torch.mean(q_val) + self.beta_rec * loss_rec

        self.opt_actor_perc.zero_grad()
        loss_ap.backward()
        nn.utils.clip_grad_norm_(
            list(self.actor.parameters()) + list(self.perception.parameters()),
            self.grad_clip,
        )
        self.opt_actor_perc.step()

        for p in self.critic.parameters():
            p.requires_grad = True

        # --- soft update ---
        soft_update(self.actor, self.actor_t, self.tau)
        soft_update(self.critic, self.critic_t, self.tau)
        soft_update(self.perception, self.perception_t, self.tau)

        return {
            "loss_critic": float(loss_q.item()),
            "loss_actor_perc": float(loss_ap.item()),
            "loss_rec": float(loss_rec.item()),
            "q_mean": float(q_val.detach().mean().item()),
        }

    @staticmethod
    def _is_segment_start(seq, t: int) -> bool:
        """True at a new contiguous depth segment (or the first well step)."""
        if hasattr(seq, "is_segment_start"):
            return bool(seq.is_segment_start(t))
        if t <= 0:
            return True
        ss = getattr(seq, "segment_start", None)
        return bool(ss[t]) if ss is not None else False

    def collect_episode(self, seq) -> Dict[str, float]:
        """Roll out one well (or multi-segment well) into the replay buffer and update."""
        self.perception.train()
        self.actor.train()

        T = seq.T
        a_prev = float(self.y_init)
        ep_reward = 0.0
        n_updates = 0
        last_stats: Dict[str, float] = {}

        for t in range(T):
            if self._is_segment_start(seq, t):
                a_prev = float(self.y_init)

            s = seq.windows[t]
            y = float(seq.toc[t])
            s_t = torch.as_tensor(s, dtype=torch.float32, device=self.device).unsqueeze(0)
            a_prev_t = torch.tensor([a_prev], dtype=torch.float32, device=self.device)

            with torch.no_grad():
                a = self.act(s_t, a_prev_t, explore=True).item()

            is_first = self._is_segment_start(seq, t)
            a_prev_tensor = torch.tensor(a_prev, dtype=torch.float32)
            y_prev_tensor = (
                None
                if is_first
                else torch.tensor(float(seq.toc[t - 1]), dtype=torch.float32)
            )
            r = compute_reward(
                torch.tensor(a, dtype=torch.float32),
                torch.tensor(y, dtype=torch.float32),
                None if is_first else a_prev_tensor,
                y_prev_tensor,
                lambda_g=self.lambda_g,
                delta_p=self.delta_p,
                delta_g=self.delta_g,
                is_first=is_first,
            ).item()

            next_is_new = (t == T - 1) or self._is_segment_start(seq, t + 1)
            done = 1.0 if next_is_new else 0.0
            if done:
                s_next = np.zeros_like(s)
            else:
                s_next = seq.windows[t + 1]

            self.buffer.add(s, a_prev, a, r, s_next, done)
            ep_reward += r
            a_prev = a

            if len(self.buffer) >= self.batch_size:
                last_stats = self.update()
                n_updates += 1

        out = {"ep_reward": ep_reward, "n_updates": float(n_updates)}
        out.update(last_stats)
        return out

    @torch.no_grad()
    def predict_sequence(
        self,
        seq,
        explore: bool = False,
    ) -> np.ndarray:
        """Autoregressive TOC prediction along depth."""
        self.perception.eval()
        self.actor.eval()
        preds = []
        a_prev = float(self.y_init)
        for t in range(seq.T):
            if self._is_segment_start(seq, t):
                a_prev = float(self.y_init)
            s_t = torch.as_tensor(
                seq.windows[t], dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            a_prev_t = torch.tensor([a_prev], dtype=torch.float32, device=self.device)
            a = self.act(s_t, a_prev_t, explore=explore).item()
            preds.append(a)
            a_prev = a
        return np.asarray(preds, dtype=np.float64)

    def freeze_perception_and_backbone(self) -> None:
        for p in self.perception.parameters():
            p.requires_grad = False
        self.actor.freeze_backbone()

    def state_dict_train(self) -> dict:
        return {
            "perception": self.perception.state_dict(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "perception_t": self.perception_t.state_dict(),
            "actor_t": self.actor_t.state_dict(),
            "critic_t": self.critic_t.state_dict(),
            "y_init": self.y_init,
            "cfg": self.cfg,
        }

    def load_state_dict_train(self, state: dict) -> None:
        self.perception.load_state_dict(state["perception"])
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.perception_t.load_state_dict(state["perception_t"])
        self.actor_t.load_state_dict(state["actor_t"])
        self.critic_t.load_state_dict(state["critic_t"])
        self.y_init = float(state.get("y_init", self.y_init))
