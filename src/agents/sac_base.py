"""Shared SAC-Lagrangian base used by state / county / district agents.

This mirrors the design in `edu-rl-agent/src/agent/sac_lagrangian.py` but
keeps the dependency surface small (no `utils.config` import) so it can
be re-used from the multiagent repo without dragging the singleton Config.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)
LOG_STD_MIN, LOG_STD_MAX = -20.0, 2.0


# --------------------------------------------------------------------------- #
@dataclass
class AgentCfg:
    state_dim: int
    action_dim: int
    action_low: float = -3.0
    action_high: float = 3.0
    hidden_dim: int = 256
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    lr_alpha: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    constraint_keys: list[str] = field(
        default_factory=lambda: [
            "equity_floor",
            "budget_envelope",
            "rawlsian_floor",
        ]
    )
    initial_lambda: float = 0.0
    lr_lambda: float = 1e-3
    max_lambda: float = 100.0


# --------------------------------------------------------------------------- #
class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int,
                 action_low: float, action_high: float):
        super().__init__()
        self.action_dim = action_dim
        self.action_low = float(action_low)
        self.action_high = float(action_high)
        self.body = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.head_mean = nn.Linear(hidden_dim, action_dim)
        self.head_log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, s: torch.Tensor):
        h = self.body(s)
        return self.head_mean(h), self.head_log_std(h).clamp(LOG_STD_MIN, LOG_STD_MAX)

    def sample(self, s: torch.Tensor):
        mean, log_std = self(s)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x = normal.rsample()
        a_tanh = torch.tanh(x)
        log_prob = normal.log_prob(x) - torch.log(1 - a_tanh.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        scale = (self.action_high - self.action_low) / 2.0
        offset = (self.action_high + self.action_low) / 2.0
        return a_tanh * scale + offset, log_prob

    @torch.no_grad()
    def sample_action(self, s: torch.Tensor, deterministic: bool = False):
        mean, log_std = self(s)
        if deterministic:
            a_tanh = torch.tanh(mean)
        else:
            std = log_std.exp()
            x = mean + std * torch.randn_like(mean)
            a_tanh = torch.tanh(x)
        scale = (self.action_high - self.action_low) / 2.0
        offset = (self.action_high + self.action_low) / 2.0
        return a_tanh * scale + offset


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, s: torch.Tensor, a: torch.Tensor):
        x = torch.cat([s, a], dim=-1)
        return self.q1(x), self.q2(x)


# --------------------------------------------------------------------------- #
class SACLagrangian:
    """Twin-critic SAC with auto-α temperature + per-constraint dual Lagrange."""

    def __init__(self, cfg: AgentCfg, device: torch.device | str = "cpu"):
        self.cfg = cfg
        self.device = torch.device(device)
        D, A = cfg.state_dim, cfg.action_dim
        self.actor = Actor(D, A, cfg.hidden_dim, cfg.action_low, cfg.action_high).to(self.device)
        self.critic = Critic(D, A, cfg.hidden_dim).to(self.device)
        self.target_critic = Critic(D, A, cfg.hidden_dim).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        for p in self.target_critic.parameters():
            p.requires_grad = False
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr_actor)
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr_critic)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.opt_alpha = torch.optim.Adam([self.log_alpha], lr=cfg.lr_alpha)
        self.target_entropy = -float(A)
        self.lambdas = {k: float(cfg.initial_lambda) for k in cfg.constraint_keys}
        self.last_metrics: dict = {}

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    # ---------------------------------------------------------------- #
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        if s.ndim == 1:
            s = s.unsqueeze(0)
        a = self.actor.sample_action(s, deterministic=deterministic).squeeze(0).cpu().numpy()
        return np.clip(a, self.cfg.action_low, self.cfg.action_high)

    def select_actions_batch(self, states: np.ndarray,
                              deterministic: bool = False) -> np.ndarray:
        s = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        a = self.actor.sample_action(s, deterministic=deterministic).cpu().numpy()
        return np.clip(a, self.cfg.action_low, self.cfg.action_high)

    # ---------------------------------------------------------------- #
    def update(self, batch: dict) -> dict:
        s = torch.as_tensor(batch["state"], dtype=torch.float32, device=self.device)
        a = torch.as_tensor(batch["action"], dtype=torch.float32, device=self.device)
        r = torch.as_tensor(batch["reward"], dtype=torch.float32, device=self.device)
        if r.ndim == 1:
            r = r.unsqueeze(-1)
        sp = torch.as_tensor(batch["next_state"], dtype=torch.float32, device=self.device)
        d = torch.as_tensor(batch["done"], dtype=torch.float32, device=self.device)
        if d.ndim == 1:
            d = d.unsqueeze(-1)

        with torch.no_grad():
            ap, lp = self.actor.sample(sp)
            tq1, tq2 = self.target_critic(sp, ap)
            target_q = torch.min(tq1, tq2) - self.alpha.detach() * lp
            y = r + (1.0 - d) * self.cfg.gamma * target_q
        q1, q2 = self.critic(s, a)
        loss_q = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        self.opt_c.zero_grad()
        loss_q.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 5.0)
        self.opt_c.step()

        a_pi, lp_pi = self.actor.sample(s)
        q1_pi, q2_pi = self.critic(s, a_pi)
        q_pi = torch.min(q1_pi, q2_pi)
        loss_actor = (self.alpha.detach() * lp_pi - q_pi).mean()
        self.opt_a.zero_grad()
        loss_actor.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 5.0)
        self.opt_a.step()

        loss_alpha = -(self.log_alpha * (lp_pi.detach() + self.target_entropy)).mean()
        self.opt_alpha.zero_grad()
        loss_alpha.backward()
        self.opt_alpha.step()

        for tp, p in zip(self.target_critic.parameters(), self.critic.parameters()):
            tp.data.mul_(1 - self.cfg.tau)
            tp.data.add_(self.cfg.tau * p.data)

        if "violations" in batch:
            self.update_lagrange(batch["violations"])

        m = {
            "loss_q": float(loss_q.item()),
            "loss_actor": float(loss_actor.item()),
            "alpha": float(self.alpha.detach().item()),
            "q_mean": float(q_pi.mean().item()),
            **{f"lambda_{k}": float(v) for k, v in self.lambdas.items()},
        }
        self.last_metrics = m
        return m

    def update_lagrange(self, violations: dict):
        for k in self.cfg.constraint_keys:
            v = violations.get(k)
            if v is None:
                continue
            mean_v = float(np.mean(np.asarray(v, dtype=np.float32)))
            new = self.lambdas[k] + self.cfg.lr_lambda * mean_v
            self.lambdas[k] = float(np.clip(new, 0.0, self.cfg.max_lambda))

    # ---------------------------------------------------------------- #
    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "lambdas": self.lambdas,
            "cfg": vars(self.cfg),
        }, path)

    def load(self, path: Path) -> bool:
        path = Path(path)
        if not path.exists():
            return False
        try:
            ck = torch.load(str(path), map_location=self.device, weights_only=False)
            self.actor.load_state_dict(ck["actor"])
            self.critic.load_state_dict(ck["critic"])
            self.target_critic.load_state_dict(ck["target_critic"])
            self.log_alpha.data.copy_(ck["log_alpha"].to(self.device))
            self.lambdas = dict(ck.get("lambdas", self.lambdas))
            return True
        except Exception as e:  # noqa: BLE001
            log.warning("agent load failed: %s", e)
            return False


# --------------------------------------------------------------------------- #
class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int, action_dim: int):
        self.capacity = int(capacity)
        self.idx = 0
        self.size = 0
        self.s = np.zeros((capacity, state_dim), dtype=np.float32)
        self.a = np.zeros((capacity, action_dim), dtype=np.float32)
        self.r = np.zeros((capacity,), dtype=np.float32)
        self.sp = np.zeros((capacity, state_dim), dtype=np.float32)
        self.d = np.zeros((capacity,), dtype=np.float32)

    def add(self, s, a, r, sp, d):
        i = self.idx
        self.s[i] = s
        self.a[i] = a
        self.r[i] = r
        self.sp[i] = sp
        self.d[i] = d
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def add_batch(self, states, actions, rewards, next_states, dones):
        n = states.shape[0]
        for k in range(n):
            self.add(states[k], actions[k], rewards[k], next_states[k], dones[k])

    def sample(self, batch_size: int) -> dict:
        idx = np.random.randint(0, self.size, size=min(batch_size, self.size))
        return {
            "state": self.s[idx],
            "action": self.a[idx],
            "reward": self.r[idx],
            "next_state": self.sp[idx],
            "done": self.d[idx],
        }
