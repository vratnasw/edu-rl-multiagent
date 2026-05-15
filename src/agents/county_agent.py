"""Parameter-shared county-level follower agents.

action_dim = 5 (within-county allocation tanh-bounded).
58 instances share one network; the county index is concatenated as a
one-hot to the observation so the agent can specialize.

Within-county equity Lagrangian: penalizes within-county outcome
inequality.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from agents.sac_base import AgentCfg, ReplayBuffer, SACLagrangian

log = logging.getLogger(__name__)


class CountyAgentPool:
    """Single SAC network shared across all counties, with county-index
    concatenation."""

    def __init__(
        self,
        per_obs_dim: int,
        n_counties: int,
        action_dim: int = 5,
        device: str = "cpu",
        buffer_size: int = 100_000,
    ):
        # Augmented observation = per_county_obs || one_hot(county_index)
        aug_dim = per_obs_dim + n_counties
        cfg = AgentCfg(
            state_dim=aug_dim,
            action_dim=action_dim,
            action_low=-1.0,
            action_high=1.0,
            constraint_keys=["within_county_equity"],
            hidden_dim=256,
        )
        self.cfg = cfg
        self.n_counties = n_counties
        self.per_obs_dim = per_obs_dim
        self.action_dim = action_dim
        self.sac = SACLagrangian(cfg, device=device)
        self.buf = ReplayBuffer(buffer_size, aug_dim, action_dim)

    def _augment(self, obs: np.ndarray, county_idx: int | np.ndarray) -> np.ndarray:
        if obs.ndim == 1:
            oh = np.zeros(self.n_counties, dtype=np.float32)
            oh[int(county_idx)] = 1.0
            return np.concatenate([obs, oh])
        # batch path
        oh = np.zeros((obs.shape[0], self.n_counties), dtype=np.float32)
        oh[np.arange(obs.shape[0]), np.asarray(county_idx).astype(int)] = 1.0
        return np.concatenate([obs, oh], axis=-1)

    def act_all(self, obs_per_county: np.ndarray,
                deterministic: bool = False) -> np.ndarray:
        """obs_per_county: (n_counties, per_obs_dim). Returns (n_counties, action_dim)."""
        idx = np.arange(self.n_counties)
        aug = self._augment(obs_per_county, idx)
        return self.sac.select_actions_batch(aug, deterministic=deterministic).astype(np.float32)

    def store_all(self, obs, actions, rewards, next_obs, dones):
        idx = np.arange(self.n_counties)
        aug_s = self._augment(obs, idx)
        aug_sp = self._augment(next_obs, idx)
        self.buf.add_batch(aug_s, actions, rewards, aug_sp, dones)

    def train_step(self, batch_size: int) -> dict | None:
        if self.buf.size < batch_size:
            return None
        batch = self.buf.sample(batch_size)
        return self.sac.update(batch)

    def update_lagrange(self, county_to_district_outcomes: dict):
        """One Lagrangian dual update averaged over all counties."""
        viols = []
        for c, outs in county_to_district_outcomes.items():
            if len(outs) <= 1:
                viols.append(0.0)
                continue
            std = float(np.std(outs))
            viols.append(std)
        self.sac.update_lagrange({"within_county_equity": np.array(viols)})

    def save(self, path: Path):
        self.sac.save(path)


# --------------------------------------------------------------------------- #
def run(fast: bool = False) -> dict:
    n_counties = 58
    per_obs_dim = 128 + 1
    pool = CountyAgentPool(per_obs_dim=per_obs_dim, n_counties=n_counties)
    obs = np.zeros((n_counties, per_obs_dim), dtype=np.float32)
    a = pool.act_all(obs)
    return {
        "action_shape": list(a.shape),
        "n_counties": n_counties,
        "per_obs_dim": per_obs_dim,
        "agent": "CountyAgentPool",
    }


if __name__ == "__main__":
    print(run())
