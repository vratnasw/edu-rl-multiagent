"""State-level Stackelberg leader.

action_dim = 58, in [0.85, 1.15] (county budget multipliers).
Lagrangian penalties for:
  - no_county_below_floor : every county must get >= 0.85
  - gini_non_increase     : Gini of district outcomes cannot rise
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from agents.sac_base import AgentCfg, ReplayBuffer, SACLagrangian

log = logging.getLogger(__name__)


class StateAgent:
    def __init__(
        self,
        obs_dim: int,
        n_counties: int = 58,
        device: str = "cpu",
        buffer_size: int = 50_000,
    ):
        cfg = AgentCfg(
            state_dim=obs_dim,
            action_dim=n_counties,
            action_low=0.85,
            action_high=1.15,
            constraint_keys=["no_county_below_floor", "gini_non_increase"],
            hidden_dim=256,
        )
        self.cfg = cfg
        self.n_counties = n_counties
        self.sac = SACLagrangian(cfg, device=device)
        self.buf = ReplayBuffer(buffer_size, obs_dim, n_counties)

    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        return self.sac.select_action(obs, deterministic=deterministic).astype(np.float32)

    def store(self, s, a, r, sp, d):
        self.buf.add(s, a, r, sp, d)

    def train_step(self, batch_size: int) -> dict | None:
        if self.buf.size < batch_size:
            return None
        batch = self.buf.sample(batch_size)
        return self.sac.update(batch)

    def update_lagrange_from_env_info(self, info: dict, county_means: np.ndarray):
        """Push state-side violations into the dual variables."""
        # no-county-below-floor: how many counties under 0.85 average reward?
        floor_viol = float(np.sum(np.maximum(0.0, 0.85 - county_means)))
        gini_viol = float(max(0.0, info.get("delta_gini", 0.0)))
        self.sac.update_lagrange({
            "no_county_below_floor": np.array([floor_viol]),
            "gini_non_increase": np.array([gini_viol]),
        })

    def save(self, path: Path):
        self.sac.save(path)


# --------------------------------------------------------------------------- #
def run(fast: bool = False) -> dict:
    """Smoke run: instantiate, sample 1 action, return shape."""
    obs_dim = 128
    agent = StateAgent(obs_dim=obs_dim)
    obs = np.zeros(obs_dim, dtype=np.float32)
    a = agent.act(obs)
    return {
        "action_dim": a.shape[0],
        "action_low": float(a.min()),
        "action_high": float(a.max()),
        "obs_dim": obs_dim,
        "agent": "StateAgent",
    }


if __name__ == "__main__":
    import logging as _l
    _l.basicConfig(level=_l.INFO)
    print(run())
