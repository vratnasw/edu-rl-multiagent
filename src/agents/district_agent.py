"""Parameter-shared district-level follower agents.

action_dim = 5, in [-3, 3] (matches Layer 6 spec).
N instances share a single network; observation is augmented with an
aggregated message from the communication module (message_dim=32) and
a per-district population/effectiveness scalar.

Equity Lagrangian: penalizes negative outcome change for the district
itself (Rawlsian floor).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from agents.sac_base import AgentCfg, ReplayBuffer, SACLagrangian

log = logging.getLogger(__name__)


class DistrictAgentPool:
    """Single SAC for all districts. Observation =
    own_state || message_aggregated || county_action[5] || state_envelope[1].
    """

    def __init__(
        self,
        own_obs_dim: int,
        message_dim: int,
        action_dim: int = 5,
        device: str = "cpu",
        buffer_size: int = 200_000,
        use_communication: bool = True,
    ):
        self.use_comm = bool(use_communication)
        aug_dim = own_obs_dim + (message_dim if self.use_comm else 0)
        cfg = AgentCfg(
            state_dim=aug_dim,
            action_dim=action_dim,
            action_low=-3.0,
            action_high=3.0,
            constraint_keys=["rawlsian_floor", "equity_floor"],
            hidden_dim=256,
        )
        self.cfg = cfg
        self.own_obs_dim = own_obs_dim
        self.message_dim = message_dim
        self.action_dim = action_dim
        self.sac = SACLagrangian(cfg, device=device)
        self.buf = ReplayBuffer(buffer_size, aug_dim, action_dim)

    def _augment(self, own_obs: np.ndarray, messages: np.ndarray | None) -> np.ndarray:
        if not self.use_comm or messages is None:
            return own_obs
        return np.concatenate([own_obs, messages], axis=-1)

    def act_all(
        self,
        own_obs: np.ndarray,
        messages: np.ndarray | None = None,
        deterministic: bool = False,
    ) -> np.ndarray:
        aug = self._augment(own_obs, messages)
        return self.sac.select_actions_batch(aug, deterministic=deterministic).astype(np.float32)

    def store_all(
        self,
        own_obs,
        messages,
        actions,
        rewards,
        next_own_obs,
        next_messages,
        dones,
    ):
        aug_s = self._augment(own_obs, messages)
        aug_sp = self._augment(next_own_obs, next_messages)
        self.buf.add_batch(aug_s, actions, rewards, aug_sp, dones)

    def train_step(self, batch_size: int) -> dict | None:
        if self.buf.size < batch_size:
            return None
        batch = self.buf.sample(batch_size)
        return self.sac.update(batch)

    def update_lagrange(self, district_rewards: np.ndarray,
                        equity_violations: np.ndarray):
        # Rawlsian floor violation: negative reward → punish
        rawl = np.maximum(0.0, -district_rewards)
        self.sac.update_lagrange({
            "rawlsian_floor": rawl,
            "equity_floor": equity_violations,
        })

    def save(self, path: Path):
        self.sac.save(path)


# --------------------------------------------------------------------------- #
def run(fast: bool = False) -> dict:
    own_obs_dim = 128 + 5 + 1  # own emb + county_action + state_envelope
    message_dim = 32
    n_districts = 50
    pool = DistrictAgentPool(own_obs_dim=own_obs_dim, message_dim=message_dim)
    own = np.zeros((n_districts, own_obs_dim), dtype=np.float32)
    msg = np.zeros((n_districts, message_dim), dtype=np.float32)
    a = pool.act_all(own, msg)
    return {
        "action_shape": list(a.shape),
        "own_obs_dim": own_obs_dim,
        "message_dim": message_dim,
        "n_districts": n_districts,
        "agent": "DistrictAgentPool",
    }


if __name__ == "__main__":
    print(run())
