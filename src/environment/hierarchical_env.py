"""Hierarchical multi-agent gymnasium/PettingZoo environment.

3 agent tiers:
  - 1 state_agent   : action_dim = 58 (county budget multipliers ∈ [0.85, 1.15])
  - 58 county_agents: action_dim = 5  (within-county allocation)
  - N district_agents: action_dim = 5 (per-district levers, ∈ [-3, 3])

Driven by the Layer-5 world model ensemble (state_dim=128, action_dim=5).
Each district has its own 128-dim "world state". After every agent in
all 3 tiers has acted, the world model rolls the district states one
step forward and we compute rewards.

Rewards
  district_r_i = ΔCAASPP_proxy_i  −  Σ_k λ_k · violation_k_i
                 (CAASPP proxy = +Δ of the first state dim, treated as
                  ELA-met-pct proxy)
  county_r_c   = Σ_{i ∈ c} w_i · district_r_i / Σ w_i   (population weighted)
  state_r      = Σ_c w_c · county_r_c / Σ w_c
                 − 0.5 · ΔGini(district_outcomes)        (state fairness)
                 − Σ_c max(0, 0.85 − budget_c) · 5.0      (no-county-<85%)

Observations
  - district sees: own_embedding[D=128] || county_action[5] || state_action_slice[1]
  - county sees:   mean(district_embs_in_county)[D=128] || state_action_slice[1]
  - state sees:    pooled embedding (mean over districts) [D=128]

budget_feasibility_check
  Given a state action a_state[58] and county actions {a_c}, returns True
  iff sum(a_state) ≤ N_counties · 1.0 (mean target = 1.0) AND every
  county respects its allocated budget envelope.

This implements the AECEnv interface from PettingZoo but we keep it
self-contained and don't rely on `from pettingzoo.utils.env import AECEnv`
for the cycle iterator — the orchestrator drives it directly.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch

log = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[2]

# --- Constants per spec ---------------------------------------------------- #
STATE_ACTION_DIM = 58
COUNTY_ACTION_DIM = 5
DISTRICT_ACTION_DIM = 5
BUDGET_LOW, BUDGET_HIGH = 0.85, 1.15
DISTRICT_ACTION_LOW, DISTRICT_ACTION_HIGH = -3.0, 3.0


def _gini(x: np.ndarray) -> float:
    """Standard Gini on non-negative array. If x has negatives, shift first."""
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size == 0:
        return 0.0
    if x.min() < 0:
        x = x - x.min()
    if x.sum() == 0:
        return 0.0
    x = np.sort(x)
    n = x.size
    cumx = np.cumsum(x)
    return float((n + 1 - 2 * (cumx.sum() / cumx[-1])) / n)


# --------------------------------------------------------------------------- #
class HierarchicalEducationEnv:
    """Hierarchical state→county→district education env."""

    metadata = {"name": "edu-hierarchical-v0"}

    def __init__(
        self,
        district_embeddings: torch.Tensor,
        world_model: Optional[torch.nn.Module] = None,
        edge_index: Optional[torch.Tensor] = None,
        n_counties: int = STATE_ACTION_DIM,
        horizon: int = 8,
        district_populations: Optional[np.ndarray] = None,
        device: str = "cpu",
        seed: int = 0,
    ):
        if district_embeddings.ndim == 3:
            x = district_embeddings[:, -1, :]
        else:
            x = district_embeddings
        self.device = torch.device(device)
        self.embeddings = x.detach().clone().float().to(self.device)
        self.n_districts = int(self.embeddings.shape[0])
        self.state_dim = int(self.embeddings.shape[1])
        self.n_counties = int(n_counties)
        self.world_model = world_model
        self.edge_index = edge_index
        self.horizon = int(horizon)

        # Assign each district to a county via stride partition.
        # Yields counties of size ~ceil(N/58).
        self.district_to_county = np.array(
            [i % self.n_counties for i in range(self.n_districts)], dtype=np.int64
        )

        # District population weights (uniform if not supplied)
        if district_populations is None:
            self.district_pop = np.ones(self.n_districts, dtype=np.float64)
        else:
            self.district_pop = np.asarray(district_populations, dtype=np.float64)
            assert self.district_pop.shape[0] == self.n_districts

        # County weight = sum of populations of its districts
        self.county_pop = np.zeros(self.n_counties, dtype=np.float64)
        for i, c in enumerate(self.district_to_county):
            self.county_pop[c] += self.district_pop[i]

        # Lagrangian (per district) — equity violations penalize district reward
        self.lambda_equity = 0.5
        self.lambda_floor = 0.5

        # Internal RNG
        self._rng = np.random.default_rng(seed)

        # Current world states for each district (initial = embeddings).
        # World model expects state_dim=128, action_dim=5 — match by trimming/padding.
        self.world_state: torch.Tensor = self._init_world_state()

        # Tracking
        self.step_count = 0
        self._initial_outcome = self._outcome_scalar(self.world_state)

        # Last-step bookkeeping for observations
        self.last_state_action: np.ndarray = np.ones(self.n_counties, dtype=np.float32)
        self.last_county_actions: np.ndarray = np.zeros(
            (self.n_counties, COUNTY_ACTION_DIM), dtype=np.float32
        )

    # ------------------------------------------------------------------ #
    def _init_world_state(self) -> torch.Tensor:
        """Project embedding to world-model state_dim. The world model expects
        state_dim=128 — embeddings happen to be 128-d already."""
        wm_state_dim = self._wm_state_dim()
        if self.state_dim == wm_state_dim:
            return self.embeddings.clone()
        # zero-pad or truncate to align dimensions
        out = torch.zeros((self.n_districts, wm_state_dim), device=self.device)
        d = min(self.state_dim, wm_state_dim)
        out[:, :d] = self.embeddings[:, :d]
        return out

    def _wm_state_dim(self) -> int:
        if self.world_model is None:
            return self.state_dim
        if hasattr(self.world_model, "members"):
            # WorldModelEnsemble — peek at first member's MemberConfig
            try:
                return int(self.world_model.members[0].cfg.state_dim)
            except Exception:
                return self.state_dim
        if hasattr(self.world_model, "state_dim"):
            return int(self.world_model.state_dim)
        return self.state_dim

    def _outcome_scalar(self, ws: torch.Tensor) -> np.ndarray:
        """Scalar proxy outcome per district from world state. We use the
        first state-dim as a stand-in for ELA-met-pct."""
        return ws[:, 0].detach().cpu().numpy()

    # ------------------------------------------------------------------ #
    # PettingZoo-style API (we keep it explicit; orchestrator drives the cycle)
    # ------------------------------------------------------------------ #
    def reset(self, seed: int | None = None) -> dict:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.world_state = self._init_world_state()
        self.step_count = 0
        self._initial_outcome = self._outcome_scalar(self.world_state)
        self.last_state_action = np.ones(self.n_counties, dtype=np.float32)
        self.last_county_actions = np.zeros(
            (self.n_counties, COUNTY_ACTION_DIM), dtype=np.float32
        )
        return self.observe_all()

    def observe_state(self) -> np.ndarray:
        """state agent: mean-pool of district embeddings."""
        return self.embeddings.mean(dim=0).cpu().numpy()

    def observe_county(self, c: int) -> np.ndarray:
        """county agent c: mean of its district embeddings || state_action[c]."""
        mask = self.district_to_county == c
        if mask.any():
            mean = self.embeddings[mask].mean(dim=0).cpu().numpy()
        else:
            mean = np.zeros(self.state_dim, dtype=np.float32)
        return np.concatenate(
            [mean.astype(np.float32),
             np.array([self.last_state_action[c]], dtype=np.float32)]
        )

    def observe_district(self, i: int) -> np.ndarray:
        """district agent i: own emb || county_action[parent_c] || state_action[parent_c]."""
        c = int(self.district_to_county[i])
        own = self.embeddings[i].cpu().numpy().astype(np.float32)
        cty = self.last_county_actions[c].astype(np.float32)
        st = np.array([self.last_state_action[c]], dtype=np.float32)
        return np.concatenate([own, cty, st])

    def observe_all(self) -> dict:
        return {
            "state": self.observe_state(),
            "county": np.stack(
                [self.observe_county(c) for c in range(self.n_counties)], axis=0
            ),
            "district": np.stack(
                [self.observe_district(i) for i in range(self.n_districts)], axis=0
            ),
        }

    # ------------------------------------------------------------------ #
    def budget_feasibility_check(
        self,
        state_action: np.ndarray,
        county_actions: Optional[np.ndarray] = None,
    ) -> bool:
        """Check that the state's budget envelope is feasible.

        Rules:
          1. Each county multiplier ∈ [0.85, 1.15]
          2. Mean budget across all counties ≤ 1.0 (no aggregate net-positive)
          3. (If county_actions given) county L2 norm respects budget envelope
        """
        if state_action.shape[-1] != self.n_counties:
            return False
        if np.any(state_action < BUDGET_LOW - 1e-6) or np.any(
            state_action > BUDGET_HIGH + 1e-6
        ):
            return False
        if float(state_action.mean()) > 1.0 + 1e-3:
            return False
        if county_actions is not None:
            # county L2 effort should scale with its allocated budget
            norms = np.linalg.norm(county_actions, axis=-1)
            limits = (state_action * 5.0)  # max 5σ effort per unit budget
            if np.any(norms > limits + 1e-3):
                return False
        return True

    # ------------------------------------------------------------------ #
    def step(
        self,
        actions: dict,
    ) -> tuple[dict, dict, dict, dict, dict]:
        """One full hierarchical step. `actions` keys:
          - 'state':    (58,) array  in [0.85, 1.15]
          - 'county':   (58, 5)
          - 'district': (N, 5)
        Returns (obs, rewards, terminations, truncations, infos)."""
        a_state = np.asarray(actions["state"], dtype=np.float32)
        a_county = np.asarray(actions["county"], dtype=np.float32)
        a_district = np.asarray(actions["district"], dtype=np.float32)

        # Clip to bounds
        a_state = np.clip(a_state, BUDGET_LOW, BUDGET_HIGH)
        a_district = np.clip(a_district, DISTRICT_ACTION_LOW, DISTRICT_ACTION_HIGH)

        # --- Apply hierarchical envelope --------------------------------- #
        # Within-county budget envelope: district action magnitudes scaled
        # by the county's budget multiplier × the county's own action vector.
        # County action acts as a per-channel multiplier ∈ tanh space.
        county_mult = np.tanh(a_county)  # (58, 5) ∈ (-1, 1)
        # For each district, scale its action by county multiplier × state envelope.
        per_d_state = a_state[self.district_to_county]  # (N,)
        per_d_county = county_mult[self.district_to_county]  # (N, 5)
        # Effective action: a_d * per_d_state * (1 + per_d_county)
        eff_action = (
            a_district
            * per_d_state[:, None]
            * (1.0 + 0.2 * per_d_county)
        )
        eff_action = np.clip(eff_action, DISTRICT_ACTION_LOW, DISTRICT_ACTION_HIGH)

        # --- Roll world model 1 step ------------------------------------- #
        wm_action_dim = COUNTY_ACTION_DIM  # 5
        if eff_action.shape[1] != wm_action_dim:
            # pad/truncate
            pad = np.zeros((self.n_districts, wm_action_dim), dtype=np.float32)
            d = min(eff_action.shape[1], wm_action_dim)
            pad[:, :d] = eff_action[:, :d]
            eff_action = pad

        a_t = torch.from_numpy(eff_action).float().to(self.device)
        with torch.no_grad():
            if self.world_model is not None:
                try:
                    next_ws, epi, ale = self.world_model.predict(self.world_state, a_t)
                except Exception as e:  # noqa: BLE001
                    log.warning("world model predict failed (%s) — using identity step", e)
                    next_ws = self.world_state + 0.01 * a_t @ torch.randn(
                        wm_action_dim, self.world_state.shape[1], device=self.device
                    )
                    epi = torch.zeros_like(next_ws)
                    ale = torch.zeros_like(next_ws)
            else:
                next_ws = self.world_state + 0.01 * a_t @ torch.randn(
                    wm_action_dim, self.world_state.shape[1], device=self.device
                )
                epi = torch.zeros_like(next_ws)
                ale = torch.zeros_like(next_ws)

        # --- Compute district rewards ------------------------------------ #
        old_outcome = self._outcome_scalar(self.world_state)
        new_outcome = self._outcome_scalar(next_ws)
        delta = new_outcome - old_outcome  # (N,)

        # Equity violation: district outcome falling >1σ below the state mean.
        state_mean = float(new_outcome.mean())
        state_std = float(new_outcome.std()) + 1e-6
        floor = state_mean - 1.0 * state_std
        equity_violations = np.maximum(0.0, floor - new_outcome)  # (N,)

        # Action-magnitude penalty (Lagrangian on excess effort)
        effort = np.linalg.norm(eff_action, axis=-1)  # (N,)
        floor_violations = np.maximum(0.0, effort - 5.0)  # (N,)

        district_rewards = (
            delta
            - self.lambda_equity * equity_violations
            - self.lambda_floor * floor_violations
        )  # (N,)

        # --- County rewards: population-weighted mean -------------------- #
        county_rewards = np.zeros(self.n_counties, dtype=np.float32)
        for c in range(self.n_counties):
            mask = self.district_to_county == c
            if not mask.any():
                continue
            w = self.district_pop[mask]
            r = district_rewards[mask]
            county_rewards[c] = float((w * r).sum() / max(w.sum(), 1e-9))

        # --- State reward: population-weighted county mean − Gini change - #
        state_aggregate = float(
            (self.county_pop * county_rewards).sum()
            / max(self.county_pop.sum(), 1e-9)
        )
        gini_now = _gini(new_outcome)
        gini_then = _gini(old_outcome)
        delta_gini = gini_now - gini_then
        # County-floor penalty (no-county-<0.85)
        floor_pen = float(np.sum(np.maximum(0.0, BUDGET_LOW - a_state)) * 5.0)
        state_reward = state_aggregate - 0.5 * delta_gini - floor_pen

        # --- Advance state ---------------------------------------------- #
        self.world_state = next_ws.detach()
        self.step_count += 1
        self.last_state_action = a_state.copy()
        self.last_county_actions = a_county.copy()

        # Termination + truncation
        done = self.step_count >= self.horizon
        term = {"state": done, "county": done, "district": done}
        trunc = {"state": False, "county": False, "district": False}

        rewards = {
            "state": float(state_reward),
            "county": county_rewards.astype(np.float32),
            "district": district_rewards.astype(np.float32),
        }
        info = {
            "epistemic_mean": float(epi.mean().item()),
            "aleatoric_mean": float(ale.mean().item()),
            "delta_outcome_mean": float(delta.mean()),
            "gini_now": gini_now,
            "delta_gini": delta_gini,
            "n_equity_violations": int((equity_violations > 0).sum()),
            "budget_feasible": self.budget_feasibility_check(a_state, a_county),
        }

        return self.observe_all(), rewards, term, trunc, info

    # ------------------------------------------------------------------ #
    def render(self, mode: str = "human") -> str:
        out = self._outcome_scalar(self.world_state)
        return (
            f"step={self.step_count}/{self.horizon} "
            f"mean_outcome={out.mean():.4f} "
            f"std={out.std():.4f} gini={_gini(out):.4f} "
            f"N_districts={self.n_districts} N_counties={self.n_counties}"
        )

    # ------------------------------------------------------------------ #
    def env_spec(self) -> dict:
        return {
            "n_agents_per_level": {
                "state": 1,
                "county": self.n_counties,
                "district": self.n_districts,
            },
            "action_spaces": {
                "state": {
                    "shape": [self.n_counties],
                    "low": BUDGET_LOW,
                    "high": BUDGET_HIGH,
                    "dtype": "float32",
                },
                "county": {
                    "shape": [self.n_counties, COUNTY_ACTION_DIM],
                    "low": -1.0,
                    "high": 1.0,
                    "dtype": "float32",
                },
                "district": {
                    "shape": [self.n_districts, DISTRICT_ACTION_DIM],
                    "low": DISTRICT_ACTION_LOW,
                    "high": DISTRICT_ACTION_HIGH,
                    "dtype": "float32",
                },
            },
            "observation_spaces": {
                "state": [self.state_dim],
                "county": [self.state_dim + 1],
                "district": [self.state_dim + COUNTY_ACTION_DIM + 1],
            },
            "reward_structure": {
                "district": "delta_outcome - lambda_equity*violation - lambda_floor*effort_excess",
                "county": "population-weighted mean of district rewards",
                "state": "population-weighted county mean - 0.5*Δgini - 5*floor_violation",
            },
            "horizon": self.horizon,
            "world_model": (
                "WorldModelEnsemble" if self.world_model is not None
                and hasattr(self.world_model, "members")
                else type(self.world_model).__name__ if self.world_model is not None
                else "identity"
            ),
        }


# --------------------------------------------------------------------------- #
def save_env_spec(env: HierarchicalEducationEnv, out_path: Path) -> dict:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    spec = env.env_spec()
    out_path.write_text(json.dumps(spec, indent=2))
    return spec


def run(fast: bool = False) -> dict:
    """Build a tiny env from real embeddings + world model, run one step,
    write the env_spec.json."""
    import torch as _t
    from utils.loaders import (
        load_world_model_checkpoint,
        load_district_embeddings,
        build_world_model_from_checkpoint,
        PlaceholderTransition,
    )

    _t.set_num_threads(8)

    emb = load_district_embeddings()
    try:
        ck = load_world_model_checkpoint()
        wm = build_world_model_from_checkpoint(ck) or PlaceholderTransition(
            state_dim=int(ck["arch"]["state_dim"]),
            action_dim=int(ck["arch"]["action_dim"]),
        )
        wm_source = "ensemble" if hasattr(wm, "members") else "placeholder"
    except Exception as e:  # noqa: BLE001
        log.warning("world model unavailable (%s) — placeholder", e)
        wm = PlaceholderTransition(state_dim=128, action_dim=5)
        wm_source = "placeholder"

    env = HierarchicalEducationEnv(
        district_embeddings=emb,
        world_model=wm,
        horizon=4 if fast else 8,
    )
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    actions = {
        "state": rng.uniform(BUDGET_LOW, BUDGET_HIGH, size=env.n_counties).astype(np.float32),
        "county": rng.uniform(-0.5, 0.5, size=(env.n_counties, COUNTY_ACTION_DIM)).astype(np.float32),
        "district": rng.uniform(-1, 1, size=(env.n_districts, DISTRICT_ACTION_DIM)).astype(np.float32),
    }
    obs, rew, term, trunc, info = env.step(actions)
    spec = save_env_spec(
        env, REPO / "results" / "environment" / "env_spec.json"
    )
    return {
        "n_districts": env.n_districts,
        "n_counties": env.n_counties,
        "world_model": wm_source,
        "step_state_reward": rew["state"],
        "step_county_reward_mean": float(np.mean(rew["county"])),
        "step_district_reward_mean": float(np.mean(rew["district"])),
        "spec_keys": list(spec.keys()),
    }


if __name__ == "__main__":
    import logging as _l
    _l.basicConfig(level=_l.INFO)
    print(run(fast=True))
