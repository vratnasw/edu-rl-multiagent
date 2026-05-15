"""Phase B smoke tests.

Verifies:
  - All 6 Phase-B modules import cleanly.
  - env.step returns valid (obs, reward, term, trunc, info).
  - budget_feasibility_check returns bool.
  - DistrictCommModule forward output shape matches (n_districts, message_dim).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "config"))


def test_module_imports():
    """All six Phase-B modules import without error."""
    from src.environment import hierarchical_env  # noqa
    from src.agents import state_agent, county_agent, district_agent  # noqa
    from src.communication import message_passing  # noqa
    from src.analysis import stackelberg_analysis  # noqa


def test_config_loads():
    sys.path.insert(0, str(REPO / "src"))
    from utils.config_loader import load_config
    cfg = load_config()
    assert isinstance(cfg, dict)
    assert "data" in cfg


def _build_small_env(world_model=None):
    """Build a small synthetic env for unit tests (no R2)."""
    import torch
    from environment.hierarchical_env import HierarchicalEducationEnv
    from utils.loaders import PlaceholderTransition
    emb = torch.randn(20, 128)
    wm = world_model or PlaceholderTransition(state_dim=128, action_dim=5)
    return HierarchicalEducationEnv(
        district_embeddings=emb,
        world_model=wm,
        n_counties=5,
        horizon=3,
        seed=0,
    )


def test_env_step_returns_valid_tuple():
    env = _build_small_env()
    obs = env.reset(seed=0)
    rng = np.random.default_rng(0)
    actions = {
        "state": rng.uniform(0.85, 1.15, size=env.n_counties).astype(np.float32),
        "county": rng.uniform(-0.5, 0.5, size=(env.n_counties, 5)).astype(np.float32),
        "district": rng.uniform(-1, 1, size=(env.n_districts, 5)).astype(np.float32),
    }
    obs_, rew, term, trunc, info = env.step(actions)
    assert "state" in obs_ and "county" in obs_ and "district" in obs_
    assert isinstance(rew["state"], float)
    assert rew["county"].shape == (env.n_counties,)
    assert rew["district"].shape == (env.n_districts,)
    assert isinstance(term["state"], bool)
    assert isinstance(trunc["state"], bool)
    assert isinstance(info, dict)


def test_budget_feasibility_check_returns_bool():
    env = _build_small_env()
    a = np.ones(env.n_counties, dtype=np.float32)
    assert env.budget_feasibility_check(a) is True
    bad = np.full(env.n_counties, 1.5, dtype=np.float32)  # > 1.15
    assert env.budget_feasibility_check(bad) is False
    # Edge: wrong shape
    assert env.budget_feasibility_check(np.ones(2, dtype=np.float32)) is False


def test_message_passing_shape():
    import torch
    from communication.message_passing import DistrictCommModule, build_knn_edges
    n_d, in_dim, msg_dim = 30, 128, 32
    x = torch.randn(n_d, in_dim)
    ei = build_knn_edges(x, k=5)
    mod = DistrictCommModule(in_dim=in_dim, message_dim=msg_dim,
                              n_rounds=3, n_heads=4)
    out = mod(x, ei)
    assert out.shape == (n_d, msg_dim), f"got {out.shape}"


def test_agents_emit_valid_actions():
    """State/county/district agents return actions of correct shape + bounds."""
    from agents.state_agent import StateAgent
    from agents.county_agent import CountyAgentPool
    from agents.district_agent import DistrictAgentPool

    obs_dim = 128
    sa = StateAgent(obs_dim=obs_dim, n_counties=58)
    a_s = sa.act(np.zeros(obs_dim, dtype=np.float32))
    assert a_s.shape == (58,)
    assert (a_s >= 0.85).all() and (a_s <= 1.15).all()

    pool_c = CountyAgentPool(per_obs_dim=obs_dim + 1, n_counties=58)
    a_c = pool_c.act_all(np.zeros((58, obs_dim + 1), dtype=np.float32))
    assert a_c.shape == (58, 5)
    assert (a_c >= -1.0).all() and (a_c <= 1.0).all()

    pool_d = DistrictAgentPool(own_obs_dim=obs_dim + 5 + 1, message_dim=32,
                                  use_communication=True)
    own = np.zeros((10, obs_dim + 5 + 1), dtype=np.float32)
    msg = np.zeros((10, 32), dtype=np.float32)
    a_d = pool_d.act_all(own, msg)
    assert a_d.shape == (10, 5)
    assert (a_d >= -3.0).all() and (a_d <= 3.0).all()
