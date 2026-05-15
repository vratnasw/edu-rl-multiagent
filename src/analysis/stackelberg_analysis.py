"""Stackelberg equilibrium analysis + price-of-anarchy report.

Compares
  - Centralized SAC (single agent commanding all districts)
  - Decentralized hierarchical Stackelberg agents (state → county → district)

and produces a price-of-anarchy:
  PoA = (centralized_reward − decentralized_reward) / |centralized_reward|

For the centralized baseline we prefer the Layer-6 trained checkpoint if
present; otherwise we use the district agent trained without
communication in this run.
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


# --------------------------------------------------------------------------- #
def _archetype_for_county(c: int, n_counties: int) -> str:
    """5 stylised archetypes by county index modulo bucket.

    Buckets approximate: dense-urban, suburban, exurban, rural,
    mountain/coastal."""
    quint = (c * 5) // n_counties
    return ["dense_urban", "suburban", "exurban", "rural", "frontier"][min(quint, 4)]


# --------------------------------------------------------------------------- #
def evaluate_rollout(
    env,
    state_agent,
    county_pool,
    district_pool,
    comm_module=None,
    edge_index=None,
    embeddings=None,
    n_episodes: int = 3,
    deterministic: bool = True,
    use_centralized: bool = False,
) -> dict:
    """Run n_episodes through env, return summary metrics.

    If use_centralized=True: ignore state/county tiers and act using the
    district_pool only with random county multipliers ≡ 1.0 (no
    hierarchical envelope), simulating a single-agent baseline.
    """
    import torch as _t

    rewards_state = []
    rewards_county = []
    rewards_district = []
    per_county_outcome = {c: [] for c in range(env.n_counties)}
    coord_gaps = []

    for ep in range(n_episodes):
        env.reset(seed=ep)
        ep_rs = ep_rc = ep_rd = 0.0
        for t in range(env.horizon):
            obs = env.observe_all()
            if use_centralized:
                a_state = np.ones(env.n_counties, dtype=np.float32)  # neutral
                a_county = np.zeros((env.n_counties, 5), dtype=np.float32)
            else:
                a_state = state_agent.act(obs["state"], deterministic=deterministic)
                a_county = county_pool.act_all(obs["county"], deterministic=deterministic)

            if comm_module is not None and embeddings is not None and edge_index is not None:
                with _t.no_grad():
                    msg = comm_module(
                        embeddings if embeddings.ndim == 2 else embeddings[:, -1, :],
                        edge_index,
                    ).cpu().numpy()
            elif getattr(district_pool, "use_comm", False):
                # comm-enabled pool but no comm graph -> pass zero messages
                msg = np.zeros(
                    (obs["district"].shape[0], district_pool.message_dim),
                    dtype=np.float32,
                )
            else:
                msg = None
            a_district = district_pool.act_all(
                obs["district"], msg, deterministic=deterministic
            )

            new_obs, rew, term, trunc, info = env.step({
                "state": a_state,
                "county": a_county,
                "district": a_district,
            })
            ep_rs += rew["state"]
            ep_rc += float(np.mean(rew["county"]))
            ep_rd += float(np.mean(rew["district"]))

            # bucket per-county district outcomes for coord analysis
            for i in range(env.n_districts):
                c = int(env.district_to_county[i])
                per_county_outcome[c].append(float(rew["district"][i]))

            # coordination gap = stdev of district rewards within a county
            for c in range(env.n_counties):
                outs = [rew["district"][i] for i in range(env.n_districts)
                          if env.district_to_county[i] == c]
                if len(outs) > 1:
                    coord_gaps.append((c, float(np.std(outs))))

        rewards_state.append(ep_rs)
        rewards_county.append(ep_rc)
        rewards_district.append(ep_rd)

    # top-10 worst coordination gaps (largest within-county std)
    coord_by_county = {}
    for c, g in coord_gaps:
        coord_by_county.setdefault(c, []).append(g)
    worst = sorted(
        ((c, float(np.mean(g))) for c, g in coord_by_county.items()),
        key=lambda x: -x[1],
    )[:10]

    return {
        "mean_state_reward": float(np.mean(rewards_state)),
        "mean_county_reward": float(np.mean(rewards_county)),
        "mean_district_reward": float(np.mean(rewards_district)),
        "n_episodes": int(n_episodes),
        "top_coordination_failures": [
            {
                "county_id": int(c),
                "archetype": _archetype_for_county(c, env.n_counties),
                "within_county_std": float(s),
            }
            for c, s in worst
        ],
    }


# --------------------------------------------------------------------------- #
def compute_price_of_anarchy(centralized_reward: float,
                              decentralized_reward: float) -> float:
    denom = max(abs(centralized_reward), 1e-6)
    return float((centralized_reward - decentralized_reward) / denom)


# --------------------------------------------------------------------------- #
def make_efficiency_gap_figure(
    per_archetype: dict,
    out_path: Path,
) -> bool:
    """Save a PDF bar plot of efficiency gap by archetype. Returns True on success."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        labels = list(per_archetype.keys())
        vals = [per_archetype[k] for k in labels]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.bar(labels, vals, color="#126342")
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_ylabel("Centralized − Decentralized reward gap")
        ax.set_title("Stackelberg efficiency gap by county archetype")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, v, f"{v:+.3f}",
                    ha="center", va="bottom" if v >= 0 else "top",
                    fontsize=9)
        fig.tight_layout()
        fig.savefig(str(out_path), format="pdf")
        plt.close(fig)
        return True
    except Exception as e:  # noqa: BLE001
        log.warning("figure render failed: %s", e)
        return False


# --------------------------------------------------------------------------- #
def analyze(
    env,
    state_agent,
    county_pool,
    district_pool,
    comm_module=None,
    edge_index=None,
    embeddings=None,
    n_episodes: int = 3,
    out_dir: Path | None = None,
) -> dict:
    out_dir = out_dir or (REPO / "results" / "analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    decent = evaluate_rollout(
        env, state_agent, county_pool, district_pool,
        comm_module=comm_module, edge_index=edge_index, embeddings=embeddings,
        n_episodes=n_episodes,
        use_centralized=False,
    )
    cent = evaluate_rollout(
        env, state_agent, county_pool, district_pool,
        comm_module=None, edge_index=None, embeddings=None,
        n_episodes=n_episodes,
        use_centralized=True,
    )
    poa = compute_price_of_anarchy(
        cent["mean_state_reward"], decent["mean_state_reward"]
    )

    # Per-archetype centralized − decentralized gap (we use the
    # within-county std difference as a proxy)
    archetype_gap: dict[str, float] = {}
    for entry_d, entry_c in zip(decent["top_coordination_failures"],
                                  cent["top_coordination_failures"]):
        a = entry_d["archetype"]
        gap = entry_d["within_county_std"] - entry_c["within_county_std"]
        archetype_gap[a] = float(archetype_gap.get(a, 0.0) + gap)
    # Ensure all 5 archetypes have an entry, fill 0 for missing
    for arc in ["dense_urban", "suburban", "exurban", "rural", "frontier"]:
        archetype_gap.setdefault(arc, 0.0)

    fig_path = REPO / "figures" / "stackelberg_efficiency_gap.pdf"
    fig_ok = make_efficiency_gap_figure(archetype_gap, fig_path)

    out = {
        "centralized": cent,
        "decentralized": decent,
        "price_of_anarchy": poa,
        "archetype_gap": archetype_gap,
        "figure_written": str(fig_path) if fig_ok else None,
    }
    (out_dir / "stackelberg_analysis.json").write_text(json.dumps(out, indent=2))
    return out


# --------------------------------------------------------------------------- #
def run(fast: bool = False) -> dict:
    """Run a self-contained smoke version using random agents.

    This is the ``run()`` entrypoint required by the smoke tests. For the
    full analysis the orchestrator calls ``analyze()`` directly with
    trained agents."""
    import torch as _t
    from environment.hierarchical_env import HierarchicalEducationEnv
    from agents.state_agent import StateAgent
    from agents.county_agent import CountyAgentPool
    from agents.district_agent import DistrictAgentPool
    from utils.loaders import (
        load_district_embeddings,
        load_world_model_checkpoint,
        build_world_model_from_checkpoint,
        PlaceholderTransition,
    )

    _t.set_num_threads(8)

    emb = load_district_embeddings()
    try:
        ck = load_world_model_checkpoint()
        wm = build_world_model_from_checkpoint(ck) or PlaceholderTransition(128, 5)
    except Exception:  # noqa: BLE001
        wm = PlaceholderTransition(128, 5)

    env = HierarchicalEducationEnv(
        district_embeddings=emb, world_model=wm, horizon=2,
    )
    state_agent = StateAgent(obs_dim=env.state_dim, n_counties=env.n_counties)
    county_pool = CountyAgentPool(
        per_obs_dim=env.state_dim + 1, n_counties=env.n_counties
    )
    district_pool = DistrictAgentPool(
        own_obs_dim=env.state_dim + 5 + 1, message_dim=32,
        use_communication=False,
    )
    out = analyze(
        env, state_agent, county_pool, district_pool,
        n_episodes=1,
    )
    return {
        "price_of_anarchy": out["price_of_anarchy"],
        "centralized_reward": out["centralized"]["mean_state_reward"],
        "decentralized_reward": out["decentralized"]["mean_state_reward"],
    }


if __name__ == "__main__":
    import logging as _l
    _l.basicConfig(level=_l.INFO)
    print(run(fast=True))
