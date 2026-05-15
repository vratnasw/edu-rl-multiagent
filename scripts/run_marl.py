"""Phase B orchestrator: train + analyze the hierarchical Stackelberg
multi-agent system, write `results/marl_paper_summary.json`.

Pipeline:
  1. Load world model + embeddings from R2 (lazy via utils.loaders).
  2. Build env, agents (state, county pool, district pool), comm module.
  3. Train for N env steps (alternating data collection + SAC updates).
  4. Run Stackelberg analysis (with-comm vs without-comm).
  5. Write summary JSON.

Flags
  --fast : 1000 env steps, batch=64, 50 SAC epochs, 1 analysis episode.
  --full : 1e6 env steps, batch=256, full per-config.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "config"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s :: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
def _load_env_dot_files() -> None:
    """Load .env from the canonical locations so R2 creds are visible."""
    for cand in (
        REPO / ".env",
        REPO.parent / ".env",
        REPO.parent / "edu-data-pipeline" / ".env",
    ):
        if cand.exists():
            try:
                from dotenv import load_dotenv  # type: ignore
                load_dotenv(cand, override=False)
            except ImportError:
                for line in cand.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, _, v = line.partition("=")
                    import os
                    os.environ.setdefault(k.strip(), v.strip())
            return


# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true",
                        help="Tiny budget (1000 steps).")
    parser.add_argument("--full", action="store_true",
                        help="Per-config budget (1e6 steps).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    _load_env_dot_files()
    torch.set_num_threads(8)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.fast:
        n_env_steps = 1000
        sac_epochs = 50
        batch_size = 64
        n_eval_episodes = 1
    elif args.full:
        n_env_steps = 1_000_000
        sac_epochs = 1000
        batch_size = 256
        n_eval_episodes = 5
    else:
        n_env_steps = 5000
        sac_epochs = 200
        batch_size = 128
        n_eval_episodes = 2

    # --- imports (after sys.path setup) ----------------------------- #
    from utils.loaders import (
        load_world_model_checkpoint,
        load_district_embeddings,
        load_spatial_edge_index,
        build_world_model_from_checkpoint,
        PlaceholderTransition,
    )
    from environment.hierarchical_env import HierarchicalEducationEnv
    from communication.message_passing import (
        DistrictCommModule,
        build_knn_edges,
        merge_edges,
        save_protocol_spec,
    )
    from agents.state_agent import StateAgent
    from agents.county_agent import CountyAgentPool
    from agents.district_agent import DistrictAgentPool
    from analysis.stackelberg_analysis import analyze

    t0 = time.time()
    log.info("=== Phase B orchestrator: fast=%s ===", args.fast)

    # ------------------------------------------------------------ #
    # 1. Load real artifacts
    # ------------------------------------------------------------ #
    emb = load_district_embeddings()
    log.info("embeddings loaded: %s", tuple(emb.shape))

    spatial_ei, _ = load_spatial_edge_index()
    log.info("spatial edge_index loaded: %s edges", spatial_ei.shape[1])

    world_model_status = "ensemble"
    try:
        ck = load_world_model_checkpoint()
        wm = build_world_model_from_checkpoint(ck)
        if wm is None:
            wm = PlaceholderTransition(
                state_dim=int(ck["arch"]["state_dim"]),
                action_dim=int(ck["arch"]["action_dim"]),
            )
            world_model_status = "placeholder_state_dict_mismatch"
            log.warning("world model load failed — using placeholder")
        else:
            log.info("world model loaded: arch=%s", ck.get("arch"))
    except Exception as e:  # noqa: BLE001
        log.warning("world model load exception (%s) — placeholder", e)
        wm = PlaceholderTransition(state_dim=128, action_dim=5)
        world_model_status = "placeholder_load_failed"

    # ------------------------------------------------------------ #
    # 2. Build env + agents
    # ------------------------------------------------------------ #
    env = HierarchicalEducationEnv(
        district_embeddings=emb,
        world_model=wm,
        horizon=4 if args.fast else 8,
        seed=args.seed,
    )
    n_districts = env.n_districts
    n_counties = env.n_counties
    log.info("env built: %d districts, %d counties", n_districts, n_counties)

    state_agent = StateAgent(obs_dim=env.state_dim, n_counties=n_counties)
    county_pool = CountyAgentPool(
        per_obs_dim=env.state_dim + 1, n_counties=n_counties, action_dim=5
    )
    district_pool = DistrictAgentPool(
        own_obs_dim=env.state_dim + 5 + 1,
        message_dim=32,
        action_dim=5,
        use_communication=True,
    )
    district_pool_no_comm = DistrictAgentPool(
        own_obs_dim=env.state_dim + 5 + 1,
        message_dim=32,
        action_dim=5,
        use_communication=False,
    )

    # ------------------------------------------------------------ #
    # 3. Build communication module
    # ------------------------------------------------------------ #
    knn_ei = build_knn_edges(emb, k=5)
    edge_index = merge_edges(spatial_ei, knn_ei, n_districts)
    comm_in_dim = env.state_dim
    comm_module = DistrictCommModule(
        in_dim=comm_in_dim, message_dim=32, n_rounds=3, n_heads=4,
    )
    save_protocol_spec(
        REPO / "results" / "communication" / "protocol_spec.json",
        n_districts=n_districts,
        n_edges=int(edge_index.shape[1]),
        extra={"spatial_edges": int(spatial_ei.shape[1])},
    )
    embeddings_2d = emb[:, -1, :] if emb.ndim == 3 else emb

    # ------------------------------------------------------------ #
    # 4. Train loop (data-collection ↔ SAC updates)
    # ------------------------------------------------------------ #
    log.info("training: %d env steps, %d SAC epochs, batch=%d",
             n_env_steps, sac_epochs, batch_size)

    obs = env.reset(seed=args.seed)
    rng = np.random.default_rng(args.seed)
    steps_completed = 0
    for step in range(n_env_steps):
        # Sample actions
        explore_eps = max(0.05, 1.0 - step / max(1, n_env_steps * 0.5))
        if rng.random() < explore_eps:
            a_state = rng.uniform(0.85, 1.15, size=n_counties).astype(np.float32)
            a_county = rng.uniform(-0.5, 0.5, size=(n_counties, 5)).astype(np.float32)
            a_district = rng.uniform(-1.0, 1.0, size=(n_districts, 5)).astype(np.float32)
        else:
            a_state = state_agent.act(obs["state"])
            a_county = county_pool.act_all(obs["county"])
            with torch.no_grad():
                msg = comm_module(embeddings_2d, edge_index).cpu().numpy()
            a_district = district_pool.act_all(obs["district"], msg)

        new_obs, rew, term, trunc, info = env.step(
            {"state": a_state, "county": a_county, "district": a_district}
        )
        done = bool(term["state"] or trunc["state"])

        # Store transitions
        state_agent.store(
            obs["state"], a_state, rew["state"],
            new_obs["state"], float(done),
        )
        county_dones = np.full(n_counties, float(done), dtype=np.float32)
        county_pool.store_all(
            obs["county"], a_county, rew["county"],
            new_obs["county"], county_dones,
        )
        district_dones = np.full(n_districts, float(done), dtype=np.float32)
        # No-comm pool also gets the data (with messages=None) so we can compare
        district_pool_no_comm.store_all(
            obs["district"], None, a_district, rew["district"],
            new_obs["district"], None, district_dones,
        )
        with torch.no_grad():
            cur_msg = comm_module(embeddings_2d, edge_index).cpu().numpy()
            next_msg = cur_msg  # embeddings static for now
        district_pool.store_all(
            obs["district"], cur_msg, a_district, rew["district"],
            new_obs["district"], next_msg, district_dones,
        )

        # Lagrangian updates
        county_means = rew["county"]
        state_agent.update_lagrange_from_env_info(info, county_means)
        county_to_outcomes = {c: [] for c in range(n_counties)}
        for i in range(n_districts):
            county_to_outcomes[int(env.district_to_county[i])].append(
                float(rew["district"][i])
            )
        county_pool.update_lagrange(county_to_outcomes)
        district_pool.update_lagrange(
            rew["district"],
            np.maximum(0.0, -rew["district"]).astype(np.float32),
        )

        obs = new_obs
        steps_completed += 1
        if done:
            obs = env.reset(seed=args.seed + step)

        if (step + 1) % max(1, n_env_steps // 5) == 0:
            log.info(
                "  step %d/%d state_r=%.4f county_r=%.4f district_r=%.4f",
                step + 1, n_env_steps, rew["state"],
                float(np.mean(rew["county"])), float(np.mean(rew["district"])),
            )

    # SAC update epochs
    log.info("SAC update phase: %d epochs", sac_epochs)
    for epoch in range(sac_epochs):
        m_s = state_agent.train_step(batch_size)
        m_c = county_pool.train_step(batch_size)
        m_d = district_pool.train_step(batch_size)
        m_dn = district_pool_no_comm.train_step(batch_size)
        if (epoch + 1) % max(1, sac_epochs // 5) == 0 and m_d is not None:
            log.info("  epoch %d/%d district_q=%.4f loss_actor=%.4f",
                     epoch + 1, sac_epochs,
                     m_d.get("q_mean", float("nan")),
                     m_d.get("loss_actor", float("nan")))

    # Save checkpoints
    ckpt_dir = REPO / "results" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    state_agent.save(ckpt_dir / "state_agent.pt")
    county_pool.save(ckpt_dir / "county_pool.pt")
    district_pool.save(ckpt_dir / "district_pool.pt")
    district_pool_no_comm.save(ckpt_dir / "district_pool_no_comm.pt")
    torch.save(
        {
            "comm_module": comm_module.state_dict(),
            "edge_index": edge_index,
        },
        str(ckpt_dir / "comm_module.pt"),
    )
    log.info("checkpoints saved to %s", ckpt_dir)

    # ------------------------------------------------------------ #
    # 5. Stackelberg analysis (centralized vs decentralized)
    # ------------------------------------------------------------ #
    analysis = analyze(
        env,
        state_agent,
        county_pool,
        district_pool,
        comm_module=comm_module,
        edge_index=edge_index,
        embeddings=embeddings_2d,
        n_episodes=n_eval_episodes,
    )

    # Communication ablation: with-comm vs no-comm reward delta
    from analysis.stackelberg_analysis import evaluate_rollout
    no_comm_eval = evaluate_rollout(
        env, state_agent, county_pool, district_pool_no_comm,
        comm_module=None, edge_index=None, embeddings=None,
        n_episodes=n_eval_episodes,
    )
    comm_delta = (
        analysis["decentralized"]["mean_state_reward"]
        - no_comm_eval["mean_state_reward"]
    )

    # Boundary-district coordination score:
    # Average within-county std on county pairs that share a spatial edge.
    boundary_pairs = set()
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    for u, v in zip(src, dst):
        cu = int(env.district_to_county[u])
        cv = int(env.district_to_county[v])
        if cu != cv:
            boundary_pairs.add((min(cu, cv), max(cu, cv)))
    # Use the per-county within-county std list from decentralized eval as the proxy
    decent_failures = analysis["decentralized"]["top_coordination_failures"]
    f_by_county = {f["county_id"]: f["within_county_std"] for f in decent_failures}
    boundary_scores = []
    for cu, cv in boundary_pairs:
        if cu in f_by_county or cv in f_by_county:
            s = (f_by_county.get(cu, 0.0) + f_by_county.get(cv, 0.0)) / 2.0
            boundary_scores.append(s)
    boundary_score = float(np.mean(boundary_scores)) if boundary_scores else 0.0

    # ------------------------------------------------------------ #
    # 6. Summary JSON
    # ------------------------------------------------------------ #
    from datetime import datetime, timezone
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "fast_mode": bool(args.fast),
        "seed": args.seed,
        "world_model_status": world_model_status,
        "training": {
            "steps_completed": int(steps_completed),
            "sac_epochs": int(sac_epochs),
            "batch_size": int(batch_size),
            "wall_time_s": round(time.time() - t0, 2),
        },
        "agents": {
            "state_action_dim": 58,
            "county_count": int(n_counties),
            "district_count": int(n_districts),
        },
        "stackelberg": {
            "price_of_anarchy": analysis["price_of_anarchy"],
            "centralized_reward": analysis["centralized"]["mean_state_reward"],
            "decentralized_reward": analysis["decentralized"]["mean_state_reward"],
            "top_coordination_failures":
                analysis["decentralized"]["top_coordination_failures"],
            "archetype_gap": analysis["archetype_gap"],
        },
        "communication_effect": {
            "boundary_district_coordination_score": boundary_score,
            "comm_vs_no_comm_reward_delta": float(comm_delta),
            "with_comm_reward": analysis["decentralized"]["mean_state_reward"],
            "without_comm_reward": no_comm_eval["mean_state_reward"],
        },
        "artifacts": {
            "checkpoints": str(ckpt_dir),
            "env_spec": str(REPO / "results" / "environment" / "env_spec.json"),
            "comm_protocol": str(REPO / "results" / "communication" / "protocol_spec.json"),
            "stackelberg_json": str(REPO / "results" / "analysis" / "stackelberg_analysis.json"),
            "figure": analysis.get("figure_written"),
        },
    }

    # Write env_spec.json as well (env didn't auto-save; we do it now)
    from environment.hierarchical_env import save_env_spec
    save_env_spec(env, REPO / "results" / "environment" / "env_spec.json")

    out = REPO / "results" / "marl_paper_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    log.info("=== summary written to %s (wall_time=%.1fs) ===",
             out, summary["training"]["wall_time_s"])
    log.info("    price_of_anarchy=%.4f  comm_delta=%.4f",
             summary["stackelberg"]["price_of_anarchy"],
             summary["communication_effect"]["comm_vs_no_comm_reward_delta"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
