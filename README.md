# edu-rl-multiagent

**Status:** SCAFFOLD (Phase A). Module files raise `NotImplementedError`.
Phase B implementation pending.

**Target journal:** JMLR or NeurIPS

## Research question

Does a Stackelberg hierarchy of policy-makers (state leader, county and district followers) with learned inter-district communication yield more equitable + better-aggregate-utility allocations than single-agent centralized RL?

## Description

Multi-agent Stackelberg RL with hierarchical state -> county -> district agents and inter-district message passing.

## Architecture overview

Three agent tiers (state action_dim=58, county=5, district=5) with GNN-based message passing (message_dim=32, 3 rounds, 4 attention heads) over the spatial graph from Layer 3.

## Layout

```
config/                # config.yaml + canonical r2_client.py
src/                   # module stubs (raise NotImplementedError)
scripts/preflight.py   # infra check (R2 + checkpoints + GPU)
scripts/run_*.py       # orchestrator stub
tests/test_smoke.py    # imports + NotImplementedError assertions
notebooks/             # exploration stub
figures/, results/     # output dirs
```

## Quickstart

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in R2 credentials
python scripts/preflight.py
python scripts/run_*.py --fast
pytest tests/
```

## Phase B will produce

- Trained multi-agent checkpoints under results/checkpoints/
- Stackelberg equilibrium analysis (best-response Jacobians, KKT residuals)
- Comparison vs. single-agent baseline on equity + aggregate metrics
- Communication-ablation table (with/without message passing)

## Operational note

The AECF national pull is running in background on this dev machine; do
not attempt parallel heavy compute (training, large embedding jobs) that
might conflict with it.
