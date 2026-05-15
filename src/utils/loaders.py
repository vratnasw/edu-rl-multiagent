"""Shared loaders for world model + district embeddings + spatial graph.

These pull from R2 (lazily) or fall back to local checkpoints. Used by
the env + agents + analysis so the heavy artifacts are downloaded once
per session.
"""
from __future__ import annotations

import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import torch

log = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[2]
_CACHE_DIR = Path(tempfile.gettempdir()) / "edu_rl_multiagent_cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
def _ensure_env_loaded() -> None:
    """Load .env from the canonical location if R2 vars aren't already set."""
    if os.environ.get("R2_ACCOUNT_ID"):
        return
    for cand in (
        REPO / ".env",
        REPO.parent / ".env",
        REPO.parent / "edu-data-pipeline" / ".env",
    ):
        if cand.exists():
            try:
                from dotenv import load_dotenv  # type: ignore
                load_dotenv(cand, override=False)
                return
            except ImportError:
                for line in cand.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, _, v = line.partition("=")
                    os.environ.setdefault(k.strip(), v.strip())
                return


def _r2_download(key: str, dest: Path) -> bool:
    """Download an R2 key to dest. Returns True on success."""
    _ensure_env_loaded()
    try:
        sys.path.insert(0, str(REPO / "config"))
        import r2_client as r2  # type: ignore
        info = r2.exists(key)
        if info is None:
            return False
        c = r2._client()
        dest.parent.mkdir(parents=True, exist_ok=True)
        c.download_file(r2.bucket_name(), key, str(dest))
        return True
    except Exception as e:  # noqa: BLE001
        log.warning("R2 download failed for %s: %s", key, e)
        return False


# --------------------------------------------------------------------------- #
def load_world_model_checkpoint() -> dict:
    """Download Layer-5 world model checkpoint and return the raw torch dict.

    Returns the loaded torch.save dict (contains 'state_dict', 'arch', ...).
    Caches locally under TEMP.
    """
    key = "checkpoints/edu-world-model/best.pt"
    dest = _CACHE_DIR / "world_model_best.pt"
    if not dest.exists():
        ok = _r2_download(key, dest)
        if not ok:
            local = REPO.parent / "edu-world-model" / "results" / "checkpoints" / "best.pt"
            if local.exists():
                import shutil
                shutil.copy(local, dest)
            else:
                raise FileNotFoundError(
                    f"world model not on R2 ({key}) nor local fallback ({local})"
                )
    return torch.load(str(dest), map_location="cpu", weights_only=False)


def load_district_embeddings() -> torch.Tensor:
    """Download Layer-4 district embeddings. Returns Tensor of shape (N, T, D)."""
    key = "embeddings/edu-gnn/district_embeddings.pt"
    dest = _CACHE_DIR / "district_embeddings.pt"
    if not dest.exists():
        ok = _r2_download(key, dest)
        if not ok:
            local = REPO.parent / "edu-gnn" / "results" / "embeddings" / "district_embeddings.pt"
            if local.exists():
                import shutil
                shutil.copy(local, dest)
            else:
                raise FileNotFoundError(
                    f"embeddings not on R2 ({key}) nor local fallback ({local})"
                )
    emb = torch.load(str(dest), map_location="cpu", weights_only=False)
    if not torch.is_tensor(emb):
        raise ValueError(f"expected Tensor, got {type(emb)}")
    return emb


def load_spatial_edge_index() -> tuple[np.ndarray, np.ndarray]:
    """Load Layer-3 spatial graph (edge_index, edge_weights).

    Returns (edge_index [2, E], edge_weights [E])."""
    ei_path = REPO.parent / "edu-spatial-rl" / "results" / "spatiotemporal" / "edge_index.npy"
    ew_path = REPO.parent / "edu-spatial-rl" / "results" / "spatiotemporal" / "edge_weights.npy"
    if not ei_path.exists():
        raise FileNotFoundError(f"edge_index not found at {ei_path}")
    ei = np.load(str(ei_path))
    ew = np.load(str(ew_path)) if ew_path.exists() else np.ones(ei.shape[1], dtype=np.float32)
    return ei, ew


# --------------------------------------------------------------------------- #
def build_world_model_from_checkpoint(
    checkpoint: dict, device: torch.device | None = None
) -> Optional[torch.nn.Module]:
    """Reconstruct WorldModelEnsemble from the saved arch. Returns None on failure
    (caller should fall back to a random-projection placeholder)."""
    device = device or torch.device("cpu")
    try:
        sys.path.insert(0, str(REPO.parent / "edu-world-model" / "src"))
        from models.world_model_ensemble import WorldModelEnsemble  # type: ignore
        from models.ensemble_member import MemberConfig  # type: ignore

        arch = checkpoint["arch"]
        member_cfg = MemberConfig(
            state_dim=int(arch["state_dim"]),
            action_dim=int(arch["action_dim"]),
            hidden_dim=int(arch.get("hidden_dim", 256)),
            num_layers=int(arch.get("num_layers", 3)),
        )
        model = WorldModelEnsemble(
            member_cfg=member_cfg,
            ensemble_size=int(arch.get("ensemble_size", 5)),
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device).eval()
        return model
    except Exception as e:  # noqa: BLE001
        log.warning("could not reconstruct world model (%s) — using placeholder", e)
        return None


class PlaceholderTransition(torch.nn.Module):
    """Random-projection fallback when the world model can't load.

    Same I/O signature as WorldModelEnsemble.predict (returns mean, epi, ale).
    Used only when there's a torch state_dict mismatch — flagged in the
    summary JSON.
    """

    def __init__(self, state_dim: int, action_dim: int, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.W_s = torch.nn.Parameter(
            (torch.randn(state_dim, state_dim, generator=g) * 0.05),
            requires_grad=False,
        )
        self.W_a = torch.nn.Parameter(
            (torch.randn(action_dim, state_dim, generator=g) * 0.05),
            requires_grad=False,
        )
        self.eye = torch.nn.Parameter(
            torch.eye(state_dim) * 0.95,
            requires_grad=False,
        )
        self.state_dim = state_dim
        self.action_dim = action_dim

    def predict(self, s: torch.Tensor, a: torch.Tensor):
        nxt = s @ self.eye + s @ self.W_s + a @ self.W_a
        epi = torch.full_like(nxt, 0.05)
        ale = torch.full_like(nxt, 0.05)
        return nxt, epi, ale

    def forward(self, s: torch.Tensor, a: torch.Tensor):
        m = self.predict(s, a)[0]
        return m.unsqueeze(0), torch.zeros_like(m).unsqueeze(0)
