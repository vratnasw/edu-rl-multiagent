"""Inter-district GNN message-passing module.

  message_dim   = 32
  n_rounds      = 3
  attention     = 4 heads (multi-head, concat → linear)

Each district broadcasts a 32-dim message; neighbors aggregate via
attention-weighted sum (graph-attention style). After 3 rounds, the
aggregated message is concatenated with the district's state embedding
and consumed by the actor.

Neighbor structure:
  - Default: Layer-3 spatial graph (edu-spatial-rl edge_index)
  - For districts not in the spatial graph, we add k=5 nearest neighbors
    by embedding similarity so every district has at least one neighbor.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)

MESSAGE_DIM = 32
N_ROUNDS = 3
N_HEADS = 4
DEFAULT_K = 5


# --------------------------------------------------------------------------- #
def build_knn_edges(
    embeddings: torch.Tensor, k: int = DEFAULT_K
) -> torch.Tensor:
    """k-NN over district embeddings (cosine). Returns edge_index [2, E]
    with self-loops excluded; both directions included so the graph is
    symmetric."""
    if embeddings.ndim == 3:
        # (N, T, D) → use the last time step
        x = embeddings[:, -1, :]
    else:
        x = embeddings
    x = F.normalize(x, dim=-1)
    sim = x @ x.T  # (N, N)
    n = sim.shape[0]
    sim.fill_diagonal_(-1.0)
    _, top = sim.topk(min(k, n - 1), dim=-1)
    src = torch.arange(n).repeat_interleave(top.shape[1])
    dst = top.flatten()
    # add reverse direction for symmetry
    ei = torch.stack(
        [torch.cat([src, dst]), torch.cat([dst, src])], dim=0
    ).long()
    return ei


def merge_edges(
    spatial_ei: np.ndarray | torch.Tensor,
    knn_ei: torch.Tensor,
    n_districts: int,
) -> torch.Tensor:
    """Union spatial + k-NN edges, clamp to [0, n_districts), dedupe."""
    if isinstance(spatial_ei, np.ndarray):
        s_ei = torch.from_numpy(spatial_ei).long()
    else:
        s_ei = spatial_ei.long()
    # clamp spatial graph to current district range
    mask = (s_ei[0] < n_districts) & (s_ei[1] < n_districts)
    s_ei = s_ei[:, mask]
    all_ei = torch.cat([s_ei, knn_ei], dim=1)
    # dedupe via hash of (u,v) pairs
    keys = all_ei[0] * (n_districts + 1) + all_ei[1]
    _, unique_idx = torch.unique(keys, return_inverse=False, return_counts=False), None
    _, idx = torch.unique(keys, return_inverse=True)
    seen: set[int] = set()
    keep: list[int] = []
    for i, k in enumerate(idx.tolist()):
        if k not in seen:
            seen.add(k); keep.append(i)
    return all_ei[:, keep]


# --------------------------------------------------------------------------- #
class MultiHeadGraphAttention(nn.Module):
    """Single round of multi-head graph attention.

    Implements GAT-style attention with `n_heads` heads. For each edge (u → v),
    attention weight α_uv ∝ exp(LeakyReLU(a·[Wh_u || Wh_v])). We softmax
    over u for each v (incoming edges to v).
    """

    def __init__(self, in_dim: int, out_dim: int, n_heads: int = N_HEADS):
        super().__init__()
        assert out_dim % n_heads == 0, "out_dim must be divisible by n_heads"
        self.n_heads = n_heads
        self.head_dim = out_dim // n_heads
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a_src = nn.Parameter(torch.empty(n_heads, self.head_dim))
        self.a_dst = nn.Parameter(torch.empty(n_heads, self.head_dim))
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_src.unsqueeze(0))
        nn.init.xavier_uniform_(self.a_dst.unsqueeze(0))
        self.out_proj = nn.Linear(out_dim, out_dim)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """h: (N, in_dim), edge_index: [2, E].
        Returns updated h: (N, out_dim)."""
        N = h.shape[0]
        Wh = self.W(h).view(N, self.n_heads, self.head_dim)  # (N, H, dh)
        src, dst = edge_index[0], edge_index[1]
        # Per-edge per-head attention logits
        e_src = (Wh[src] * self.a_src.unsqueeze(0)).sum(-1)  # (E, H)
        e_dst = (Wh[dst] * self.a_dst.unsqueeze(0)).sum(-1)  # (E, H)
        e = F.leaky_relu(e_src + e_dst, negative_slope=0.2)  # (E, H)
        # Softmax over incoming edges for each dst
        # subtract per-(dst,head) max for numerical stability
        e_max = torch.full((N, self.n_heads), -1e9, device=h.device)
        e_max = e_max.scatter_reduce(0, dst.unsqueeze(-1).expand(-1, self.n_heads),
                                       e, reduce="amax", include_self=True)
        e_shift = e - e_max[dst]
        e_exp = e_shift.exp()
        # Denominator: sum per dst
        denom = torch.zeros((N, self.n_heads), device=h.device)
        denom = denom.scatter_add_(0,
                                     dst.unsqueeze(-1).expand(-1, self.n_heads),
                                     e_exp) + 1e-12
        alpha = e_exp / denom[dst]  # (E, H)
        # Weighted message from src
        m = Wh[src] * alpha.unsqueeze(-1)  # (E, H, dh)
        # Aggregate at dst
        out = torch.zeros((N, self.n_heads, self.head_dim), device=h.device)
        out = out.scatter_add_(
            0,
            dst.view(-1, 1, 1).expand(-1, self.n_heads, self.head_dim),
            m,
        )
        out = out.reshape(N, self.n_heads * self.head_dim)
        out = self.out_proj(out)
        return F.elu(out)


# --------------------------------------------------------------------------- #
class DistrictCommModule(nn.Module):
    """Stack of `n_rounds` multi-head GAT layers over the district graph.

    Input:  district embedding (N, in_dim)
    Output: aggregated message (N, message_dim)
    """

    def __init__(
        self,
        in_dim: int,
        message_dim: int = MESSAGE_DIM,
        n_rounds: int = N_ROUNDS,
        n_heads: int = N_HEADS,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.message_dim = message_dim
        self.n_rounds = n_rounds
        self.n_heads = n_heads
        # Project input → message_dim, then stack n_rounds GAT layers
        self.proj_in = nn.Linear(in_dim, message_dim)
        self.layers = nn.ModuleList(
            [
                MultiHeadGraphAttention(message_dim, message_dim, n_heads)
                for _ in range(n_rounds)
            ]
        )
        self.layer_norm = nn.LayerNorm(message_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.proj_in(x)
        for layer in self.layers:
            h_new = layer(h, edge_index)
            # residual + layernorm
            h = self.layer_norm(h + h_new)
        return h  # (N, message_dim)


# --------------------------------------------------------------------------- #
def save_protocol_spec(
    out_path: Path,
    n_districts: int,
    n_edges: int,
    extra: dict | None = None,
) -> dict:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    spec = {
        "message_dim": MESSAGE_DIM,
        "n_communication_rounds": N_ROUNDS,
        "attention_heads": N_HEADS,
        "knn_k_fallback": DEFAULT_K,
        "n_districts": int(n_districts),
        "n_edges_total": int(n_edges),
        "aggregation": "multi-head graph-attention (GAT-style)",
        "round_residual": "LayerNorm(h + GAT(h))",
    }
    if extra:
        spec.update(extra)
    out_path.write_text(json.dumps(spec, indent=2))
    return spec


# --------------------------------------------------------------------------- #
def run(fast: bool = False) -> dict:
    """Build a comm module from real spatial + embedding data and
    do a single forward pass to validate shapes. Returns a small dict."""
    from utils.loaders import load_district_embeddings, load_spatial_edge_index

    emb = load_district_embeddings()
    spatial_ei, _ = load_spatial_edge_index()
    if emb.ndim == 3:
        x = emb[:, -1, :]
    else:
        x = emb
    n_districts, in_dim = x.shape
    knn_ei = build_knn_edges(emb)
    ei = merge_edges(spatial_ei, knn_ei, n_districts)
    mod = DistrictCommModule(in_dim=in_dim)
    out = mod(x, ei)
    repo = Path(__file__).resolve().parents[2]
    save_protocol_spec(
        repo / "results" / "communication" / "protocol_spec.json",
        n_districts=n_districts,
        n_edges=int(ei.shape[1]),
    )
    return {
        "n_districts": int(n_districts),
        "in_dim": int(in_dim),
        "out_shape": list(out.shape),
        "n_edges": int(ei.shape[1]),
    }


if __name__ == "__main__":
    import logging as _l
    _l.basicConfig(level=_l.INFO)
    print(run())
