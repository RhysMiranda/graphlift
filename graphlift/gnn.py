"""A heterogeneous GraphSAGE, the trained-GNN rung of the ladder.

Optional: needs PyTorch (`pip install graphlift[gnn]`), nothing else. Written in
plain PyTorch rather than PyTorch Geometric so it installs anywhere torch does.

Architecture (Hamilton et al., 2017, made relation-aware):

    h_t^(0)   = ReLU(W_in,t [x_t, log(1 + degree)])          one input layer per node type
    h_t^(k+1) = ReLU(W_self,t h_t^(k)                         for every node type t
                     + sum over relations r out of t of
                         W_mean,r mean_{j in N_r(i)} h^(k)_j
                       + W_max,r  max_{j in S_r(i)}  h^(k)_j)
    score     = W_out h_target^(K)

Mean runs over every neighbour (a sparse product, cheap at any degree). Max runs
over at most `max_sample` neighbours per node, resampled every epoch, which is
GraphSAGE's own neighbour sampling and keeps memory bounded on graphs with
millions of edges. Mean and max together, three layers: the configuration Syn
Env's ablations found best (max sees a ring hiding among benign neighbours, mean
can count).

Training is full-batch with class-weighted cross-entropy and Adam. A slice of the
training targets is held out for early stopping on validation PR-AUC: random
groups under a grouped split, the latest training periods under a temporal one.
Unlabelled targets stay in the graph and pass messages but never enter the loss.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score

from .data import GraphData, encode
from .structure import typed_relations

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAVE_TORCH = True
except ImportError:          # the rest of graphlift works without it
    HAVE_TORCH = False


def available() -> bool:
    return HAVE_TORCH


class _Graph:
    """Everything that doesn't change between training runs, built once."""

    def __init__(self, g: GraphData, device: str):
        rels = typed_relations(g)
        self.types = list(g.nodes)
        self.target = g.target
        self.n = {t: len(g.nodes[t]) for t in self.types}
        self.x = {}
        for t in self.types:
            x, _ = encode(g, t)
            deg = np.zeros(self.n[t], dtype=np.float32)
            for r in rels:
                if r.src == t:
                    deg += np.asarray(r.m.sum(axis=1)).ravel()
            x = np.hstack([x, np.log1p(deg)[:, None]]).astype(np.float32)
            self.x[t] = torch.tensor(x, device=device)
        self.rels = []
        for r in rels:
            m = r.m.tocsr()
            if m.nnz == 0:
                continue
            deg = np.diff(m.indptr)
            rows = np.repeat(np.arange(m.shape[0]), deg)
            vals = (1.0 / np.maximum(deg, 1))[rows].astype(np.float32)
            mean = torch.sparse_coo_tensor(
                torch.tensor(np.vstack([rows, m.indices]), dtype=torch.long),
                torch.tensor(vals), size=m.shape, device=device,
                check_invariants=True).coalesce()   # indices come from a valid CSR
            self.rels.append({"name": r.name, "src": r.src, "dst": r.dst, "mean": mean,
                              "indptr": m.indptr, "indices": m.indices, "deg": deg})
        self.device = device

    def sample(self, rng: np.random.Generator, cap: int) -> list:
        """(row, col) index tensors per relation, at most `cap` neighbours per row."""
        out = []
        for r in self.rels:
            deg, ptr, idx = r["deg"], r["indptr"], r["indices"]
            rows_all = r.setdefault("rows_all", np.repeat(np.arange(len(deg)), deg))
            if deg.max() <= cap:
                rows, cols = rows_all, idx
            else:
                # Random key per edge; keep each row's `cap` smallest keys. Rows are
                # already contiguous in CSR order, so a stable sort on the key
                # within rows gives a uniform sample without replacement.
                order = np.lexsort((rng.random(len(idx)), rows_all))
                rank = np.arange(len(idx)) - ptr[:-1][rows_all]
                keep = order[rank < cap]
                rows, cols = rows_all[keep], idx[keep]
            out.append((torch.tensor(rows, dtype=torch.long, device=self.device),
                        torch.tensor(cols, dtype=torch.long, device=self.device)))
        return out


if HAVE_TORCH:
    class HeteroSAGE(nn.Module):
        def __init__(self, G: _Graph, hidden: int = 64, layers: int = 3, dropout: float = 0.2):
            super().__init__()
            self.G, self.layers, self.dropout = G, layers, dropout
            self.inp = nn.ModuleDict({t: nn.Linear(G.x[t].shape[1], hidden) for t in G.types})
            self.self_ = nn.ModuleList(nn.ModuleDict({t: nn.Linear(hidden, hidden) for t in G.types})
                                       for _ in range(layers))
            key = lambda i: f"r{i}"
            self.mean = nn.ModuleList(nn.ModuleDict({key(i): nn.Linear(hidden, hidden, bias=False)
                                                     for i in range(len(G.rels))}) for _ in range(layers))
            self.max = nn.ModuleList(nn.ModuleDict({key(i): nn.Linear(hidden, hidden, bias=False)
                                                    for i in range(len(G.rels))}) for _ in range(layers))
            self.out = nn.Linear(hidden, 1)

        def forward(self, sampled: list) -> "torch.Tensor":
            G = self.G
            h = {t: F.relu(self.inp[t](G.x[t])) for t in G.types}
            for k in range(self.layers):
                new = {t: self.self_[k][t](h[t]) for t in G.types}
                for i, r in enumerate(G.rels):
                    src, dst = r["src"], r["dst"]
                    new[src] = new[src] + self.mean[k][f"r{i}"](torch.sparse.mm(r["mean"], h[dst]))
                    rows, cols = sampled[i]
                    if len(rows):
                        mx = torch.zeros(G.n[src], h[dst].shape[1], device=h[dst].device)
                        mx = mx.scatter_reduce(0, rows[:, None].expand(-1, h[dst].shape[1]),
                                               h[dst][cols], reduce="amax", include_self=False)
                        new[src] = new[src] + self.max[k][f"r{i}"](mx)
                h = {t: F.dropout(F.relu(v), self.dropout, self.training) for t, v in new.items()}
            return self.out(h[G.target]).squeeze(-1)


def _val_split(train_idx: np.ndarray, groups: np.ndarray | None, times: np.ndarray | None,
               rng: np.random.Generator, frac: float = 0.15):
    """Hold out part of the training targets for early stopping."""
    if times is not None:                       # temporal: the latest training periods
        cut = np.quantile(times[train_idx], 1 - frac)
        val = times[train_idx] > cut
        if 0 < val.sum() < len(val):
            return train_idx[~val], train_idx[val]
    gs = groups[train_idx] if groups is not None else train_idx.astype(str)
    uniq = np.unique(gs)
    held = set(rng.choice(uniq, max(1, int(len(uniq) * frac)), replace=False))
    val = np.array([x in held for x in gs])
    return train_idx[~val], train_idx[val]


def predict(g: GraphData, splits: list, seed: int, lab_idx: np.ndarray, y: np.ndarray,
            groups: np.ndarray | None = None, times: np.ndarray | None = None,
            epochs: int = 300, hidden: int = 64, layers: int = 3, lr: float = 0.005,
            max_sample: int = 10, patience: int = 60, cache: dict | None = None,
            device: str | None = None, verbose: bool = False) -> np.ndarray:
    """Out-of-split GNN scores for the labelled targets, NaN where not tested.

    `splits` holds (train_mask, test_mask) pairs over the labelled targets, the
    same pairs the tree rungs use."""
    if not HAVE_TORCH:
        raise ImportError("The GNN rung needs PyTorch: pip install graphlift[gnn]")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    cache = cache if cache is not None else {}
    if "G" not in cache:
        cache["G"] = _Graph(g, device)
    G = cache["G"]
    yt = torch.tensor(y, dtype=torch.float32, device=device)
    tgt_idx = torch.tensor(lab_idx, dtype=torch.long, device=device)
    p = np.full(len(y), np.nan)

    for fi, (tr, te) in enumerate(splits):
        torch.manual_seed(seed * 1000 + fi)
        rng = np.random.default_rng(seed * 1000 + fi)
        tr_i, va_i = _val_split(np.flatnonzero(tr), groups, times, rng)
        model = HeteroSAGE(G, hidden, layers).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
        pos = max(float(y[tr_i].sum()), 1.0)
        w = torch.tensor((len(tr_i) - pos) / pos, device=device)
        eval_sample = G.sample(np.random.default_rng(12345), max_sample)
        best, best_state, since = -1.0, None, 0
        tr_t = torch.tensor(tr_i, device=device)
        for ep in range(1, epochs + 1):
            model.train()
            opt.zero_grad()
            logit = model(G.sample(rng, max_sample))[tgt_idx[tr_t]]
            loss = F.binary_cross_entropy_with_logits(logit, yt[tr_t], pos_weight=w)
            loss.backward()
            opt.step()
            if ep % 5 == 0:
                model.eval()
                with torch.no_grad():
                    s = model(eval_sample)[tgt_idx].cpu().numpy()
                ap = average_precision_score(y[va_i], s[va_i]) if 0 < y[va_i].sum() < len(va_i) else -loss.item()
                if ap > best:
                    best, since = ap, 0
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                else:
                    since += 5
                    if since >= patience:
                        break
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            s = torch.sigmoid(model(eval_sample)[tgt_idx]).cpu().numpy()
        p[te] = s[te]
        if verbose:
            print(f"[graphlift]   gnn split {fi + 1}/{len(splits)}: stopped at epoch {ep}, "
                  f"val PR-AUC {best:.3f}")
    return p
