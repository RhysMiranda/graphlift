"""Turning graph structure into features, at two levels of ambition.

1. `hub_features`: what a careful analyst can compute WITHOUT message passing.
   For every short path out of a target node (application -> device,
   application -> person -> email, application -> bank -> bank, ...) it counts how
   many other targets share the entity at the end of that path, and how tightly
   in time they arrived. This is the strong non-graph baseline. If a graph model
   cannot beat a gradient-boosted tree on these counts, the graph adds nothing
   that a few SQL joins would not.

2. `propagate`: what message passing adds on top. Every node's attributes are
   aggregated over its neighbours (mean and max, per neighbour type), then over
   their neighbours, for several hops, in the style of SIGN (Frasca et al.,
   2020). Mean asks "what is my neighbourhood like on average"; max asks "is ANY
   neighbour extreme", which is what a ring hiding among benign neighbours looks
   like. The result is a fixed-size vector per hop that any classifier can use.

Both use only attributes and edges, never labels.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import PCA

from .data import GraphData, encode


# --------------------------------------------------------------------------- #
# typed sparse relations
# --------------------------------------------------------------------------- #
@dataclass
class Rel:
    """One relation viewed from one side: rows are `src` nodes, cols `dst` nodes."""
    name: str
    src: str
    dst: str
    m: sp.csr_matrix


def typed_relations(g: GraphData) -> list[Rel]:
    """Every relation in both directions, as binary type-to-type matrices."""
    rels = []
    sizes = {t: len(df) for t, df in g.nodes.items()}
    for (st, rel, dt), e in g.edges.groupby(["src_type", "relation", "dst_type"]):
        si = g.global_ids(st, e["src_id"]) - g.offset[st]
        di = g.global_ids(dt, e["dst_id"]) - g.offset[dt]
        ok = ~(pd.isna(si) | pd.isna(di))
        si, di = si[ok].astype(int), di[ok].astype(int)
        m = sp.csr_matrix((np.ones(len(si), dtype=np.float32), (si, di)),
                          shape=(sizes[st], sizes[dt]))
        m.data[:] = 1.0                       # duplicates collapse to one edge
        m.sum_duplicates()
        m.data[:] = 1.0
        rels.append(Rel(rel, st, dt, m))
        rels.append(Rel(f"{rel}^-1", dt, st, m.T.tocsr()))
    return rels


def adjacency(g: GraphData) -> sp.csr_matrix:
    """Undirected, unweighted adjacency over all nodes of all types."""
    rows, cols = [], []
    for (st, dt), e in g.edges.groupby(["src_type", "dst_type"]):
        si = g.global_ids(st, e["src_id"])
        di = g.global_ids(dt, e["dst_id"])
        ok = ~(pd.isna(si) | pd.isna(di))
        rows.append(si[ok].astype(int))
        cols.append(di[ok].astype(int))
    r = np.concatenate(rows)
    c = np.concatenate(cols)
    a = sp.csr_matrix((np.ones(len(r), dtype=np.float32), (r, c)), shape=(g.n_nodes,) * 2)
    a = ((a + a.T) > 0).astype(np.float32).tocsr()
    a.setdiag(0)
    a.eliminate_zeros()
    return a


# --------------------------------------------------------------------------- #
# 1. hub features: the strong non-graph baseline
# --------------------------------------------------------------------------- #
@dataclass
class HubPath:
    name: str          # e.g. "submits^-1 > has_email"
    hub_type: str
    reach: sp.csr_matrix   # targets x hubs, 1 where the target reaches the hub


def hub_paths(g: GraphData, max_len: int = 2) -> list[HubPath]:
    """All paths of length 1..max_len out of the target type that end on a
    non-target node and never pass back through a target."""
    rels = typed_relations(g)
    by_src: dict[str, list[Rel]] = {}
    for r in rels:
        by_src.setdefault(r.src, []).append(r)

    out: list[HubPath] = []
    frontier = [(r.name, r.dst, r.m) for r in by_src.get(g.target, []) if r.dst != g.target]
    for depth in range(1, max_len + 1):
        nxt = []
        for name, t, m in frontier:
            reach = (m > 0).astype(np.float32).tocsr()
            if reach.nnz:
                out.append(HubPath(name, t, reach))
            if depth < max_len:
                for r in by_src.get(t, []):
                    if r.dst == g.target:
                        continue
                    # Don't walk straight back along the edge we came in on.
                    if r.name.endswith("^-1") and name.split(" > ")[-1] == r.name[:-3]:
                        continue
                    if name.split(" > ")[-1] == r.name + "^-1":
                        continue
                    nxt.append((f"{name} > {r.name}", r.dst, (m @ r.m)))
        frontier = nxt
    return out


def _col_reduce(m: sp.csc_matrix, values: np.ndarray, fn) -> np.ndarray:
    """Per-column reduction of `values[row]` over the stored entries of m."""
    out = np.full(m.shape[1], np.nan)
    nz = np.flatnonzero(np.diff(m.indptr) > 0)
    if len(nz):
        v = values[m.indices]
        red = fn.reduceat(v, m.indptr[:-1][nz])
        out[nz] = red
    return out


def _row_reduce(m: sp.csr_matrix, fill: float, fn) -> np.ndarray:
    out = np.full(m.shape[0], fill, dtype=float)
    nz = np.flatnonzero(np.diff(m.indptr) > 0)
    if len(nz):
        out[nz] = fn.reduceat(m.data, m.indptr[:-1][nz])
    return out


def hub_features(g: GraphData, paths: list[HubPath] | None = None
                 ) -> tuple[np.ndarray, list[str]]:
    """Per target and per path: how shared are the hubs it reaches, and how
    bursty was the sharing.

      max_share   largest number of targets on any hub reached by this path
      mean_share  average over the hubs reached
      n_shared    how many of the hubs reached are shared with another target
      min_span    smallest arrival-time span among shared hubs (days), if a time
                  column is declared; a ring is worked in a burst
    """
    paths = paths if paths is not None else hub_paths(g)
    t = None
    if g.time and g.time in g.targets:
        t = g.targets[g.time].to_numpy(dtype=float)
    cols, names = [], []
    for p in paths:
        share = np.asarray(p.reach.sum(axis=0)).ravel()            # targets per hub
        r = p.reach.copy()
        r.data = share[r.indices].astype(float)                    # entry = hub share
        mx = _row_reduce(r, 0.0, np.maximum)
        n_hubs = np.diff(r.indptr)
        mean = np.divide(np.asarray(r.sum(axis=1)).ravel(), n_hubs,
                         out=np.zeros(r.shape[0]), where=n_hubs > 0)
        r2 = r.copy()
        r2.data = (r2.data >= 2).astype(float)
        n_shared = np.asarray(r2.sum(axis=1)).ravel()
        cols += [mx, mean, n_shared]
        names += [f"{p.name}:max_share", f"{p.name}:mean_share", f"{p.name}:n_shared"]
        if t is not None:
            c = p.reach.tocsc()
            span = _col_reduce(c, t, np.maximum) - _col_reduce(c, t, np.minimum)
            span[share < 2] = np.nan
            rs = p.reach.copy()
            rs.data = span[rs.indices]
            rs.data = np.where(np.isnan(rs.data), 1e6, rs.data)
            ms = _row_reduce(rs, 1e6, np.minimum)
            cols.append(np.where(ms >= 1e6, -1.0, ms))
            names.append(f"{p.name}:min_span")
    return np.column_stack(cols).astype(np.float32), names


# --------------------------------------------------------------------------- #
# 2. propagation: what message passing adds
# --------------------------------------------------------------------------- #
def _neighbour_max(a: sp.csr_matrix, h: np.ndarray) -> np.ndarray:
    out = np.zeros((a.shape[0], h.shape[1]), dtype=np.float32)
    nz = np.flatnonzero(np.diff(a.indptr) > 0)
    if len(nz):
        out[nz] = np.maximum.reduceat(h[a.indices], a.indptr[:-1][nz], axis=0)
    return out


def _neighbour_mean(a: sp.csr_matrix, h: np.ndarray) -> np.ndarray:
    deg = np.asarray(a.sum(axis=1)).ravel()
    s = a @ h
    return np.divide(s, deg[:, None], out=np.zeros_like(s), where=deg[:, None] > 0)


def _reduce(x: np.ndarray, dim: int, seed: int) -> np.ndarray:
    x = x - x.mean(0)
    sd = x.std(0)
    x = x[:, sd > 1e-9] / sd[sd > 1e-9]
    k = min(dim, x.shape[1], x.shape[0] - 1)
    return PCA(n_components=k, random_state=seed).fit_transform(x).astype(np.float32)


def propagate(g: GraphData, hops: int = 3, dim: int = 48, seed: int = 0,
              a: sp.csr_matrix | None = None) -> tuple[list[np.ndarray], list[str]]:
    """Multi-hop neighbourhood summaries for the target nodes.

    Hop 1 aggregates neighbours' own attributes separately per neighbour type,
    so a device's fields are never averaged with a bank's. Later hops aggregate
    the previous hop's summaries over all neighbours. Each hop is compressed to
    `dim` columns with PCA, fitted without labels.

    Returns one (n_targets x dim) block per hop, plus column names."""
    a = a if a is not None else adjacency(g)
    n = g.n_nodes

    typed = []
    for t, df in g.nodes.items():
        x_t, _ = encode(g, t)
        lo, hi = g.offset[t], g.offset[t] + len(df)
        a_t = a[:, lo:hi].tocsr()
        deg_t = np.asarray(a_t.sum(axis=1)).ravel()[:, None].astype(np.float32)
        typed.append(np.log1p(deg_t))
        if x_t.shape[1]:
            typed.append(_neighbour_mean(a_t, x_t))
            typed.append(_neighbour_max(a_t, x_t))
    h = _reduce(np.hstack(typed), dim, seed)

    tgt = slice(g.offset[g.target], g.offset[g.target] + len(g.targets))
    blocks, names = [h[tgt]], [f"hop1_{i}" for i in range(h.shape[1])]
    for k in range(2, hops + 1):
        h = _reduce(np.hstack([_neighbour_mean(a, h), _neighbour_max(a, h)]), dim, seed)
        blocks.append(h[tgt])
        names += [f"hop{k}_{i}" for i in range(h.shape[1])]
    assert all(b.shape[0] == len(g.targets) for b in blocks) and n == a.shape[0]
    return blocks, names
