"""Checks that run before any model is trusted.

Three ways a graph benchmark lies to you:

1. **Split leakage.** Members of one fraud ring land on both sides of the
   train/test boundary, so the test set is partly a memory test. graphlift splits
   by group (a ring, a household, a connected component) and reports how much of
   the test set can still reach a training node through a shared entity.
2. **Label attributes.** A neighbour carries a column that IS the answer
   (`bank.is_mule`, `document.is_forged`). Any model that aggregates neighbours
   reads it straight off. graphlift scans every attribute within two hops and
   flags any that predicts the label almost perfectly on its own.
3. **One giant component.** If every target is connected to every other, no
   split is leak-free, and the report says so instead of pretending.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import roc_auc_score

from .data import GraphData, MAX_CATEGORIES
from .structure import HubPath


def components(g: GraphData, a: sp.csr_matrix) -> np.ndarray:
    """Connected-component id for every target node, over all edges."""
    _, lab = connected_components(a, directed=False)
    lo = g.offset[g.target]
    return lab[lo:lo + len(g.targets)]


def resolve_groups(g: GraphData, a: sp.csr_matrix) -> tuple[np.ndarray, dict]:
    """Groups that must stay on one side of a split, and a note on where they
    came from. A declared group column wins; otherwise connected components."""
    comp = components(g, a)
    sizes = pd.Series(comp).value_counts()
    info = {"n_components": int(len(sizes)),
            "largest_component_share": float(sizes.iloc[0] / len(comp))}
    if g.group and g.group in g.targets:
        grp = g.targets[g.group].astype(str).to_numpy()
        gs = pd.Series(grp).value_counts()
        info.update(source=f"declared column `{g.group}`", n_groups=int(len(gs)),
                    largest_group=int(gs.iloc[0]),
                    singleton_share=float((gs == 1).sum() / len(grp)))
        return grp, info
    if info["largest_component_share"] > 0.5:
        # Grouping by component would put most of the graph in one fold.
        info.update(source="individual targets (no leak-free grouping exists)",
                    n_groups=int(g.labelled.sum()), largest_group=1, singleton_share=1.0,
                    warning=("Over half the targets sit in one connected component, so no "
                             "split can keep connected targets apart and the split falls "
                             "back to individual targets, as most published results on "
                             "this kind of graph do. Treat absolute scores as optimistic; "
                             "the comparison between models stays fair because every "
                             "model sees the same split."))
        return np.arange(len(comp)).astype(str), info
    info.update(source="connected components", n_groups=info["n_components"],
                largest_group=int(sizes.iloc[0]),
                singleton_share=float((sizes == 1).sum() / len(comp)))
    return comp.astype(str), info


def exposure(paths: list[HubPath], y: np.ndarray, train: np.ndarray,
             test: np.ndarray, direct: sp.csr_matrix | None = None) -> dict:
    """Share of test targets that share an entity (within the hub paths) with a
    training target, or link to one directly, and the same for a training
    POSITIVE. The second is the channel through which a label-aware model can
    copy answers across the split."""
    n_t = len(y)
    any_train = np.zeros(int(test.sum()), bool)
    any_pos = np.zeros(int(test.sum()), bool)
    if paths:
        reach = sp.hstack([p.reach for p in paths]).tocsr()
        hub_train = np.asarray(reach[train].sum(axis=0)).ravel() > 0
        hub_pos = np.asarray(reach[train & (y == 1)].sum(axis=0)).ravel() > 0
        rt = reach[test]
        any_train |= (rt @ hub_train.astype(np.float32)) > 0
        any_pos |= (rt @ hub_pos.astype(np.float32)) > 0
    if direct is not None and direct.nnz:
        dt = direct[test]
        any_train |= (dt @ train.astype(np.float32)) > 0
        any_pos |= (dt @ (train & (y == 1)).astype(np.float32)) > 0
    n = max(int(test.sum()), 1)
    return {"test_touches_train": float(any_train.sum() / n),
            "test_touches_train_positive": float(any_pos.sum() / n)}


def attribute_scan(g: GraphData, a: sp.csr_matrix, threshold: float = 0.9,
                   top: int = 8) -> list[dict]:
    """How well does each single neighbour attribute (max over 1 and 2 hops)
    separate the classes on its own? Anything near-perfect is probably a label.

    Declared exclusions are scanned too, so the report shows what WOULD have
    leaked had they not been excluded."""
    lab = g.labelled
    y = g.y[lab]
    lo = g.offset[g.target]
    tgt = slice(lo, lo + len(g.targets))
    a_t = a[tgt][lab]
    others = [t for t in g.nodes if t != g.target]
    # Two hops out from every target, restricted to non-target columns. Skipped
    # when there are no other node types (a graph of targets linked directly).
    two = None
    if others:
        keep = np.zeros(a.shape[0], dtype=np.float32)
        for t in others:
            keep[g.offset[t]:g.offset[t] + len(g.nodes[t])] = 1.0
        two = ((a_t @ (a @ sp.diags(keep))) > 0).astype(np.float32).tocsr()
    rows = []
    # The target's own columns first: a label copied into a feature column is
    # the most common leak of all.
    own = g.targets
    reserved = {"id", g.label, g.group, g.time, g.breakdown}
    for c in own.columns:
        if c in reserved:
            continue
        s = own[c]
        if not pd.api.types.is_numeric_dtype(s):
            if not (1 < s.nunique() <= MAX_CATEGORIES):
                continue
            s = pd.Series(pd.factorize(s)[0], index=s.index)
        v = np.nan_to_num(s.to_numpy(dtype=float), nan=0.0)[lab]
        if np.std(v) == 0:
            continue
        auc = roc_auc_score(y, v)
        auc = max(auc, 1 - auc)
        rows.append({"attribute": f"{g.target}.{c}", "auc": round(float(auc), 3),
                     "via": "own column", "excluded": c in set(g.exclude.get(g.target, [])),
                     "flag": bool(auc >= threshold)})
    for t, df in g.nodes.items():
        if t == g.target:
            continue
        o = g.offset[t]
        hop1 = a_t[:, o:o + len(df)].tocsr()
        hop2 = two[:, o:o + len(df)].tocsr() if two is not None else sp.csr_matrix(hop1.shape)
        excluded = set(g.exclude.get(t, []))
        for c in df.columns:
            if c == "id":
                continue
            s = df[c]
            if not pd.api.types.is_numeric_dtype(s):
                if not (1 < s.nunique() <= MAX_CATEGORIES):
                    continue
                s = pd.Series(pd.factorize(s)[0], index=s.index)
            v = s.to_numpy(dtype=float)
            if np.nanstd(v) == 0:
                continue
            v = np.nan_to_num(v, nan=np.nanmin(v))
            best = 0.5
            for hop, m in (("1-hop", hop1), ("2-hop", hop2)):
                if m.nnz == 0:
                    continue
                for fn, nm in ((np.maximum, "max"), (np.minimum, "min")):
                    out = np.zeros(m.shape[0])
                    nz = np.flatnonzero(np.diff(m.indptr) > 0)
                    if not len(nz):
                        continue
                    out[nz] = fn.reduceat(v[m.indices], m.indptr[:-1][nz])
                    out[np.diff(m.indptr) == 0] = np.nan
                    ok = ~np.isnan(out)
                    if ok.sum() < 10 or len(np.unique(y[ok])) < 2:
                        continue
                    auc = roc_auc_score(y[ok], out[ok])
                    auc = max(auc, 1 - auc) * ok.mean() + 0.5 * (1 - ok.mean())
                    if auc > best:
                        best, where = auc, f"{nm} over {hop} {t}"
            if best > 0.5:
                rows.append({"attribute": f"{t}.{c}", "auc": round(float(best), 3),
                             "via": where, "excluded": c in excluded,
                             "flag": bool(best >= threshold)})
    rows.sort(key=lambda r: -r["auc"])
    flagged = [r for r in rows if r["flag"] or r["excluded"]]
    rest = [r for r in rows if not (r["flag"] or r["excluded"])][:top]
    return flagged + rest
