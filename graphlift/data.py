"""Loading a graph and turning node attributes into numbers.

graphlift works on any heterogeneous graph stored as plain tables:

    <dir>/nodes/<type>.csv   id column + attribute columns, one file per node type
    <dir>/edges.csv          src_type, src_id, relation, dst_type, dst_id
    <dir>/graphlift.json     {"target": ..., "label": ..., "group": ..., ...}

Nothing here knows about fraud. A dataset declares which node type is being
classified, which column is its label, and which attributes are really labels in
disguise (an `is_mule` flag on a bank account is the answer, not an observation).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Strings with more distinct values than this are identifiers or free text
# (names, hashes, street addresses) and are dropped rather than one-hot encoded.
MAX_CATEGORIES = 16


@dataclass
class GraphData:
    nodes: dict[str, pd.DataFrame]
    edges: pd.DataFrame
    target: str
    label: str
    group: str | None = None
    time: str | None = None
    breakdown: str | None = None
    exclude: dict[str, list[str]] = field(default_factory=dict)
    undirected: list[str] = field(default_factory=list)
    name: str = "graph"

    # Filled by `index()`.
    offset: dict[str, int] = field(default_factory=dict)
    n_nodes: int = 0

    def index(self) -> "GraphData":
        """Give every node a global integer index, types laid out back to back."""
        off = 0
        for t, df in self.nodes.items():
            self.offset[t] = off
            off += len(df)
        self.n_nodes = off
        return self

    # -- convenience ------------------------------------------------------------
    @property
    def targets(self) -> pd.DataFrame:
        return self.nodes[self.target]

    @property
    def labelled(self) -> np.ndarray:
        """Targets that carry a usable 0/1 label. Unlabelled targets stay in the
        graph as neighbours but are never trained or scored on."""
        v = pd.to_numeric(self.targets[self.label], errors="coerce").to_numpy()
        return np.isin(v, (0, 1))

    @property
    def y(self) -> np.ndarray:
        """Labels for ALL targets; unlabelled ones are -1. Index with `labelled`."""
        v = pd.to_numeric(self.targets[self.label], errors="coerce").to_numpy()
        return np.where(np.isin(v, (0, 1)), v, -1).astype(int)

    def global_ids(self, ntype: str, ids: pd.Series) -> np.ndarray:
        lookup = pd.Series(np.arange(len(self.nodes[ntype])) + self.offset[ntype],
                           index=self.nodes[ntype]["id"].to_numpy())
        return lookup.reindex(ids.to_numpy()).to_numpy()

    def summary(self) -> dict:
        y = self.y[self.labelled]
        return {
            "name": self.name,
            "target": self.target,
            "n_targets": int(len(self.targets)),
            "n_labelled": int(len(y)),
            "n_positive": int(y.sum()),
            "prevalence": float(y.mean()),
            "node_types": {t: int(len(d)) for t, d in self.nodes.items()},
            "n_edges": int(len(self.edges)),
            "relations": sorted(self.edges["relation"].unique().tolist()),
        }


def load_dir(path: str) -> GraphData:
    """Read the on-disk format described at the top of this module."""
    with open(os.path.join(path, "graphlift.json")) as f:
        meta = json.load(f)
    nodes = {}
    for fn in sorted(os.listdir(os.path.join(path, "nodes"))):
        if fn.endswith(".csv"):
            nodes[fn[:-4]] = pd.read_csv(os.path.join(path, "nodes", fn), low_memory=False)
    edges = pd.read_csv(os.path.join(path, "edges.csv"))
    g = GraphData(nodes=nodes, edges=edges, target=meta["target"], label=meta["label"],
                  group=meta.get("group"), time=meta.get("time"),
                  breakdown=meta.get("breakdown"), exclude=meta.get("exclude", {}),
                  undirected=meta.get("undirected", []),
                  name=meta.get("name", os.path.basename(path.rstrip("/"))))
    for t, df in g.nodes.items():
        if "id" not in df.columns:
            raise ValueError(f"nodes/{t}.csv has no 'id' column")
        if df["id"].duplicated().any():
            raise ValueError(f"nodes/{t}.csv has duplicate ids")
    unknown = set(edges["src_type"]).union(edges["dst_type"]) - set(nodes)
    if unknown:
        raise ValueError(f"edges.csv refers to node types with no file: {sorted(unknown)}")
    return g.index()


def reserved_columns(g: GraphData, ntype: str) -> set[str]:
    """Columns that must never become features for this node type."""
    cols = {"id", *g.exclude.get(ntype, [])}
    if ntype == g.target:
        cols |= {c for c in (g.label, g.group, g.time, g.breakdown) if c}
    return cols


def encode(g: GraphData, ntype: str) -> tuple[np.ndarray, list[str]]:
    """Numeric feature matrix for one node type, standardised column by column.

    Numbers are kept, booleans become 0/1, low-cardinality strings are one-hot
    encoded, and everything else is dropped. Missing values become 0 after
    standardising, i.e. the column mean."""
    df = g.nodes[ntype]
    skip = reserved_columns(g, ntype)
    blocks, names = [], []
    for c in df.columns:
        if c in skip:
            continue
        s = df[c]
        if pd.api.types.is_bool_dtype(s):
            s = s.astype(float)
        if pd.api.types.is_numeric_dtype(s):
            v = s.to_numpy(dtype=float)
            sd = np.nanstd(v)
            if not np.isfinite(sd) or sd == 0:
                continue
            blocks.append(((v - np.nanmean(v)) / sd)[:, None])
            names.append(c)
        else:
            k = s.nunique(dropna=True)
            if 1 < k <= MAX_CATEGORIES:
                d = pd.get_dummies(s, prefix=c, dtype=float)
                blocks.append(d.to_numpy())
                names.extend(d.columns.tolist())
    if not blocks:
        return np.zeros((len(df), 0), dtype=np.float32), []
    x = np.nan_to_num(np.hstack(blocks), nan=0.0).astype(np.float32)
    return x, names
