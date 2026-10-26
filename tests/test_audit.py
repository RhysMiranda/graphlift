"""Two planted worlds: one where structure carries the signal, one where it can't."""
import json
import os

import numpy as np
import pandas as pd

from graphlift import audit, load_dir


def _write(tmp, structured: bool, seed: int = 0):
    rng = np.random.default_rng(seed)
    apps, edges, dev = [], [], []
    n_dev = 0
    gid = 0
    for _ in range(150):                      # small clusters of applications
        k = int(rng.integers(1, 5))
        bad = rng.random() < 0.3
        d = f"d{n_dev}"
        n_dev += 1
        dev.append({"id": d, "proxy": int(bad and structured) if k > 1 else 0})
        for _ in range(k):
            i = f"a{len(apps)}"
            # Labels only show up in the shared device's attribute when `structured`.
            y = int(bad) if structured else int(rng.random() < 0.3)
            apps.append({"id": i, "x": rng.normal(), "y": y, "group": f"g{gid}"})
            edges.append(("app", i, "uses", "device", d))
        gid += 1
    os.makedirs(tmp / "nodes", exist_ok=True)
    pd.DataFrame(apps).to_csv(tmp / "nodes" / "app.csv", index=False)
    pd.DataFrame(dev).to_csv(tmp / "nodes" / "device.csv", index=False)
    pd.DataFrame(edges, columns=["src_type", "src_id", "relation", "dst_type", "dst_id"]
                 ).to_csv(tmp / "edges.csv", index=False)
    json.dump({"target": "app", "label": "y", "group": "group"}, open(tmp / "graphlift.json", "w"))
    return tmp


def test_graph_helps_when_signal_is_in_the_neighbourhood(tmp_path):
    g = load_dir(str(_write(tmp_path, structured=True)))
    r = audit.run(g, seeds=2, folds=3, hops=2, dim=8, verbose=False)
    assert r.metrics["graph"]["pr_auc_mean"] > r.metrics["own"]["pr_auc_mean"] + 0.1


def test_nothing_to_find_when_labels_are_random(tmp_path):
    g = load_dir(str(_write(tmp_path, structured=False)))
    r = audit.run(g, seeds=2, folds=3, hops=2, dim=8, verbose=False)
    assert r.verdict != "graph helps"


def test_direct_links_and_unlabelled_targets(tmp_path):
    """Targets linked straight to each other (YelpChi-style), some unlabelled."""
    rng = np.random.default_rng(1)
    n = 400
    y = (rng.random(n) < 0.2).astype(float)
    # A weak, noisy signal on each node; averaging same-class neighbours sharpens it.
    x = rng.normal(loc=0.7 * y, scale=1.5)
    # Positives link mostly to positives on relation "same", at random on "noise".
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    rows = []
    for i in range(n):
        pool = pos if (y[i] == 1 and rng.random() < 0.8) else neg
        for j in rng.choice(pool, 3):
            if i != j:
                rows.append(("t", i, "same", "t", int(j)))
        for j in rng.integers(0, n, 3):
            if i != j:
                rows.append(("t", i, "noise", "t", int(j)))
    y[:60] = np.nan                                   # unlabelled, still in the graph
    os.makedirs(tmp_path / "nodes", exist_ok=True)
    pd.DataFrame({"id": np.arange(n), "x": x, "y": y}).to_csv(tmp_path / "nodes" / "t.csv", index=False)
    pd.DataFrame(rows, columns=["src_type", "src_id", "relation", "dst_type", "dst_id"]
                 ).to_csv(tmp_path / "edges.csv", index=False)
    json.dump({"target": "t", "label": "y", "undirected": ["same", "noise"]},
              open(tmp_path / "graphlift.json", "w"))
    g = load_dir(str(tmp_path))
    assert g.labelled.sum() == n - 60
    r = audit.run(g, seeds=2, folds=3, hops=2, dim=8, verbose=False)
    assert r.dataset["n_labelled"] == n - 60
    assert np.isnan(r.oof["graph"][:60]).all() and not np.isnan(r.oof["graph"][60:]).any()
    assert r.metrics["graph"]["pr_auc_mean"] > r.metrics["own"]["pr_auc_mean"] + 0.05
    assert r.groups["n_groups"] == n - 60
