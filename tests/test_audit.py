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
