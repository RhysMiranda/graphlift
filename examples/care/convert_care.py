"""Convert the YelpChi and Amazon fraud graphs to graphlift's on-disk format.

Both come from CARE-GNN (Dou et al., CIKM 2020) and are the two most-used
benchmarks for GNN fraud detection:

    git clone --depth 1 https://github.com/YingtongDou/CARE-GNN
    unzip CARE-GNN/data/YelpChi.zip -d raw && unzip CARE-GNN/data/Amazon.zip -d raw
    python examples/care/convert_care.py raw/YelpChi.mat data/yelpchi
    python examples/care/convert_care.py raw/Amazon.mat  data/amazon

YelpChi: 45,954 hotel and restaurant reviews, spam vs not, linked by
    R-U-R  written by the same user
    R-T-R  same product, same month, same star rating
    R-S-R  same product, same star rating
Amazon: 11,944 users of musical-instrument reviews, fraudulent vs not, linked by
    U-P-U  reviewed at least one same product
    U-S-U  gave at least one same star rating within a week
    U-V-U  top-5% mutual review text similarity

Amazon's first 3,305 users are unlabelled (CARE-GNN's own training code skips
them). They stay in the graph as neighbours but get no label here.

Every relation is symmetric, so each pair is written once and the relation is
declared `undirected` in graphlift.json.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.sparse as sp

SPECS = {
    "yelpchi": {"target": "review", "rels": {"net_rur": "R-U-R", "net_rtr": "R-T-R", "net_rsr": "R-S-R"},
                "unlabelled": 0, "name": "YelpChi (spam reviews)"},
    "amazon": {"target": "user", "rels": {"net_upu": "U-P-U", "net_usu": "U-S-U", "net_uvu": "U-V-U"},
               "unlabelled": 3305, "name": "Amazon (fraudulent users)"},
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mat")
    p.add_argument("out")
    a = p.parse_args()
    key = "amazon" if "amazon" in os.path.basename(a.mat).lower() else "yelpchi"
    spec = SPECS[key]
    m = sio.loadmat(a.mat)
    x = m["features"].toarray() if sp.issparse(m["features"]) else np.asarray(m["features"])
    y = m["label"].ravel().astype(float)
    y[: spec["unlabelled"]] = np.nan
    n = len(y)

    os.makedirs(os.path.join(a.out, "nodes"), exist_ok=True)
    nodes = pd.DataFrame(x, columns=[f"f{i:02d}" for i in range(x.shape[1])])
    nodes.insert(0, "id", np.arange(n))
    nodes["label"] = y
    nodes.to_csv(os.path.join(a.out, "nodes", f"{spec['target']}.csv"), index=False)

    parts = []
    for mk, rel in spec["rels"].items():
        up = sp.triu(m[mk].tocsr(), k=1).tocoo()
        parts.append(pd.DataFrame({"src_type": spec["target"], "src_id": up.row, "relation": rel,
                                   "dst_type": spec["target"], "dst_id": up.col}))
        print(f"{rel}: {up.nnz:,} undirected edges")
    pd.concat(parts).to_csv(os.path.join(a.out, "edges.csv"), index=False)

    meta = {"name": spec["name"], "target": spec["target"], "label": "label",
            "undirected": list(spec["rels"].values()),
            "notes": "CARE-GNN release (Dou et al., CIKM 2020)."
                     + (f" Targets 0-{spec['unlabelled'] - 1} are unlabelled." if spec["unlabelled"] else "")}
    with open(os.path.join(a.out, "graphlift.json"), "w") as f:
        json.dump(meta, f, indent=2)
    lab = ~np.isnan(y)
    print(f"{spec['name']}: {n:,} nodes, {int(lab.sum()):,} labelled, "
          f"{int(np.nansum(y)):,} positive ({np.nanmean(y):.1%})")


if __name__ == "__main__":
    main()
