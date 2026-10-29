"""Convert the Elliptic Bitcoin dataset to graphlift's on-disk format.

Download "Elliptic Data Set" (ellipticco) from Kaggle and unzip it, then:

    python examples/elliptic/convert_elliptic.py path/to/elliptic_bitcoin_dataset data/elliptic

203,769 Bitcoin transactions; an edge means money flowed from one transaction
to the next. 4,545 are labelled illicit, 42,019 licit, the rest unknown
(kept in the graph as neighbours, never scored). Transactions fall into 49
time steps about two weeks apart, and no edge crosses a time step.

Each transaction has 166 features after its id:
    column 1          the time step
    columns 2-94      93 "local" features: the transaction itself
    columns 95-166    72 "aggregated" features: statistics of its one-hop
                      neighbours, computed by the dataset's authors

The aggregated block is graph information baked into the table. So the
converter writes two configs over the same files:

    graphlift.json        all features, as the dataset ships
    graphlift_local.json  the 72 aggregated columns excluded

and the audit is run on each:

    graphlift audit data/elliptic
    graphlift audit data/elliptic --meta graphlift_local.json --out data/elliptic/graphlift_report_local

Both use the dataset's standard temporal split (Weber et al., 2019): train on
time steps 1-34, test on 35-49.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

N_LOCAL = 93
N_AGG = 72


def main():
    p = argparse.ArgumentParser()
    p.add_argument("src", help="folder with elliptic_txs_{features,classes,edgelist}.csv")
    p.add_argument("out")
    a = p.parse_args()

    feats = pd.read_csv(os.path.join(a.src, "elliptic_txs_features.csv"), header=None)
    if feats.shape[1] != 2 + N_LOCAL + N_AGG:
        raise ValueError(f"expected {2 + N_LOCAL + N_AGG} columns, found {feats.shape[1]}")
    local = [f"local_{i:02d}" for i in range(N_LOCAL)]
    agg = [f"agg_{i:02d}" for i in range(N_AGG)]
    feats.columns = ["id", "time_step"] + local + agg

    cls = pd.read_csv(os.path.join(a.src, "elliptic_txs_classes.csv"))
    label = cls["class"].astype(str).map({"1": 1.0, "2": 0.0})      # unknown -> NaN
    feats = feats.merge(pd.DataFrame({"id": cls["txId"], "illicit": label}), on="id", how="left")

    os.makedirs(os.path.join(a.out, "nodes"), exist_ok=True)
    # Six significant figures keep every feature's information and halve the file.
    feats.to_csv(os.path.join(a.out, "nodes", "tx.csv"), index=False, float_format="%.6g")

    e = pd.read_csv(os.path.join(a.src, "elliptic_txs_edgelist.csv"))
    pd.DataFrame({"src_type": "tx", "src_id": e["txId1"], "relation": "pays",
                  "dst_type": "tx", "dst_id": e["txId2"]}).to_csv(
        os.path.join(a.out, "edges.csv"), index=False)

    base = {"target": "tx", "label": "illicit", "time": "time_step",
            "split": "temporal", "temporal_cutoff": 34}
    full = {**base, "name": "Elliptic (illicit Bitcoin), all features",
            "notes": "Weber et al., KDD 2019 workshop. Includes the 72 aggregated "
                     "neighbourhood features the dataset ships with."}
    loc = {**base, "name": "Elliptic (illicit Bitcoin), local features only",
           "exclude": {"tx": agg},
           "notes": "Weber et al., KDD 2019 workshop. The 72 aggregated neighbourhood "
                    "features are excluded, so all graph information comes from the edges."}
    for fn, meta in (("graphlift.json", full), ("graphlift_local.json", loc)):
        with open(os.path.join(a.out, fn), "w") as f:
            json.dump(meta, f, indent=2)

    lab = feats["illicit"].notna()
    print(f"{len(feats):,} transactions, {len(e):,} edges, {int(lab.sum()):,} labelled, "
          f"{int(feats['illicit'].sum()):,} illicit ({feats.loc[lab, 'illicit'].mean():.1%})")
    test = lab & (feats["time_step"] > 34)
    print(f"temporal split: {int((lab & ~test).sum()):,} train (steps 1-34), "
          f"{int(test.sum()):,} test (steps 35-49), "
          f"{feats.loc[test, 'illicit'].mean():.1%} illicit in test")


if __name__ == "__main__":
    main()
