"""The audit: does the graph find anything a non-graph model cannot?

Four models, each a gradient-boosted tree, each seeing more than the last:

    own      the target's own attributes only
    hubs     how shared its entities are, and how bursty (no own attributes)
    own+hubs both: the strongest model that never passes a message
    graph    own+hubs plus multi-hop neighbourhood summaries

Every score is out-of-fold under a grouped split, repeated over seeds.
**Lift** is graph minus the best of the first three. If lift is indistinguishable
from zero, the graph is decorative: its signal is a handful of counts.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from .data import GraphData, encode
from .leakage import attribute_scan, exposure, resolve_groups
from .structure import adjacency, hub_features, hub_paths, propagate

BASELINES = ("own", "hubs", "own+hubs")
MODELS = BASELINES + ("graph",)


@dataclass
class AuditResult:
    dataset: dict
    groups: dict
    split_exposure: dict
    attributes: list
    metrics: dict                    # model -> {pr_auc_mean, pr_auc_std, ...}
    lift: dict
    verdict: str
    random_split: dict
    breakdown: dict = field(default_factory=dict)
    oof: dict = field(default_factory=dict)       # model -> mean OOF score per target
    feature_counts: dict = field(default_factory=dict)
    settings: dict = field(default_factory=dict)
    seconds: float = 0.0


def _folds(groups: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Assign whole groups to k folds at random; returns a fold id per target."""
    uniq = np.unique(groups)
    perm = np.random.default_rng(seed).permutation(len(uniq))
    fold_of = dict(zip(uniq[perm], np.arange(len(uniq)) % k))
    return np.array([fold_of[g] for g in groups])


def _model(seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0,
        early_stopping=False, random_state=seed)


def _oof(x: np.ndarray, y: np.ndarray, fold: np.ndarray, seed: int) -> np.ndarray:
    p = np.zeros(len(y))
    for f in np.unique(fold):
        te = fold == f
        tr = ~te
        if len(np.unique(y[tr])) < 2:
            p[te] = y[tr].mean()
            continue
        p[te] = _model(seed).fit(x[tr], y[tr]).predict_proba(x[te])[:, 1]
    return p


def run(g: GraphData, seeds: int = 3, folds: int = 5, hops: int = 3, dim: int = 48,
        verbose: bool = True) -> AuditResult:
    t0 = time.time()
    say = print if verbose else (lambda *a, **k: None)
    y = g.y
    if len(np.unique(y)) < 2:
        raise ValueError("The label has only one class.")

    say(f"[graphlift] {g.name}: {len(y):,} targets, {y.sum():,} positive ({y.mean():.1%})")
    a = adjacency(g)
    groups, ginfo = resolve_groups(g, a)
    say(f"[graphlift] groups from {ginfo['source']}: {ginfo['n_groups']:,}")

    say("[graphlift] building features")
    x_own, own_names = encode(g, g.target)
    paths = hub_paths(g)
    x_hub, hub_names = hub_features(g, paths)
    prop, prop_names = propagate(g, hops=hops, dim=dim, a=a)
    xs = {
        "own": x_own,
        "hubs": x_hub,
        "own+hubs": np.hstack([x_own, x_hub]),
        "graph": np.hstack([x_own, x_hub, *prop]),
    }

    say("[graphlift] scanning attributes for label leaks")
    attrs = attribute_scan(g, a)

    per_seed = {m: {"pr": [], "roc": []} for m in MODELS}
    oof_sum = {m: np.zeros(len(y)) for m in MODELS}
    expo = []
    for s in range(seeds):
        fold = _folds(groups, folds, seed=s)
        for f in np.unique(fold):
            te = fold == f
            expo.append(exposure(paths, y, ~te, te))
        for m in MODELS:
            p = _oof(xs[m], y, fold, seed=s)
            oof_sum[m] += p
            per_seed[m]["pr"].append(average_precision_score(y, p))
            per_seed[m]["roc"].append(roc_auc_score(y, p))
        say(f"[graphlift] seed {s}: " + "  ".join(
            f"{m} {per_seed[m]['pr'][-1]:.3f}" for m in MODELS))

    metrics = {m: {"pr_auc_mean": float(np.mean(v["pr"])), "pr_auc_std": float(np.std(v["pr"])),
                   "roc_auc_mean": float(np.mean(v["roc"])), "roc_auc_std": float(np.std(v["roc"])),
                   "pr_auc_seeds": [float(z) for z in v["pr"]]}
               for m, v in per_seed.items()}
    best = max(BASELINES, key=lambda m: metrics[m]["pr_auc_mean"])
    lifts = np.array(per_seed["graph"]["pr"]) - np.array(per_seed[best]["pr"])
    lift = {"best_baseline": best, "pr_auc_lift_mean": float(lifts.mean()),
            "pr_auc_lift_std": float(lifts.std()), "seeds_positive": int((lifts > 0).sum()),
            "n_seeds": seeds, "prevalence": float(y.mean()),
            "graph_over_baseline_ratio": float(metrics["graph"]["pr_auc_mean"]
                                               / max(metrics[best]["pr_auc_mean"], 1e-9))}
    verdict = _verdict(lift)

    # How much would a careless random split have flattered the models?
    rnd = {}
    fold_r = np.random.default_rng(0).integers(0, folds, len(y))
    for m in (best, "graph"):
        rnd[m] = float(average_precision_score(y, _oof(xs[m], y, fold_r, seed=0)))
    expo_r = [exposure(paths, y, fold_r != f, fold_r == f) for f in range(folds)]
    random_split = {"pr_auc": rnd,
                    "grouped_pr_auc": {m: metrics[m]["pr_auc_mean"] for m in rnd},
                    "test_touches_train_positive": float(np.mean(
                        [e["test_touches_train_positive"] for e in expo_r]))}

    split_exposure = {k: float(np.mean([e[k] for e in expo])) for k in expo[0]}
    oof = {m: v / seeds for m, v in oof_sum.items()}
    breakdown = _breakdown(g, y, oof)

    return AuditResult(
        dataset=g.summary(), groups=ginfo, split_exposure=split_exposure, attributes=attrs,
        metrics=metrics, lift=lift, verdict=verdict, random_split=random_split,
        breakdown=breakdown, oof=oof,
        feature_counts={"own": len(own_names), "hubs": len(hub_names),
                        "hub_paths": len(paths), "propagated": len(prop_names)},
        settings={"seeds": seeds, "folds": folds, "hops": hops, "dim": dim},
        seconds=time.time() - t0)


def _verdict(lift: dict) -> str:
    m, s = lift["pr_auc_lift_mean"], lift["pr_auc_lift_std"]
    if m > max(2 * s, 0.01) and lift["seeds_positive"] == lift["n_seeds"]:
        return "graph helps"
    if m < -max(2 * s, 0.01):
        return "graph hurts"
    return "graph is decorative"


def _breakdown(g: GraphData, y: np.ndarray, oof: dict) -> dict:
    """Recall per category when each model flags as many targets as there are
    positives. Only computed if the dataset declares a `breakdown` column."""
    if not g.breakdown or g.breakdown not in g.targets:
        return {}
    cat = g.targets[g.breakdown].astype(str).to_numpy()
    k = int(y.sum())
    out = {}
    for m, p in oof.items():
        flagged = np.zeros(len(y), bool)
        flagged[np.argsort(-p)[:k]] = True
        for c in np.unique(cat[y == 1]):
            sel = (cat == c) & (y == 1)
            out.setdefault(c, {"n": int(sel.sum())})[m] = float(flagged[sel].mean())
    return dict(sorted(out.items(), key=lambda kv: -kv[1]["n"]))
