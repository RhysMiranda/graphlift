"""The audit: does the graph find anything a non-graph model cannot?

Four models, each a gradient-boosted tree, each seeing more than the last:

    own      the target's own attributes only
    hubs     how shared its entities are (and how bursty), or for graphs whose
             targets link directly, how many neighbours it has per relation
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
import pandas as pd
import scipy.sparse as sp
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from . import gnn as gnn_mod
from .data import GraphData, encode
from .leakage import attribute_scan, exposure, resolve_groups
from .structure import adjacency, hub_features, hub_paths, propagate, typed_relations

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
    by_time: dict = field(default_factory=dict)    # temporal split: PR-AUC per test period
    gnn_lift: dict = field(default_factory=dict)   # the trained GNN's lift, when it was run
    oof: dict = field(default_factory=dict)       # model -> mean OOF score per target (NaN if unlabelled)
    feature_counts: dict = field(default_factory=dict)
    settings: dict = field(default_factory=dict)
    seconds: float = 0.0


def _folds(groups: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Assign whole groups to k folds at random; returns a fold id per target."""
    uniq, inv = np.unique(groups, return_inverse=True)
    perm = np.random.default_rng(seed).permutation(len(uniq))
    fold_of_group = np.empty(len(uniq), dtype=int)
    fold_of_group[perm] = np.arange(len(uniq)) % k
    return fold_of_group[inv]


def _model(seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0,
        early_stopping=False, random_state=seed)


def _predict(x: np.ndarray, y: np.ndarray, splits: list, seed: int) -> np.ndarray:
    """Scores for every target that is in some split's test side; NaN elsewhere."""
    p = np.full(len(y), np.nan)
    for tr, te in splits:
        if x.shape[1] == 0 or len(np.unique(y[tr])) < 2:
            p[te] = y[tr].mean()
            continue
        p[te] = _model(seed).fit(x[tr], y[tr]).predict_proba(x[te])[:, 1]
    return p


def _kfold(fold: np.ndarray) -> list:
    return [(fold != f, fold == f) for f in np.unique(fold)]


def _full(mask_lab: np.ndarray, lab_idx: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros(n, bool)
    out[lab_idx[mask_lab]] = True
    return out


def run(g: GraphData, seeds: int = 3, folds: int = 5, hops: int = 3, dim: int = 48,
        verbose: bool = True, gnn: bool = False, gnn_epochs: int = 300) -> AuditResult:
    t0 = time.time()
    say = print if verbose else (lambda *a, **k: None)
    lab = g.labelled
    lab_idx = np.flatnonzero(lab)
    y_all = g.y
    y = y_all[lab]
    n_t = len(y_all)
    if len(np.unique(y)) < 2:
        raise ValueError("The label has only one class among labelled targets.")

    extra = f" ({n_t - len(y):,} unlabelled kept as neighbours)" if len(y) < n_t else ""
    say(f"[graphlift] {g.name}: {len(y):,} labelled targets, {y.sum():,} positive "
        f"({y.mean():.1%}){extra}")
    rels = typed_relations(g)
    a = adjacency(g)
    groups, ginfo = resolve_groups(g, a)
    groups = groups[lab]
    say(f"[graphlift] groups from {ginfo['source']}: {len(np.unique(groups)):,}")

    say("[graphlift] building features")
    x_own, own_names = encode(g, g.target)
    paths = hub_paths(g, rels=rels)
    x_hub, hub_names = hub_features(g, paths, rels=rels)
    prop, prop_names = propagate(g, hops=hops, dim=dim, a=a, rels=rels)
    xs = {
        "own": x_own,
        "hubs": x_hub,
        "own+hubs": np.hstack([x_own, x_hub]),
        "graph": np.hstack([x_own, x_hub, *prop]),
    }
    xs = {m: x[lab] for m, x in xs.items()}
    direct = None
    tt = [r.m for r in rels if r.src == g.target and r.dst == g.target]
    if tt:
        direct = (sum(tt) > 0).astype(np.float32).tocsr()

    say("[graphlift] scanning attributes for label leaks")
    attrs = attribute_scan(g, a)

    # ---- the split ----------------------------------------------------------
    temporal = g.split == "temporal"
    if temporal:
        if not g.time or g.time not in g.targets:
            raise ValueError("A temporal split needs a `time` column on the target.")
        t_lab = pd.to_numeric(g.targets[g.time], errors="coerce").to_numpy()[lab]
        cut = g.temporal_cutoff if g.temporal_cutoff is not None else float(np.nanquantile(t_lab, 0.7))
        tr_t = t_lab <= cut
        ginfo["split"] = (f"temporal: train on `{g.time}` <= {cut:g} ({int(tr_t.sum()):,} targets), "
                          f"test on later ({int((~tr_t).sum()):,} targets)")
        seed_splits = [[(tr_t, ~tr_t)]]          # one fixed split; the trees are deterministic
    else:
        seed_splits = [_kfold(_folds(groups, folds, seed=s)) for s in range(seeds)]
        ginfo["split"] = f"grouped {folds}-fold by {ginfo['source']}, {seeds} seeds"
    say(f"[graphlift] split: {ginfo['split']}")

    if gnn and not gnn_mod.available():
        raise ImportError("--gnn needs PyTorch: pip install graphlift[gnn]")
    run_models = MODELS + (("gnn",) if gnn else ())
    gnn_cache: dict = {}
    per_seed = {m: {"pr": [], "roc": []} for m in run_models}
    oof_sum = {m: np.zeros(len(y)) for m in run_models}
    expo = []
    preds = {}
    for s, splits in enumerate(seed_splits):
        for tr, te in splits:
            expo.append(exposure(paths, y_all, _full(tr, lab_idx, n_t), _full(te, lab_idx, n_t), direct))
        for m in run_models:
            if m == "gnn":
                p = gnn_mod.predict(g, splits, seed=s, lab_idx=lab_idx, y=y,
                                    groups=None if temporal else groups,
                                    times=t_lab if temporal else None,
                                    epochs=gnn_epochs, cache=gnn_cache, verbose=verbose)
            else:
                p = _predict(xs[m], y, splits, seed=s)
            ev = ~np.isnan(p)
            preds[m] = p
            oof_sum[m] += np.nan_to_num(p)
            per_seed[m]["pr"].append(average_precision_score(y[ev], p[ev]))
            per_seed[m]["roc"].append(roc_auc_score(y[ev], p[ev]))
        say(f"[graphlift] {'test' if temporal else f'seed {s}'}: " + "  ".join(
            f"{m} {per_seed[m]['pr'][-1]:.3f}" for m in run_models))
    ev = ~np.isnan(preds["graph"])
    n_runs = len(seed_splits)

    if temporal:
        # Uncertainty from a paired bootstrap over test targets, stratified by class.
        rng = np.random.default_rng(0)
        yt = y[ev]
        pos, neg = np.flatnonzero(yt == 1), np.flatnonzero(yt == 0)
        boots = {m: [] for m in run_models}
        for _ in range(200):
            idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
            for m in run_models:
                boots[m].append(average_precision_score(yt[idx], preds[m][ev][idx]))
        metrics = {m: {"pr_auc_mean": per_seed[m]["pr"][0], "pr_auc_std": float(np.std(boots[m])),
                       "roc_auc_mean": per_seed[m]["roc"][0], "roc_auc_std": 0.0,
                       "pr_auc_seeds": per_seed[m]["pr"]} for m in run_models}
        best = max(BASELINES, key=lambda m: metrics[m]["pr_auc_mean"])

        def lift_of(m):
            diffs = np.array(boots[m]) - np.array(boots[best])
            return {"best_baseline": best,
                    "pr_auc_lift_mean": metrics[m]["pr_auc_mean"] - metrics[best]["pr_auc_mean"],
                    "pr_auc_lift_std": float(diffs.std()), "seeds_positive": int((diffs > 0).sum()),
                    "n_seeds": len(diffs), "uncertainty": "bootstrap"}
    else:
        metrics = {m: {"pr_auc_mean": float(np.mean(v["pr"])), "pr_auc_std": float(np.std(v["pr"])),
                       "roc_auc_mean": float(np.mean(v["roc"])), "roc_auc_std": float(np.std(v["roc"])),
                       "pr_auc_seeds": [float(z) for z in v["pr"]]}
                   for m, v in per_seed.items()}
        best = max(BASELINES, key=lambda m: metrics[m]["pr_auc_mean"])

        def lift_of(m):
            lifts = np.array(per_seed[m]["pr"]) - np.array(per_seed[best]["pr"])
            return {"best_baseline": best, "pr_auc_lift_mean": float(lifts.mean()),
                    "pr_auc_lift_std": float(lifts.std()), "seeds_positive": int((lifts > 0).sum()),
                    "n_seeds": seeds, "uncertainty": "seeds"}

    def finish(lf, m):
        lf.update(prevalence=float(y[ev].mean()),
                  graph_over_baseline_ratio=float(metrics[m]["pr_auc_mean"]
                                                  / max(metrics[best]["pr_auc_mean"], 1e-9)))
        lf["verdict"] = _verdict(lf)
        return lf

    lift = finish(lift_of("graph"), "graph")
    verdict = lift["verdict"]
    gnn_lift = finish(lift_of("gnn"), "gnn") if gnn else {}

    # How much would a careless random split have flattered the models?
    rnd = {}
    rnd_splits = _kfold(np.random.default_rng(0).integers(0, folds, len(y)))
    for m in (best, "graph"):
        rnd[m] = float(average_precision_score(y, _predict(xs[m], y, rnd_splits, seed=0)))
    expo_r = [exposure(paths, y_all, _full(tr, lab_idx, n_t), _full(te, lab_idx, n_t), direct)
              for tr, te in rnd_splits]
    random_split = {"pr_auc": rnd,
                    "split_pr_auc": {m: metrics[m]["pr_auc_mean"] for m in rnd},
                    "test_touches_train_positive": float(np.mean(
                        [e["test_touches_train_positive"] for e in expo_r]))}

    by_time = {}
    if temporal:
        tt = t_lab[ev]
        for t in np.unique(tt):
            sel = tt == t
            yy = y[ev][sel]
            row = {"n": int(sel.sum()), "positive": int(yy.sum())}
            for m in (best, "graph") + (("gnn",) if gnn else ()):
                row[m] = float(average_precision_score(yy, preds[m][ev][sel])) if 0 < yy.sum() < len(yy) else None
            by_time[f"{t:g}"] = row

    split_exposure = {k: float(np.mean([e[k] for e in expo])) for k in expo[0]}
    oof = {}
    for m, v in oof_sum.items():
        full = np.full(n_t, np.nan)
        vals = v / n_runs
        vals[~ev] = np.nan
        full[lab] = vals
        oof[m] = full
    breakdown = _breakdown(g, y, {m: v / n_runs for m, v in oof_sum.items()}, lab, ev)

    return AuditResult(
        dataset=g.summary(), groups=ginfo, split_exposure=split_exposure, attributes=attrs,
        metrics=metrics, lift=lift, verdict=verdict, random_split=random_split,
        breakdown=breakdown, by_time=by_time, gnn_lift=gnn_lift, oof=oof,
        feature_counts={"own": len(own_names), "hubs": len(hub_names),
                        "hub_paths": len(paths), "propagated": len(prop_names)},
        settings={"seeds": n_runs, "folds": folds, "hops": hops, "dim": dim, "split": g.split,
                  "gnn": bool(gnn), "gnn_epochs": gnn_epochs if gnn else None},
        seconds=time.time() - t0)


def _verdict(lift: dict) -> str:
    m, s = lift["pr_auc_lift_mean"], lift["pr_auc_lift_std"]
    consistent = lift["seeds_positive"] >= (0.95 * lift["n_seeds"] if lift.get("uncertainty") == "bootstrap"
                                            else lift["n_seeds"])
    if m > max(2 * s, 0.01) and consistent:
        return "graph helps"
    if m < -max(2 * s, 0.01):
        return "graph hurts"
    return "graph is decorative"


def _breakdown(g: GraphData, y: np.ndarray, oof: dict, lab: np.ndarray, ev: np.ndarray) -> dict:
    """Recall per category when each model flags as many targets as there are
    positives. Only computed if the dataset declares a `breakdown` column."""
    if not g.breakdown or g.breakdown not in g.targets:
        return {}
    cat = g.targets[g.breakdown].astype(str).to_numpy()[lab][ev]
    y = y[ev]
    oof = {m: p[ev] for m, p in oof.items()}
    k = int(y.sum())
    out = {}
    for m, p in oof.items():
        flagged = np.zeros(len(y), bool)
        flagged[np.argsort(-p)[:k]] = True
        for c in np.unique(cat[y == 1]):
            sel = (cat == c) & (y == 1)
            out.setdefault(c, {"n": int(sel.sum())})[m] = float(flagged[sel].mean())
    return dict(sorted(out.items(), key=lambda kv: -kv[1]["n"]))
