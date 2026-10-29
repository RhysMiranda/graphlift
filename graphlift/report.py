"""Render an AuditResult as a Markdown report card and as JSON."""
from __future__ import annotations

import json
from dataclasses import asdict

from .audit import BASELINES, MODELS, AuditResult

LABELS = {"own": "Own attributes only", "hubs": "Shared-entity counts only",
          "own+hubs": "Own + shared-entity counts", "graph": "Graph (multi-hop)",
          "gnn": "GraphSAGE (trained GNN)"}


def _models(r) -> list:
    return [m for m in MODELS + ("gnn",) if m in r.metrics]


def _lift_text(L: dict) -> str:
    return (f"{L['pr_auc_lift_mean']:+.3f} ± {L['pr_auc_lift_std']:.3f} PR-AUC** "
            + (f"(positive in {100 * L['seeds_positive'] / L['n_seeds']:.0f}% of {L['n_seeds']} paired "
               "bootstrap resamples of the test set). "
               if L.get("uncertainty") == "bootstrap" else
               f"({L['seeds_positive']} of {L['n_seeds']} seeds positive). "))


def to_json(r: AuditResult, path: str) -> None:
    d = asdict(r)
    d.pop("oof")
    with open(path, "w") as f:
        json.dump(d, f, indent=2)


def from_json(path: str) -> AuditResult:
    """Load a saved report.json back into an AuditResult (without per-target scores)."""
    with open(path) as f:
        d = json.load(f)
    fields = AuditResult.__dataclass_fields__
    return AuditResult(**{k: v for k, v in d.items() if k in fields})


def to_markdown(r: AuditResult) -> str:
    d, L, M = r.dataset, r.lift, r.metrics
    pct = lambda v: f"{100 * v:.1f}%"
    lines = [
        f"# graphlift report: {d['name']}",
        "",
        f"**Verdict: {r.verdict}.** "
        + _verdict_sentence(r),
        "",
        "## Does the graph help?",
        "",
        "| Model | PR-AUC | ROC-AUC |",
        "|---|---|---|",
    ]
    for m in _models(r):
        star = " (best non-graph)" if m == L["best_baseline"] else ""
        lines.append(f"| {LABELS[m]}{star} | {M[m]['pr_auc_mean']:.3f} ± {M[m]['pr_auc_std']:.3f}"
                     f" | {M[m]['roc_auc_mean']:.3f} ± {M[m]['roc_auc_std']:.3f} |")
    lines += [
        "",
        "**Lift: " + _lift_text(L)
        + (f"\n\n**GraphSAGE lift: " + _lift_text(r.gnn_lift) + f"Verdict for the trained GNN: "
           f"{r.gnn_lift['verdict']}. " if getattr(r, "gnn_lift", None) else "")
        + f"The PR-AUC floor is the prevalence, {pct(L['prevalence'])}, so compare scores "
        "within this table, not across datasets.",
        "",
        _split_sentence(r, d),
        "",
        "## Is the split honest?",
        "",
        f"- Groups: {r.groups['n_groups']:,} from {r.groups['source']}; the largest holds "
        f"{r.groups['largest_group']} target{'s' if r.groups['largest_group'] != 1 else ''}.",
        f"- Largest connected component: {pct(r.groups['largest_component_share'])} of targets.",
        f"- Test targets linked to a training target (directly or through a shared entity): "
        f"{pct(r.split_exposure['test_touches_train'])}; to a training positive: "
        f"{pct(r.split_exposure['test_touches_train_positive'])}.",
        f"- A random split would link {pct(r.random_split['test_touches_train_positive'])} of "
        "test targets to a training positive, and score:",
    ]
    split_name = "temporal" if r.settings.get("split") == "temporal" else "grouped"
    ref = r.random_split.get("split_pr_auc", r.random_split.get("grouped_pr_auc", {}))
    for m, v in r.random_split["pr_auc"].items():
        g = ref[m]
        lines.append(f"  - {LABELS[m]}: {v:.3f} random vs {g:.3f} {split_name} ({v - g:+.3f})")
    if "warning" in r.groups:
        lines += ["", f"> **Warning.** {r.groups['warning']}"]

    lines += ["", "## Attributes that predict the label on their own", "",
              "Each neighbour attribute alone, as max or min over 1 or 2 hops. "
              "Anything near 1.0 is probably a label, not an observation.", "",
              "| Attribute | AUC alone | Via | Status |", "|---|---|---|---|"]
    for a in r.attributes:
        status = ("excluded (label)" if a["excluded"] else
                  "**FLAG: check this**" if a["flag"] else "used")
        lines.append(f"| `{a['attribute']}` | {a['auc']:.3f} | {a['via']} | {status} |")

    if getattr(r, "by_time", None):
        best = L["best_baseline"]
        cols = [best, "graph"] + (["gnn"] if "gnn" in r.metrics else [])
        lines += ["", "## Over time", "",
                  "PR-AUC within each test period. Periods with no positives are blank.", "",
                  "| Period | Targets | Positive | " + " | ".join(LABELS[c] for c in cols)
                  + " | Graph minus best |", "|---|---|---|" + "---|" * (len(cols) + 1)]
        for t, v in r.by_time.items():
            vals = [v.get(c) for c in cols]
            cells = [f"{x:.3f}" if x is not None else "" for x in vals]
            b_, g_ = vals[0], vals[1]
            diff = f"{g_ - b_:+.3f}" if b_ is not None and g_ is not None else ""
            lines.append(f"| {t} | {v['n']:,} | {v['positive']:,} | " + " | ".join(cells) + f" | {diff} |")

    if r.breakdown:
        lines += ["", "## Recall by category", "",
                  "Share of each category's positives flagged when every model flags as many "
                  "targets as there are positives.", "",
                  "| Category | n | " + " | ".join(LABELS[m] for m in _models(r)) + " |",
                  "|---|---|" + "---|" * len(_models(r))]
        for c, v in r.breakdown.items():
            lines.append(f"| {c} | {v['n']} | " + " | ".join(f"{v[m]:.2f}" for m in _models(r)) + " |")

    fc = r.feature_counts
    gnn_line = ([f"- GraphSAGE: a relation-aware GraphSAGE trained end to end (3 layers, mean over all "
                 f"neighbours plus max over up to 10 sampled, early-stopped on held-out training "
                 f"targets, at most {r.settings.get('gnn_epochs')} epochs)."]
                if r.settings.get("gnn") else [])
    lines += ["", "## What each model saw", "",
              f"- Own attributes: {fc['own']} columns.",
              f"- Shared-entity and direct-link counts: {fc['hubs']} columns ({fc['hub_paths']} "
              "entity paths of length 1–2, plus neighbour and two-step walk counts for every "
              "relation that links targets directly).",
              f"- Graph: {fc['propagated']} extra columns, mean and max neighbourhood "
              f"summaries over {r.settings['hops']} hops (SIGN-style propagation, PCA to "
              f"{r.settings['dim']} per hop, no labels used), fed to the same gradient-boosted trees.",
              *gnn_line,
              "",
              f"Ran in {r.seconds:.0f} s."]
    return "\n".join(lines) + "\n"


def _split_sentence(r: AuditResult, d: dict) -> str:
    unl = (f" Only the {d['n_labelled']:,} labelled targets of {d['n_targets']:,} are scored."
           if d.get("n_labelled", d["n_targets"]) < d["n_targets"] else "")
    if r.settings.get("split") == "temporal":
        return (f"Split: {r.groups.get('split', 'temporal')}. Models are trained once on the "
                "past and scored on the future; ± is the spread over 200 paired bootstrap "
                "resamples of the test set." + unl)
    return (f"Out-of-fold scores, {r.settings['folds']}-fold split by {r.groups['source']}, "
            f"mean ± std over {r.settings['seeds']} seeds." + unl)


def _verdict_sentence(r: AuditResult) -> str:
    L, M = r.lift, r.metrics
    b = L["best_baseline"]
    if r.verdict == "graph helps":
        return (f"Multi-hop structure lifts PR-AUC from {M[b]['pr_auc_mean']:.3f} to "
                f"{M['graph']['pr_auc_mean']:.3f} over the best model that never passes a "
                f"message ({L['graph_over_baseline_ratio']:.2f}x).")
    if r.verdict == "graph hurts":
        return "The graph model scores below a model using simple shared-entity counts."
    lead = (f"The graph model's edge over the best non-graph model ({LABELS[b].lower()}) is "
            f"{L['pr_auc_lift_mean']:+.3f} PR-AUC, ")
    if abs(L["pr_auc_lift_mean"]) <= 2 * L["pr_auc_lift_std"]:
        lead += "within noise."
    else:
        lead += "consistent but below the 0.01 that graphlift treats as a practical difference."
    where = {"own": " What the graph knows is already in the targets' own columns.",
             "hubs": " Counts of neighbours and shared entities capture what the graph knows.",
             "own+hubs": " Own columns plus neighbour and shared-entity counts capture what "
                         "the graph knows."}[b]
    return lead + where


def index_markdown(results: list[AuditResult]) -> str:
    """One row per dataset: how much the graph adds, and how much of that a
    model gets from counts alone, without passing a single message."""
    lines = [
        "| Dataset | Labelled targets | Positive | Own attributes | Best non-graph | Graph (propagation) | Lift | GraphSAGE | GraphSAGE lift | Share of the graph's gain from counts | Verdict |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(results, key=lambda r: -r.lift["pr_auc_lift_mean"]):
        d, M, L = r.dataset, r.metrics, r.lift
        own = M["own"]["pr_auc_mean"]
        base = M[L["best_baseline"]]["pr_auc_mean"]
        graph = M["graph"]["pr_auc_mean"]
        total = graph - own
        share = (base - own) / total if total > 0.005 else float("nan")
        share_txt = "n/a" if share != share else f"{100 * min(max(share, 0), 1):.0f}%"
        G = getattr(r, "gnn_lift", None) or {}
        gnn_txt = (f"{M['gnn']['pr_auc_mean']:.3f} | {G['pr_auc_lift_mean']:+.3f} ± {G['pr_auc_lift_std']:.3f}"
                   if G else "not run | –")
        lines.append(
            f"| {d['name']} | {d.get('n_labelled', d['n_targets']):,} | {100 * d['prevalence']:.1f}% "
            f"| {own:.3f} | {base:.3f} | {graph:.3f} "
            f"| {L['pr_auc_lift_mean']:+.3f} ± {L['pr_auc_lift_std']:.3f} | {gnn_txt} | {share_txt} | {r.verdict} |")
    lines += ["",
              "PR-AUC, out-of-fold, mean over seeds. *Best non-graph* is the strongest model that "
              "never passes a message (own attributes, counts of shared entities or direct links, "
              "or both). *Lift* is graph minus best non-graph; *GraphSAGE lift* is the trained GNN "
              "minus the same baseline. *Share from counts* is how much of "
              "the graph model's gain over own attributes alone the counts model already gets."]
    return "\n".join(lines) + "\n"
