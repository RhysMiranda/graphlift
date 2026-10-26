"""Render an AuditResult as a Markdown report card and as JSON."""
from __future__ import annotations

import json
from dataclasses import asdict

from .audit import BASELINES, MODELS, AuditResult

LABELS = {"own": "Own attributes only", "hubs": "Shared-entity counts only",
          "own+hubs": "Own + shared-entity counts", "graph": "Graph (multi-hop)"}


def to_json(r: AuditResult, path: str) -> None:
    d = asdict(r)
    d.pop("oof")
    with open(path, "w") as f:
        json.dump(d, f, indent=2)


def from_json(path: str) -> AuditResult:
    """Load a saved report.json back into an AuditResult (without per-target scores)."""
    with open(path) as f:
        d = json.load(f)
    return AuditResult(**d)


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
    for m in MODELS:
        star = " (best non-graph)" if m == L["best_baseline"] else ""
        lines.append(f"| {LABELS[m]}{star} | {M[m]['pr_auc_mean']:.3f} ± {M[m]['pr_auc_std']:.3f}"
                     f" | {M[m]['roc_auc_mean']:.3f} ± {M[m]['roc_auc_std']:.3f} |")
    lines += [
        "",
        f"**Lift: {L['pr_auc_lift_mean']:+.3f} ± {L['pr_auc_lift_std']:.3f} PR-AUC** "
        f"({L['seeds_positive']} of {L['n_seeds']} seeds positive). "
        f"The PR-AUC floor is the prevalence, {pct(L['prevalence'])}, so compare scores "
        "within this table, not across datasets.",
        "",
        f"Out-of-fold scores, {r.settings['folds']}-fold split by {r.groups['source']}, "
        f"mean ± std over {r.settings['seeds']} seeds"
        + (f", on the {d['n_labelled']:,} labelled targets of {d['n_targets']:,}."
           if d.get('n_labelled', d['n_targets']) < d['n_targets'] else "."),
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
    for m, v in r.random_split["pr_auc"].items():
        g = r.random_split["grouped_pr_auc"][m]
        lines.append(f"  - {LABELS[m]}: {v:.3f} random vs {g:.3f} grouped ({v - g:+.3f})")
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

    if r.breakdown:
        lines += ["", "## Recall by category", "",
                  "Share of each category's positives flagged when every model flags as many "
                  "targets as there are positives.", "",
                  "| Category | n | " + " | ".join(LABELS[m] for m in MODELS) + " |",
                  "|---|---|" + "---|" * len(MODELS)]
        for c, v in r.breakdown.items():
            lines.append(f"| {c} | {v['n']} | " + " | ".join(f"{v[m]:.2f}" for m in MODELS) + " |")

    fc = r.feature_counts
    lines += ["", "## What each model saw", "",
              f"- Own attributes: {fc['own']} columns.",
              f"- Shared-entity and direct-link counts: {fc['hubs']} columns ({fc['hub_paths']} "
              "entity paths of length 1–2, plus neighbour and two-step walk counts for every "
              "relation that links targets directly).",
              f"- Graph: {fc['propagated']} extra columns, mean and max neighbourhood "
              f"summaries over {r.settings['hops']} hops (SIGN-style propagation, PCA to "
              f"{r.settings['dim']} per hop, no labels used).",
              "",
              f"Ran in {r.seconds:.0f} s."]
    return "\n".join(lines) + "\n"


def _verdict_sentence(r: AuditResult) -> str:
    L, M = r.lift, r.metrics
    b = L["best_baseline"]
    if r.verdict == "graph helps":
        return (f"Multi-hop structure lifts PR-AUC from {M[b]['pr_auc_mean']:.3f} to "
                f"{M['graph']['pr_auc_mean']:.3f} over the best model that never passes a "
                f"message ({L['graph_over_baseline_ratio']:.2f}x).")
    if r.verdict == "graph hurts":
        return "The graph model scores below a model using simple shared-entity counts."
    return (f"The best non-graph model ({LABELS[b].lower()}) is within noise of the graph "
            "model. The structure's signal is captured by a few counts.")


def index_markdown(results: list[AuditResult]) -> str:
    """One row per dataset: how much the graph adds, and how much of that a
    model gets from counts alone, without passing a single message."""
    lines = [
        "| Dataset | Labelled targets | Positive | Own attributes | Best non-graph | Graph | Lift | Share of the graph's gain from counts | Verdict |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(results, key=lambda r: -r.lift["pr_auc_lift_mean"]):
        d, M, L = r.dataset, r.metrics, r.lift
        own = M["own"]["pr_auc_mean"]
        base = M[L["best_baseline"]]["pr_auc_mean"]
        graph = M["graph"]["pr_auc_mean"]
        total = graph - own
        share = (base - own) / total if total > 0.005 else float("nan")
        share_txt = "n/a" if share != share else f"{100 * min(max(share, 0), 1):.0f}%"
        lines.append(
            f"| {d['name']} | {d.get('n_labelled', d['n_targets']):,} | {100 * d['prevalence']:.1f}% "
            f"| {own:.3f} | {base:.3f} | {graph:.3f} "
            f"| {L['pr_auc_lift_mean']:+.3f} ± {L['pr_auc_lift_std']:.3f} | {share_txt} | {r.verdict} |")
    lines += ["",
              "PR-AUC, out-of-fold, mean over seeds. *Best non-graph* is the strongest model that "
              "never passes a message (own attributes, counts of shared entities or direct links, "
              "or both). *Lift* is graph minus best non-graph. *Share from counts* is how much of "
              "the graph model's gain over own attributes alone the counts model already gets."]
    return "\n".join(lines) + "\n"
