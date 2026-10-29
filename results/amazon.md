# graphlift report: Amazon (fraudulent users)

**Verdict: graph is decorative.** The graph model's edge over the best non-graph model (own + shared-entity counts) is +0.006 PR-AUC, consistent but below the 0.01 that graphlift treats as a practical difference. Own columns plus neighbour and shared-entity counts capture what the graph knows.

## Does the graph help?

| Model | PR-AUC | ROC-AUC |
|---|---|---|
| Own attributes only | 0.915 ± 0.002 | 0.981 ± 0.000 |
| Shared-entity counts only | 0.502 ± 0.005 | 0.848 ± 0.004 |
| Own + shared-entity counts (best non-graph) | 0.935 ± 0.001 | 0.986 ± 0.000 |
| Graph (multi-hop) | 0.941 ± 0.001 | 0.988 ± 0.000 |
| GraphSAGE (trained GNN) | 0.910 ± 0.003 | 0.976 ± 0.000 |

**Lift: +0.006 ± 0.001 PR-AUC** (3 of 3 seeds positive; 95% interval [+0.001, +0.013] from 200 resamples of the groups). 

**GraphSAGE lift: -0.025 ± 0.004 PR-AUC** (0 of 3 seeds positive; 95% interval [-0.034, -0.006] from 200 resamples of the groups). Verdict for the trained GNN: graph hurts. The PR-AUC floor is the prevalence, 9.5%, so compare scores within this table, not across datasets.

Out-of-fold scores, 5-fold split by individual targets (no leak-free grouping exists), mean ± std over 3 seeds. Only the 8,639 labelled targets of 11,944 are scored.

## Is the split honest?

- Groups: 8,639 from individual targets (no leak-free grouping exists); the largest holds 1 target.
- Largest connected component: 100.0% of targets.
- Test targets linked to a training target (directly or through a shared entity): 100.0%; to a training positive: 96.3%.
- A random split would link 96.2% of test targets to a training positive, and score:
  - Own + shared-entity counts: 0.933 random vs 0.935 grouped (-0.001)
  - Graph (multi-hop): 0.941 random vs 0.941 grouped (+0.000)

> **Warning.** Over half the targets sit in one connected component, so no split can keep connected targets apart and the split falls back to individual targets, as most published results on this kind of graph do. Treat absolute scores as optimistic; the comparison between models stays fair because every model sees the same split.

## Attributes that predict the label on their own

Each neighbour attribute alone, as max or min over 1 or 2 hops. Anything near 1.0 is probably a label, not an observation.

| Attribute | AUC alone | Via | Status |
|---|---|---|---|
| `user.f19` | 0.883 | own column | used |
| `user.f18` | 0.870 | own column | used |
| `user.f15` | 0.866 | own column | used |
| `user.f13` | 0.852 | own column | used |
| `user.f16` | 0.838 | own column | used |
| `user.f17` | 0.817 | own column | used |
| `user.f12` | 0.812 | own column | used |
| `user.f06` | 0.802 | own column | used |

## What each model saw

- Own attributes: 25 columns.
- Shared-entity and direct-link counts: 6 columns (0 entity paths of length 1–2, plus neighbour and two-step walk counts for every relation that links targets directly).
- Graph: 297 extra columns, mean and max neighbourhood summaries over 3 hops (SIGN-style propagation, PCA to 48 per hop, no labels used), fed to the same gradient-boosted trees.
- GraphSAGE: a relation-aware GraphSAGE trained end to end (3 layers, mean over all neighbours plus max over up to 10 sampled, early-stopped on held-out training targets, at most 300 epochs).

Ran in 1853 s.
