# graphlift report: Amazon (fraudulent users)

**Verdict: graph is decorative.** The best non-graph model (own + shared-entity counts) is within noise of the graph model. The structure's signal is captured by a few counts.

## Does the graph help?

| Model | PR-AUC | ROC-AUC |
|---|---|---|
| Own attributes only | 0.915 ± 0.001 | 0.981 ± 0.001 |
| Shared-entity counts only | 0.501 ± 0.003 | 0.848 ± 0.004 |
| Own + shared-entity counts (best non-graph) | 0.935 ± 0.001 | 0.986 ± 0.000 |
| Graph (multi-hop) | 0.940 ± 0.001 | 0.987 ± 0.001 |

**Lift: +0.005 ± 0.001 PR-AUC** (3 of 3 seeds positive). The PR-AUC floor is the prevalence, 9.5%, so compare scores within this table, not across datasets.

Out-of-fold scores, 5-fold split by individual targets (no leak-free grouping exists), mean ± std over 3 seeds, on the 8,639 labelled targets of 11,944.

## Is the split honest?

- Groups: 8,639 from individual targets (no leak-free grouping exists); the largest holds 1 target.
- Largest connected component: 100.0% of targets.
- Test targets linked to a training target (directly or through a shared entity): 100.0%; to a training positive: 96.3%.
- A random split would link 96.2% of test targets to a training positive, and score:
  - Own + shared-entity counts: 0.933 random vs 0.935 grouped (-0.002)
  - Graph (multi-hop): 0.941 random vs 0.940 grouped (+0.001)

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
- Graph: 297 extra columns, mean and max neighbourhood summaries over 3 hops (SIGN-style propagation, PCA to 48 per hop, no labels used).

Ran in 324 s.
