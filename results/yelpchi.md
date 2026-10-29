# graphlift report: YelpChi (spam reviews)

**Verdict: graph helps.** Multi-hop structure lifts PR-AUC from 0.879 to 0.913 over the best model that never passes a message (1.04x).

## Does the graph help?

| Model | PR-AUC | ROC-AUC |
|---|---|---|
| Own attributes only | 0.842 ± 0.001 | 0.953 ± 0.000 |
| Shared-entity counts only | 0.331 ± 0.002 | 0.761 ± 0.000 |
| Own + shared-entity counts (best non-graph) | 0.879 ± 0.000 | 0.966 ± 0.000 |
| Graph (multi-hop) | 0.913 ± 0.001 | 0.976 ± 0.000 |
| GraphSAGE (trained GNN) | 0.784 ± 0.004 | 0.943 ± 0.001 |

**Lift: +0.034 ± 0.001 PR-AUC** (3 of 3 seeds positive; 95% interval [+0.029, +0.037] from 200 resamples of the groups). 

**GraphSAGE lift: -0.095 ± 0.004 PR-AUC** (0 of 3 seeds positive; 95% interval [-0.074, -0.060] from 200 resamples of the groups). Verdict for the trained GNN: graph hurts. The PR-AUC floor is the prevalence, 14.5%, so compare scores within this table, not across datasets.

Out-of-fold scores, 5-fold split by individual targets (no leak-free grouping exists), mean ± std over 3 seeds.

## Is the split honest?

- Groups: 45,954 from individual targets (no leak-free grouping exists); the largest holds 1 target.
- Largest connected component: 99.9% of targets.
- Test targets linked to a training target (directly or through a shared entity): 100.0%; to a training positive: 97.4%.
- A random split would link 97.3% of test targets to a training positive, and score:
  - Own + shared-entity counts: 0.877 random vs 0.879 grouped (-0.002)
  - Graph (multi-hop): 0.913 random vs 0.913 grouped (+0.001)

> **Warning.** Over half the targets sit in one connected component, so no split can keep connected targets apart and the split falls back to individual targets, as most published results on this kind of graph do. Treat absolute scores as optimistic; the comparison between models stays fair because every model sees the same split.

## Attributes that predict the label on their own

Each neighbour attribute alone, as max or min over 1 or 2 hops. Anything near 1.0 is probably a label, not an observation.

| Attribute | AUC alone | Via | Status |
|---|---|---|---|
| `review.f05` | 0.664 | own column | used |
| `review.f08` | 0.600 | own column | used |
| `review.f21` | 0.597 | own column | used |
| `review.f20` | 0.594 | own column | used |
| `review.f02` | 0.588 | own column | used |
| `review.f01` | 0.566 | own column | used |
| `review.f15` | 0.563 | own column | used |
| `review.f23` | 0.563 | own column | used |

## What each model saw

- Own attributes: 32 columns.
- Shared-entity and direct-link counts: 6 columns (0 entity paths of length 1–2, plus neighbour and two-step walk counts for every relation that links targets directly).
- Graph: 339 extra columns, mean and max neighbourhood summaries over 3 hops (SIGN-style propagation, PCA to 48 per hop, no labels used), fed to the same gradient-boosted trees.
- GraphSAGE: a relation-aware GraphSAGE trained end to end (3 layers, mean over all neighbours plus max over up to 10 sampled, early-stopped on held-out training targets, at most 300 epochs).

Ran in 2807 s.
