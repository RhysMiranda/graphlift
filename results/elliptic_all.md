# graphlift report: Elliptic (illicit Bitcoin), all features

**Verdict: graph is decorative.** The graph model's edge over the best non-graph model (own + shared-entity counts) is +0.010 PR-AUC, consistent but below the 0.01 that graphlift treats as a practical difference. Own columns plus neighbour and shared-entity counts capture what the graph knows.

## Does the graph help?

| Model | PR-AUC | ROC-AUC |
|---|---|---|
| Own attributes only | 0.807 ± 0.010 | 0.943 ± 0.000 |
| Shared-entity counts only | 0.119 ± 0.004 | 0.681 ± 0.000 |
| Own + shared-entity counts (best non-graph) | 0.807 ± 0.010 | 0.944 ± 0.000 |
| Graph (multi-hop) | 0.817 ± 0.010 | 0.953 ± 0.000 |
| GraphSAGE (trained GNN) | 0.379 ± 0.012 | 0.890 ± 0.000 |

**Lift: +0.010 ± 0.002 PR-AUC** (positive in 100% of 200 paired bootstrap resamples of the test set). 

**GraphSAGE lift: -0.428 ± 0.012 PR-AUC** (positive in 0% of 200 paired bootstrap resamples of the test set). Verdict for the trained GNN: graph hurts. The PR-AUC floor is the prevalence, 6.5%, so compare scores within this table, not across datasets.

Split: temporal: train on `time_step` <= 34 (29,894 targets), test on later (16,670 targets). Models are trained once on the past and scored on the future; ± is the spread over 200 paired bootstrap resamples of the test set. Only the 46,564 labelled targets of 203,769 are scored.

## Is the split honest?

- Groups: 49 from connected components; the largest holds 7880 targets.
- Largest connected component: 3.9% of targets.
- Test targets linked to a training target (directly or through a shared entity): 0.0%; to a training positive: 0.0%.
- A random split would link 4.4% of test targets to a training positive, and score:
  - Own + shared-entity counts: 0.989 random vs 0.807 temporal (+0.182)
  - Graph (multi-hop): 0.988 random vs 0.817 temporal (+0.170)

## Attributes that predict the label on their own

Each neighbour attribute alone, as max or min over 1 or 2 hops. Anything near 1.0 is probably a label, not an observation.

| Attribute | AUC alone | Via | Status |
|---|---|---|---|
| `tx.local_89` | 0.843 | own column | used |
| `tx.local_52` | 0.836 | own column | used |
| `tx.local_54` | 0.824 | own column | used |
| `tx.local_04` | 0.822 | own column | used |
| `tx.local_13` | 0.822 | own column | used |
| `tx.local_22` | 0.815 | own column | used |
| `tx.local_28` | 0.815 | own column | used |
| `tx.agg_38` | 0.815 | own column | used |

## Over time

PR-AUC within each test period. Periods with no positives are blank.

| Period | Targets | Positive | Own + shared-entity counts | Graph (multi-hop) | GraphSAGE (trained GNN) | Graph minus best |
|---|---|---|---|---|---|---|
| 35 | 1,341 | 182 | 0.992 | 0.995 | 0.641 | +0.003 |
| 36 | 1,708 | 33 | 0.991 | 0.987 | 0.205 | -0.005 |
| 37 | 498 | 40 | 0.899 | 0.914 | 0.337 | +0.015 |
| 38 | 756 | 111 | 0.957 | 0.960 | 0.326 | +0.003 |
| 39 | 1,183 | 81 | 0.959 | 0.962 | 0.710 | +0.003 |
| 40 | 1,211 | 112 | 0.765 | 0.781 | 0.455 | +0.016 |
| 41 | 1,132 | 116 | 0.967 | 0.964 | 0.312 | -0.003 |
| 42 | 2,154 | 239 | 0.893 | 0.898 | 0.686 | +0.005 |
| 43 | 1,370 | 24 | 0.051 | 0.065 | 0.025 | +0.014 |
| 44 | 1,591 | 24 | 0.044 | 0.041 | 0.032 | -0.003 |
| 45 | 1,221 | 5 | 0.013 | 0.012 | 0.015 | -0.001 |
| 46 | 712 | 2 | 0.262 | 0.178 | 0.508 | -0.084 |
| 47 | 846 | 22 | 0.056 | 0.067 | 0.045 | +0.011 |
| 48 | 471 | 36 | 0.255 | 0.347 | 0.309 | +0.093 |
| 49 | 476 | 56 | 0.281 | 0.330 | 0.111 | +0.049 |

## What each model saw

- Own attributes: 165 columns.
- Shared-entity and direct-link counts: 4 columns (0 entity paths of length 1–2, plus neighbour and two-step walk counts for every relation that links targets directly).
- Graph: 806 extra columns, mean and max neighbourhood summaries over 3 hops (SIGN-style propagation, PCA to 48 per hop, no labels used), fed to the same gradient-boosted trees.
- GraphSAGE: a relation-aware GraphSAGE trained end to end (3 layers, mean over all neighbours plus max over up to 10 sampled, early-stopped on held-out training targets, at most 300 epochs).

Ran in 883 s.
