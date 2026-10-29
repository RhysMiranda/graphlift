# graphlift report: Elliptic (illicit Bitcoin), local features only

**Verdict: graph helps.** Multi-hop structure lifts PR-AUC from 0.790 to 0.811 over the best model that never passes a message (1.03x).

## Does the graph help?

| Model | PR-AUC | ROC-AUC |
|---|---|---|
| Own attributes only (best non-graph) | 0.790 ± 0.011 | 0.933 ± 0.000 |
| Shared-entity counts only | 0.119 ± 0.004 | 0.681 ± 0.000 |
| Own + shared-entity counts | 0.788 ± 0.011 | 0.933 ± 0.000 |
| Graph (multi-hop) | 0.811 ± 0.010 | 0.948 ± 0.000 |
| GraphSAGE (trained GNN) | 0.555 ± 0.016 | 0.895 ± 0.000 |

**Lift: +0.021 ± 0.003 PR-AUC** (positive in 100% of 200 paired bootstrap resamples of the test set). 

**GraphSAGE lift: -0.234 ± 0.014 PR-AUC** (positive in 0% of 200 paired bootstrap resamples of the test set). Verdict for the trained GNN: graph hurts. The PR-AUC floor is the prevalence, 6.5%, so compare scores within this table, not across datasets.

Split: temporal: train on `time_step` <= 34 (29,894 targets), test on later (16,670 targets). Models are trained once on the past and scored on the future; ± is the spread over 200 paired bootstrap resamples of the test set. Only the 46,564 labelled targets of 203,769 are scored.

## Is the split honest?

- Groups: 49 from connected components; the largest holds 7880 targets.
- Largest connected component: 3.9% of targets.
- Test targets linked to a training target (directly or through a shared entity): 0.0%; to a training positive: 0.0%.
- A random split would link 4.4% of test targets to a training positive, and score:
  - Own attributes only: 0.983 random vs 0.790 temporal (+0.193)
  - Graph (multi-hop): 0.985 random vs 0.811 temporal (+0.174)

## Attributes that predict the label on their own

Each neighbour attribute alone, as max or min over 1 or 2 hops. Anything near 1.0 is probably a label, not an observation.

| Attribute | AUC alone | Via | Status |
|---|---|---|---|
| `tx.agg_38` | 0.815 | own column | excluded (label) |
| `tx.agg_44` | 0.780 | own column | excluded (label) |
| `tx.agg_50` | 0.777 | own column | excluded (label) |
| `tx.agg_62` | 0.772 | own column | excluded (label) |
| `tx.agg_48` | 0.754 | own column | excluded (label) |
| `tx.agg_67` | 0.745 | own column | excluded (label) |
| `tx.agg_36` | 0.717 | own column | excluded (label) |
| `tx.agg_69` | 0.714 | own column | excluded (label) |
| `tx.agg_56` | 0.707 | own column | excluded (label) |
| `tx.agg_60` | 0.704 | own column | excluded (label) |
| `tx.agg_01` | 0.703 | own column | excluded (label) |
| `tx.agg_02` | 0.688 | own column | excluded (label) |
| `tx.agg_68` | 0.686 | own column | excluded (label) |
| `tx.agg_03` | 0.675 | own column | excluded (label) |
| `tx.agg_25` | 0.671 | own column | excluded (label) |
| `tx.agg_27` | 0.663 | own column | excluded (label) |
| `tx.agg_14` | 0.642 | own column | excluded (label) |
| `tx.agg_31` | 0.639 | own column | excluded (label) |
| `tx.agg_08` | 0.634 | own column | excluded (label) |
| `tx.agg_26` | 0.633 | own column | excluded (label) |
| `tx.agg_54` | 0.633 | own column | excluded (label) |
| `tx.agg_32` | 0.623 | own column | excluded (label) |
| `tx.agg_33` | 0.621 | own column | excluded (label) |
| `tx.agg_20` | 0.617 | own column | excluded (label) |
| `tx.agg_13` | 0.600 | own column | excluded (label) |
| `tx.agg_18` | 0.578 | own column | excluded (label) |
| `tx.agg_55` | 0.566 | own column | excluded (label) |
| `tx.agg_15` | 0.564 | own column | excluded (label) |
| `tx.agg_66` | 0.563 | own column | excluded (label) |
| `tx.agg_19` | 0.551 | own column | excluded (label) |
| `tx.agg_51` | 0.550 | own column | excluded (label) |
| `tx.agg_24` | 0.549 | own column | excluded (label) |
| `tx.agg_12` | 0.544 | own column | excluded (label) |
| `tx.agg_37` | 0.544 | own column | excluded (label) |
| `tx.agg_70` | 0.537 | own column | excluded (label) |
| `tx.agg_71` | 0.536 | own column | excluded (label) |
| `tx.agg_57` | 0.533 | own column | excluded (label) |
| `tx.agg_61` | 0.533 | own column | excluded (label) |
| `tx.agg_43` | 0.527 | own column | excluded (label) |
| `tx.agg_63` | 0.526 | own column | excluded (label) |
| `tx.agg_21` | 0.525 | own column | excluded (label) |
| `tx.agg_46` | 0.524 | own column | excluded (label) |
| `tx.agg_47` | 0.522 | own column | excluded (label) |
| `tx.agg_00` | 0.519 | own column | excluded (label) |
| `tx.agg_40` | 0.516 | own column | excluded (label) |
| `tx.agg_41` | 0.515 | own column | excluded (label) |
| `tx.agg_04` | 0.513 | own column | excluded (label) |
| `tx.agg_64` | 0.512 | own column | excluded (label) |
| `tx.agg_39` | 0.511 | own column | excluded (label) |
| `tx.agg_45` | 0.511 | own column | excluded (label) |
| `tx.agg_65` | 0.511 | own column | excluded (label) |
| `tx.agg_29` | 0.510 | own column | excluded (label) |
| `tx.agg_28` | 0.509 | own column | excluded (label) |
| `tx.agg_05` | 0.508 | own column | excluded (label) |
| `tx.agg_06` | 0.508 | own column | excluded (label) |
| `tx.agg_09` | 0.507 | own column | excluded (label) |
| `tx.agg_42` | 0.507 | own column | excluded (label) |
| `tx.agg_59` | 0.507 | own column | excluded (label) |
| `tx.agg_52` | 0.506 | own column | excluded (label) |
| `tx.agg_58` | 0.506 | own column | excluded (label) |
| `tx.agg_07` | 0.505 | own column | excluded (label) |
| `tx.agg_22` | 0.505 | own column | excluded (label) |
| `tx.agg_23` | 0.505 | own column | excluded (label) |
| `tx.agg_53` | 0.505 | own column | excluded (label) |
| `tx.agg_34` | 0.503 | own column | excluded (label) |
| `tx.agg_35` | 0.503 | own column | excluded (label) |
| `tx.agg_49` | 0.503 | own column | excluded (label) |
| `tx.agg_10` | 0.502 | own column | excluded (label) |
| `tx.agg_11` | 0.502 | own column | excluded (label) |
| `tx.agg_16` | 0.502 | own column | excluded (label) |
| `tx.agg_17` | 0.501 | own column | excluded (label) |
| `tx.agg_30` | 0.501 | own column | excluded (label) |
| `tx.local_89` | 0.843 | own column | used |
| `tx.local_52` | 0.836 | own column | used |
| `tx.local_54` | 0.824 | own column | used |
| `tx.local_04` | 0.822 | own column | used |
| `tx.local_13` | 0.822 | own column | used |
| `tx.local_22` | 0.815 | own column | used |
| `tx.local_28` | 0.815 | own column | used |
| `tx.local_40` | 0.803 | own column | used |

## Over time

PR-AUC within each test period. Periods with no positives are blank.

| Period | Targets | Positive | Own attributes only | Graph (multi-hop) | GraphSAGE (trained GNN) | Graph minus best |
|---|---|---|---|---|---|---|
| 35 | 1,341 | 182 | 0.992 | 0.995 | 0.960 | +0.003 |
| 36 | 1,708 | 33 | 0.990 | 0.974 | 0.819 | -0.015 |
| 37 | 498 | 40 | 0.851 | 0.874 | 0.659 | +0.023 |
| 38 | 756 | 111 | 0.961 | 0.962 | 0.921 | +0.001 |
| 39 | 1,183 | 81 | 0.943 | 0.964 | 0.886 | +0.021 |
| 40 | 1,211 | 112 | 0.744 | 0.772 | 0.754 | +0.029 |
| 41 | 1,132 | 116 | 0.958 | 0.960 | 0.848 | +0.002 |
| 42 | 2,154 | 239 | 0.885 | 0.897 | 0.633 | +0.012 |
| 43 | 1,370 | 24 | 0.026 | 0.056 | 0.018 | +0.030 |
| 44 | 1,591 | 24 | 0.056 | 0.054 | 0.038 | -0.001 |
| 45 | 1,221 | 5 | 0.022 | 0.026 | 0.007 | +0.004 |
| 46 | 712 | 2 | 0.266 | 0.083 | 0.257 | -0.184 |
| 47 | 846 | 22 | 0.040 | 0.061 | 0.060 | +0.021 |
| 48 | 471 | 36 | 0.146 | 0.270 | 0.219 | +0.124 |
| 49 | 476 | 56 | 0.229 | 0.250 | 0.108 | +0.021 |

## What each model saw

- Own attributes: 93 columns.
- Shared-entity and direct-link counts: 4 columns (0 entity paths of length 1–2, plus neighbour and two-step walk counts for every relation that links targets directly).
- Graph: 518 extra columns, mean and max neighbourhood summaries over 3 hops (SIGN-style propagation, PCA to 48 per hop, no labels used), fed to the same gradient-boosted trees.
- GraphSAGE: a relation-aware GraphSAGE trained end to end (3 layers, mean over all neighbours plus max over up to 10 sampled, early-stopped on held-out training targets, at most 300 epochs).

Ran in 1148 s.
