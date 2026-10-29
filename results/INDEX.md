| Dataset | Labelled targets | Positive | Own attributes | Best non-graph | Graph (propagation) | Lift | GraphSAGE | GraphSAGE lift | Share of the graph's gain from counts | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| Syn Env (balanced, seed 2) | 2,820 | 25.8% | 0.466 | 0.659 | 0.737 | +0.078 ± 0.012 | 0.738 | +0.078 ± 0.007 | 71% | graph helps |
| YelpChi (spam reviews) | 45,954 | 14.5% | 0.841 | 0.879 | 0.913 | +0.034 ± 0.001 | not run |  | 53% | graph helps |
| Elliptic (illicit Bitcoin), local features only | 46,564 | 9.8% | 0.790 | 0.790 | 0.811 | +0.021 ± 0.003 | 0.555 | -0.234 ± 0.014 | 0% | graph helps |
| Elliptic (illicit Bitcoin), all features | 46,564 | 9.8% | 0.807 | 0.807 | 0.817 | +0.010 ± 0.002 | 0.379 | -0.428 ± 0.012 | 6% | graph is decorative |
| Amazon (fraudulent users) | 8,639 | 9.5% | 0.915 | 0.935 | 0.940 | +0.005 ± 0.001 | not run |  | 79% | graph is decorative |

PR-AUC, out-of-fold, mean over seeds. *Best non-graph* is the strongest model that never passes a message (own attributes, counts of shared entities or direct links, or both). *Lift* is graph minus best non-graph; *GraphSAGE lift* is the trained GNN minus the same baseline. *Share from counts* is how much of the graph model's gain over own attributes alone the counts model already gets.
