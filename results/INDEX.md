| Dataset | Labelled targets | Positive | Own attributes | Best non-graph | Graph | Lift | Share of the graph's gain from counts | Verdict |
|---|---|---|---|---|---|---|---|---|
| Syn Env (balanced, seed 2) | 2,820 | 25.8% | 0.466 | 0.659 | 0.737 | +0.078 ± 0.012 | 71% | graph helps |
| YelpChi (spam reviews) | 45,954 | 14.5% | 0.841 | 0.879 | 0.913 | +0.034 ± 0.001 | 53% | graph helps |
| Amazon (fraudulent users) | 8,639 | 9.5% | 0.915 | 0.935 | 0.940 | +0.005 ± 0.001 | 79% | graph is decorative |

PR-AUC, out-of-fold, mean over seeds. *Best non-graph* is the strongest model that never passes a message (own attributes, counts of shared entities or direct links, or both). *Lift* is graph minus best non-graph. *Share from counts* is how much of the graph model's gain over own attributes alone the counts model already gets.
