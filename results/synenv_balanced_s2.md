# graphlift report: Syn Env (balanced, seed 2)

**Verdict: graph helps.** Multi-hop structure lifts PR-AUC from 0.659 to 0.738 over the best model that never passes a message (1.12x).

## Does the graph help?

| Model | PR-AUC | ROC-AUC |
|---|---|---|
| Own attributes only | 0.466 ± 0.011 | 0.741 ± 0.003 |
| Shared-entity counts only (best non-graph) | 0.659 ± 0.007 | 0.867 ± 0.000 |
| Own + shared-entity counts | 0.652 ± 0.004 | 0.869 ± 0.001 |
| Graph (multi-hop) | 0.738 ± 0.007 | 0.900 ± 0.002 |

**Lift: +0.078 ± 0.010 PR-AUC** (3 of 3 seeds positive). The PR-AUC floor is the prevalence, 25.8%, so compare scores within this table, not across datasets.

Out-of-fold scores, grouped 5-fold split, mean ± std over 3 seeds.

## Is the split honest?

- Groups: 839 from declared column `group`; the largest holds 21 targets.
- Largest connected component: 99.4% of targets.
- Test targets sharing an entity with a training target: 92.5%; with a training positive: 59.5%.
- A random split would put 76.7% of test targets next to a training positive, and score:
  - Shared-entity counts only: 0.708 random vs 0.659 grouped (+0.049)
  - Graph (multi-hop): 0.749 random vs 0.738 grouped (+0.011)

## Attributes that predict the label on their own

Each neighbour attribute alone, as max or min over 1 or 2 hops. Anything near 1.0 is probably a label, not an observation.

| Attribute | AUC alone | Via | Status |
|---|---|---|---|
| `application.fraud_true` | 0.938 | own column | excluded (label) |
| `document.is_forged` | 0.703 | max over 1-hop document | excluded (label) |
| `bank.is_mule` | 0.582 | max over 1-hop bank | excluded (label) |
| `transaction.is_laundering` | 0.555 | max over 2-hop transaction | excluded (label) |
| `device.fraud_count` | 0.547 | max over 1-hop device | excluded (label) |
| `employee.prior_fraud_flags` | 0.509 | max over 2-hop employee | excluded (label) |
| `employee.is_corrupt` | 0.509 | max over 2-hop employee | excluded (label) |
| `application.is_sme` | 0.723 | own column | used |
| `address.addr_type` | 0.723 | max over 2-hop address | used |
| `document.font_inconsistency` | 0.697 | max over 1-hop document | used |
| `business.n_employees` | 0.667 | max over 2-hop business | used |
| `business.annual_revenue` | 0.661 | max over 2-hop business | used |
| `business.business_age_years` | 0.630 | max over 2-hop business | used |
| `address.postal` | 0.592 | max over 2-hop address | used |
| `bank.avg_balance` | 0.576 | min over 1-hop bank | used |

## Recall by category

Share of each category's positives flagged when every model flags as many targets as there are positives.

| Category | n | Own attributes only | Shared-entity counts only | Own + shared-entity counts | Graph (multi-hop) |
|---|---|---|---|---|---|
| money_mule | 125 | 0.53 | 0.90 | 0.92 | 0.94 |
| device_farm | 106 | 0.57 | 0.64 | 0.66 | 0.66 |
| loan_stacking | 101 | 0.27 | 0.55 | 0.60 | 0.56 |
| identity_match | 91 | 0.57 | 0.77 | 0.78 | 0.79 |
| phantom_employer | 86 | 0.56 | 0.56 | 0.45 | 0.64 |
| document_reuse | 79 | 0.61 | 0.68 | 0.63 | 0.89 |
| shell_company | 44 | 0.07 | 0.64 | 0.66 | 0.91 |
| nested_mule | 38 | 0.50 | 0.92 | 0.92 | 0.95 |
| legit | 36 | 0.17 | 0.14 | 0.08 | 0.06 |
| collusion | 22 | 0.68 | 0.68 | 0.68 | 0.77 |

## What each model saw

- Own attributes: 56 columns.
- Shared-entity counts: 72 columns over 18 paths of length 1–2 out of each target (how many other targets share the entity at the end, and how tightly in time).
- Graph: 144 extra columns, mean and max neighbourhood summaries over 3 hops (SIGN-style propagation, PCA to 48 per hop, no labels used).

Ran in 240 s.
