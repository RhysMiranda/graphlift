# graphlift

**Does your graph actually help?**

Graph neural networks are often reported against weak baselines: a model that
sees only each node's own features. But much of what a graph knows can be
computed with a few joins, such as how many other applications share this
device, or how quickly they arrived. If a graph model can't beat a
gradient-boosted tree given those counts, the graph is decorative.

graphlift runs that comparison on any graph dataset, under a split that keeps
connected cases together where the graph allows it, and reports the gap.

## The Graph Lift Index

| Dataset | Labelled | Fraud | Own attributes | Best non-graph | Graph | Lift | Gain already in counts | Verdict |
|---|---|---|---|---|---|---|---|---|
| Syn Env (synthetic SME loans) | 2,820 | 25.8% | 0.466 | 0.659 | **0.737** | **+0.078** ± 0.012 | 71% | graph helps |
| YelpChi (spam reviews) | 45,954 | 14.5% | 0.841 | 0.879 | **0.913** | **+0.034** ± 0.001 | 53% | graph helps |
| Amazon (fraudulent users) | 8,639 | 9.5% | 0.915 | 0.935 | 0.940 | +0.005 ± 0.001 | 79% | graph is decorative |

PR-AUC, out-of-fold, mean ± std over 3 seeds. *Best non-graph* is the strongest
model that never passes a message. *Gain already in counts* is how much of the
graph model's improvement over own attributes a model gets from neighbour and
shared-entity counts alone. Full table: [`results/INDEX.md`](results/INDEX.md).

**What it shows so far:** on every dataset, most of what "the graph" adds over a
node's own attributes is a handful of counts. On Amazon that is nearly all of
it. YelpChi is the clearer case where message passing earns its place, and
even there counts deliver half the gain. The graph scores here match or exceed
are in line with the best published tree-plus-neighbours model on both
benchmarks (GADBench reports 91.1 on YelpChi and 93.3 on Amazon, under its own
split), so the graph rung is not a strawman.

Play the Syn Env result instead of reading it:
**[Find the Ring](https://rhysmiranda.github.io/graphlift/)**, a 90-second
browser game built on graphlift's scores.

## My very own Syn Env (synthetic SME loan fraud)

Syn Env embeds organised fraud rings in a loan network as structure rather
than feature values, with legitimate look-alike clusters, camouflage edges and
noisy labels. Grouped 5-fold split that keeps every ring and cluster on one side:

| Model | What it sees | PR-AUC |
|---|---|---|
| Own attributes | the application's own row | 0.466 ± 0.011 |
| Shared-entity counts | how many others share each device, account, address… and how close in time | 0.659 ± 0.007 |
| Own + counts | both | 0.652 ± 0.004 |
| **Graph** | the above plus per-relation neighbour summaries and three hops of propagation | **0.737 ± 0.006** |

Reports: [`synenv_balanced_s2`](results/synenv_balanced_s2.md),
[`yelpchi`](results/yelpchi.md), [`amazon`](results/amazon.md).

## Quick start

```bash
pip install -e .
graphlift audit path/to/dataset          # full audit -> graphlift_report/
graphlift report path/to/report.json     # re-render a saved report
graphlift index results/*.json           # one table across datasets
```

To reproduce the YelpChi and Amazon rows, see
[`examples/care/convert_care.py`](examples/care/convert_care.py).

A dataset is a folder of plain tables:

```
nodes/<type>.csv    an `id` column plus attributes, one file per node type
edges.csv           src_type, src_id, relation, dst_type, dst_id
graphlift.json      what to classify and which columns are off limits
```

```json
{
  "target": "application",
  "label": "fraud_bool",
  "group": "group",
  "time": "applied_at",
  "breakdown": "typology",
  "exclude": {"bank": ["is_mule"], "document": ["is_forged"]}
}
```

- `group` (optional): ids that must stay on one side of the split, such as a
  case or cluster. Without it, graphlift uses connected components and warns
  you if one component swallows most of the graph.
- `time` (optional): enables burst features ("these five arrived within two days").
- `exclude`: attributes that are labels in disguise. graphlift also scans for
  ones you missed.

## What the report contains

1. **The ladder.** Four gradient-boosted models, each seeing more than the last:
   own attributes, shared-entity counts, both, then the graph. Lift is the graph
   minus the best of the other three, with its spread across seeds and a verdict:
   *graph helps*, *graph is decorative*, or *graph hurts*.
2. **Split honesty.** How many test nodes still touch a training node, or a
   training positive, through a shared entity, and how much a careless random
   split would have inflated each score.
3. **Label leaks.** Every attribute within two hops, scored on its own. Anything
   that separates the classes almost perfectly is flagged, because it is
   probably an answer, not an observation.
4. **Recall by category**, when a `breakdown` column is declared.

## How the models are built

- **Shared-entity counts** walk every path of length 1–2 out of a target node
  (application → device, application → applicant → email, application → bank →
  bank) and record how many other targets share the entity at the end, and the
  tightest arrival-time span among them.
- **Direct-link counts** cover graphs whose targets link straight to each other
  (reviews by the same user): neighbours per relation and two-step walks.
- **The graph model** adds the mean and max of neighbours' attributes for every
  relation, uncompressed, then SIGN-style propagation (Frasca et al., 2020) for
  further hops: mean and max of the previous hop over all neighbours, each
  compressed with PCA. No labels are used to
  build features. Max matters because a ring hides among benign neighbours;
  mean matters because some patterns are counts.

A trained GNN (GraphSAGE via PyTorch Geometric) is planned as an optional
fourth rung. The propagation model is used first because it needs no GPU, gives
the same answer on every machine, and is a strong graph baseline in its own right.

## How this relates to prior work

GADBench (Tang et al., NeurIPS 2023) benchmarked 29 models for supervised graph
anomaly detection and found that gradient-boosted trees given simple
neighbourhood averages beat purpose-built fraud GNNs on most datasets. graphlift
takes that result as its starting point rather than its claim. What it adds:

- **A counts-only rung.** GADBench compares trees with and without neighbour
  aggregation. graphlift inserts a model that sees how many neighbours or shared
  entities a node has, and how bursty they are, but never their attributes.
  That separates "the graph's signal is a degree count" from "the graph's
  signal needs message passing".
- **Split and label audits.** How much of the test set can reach a training
  positive through a shared entity, how much a random split inflates each
  score, and which neighbour attributes are labels in disguise.
- **Any dataset, one command.** Plain CSV tables in, a report card out, with
  entity-sharing graphs (applications sharing devices and accounts) and
  directly linked graphs (reviews by the same user) handled the same way.

## Roadmap

- [x] Audit core: ladder, grouped split, exposure, label-leak scan
- [x] Syn Env loader and first report
- [x] YelpChi and Amazon (CARE-GNN releases), graphs with direct links and unlabelled nodes
- [x] `graphlift index`: the cross-dataset table
- [ ] Loaders: Elliptic, IBM AMLworld, DGraph-Fin, T-Finance
- [ ] Optional GraphSAGE rung (PyTorch Geometric)
- [ ] The Graph Lift Index: one table across every public fraud graph
- [ ] PyPI release

## Licence

MIT
