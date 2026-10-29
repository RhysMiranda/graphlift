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

| Dataset | Split | Labelled | Fraud | Own attributes | Best non-graph | Graph (propagation) | Lift | GraphSAGE | GraphSAGE lift |
|---|---|---|---|---|---|---|---|---|---|
| Syn Env (synthetic SME loans) | grouped by ring | 2,820 | 25.8% | 0.466 | 0.659 | **0.737** | **+0.078** ± 0.012 | **0.738** | **+0.078** ± 0.007 |
| YelpChi (spam reviews) | random | 45,954 | 14.5% | 0.842 | 0.879 | **0.913** | **+0.034** ± 0.001 | 0.784 | −0.095 ± 0.004 |
| Elliptic (illicit Bitcoin), local features | past → future | 46,564 | 9.8% | 0.790 | 0.790 | **0.811** | **+0.021** ± 0.003 | 0.555 | −0.234 ± 0.014 |
| Elliptic, all features | past → future | 46,564 | 9.8% | 0.807 | 0.807 | 0.817 | +0.010 ± 0.002 | 0.379 | −0.428 ± 0.012 |
| Amazon (fraudulent users) | random | 8,639 | 9.5% | 0.915 | 0.935 | 0.941 | +0.006 ± 0.001 | 0.910 | −0.025 ± 0.004 |

PR-AUC. *Best non-graph* is the strongest model that never passes a message: own
attributes, counts of neighbours and shared entities, or both. *Graph
(propagation)* feeds multi-hop neighbourhood summaries to the same
gradient-boosted trees; *GraphSAGE* is a GNN trained end to end. Each *lift* is
that model minus the best non-graph model; ± is the spread over 3 seeds, or over
200 paired bootstrap resamples of the test set for the past → future split.
GraphSAGE runs on a GPU ([reproduce on Colab](https://colab.research.google.com/github/RhysMiranda/graphlift/blob/main/notebooks/graphlift_gpu_colab.ipynb)). Full reports in
[`results/`](https://github.com/RhysMiranda/graphlift/tree/main/results/), auto-generated table in [`results/INDEX.md`](https://github.com/RhysMiranda/graphlift/blob/main/results/INDEX.md).

**What it shows so far**

- **Most of a graph's value is cheap to get.** On Syn Env, YelpChi and Amazon,
  between half and four-fifths of what the graph model adds over a node's own
  attributes comes from simple neighbour and shared-entity counts. On Amazon
  that leaves almost nothing for message passing.
- **Pre-computed neighbour features close most of the gap.** Elliptic ships 72
  columns summarising each transaction's neighbours. With them, the non-graph
  model rises from 0.790 to 0.807, which is 81% of the 0.021 the graph model
  found on local features alone, and the remaining lift drops to +0.010.
- **The split matters more than the model.** On Elliptic, the same models score
  0.983 under a random split and 0.790 when trained on the past and tested on
  the future. That gap is more than twice the largest graph lift in the table.
  It is not graph leakage: only 4% of random-split test transactions link to a
  training fraud.
- **A trained GNN is not automatically the best graph model.** On Syn Env,
  GraphSAGE and the tree-based propagation model land on the same lift
  (+0.078), two very different methods agreeing that the structure carries real
  signal. On Elliptic's past → future split, GraphSAGE falls far below every
  tree model (0.555 against 0.790 on local features). The same network scores
  0.367 with message passing switched off (a plain MLP) and 0.464 with one layer
  of it, so message passing is helping the neural model; what fails is neural networks on these features
  when the future looks different from the past. Gradient-boosted trees hold up
  much better. On YelpChi and Amazon GraphSAGE also trails the trees (0.784
  against 0.913, and 0.910 against 0.941), though it beats GADBench's published
  GraphSAGE on both (46.6 and 82.5).
- **Every model fails when the world changes.** On Elliptic, PR-AUC falls from
  about 0.9 to under 0.3 from time step 43, the dark-market shutdown described
  in the dataset's original paper (Weber et al., 2019), and the graph does not
  rescue it.

The propagation scores are in line with the best published tree-plus-neighbours
results on the shared benchmarks (GADBench reports 91.1 on YelpChi and 93.3 on
Amazon under its own split), so the graph rung is not a strawman.

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

Reports: [`synenv_balanced_s2`](https://github.com/RhysMiranda/graphlift/blob/main/results/synenv_balanced_s2.md),
[`yelpchi`](https://github.com/RhysMiranda/graphlift/blob/main/results/yelpchi.md), [`amazon`](https://github.com/RhysMiranda/graphlift/blob/main/results/amazon.md).

## Quick start

```bash
pip install graphlift                    # or: pip install "graphlift[gnn]" for the GraphSAGE rung
graphlift audit path/to/dataset          # full audit -> graphlift_report/
graphlift audit path/to/dataset --gnn    # ...plus a trained GraphSAGE
graphlift report path/to/report.json     # re-render a saved report
graphlift index results/*.json           # one table across datasets
```

The GraphSAGE rung is plain PyTorch, with no PyTorch Geometric needed, and uses
a GPU when one is available.

To reproduce the other rows, see
[`examples/care/convert_care.py`](https://github.com/RhysMiranda/graphlift/blob/main/examples/care/convert_care.py) (YelpChi, Amazon) and
[`examples/elliptic/convert_elliptic.py`](https://github.com/RhysMiranda/graphlift/blob/main/examples/elliptic/convert_elliptic.py) (Elliptic).
A dataset can also declare a past → future split; see below.

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
- `split` (optional): `"grouped"` (default) or `"temporal"`, which trains on targets
  with `time` up to `temporal_cutoff` and tests on the rest. Override per run with
  `--split` and `--cutoff`; pick a different config file with `--meta`.
- `exclude`: attributes that are labels in disguise. graphlift also scans for
  ones you missed.

## What the report contains

1. **The ladder.** Four gradient-boosted models, each seeing more than the last:
   own attributes, shared-entity counts, both, then the graph. Lift is the graph
   minus the best of the other three, with a verdict: *graph helps*, *graph is
   decorative*, or *graph hurts*. A verdict either way needs a gain of at least
   0.01 PR-AUC that is clear of two kinds of noise: re-running the split with
   new seeds, and resampling the groups themselves (200 resamples, 95% of them
   pointing the same way). On random labels this keeps false "helps" verdicts
   at zero in 20 trials.
2. **Split honesty.** How many test nodes still touch a training node, or a
   training positive, through a shared entity or a direct link, and how much a
   careless random split would have inflated each score. Past → future splits
   also get a period-by-period table.
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

- **GraphSAGE** (optional, `--gnn`) is a relation-aware GraphSAGE (Hamilton et
  al., 2017) in plain PyTorch: three layers, each adding the mean over every
  neighbour and the max over up to 10 sampled neighbours, separately per
  relation. It trains full-batch with class-weighted loss and stops early on
  held-out training targets: random groups under a grouped split, the latest
  training periods under a past → future one. It sees exactly the features and
  edges the other rungs see, and is scored on the same splits.

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
- [x] Elliptic, with past → future splits and bootstrap uncertainty
- [ ] Loaders: IBM AMLworld, DGraph-Fin, T-Finance
- [x] Optional GraphSAGE rung (plain PyTorch, GPU when available)
- [x] GraphSAGE on YelpChi and Amazon (GPU run)
- [ ] The Graph Lift Index: one table across every public fraud graph
- [ ] PyPI release (package and release workflow ready)

## Licence

MIT
