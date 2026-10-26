# graphlift

**Does your graph actually help?**

Graph neural networks are often reported against weak baselines: a model that
sees only each node's own features. But much of what a graph knows can be
computed with a few joins, such as how many other applications share this
device, or how quickly they arrived. If a graph model can't beat a
gradient-boosted tree given those counts, the graph is decorative.

graphlift runs that comparison on any graph dataset, under a split that keeps
connected cases together, and reports the gap.

## First result: My very own Syn Env (synthetic SME loan fraud)

2,820 applications, 25.8% labelled fraud, grouped 5-fold split, 3 seeds:

| Model | What it sees | PR-AUC |
|---|---|---|
| Own attributes | the application's own row | 0.466 ± 0.011 |
| Shared-entity counts | how many others share each device, account, address… and how close in time | 0.659 ± 0.007 |
| Own + counts | both | 0.652 ± 0.004 |
| **Graph** | the above plus three hops of neighbourhood summaries | **0.738 ± 0.007** |

**Lift: +0.078 ± 0.010 PR-AUC over the best non-graph model, positive on all 3 seeds.**
Full report: [`results/synenv_balanced_s2.md`](results/synenv_balanced_s2.md).

Play the result instead of reading it: **Find the Ring**, a 90-second browser
game built on these exact scores (link to be added when public).

## Quick start

```bash
pip install -e .
graphlift audit path/to/dataset
```

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
- **The graph model** adds SIGN-style propagation (Frasca et al., 2020): mean and
  max of neighbours' attributes per neighbour type, then mean and max of those
  summaries for further hops, each compressed with PCA. No labels are used to
  build features. Max matters because a ring hides among benign neighbours;
  mean matters because some patterns are counts.

A trained GNN (GraphSAGE via PyTorch Geometric) is planned as an optional
fourth rung. The propagation model is used first because it needs no GPU, gives
the same answer on every machine, and is a strong graph baseline in its own right.

## Roadmap

- [x] Audit core: ladder, grouped split, exposure, label-leak scan
- [x] Syn Env loader and first report
- [ ] Loaders: Elliptic, IBM AMLworld, DGraph-Fin, YelpChi, Amazon, T-Finance
- [ ] Optional GraphSAGE rung (PyTorch Geometric)
- [ ] The Graph Lift Index: one table across every public fraud graph
- [ ] PyPI release

## Licence

MIT
