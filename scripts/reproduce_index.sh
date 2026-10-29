#!/usr/bin/env bash
# Rebuild the Graph Lift Index from scratch, GraphSAGE included.
#
#   pip install "graphlift[gnn]"
#   bash scripts/reproduce_index.sh /path/to/raw
#
# /path/to/raw must contain:
#   YelpChi.mat, Amazon.mat          from github.com/YingtongDou/CARE-GNN (data/*.zip)
#   elliptic_bitcoin_dataset/        from Kaggle, "Elliptic Data Set" (ellipticco)
#   synenv_balanced_s2/              exported with examples/synenv/export_synenv.py
#
# On a recent NVIDIA GPU this takes about 15 minutes; on a laptop CPU, several hours.
set -euo pipefail
RAW=${1:?usage: reproduce_index.sh /path/to/raw}
OUT=${2:-data}
mkdir -p "$OUT" results

python examples/care/convert_care.py "$RAW/YelpChi.mat" "$OUT/yelpchi"
python examples/care/convert_care.py "$RAW/Amazon.mat" "$OUT/amazon"
python examples/elliptic/convert_elliptic.py "$RAW/elliptic_bitcoin_dataset" "$OUT/elliptic"

graphlift audit "$RAW/synenv_balanced_s2" --gnn --out "$OUT/synenv_report"
graphlift audit "$OUT/yelpchi" --gnn --out "$OUT/yelpchi_report"
graphlift audit "$OUT/amazon" --gnn --out "$OUT/amazon_report"
graphlift audit "$OUT/elliptic" --gnn --meta graphlift_local.json --out "$OUT/elliptic_local_report"
graphlift audit "$OUT/elliptic" --gnn --out "$OUT/elliptic_all_report"

for r in synenv yelpchi amazon elliptic_local elliptic_all; do
  cp "$OUT/${r}_report/report.md" "results/$r.md"
  cp "$OUT/${r}_report/report.json" "results/$r.json"
done
mv results/synenv.md results/synenv_balanced_s2.md
mv results/synenv.json results/synenv_balanced_s2.json
graphlift index results/*.json --out results/INDEX.md
