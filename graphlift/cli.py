"""Command line: `graphlift audit <dataset-dir>`."""
from __future__ import annotations

import argparse
import os

from . import audit, report
from .data import load_dir


def main(argv=None):
    p = argparse.ArgumentParser(prog="graphlift",
                                description="Does your graph actually help?")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("audit", help="run the full audit on a dataset directory")
    a.add_argument("path")
    a.add_argument("--seeds", type=int, default=3)
    a.add_argument("--folds", type=int, default=5)
    a.add_argument("--hops", type=int, default=3)
    a.add_argument("--dim", type=int, default=48)
    a.add_argument("--out", default=None, help="folder for report.md and report.json")
    a.add_argument("--meta", default="graphlift.json",
                   help="config file inside the dataset folder (default: graphlift.json)")
    a.add_argument("--split", choices=["grouped", "temporal"], default=None,
                   help="override the split declared in the config")
    a.add_argument("--cutoff", type=float, default=None,
                   help="temporal split: train on time <= this value")
    a.add_argument("--gnn", action="store_true",
                   help="also train a GraphSAGE (needs PyTorch: pip install graphlift[gnn])")
    a.add_argument("--gnn-epochs", type=int, default=300)
    rp = sub.add_parser("report", help="re-render report.md from a saved report.json")
    rp.add_argument("json")
    ix = sub.add_parser("index", help="one table across several saved report.json files")
    ix.add_argument("jsons", nargs="+")
    ix.add_argument("--out", default=None, help="write the table to this Markdown file")
    args = p.parse_args(argv)

    if args.cmd == "index":
        md = report.index_markdown([report.from_json(j) for j in args.jsons])
        if args.out:
            with open(args.out, "w") as f:
                f.write(md)
        print(md)
        return

    if args.cmd == "report":
        r = report.from_json(args.json)
        md = report.to_markdown(r)
        with open(os.path.join(os.path.dirname(os.path.abspath(args.json)), "report.md"), "w") as f:
            f.write(md)
        print(md)
        return

    g = load_dir(args.path, args.meta)
    if args.split:
        g.split = args.split
    if args.cutoff is not None:
        g.temporal_cutoff = args.cutoff
    r = audit.run(g, seeds=args.seeds, folds=args.folds, hops=args.hops, dim=args.dim,
                  gnn=args.gnn, gnn_epochs=args.gnn_epochs)
    out = args.out or os.path.join(args.path, "graphlift_report")
    os.makedirs(out, exist_ok=True)
    md = report.to_markdown(r)
    with open(os.path.join(out, "report.md"), "w") as f:
        f.write(md)
    report.to_json(r, os.path.join(out, "report.json"))
    scores = g.targets[["id", g.label]].copy()
    for m, p in r.oof.items():
        scores[f"score_{m}"] = p
    scores.to_csv(os.path.join(out, "scores.csv"), index=False)
    print()
    print(md)


if __name__ == "__main__":
    main()
