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
    args = p.parse_args(argv)

    g = load_dir(args.path)
    r = audit.run(g, seeds=args.seeds, folds=args.folds, hops=args.hops, dim=args.dim)
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
