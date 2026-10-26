"""Cut playable boards out of a Syn Env graph scored by graphlift.

    python findthering/build_boards.py runs/synenv_balanced_s2 --boards 3

Each board is a neighbourhood of about 100 applications: a handful of fraud
rings drawn at random, the legitimate clusters those rings blend into (their
camouflage edges point there), more legitimate clusters, and lone applicants.

Every score shown in the game is graphlift's out-of-fold score: the model that
scored an application never trained on that application's ring or cluster.
Rings are drawn at random per board, not picked to flatter any model.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import networkx as nx
import numpy as np
import pandas as pd

# How each entity is described to a player.
KIND = {
    "bank": "Bank account",
    "device": "Device",
    "document": "Document",
    "person": "Applicant",
    "business": "Business",
    "employee": "Loan officer",
    "email": "Email address",
    "address": "Address",
    "employer": "Employer",
    "collector": "Receiving account",
}

RING_STORY = {
    "money_mule": ("Mule account",
                   "Different applicants, each with their own device and documents, all pay their loans into one bank account."),
    "nested_mule": ("Layered funnel",
                    "Each bank account looks ordinary and takes only a few applications. All of them forward money to one receiving account that has no applications of its own."),
    "device_farm": ("Device farm",
                    "Applications from many different people, all filed from the same device within days."),
    "loan_stacking": ("Loan stacking",
                      "One borrower files several loans close together, before any lender can report the others."),
    "identity_match": ("Shared identity",
                       "Several applicants are one identity: same email, same tax ID, same birthday."),
    "document_reuse": ("Reused forgery",
                       "The same forged document is attached to several unrelated applications."),
    "shell_company": ("Shell companies",
                      "Businesses with one employee and almost no revenue, all registered at the same address."),
    "phantom_employer": ("Phantom employer",
                         "Applicants all claim the same employer, a company with no staff, no owner and no history."),
    "collusion": ("Inside job",
                  "One loan officer processed every application in the ring, from one shared device."),
}

APP_COLS = ["requested_amount", "loan_purpose", "customer_age", "applied_at", "is_sme",
            "income", "credit_risk_score"]


def load(run: str):
    nodes = {f[:-4]: pd.read_csv(os.path.join(run, "nodes", f), low_memory=False)
             for f in os.listdir(os.path.join(run, "nodes")) if f.endswith(".csv")}
    edges = pd.read_csv(os.path.join(run, "edges.csv"))
    scores = pd.read_csv(os.path.join(run, "graphlift_report", "scores.csv"))
    rep = json.load(open(os.path.join(run, "graphlift_report", "report.json")))
    return nodes, edges, scores, rep


def entity_links(nodes, edges):
    """app -> set of (entity_id, kind), collapsing person/business hops, plus
    bank -> collector links for the layered funnel."""
    links = defaultdict(set)
    person_apps = defaultdict(set)
    biz_apps = defaultdict(set)
    for r in edges.itertuples(index=False):
        if r.src_type == "application":
            links[r.src_id].add((r.dst_id, r.dst_type))
        elif r.dst_type == "application":
            kind = r.src_type
            links[r.dst_id].add((r.src_id, kind))
            if r.relation == "submits":
                person_apps[r.src_id].add(r.dst_id)
            if r.relation == "applies_for":
                biz_apps[r.src_id].add(r.dst_id)
    bank_to = defaultdict(set)
    for r in edges.itertuples(index=False):
        if r.src_type == "person" and r.relation in ("has_email", "resides_at", "employed_by"):
            kind = {"has_email": "email", "resides_at": "address", "employed_by": "employer"}[r.relation]
            for a in person_apps.get(r.src_id, ()):
                links[a].add((r.dst_id, kind))
        elif r.src_type == "business" and r.relation == "registered_at":
            for a in biz_apps.get(r.src_id, ()):
                links[a].add((r.dst_id, "address"))
        elif r.relation == "transfers_to":
            bank_to[r.src_id].add(r.dst_id)
    return links, bank_to


def build_board(rng, apps, links, bank_to, n_rings=6, size=100):
    by_group = apps.groupby("group")
    ring_groups = [g for g, d in by_group if d.fraud_true.max() == 1]
    benign_groups = [g for g, d in by_group if d.fraud_true.max() == 0 and len(d) >= 2]
    singles = [g for g, d in by_group if d.fraud_true.max() == 0 and len(d) == 1]

    ring_typ = {g: by_group.get_group(g).typology.mode()[0] for g in ring_groups}
    ring_len = {g: len(by_group.get_group(g)) for g in ring_groups}
    typs = sorted({t for t in ring_typ.values()})
    chosen_typs = rng.choice(typs, size=min(n_rings, len(typs)), replace=False)
    rings = []
    for t in chosen_typs:
        cand = [g for g in ring_groups if ring_typ[g] == t and 3 <= ring_len[g] <= 9]
        cand = cand or [g for g in ring_groups if ring_typ[g] == t]
        rings.append(cand[int(rng.integers(len(cand)))])

    chosen = set(rings)
    app_group = dict(zip(apps.id, apps.group))
    # Legit clusters the rings hide behind: share an entity with a ring member.
    ent_groups = defaultdict(set)
    for a, ents in links.items():
        for e, _ in ents:
            ent_groups[e].add(app_group[a])
    ring_apps = apps[apps.group.isin(rings)].id
    nearby = set()
    for a in ring_apps:
        for e, _ in links[a]:
            nearby |= {g for g in ent_groups[e] if g in set(benign_groups)}
    nearby = sorted(nearby)
    rng.shuffle(nearby)
    n_now = lambda: int(apps.group.isin(chosen).sum())
    for g in nearby[:8]:
        chosen.add(g)
    pool = [g for g in benign_groups if g not in chosen]
    rng.shuffle(pool)
    for g in pool:
        if n_now() >= int(size * 0.72):
            break
        chosen.add(g)
    rng.shuffle(singles)
    for g in singles:
        if n_now() >= size:
            break
        chosen.add(g)
    return rings, apps[apps.group.isin(chosen)].copy()


def pack_layout(G, seed: int, size: float = 1000.0, margin: float = 40.0,
                min_gap: float = 40.0) -> dict:
    """ForceAtlas2 with strong gravity over the whole board, lone applicants
    included, so the chart fills its square evenly instead of trailing off in
    long chains. Then a few passes push apart any two marks that would touch,
    and everything is scaled uniformly into the square (never stretched)."""
    pos = nx.forceatlas2_layout(G, max_iter=1500, gravity=2.0, strong_gravity=True,
                                scaling_ratio=2.0, seed=seed)
    nodes = list(G)
    xy = np.array([pos[v] for v in nodes], dtype=float)
    for _ in range(3):                      # fit, separate, refit
        lo, hi = xy.min(0), xy.max(0)
        s = (size - 2 * margin) / max((hi - lo).max(), 1e-9)
        xy = (xy - lo) * s
        for _ in range(60):
            d = xy[:, None, :] - xy[None, :, :]
            dist = np.sqrt((d ** 2).sum(-1)) + np.eye(len(xy)) * 1e9
            close = dist < min_gap
            if not close.any():
                break
            push = np.where(close[..., None], d / dist[..., None] * (min_gap - dist[..., None]) / 2, 0)
            xy += push.sum(1)
    lo, hi = xy.min(0), xy.max(0)
    s = (size - 2 * margin) / max((hi - lo).max(), 1e-9)
    off = (size - (hi - lo) * s) / 2
    xy = (xy - lo) * s + off
    return {v: xy[i] for i, v in enumerate(nodes)}


def board_payload(idx, rings, sub, links, bank_to, best, rep):
    ids = set(sub.id)
    ent_apps = defaultdict(set)
    ent_kind = {}
    for a in ids:
        for e, k in links[a]:
            ent_apps[e].add(a)
            ent_kind.setdefault(e, k)
    # Receiving accounts of the layered funnel: shared via visible banks.
    coll_banks = defaultdict(set)
    for e, k in list(ent_kind.items()):
        if k == "bank":
            for c in bank_to.get(e, ()):
                coll_banks[c].add(e)
    visible = {e for e, s in ent_apps.items() if len(s) >= 2}
    for c, banks in coll_banks.items():
        reach = set().union(*(ent_apps[b] for b in banks))
        if len(reach) >= 2:
            visible.add(c)
            ent_kind[c] = "collector"
            visible |= {b for b in banks}

    G = nx.Graph()
    app_ix = {a: i for i, a in enumerate(sorted(ids))}
    for a in ids:
        G.add_node(a)
    links_out = []
    for e in sorted(visible):
        G.add_node(e)
        for a in sorted(ent_apps.get(e, ())):
            G.add_edge(a, e)
            links_out.append([app_ix[a], e])
    for c, banks in coll_banks.items():
        if c in visible:
            for b in banks:
                G.add_edge(b, c)
                links_out.append([b, c])
    pos = pack_layout(G, seed=idx + 7)
    norm = lambda p: [round(float(p[0]), 1), round(float(p[1]), 1)]

    ring_ix = {g: i for i, g in enumerate(rings)}
    a_rows = []
    s = sub.set_index("id")
    for a in sorted(ids):
        r = s.loc[a]
        a_rows.append({
            "id": a.replace("app_", ""),
            "xy": norm(pos[a]),
            "fraud": int(r.fraud_true),
            "ring": ring_ix.get(r.group, -1),
            "amount": round(float(r.requested_amount)),
            "purpose": str(r.loan_purpose).replace("_", " "),
            "age": int(round(float(r.customer_age))),
            "day": int(float(r.applied_at)) + 1,
            "sme": int(r.is_sme),
            "s": {"own": round(float(r.score_own), 4), "hubs": round(float(r[f"score_{best}"]), 4),
                  "graph": round(float(r.score_graph), 4)},
        })
    ents = []
    for e in sorted(visible):
        ents.append({"id": e, "kind": ent_kind[e], "xy": norm(pos[e]),
                     "n": len(ent_apps.get(e, ())) or len(
                         set().union(*(ent_apps[b] for b in coll_banks.get(e, ()))))})
    ent_ix = {e["id"]: i for i, e in enumerate(ents)}
    edges_out = []
    for u, v in links_out:
        if isinstance(u, int):
            edges_out.append([u, ent_ix[v], 0])          # app -> entity
        else:
            edges_out.append([ent_ix[u], ent_ix[v], 1])  # entity -> entity
    ring_rows = []
    for g in rings:
        mem = sub[sub.group == g]
        typ = mem.typology.mode()[0]
        title, story = RING_STORY[typ]
        ring_rows.append({"typology": typ, "title": title, "story": story,
                          "size": int(len(mem))})
    return {"apps": a_rows, "ents": ents, "edges": edges_out, "rings": ring_rows}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("--boards", type=int, default=3)
    p.add_argument("--out", default="findthering/boards.json")
    a = p.parse_args()
    nodes, edges, scores, rep = load(a.run)
    apps = nodes["application"].merge(scores.drop(columns=["fraud_bool"]), on="id")
    links, bank_to = entity_links(nodes, edges)
    best = rep["lift"]["best_baseline"]
    boards = []
    for i in range(a.boards):
        rng = np.random.default_rng(100 + i)
        rings, sub = build_board(rng, apps, links, bank_to)
        b = board_payload(i, rings, sub, links, bank_to, best, rep)
        boards.append(b)
        f = sum(x["fraud"] for x in b["apps"])
        print(f"board {i}: {len(b['apps'])} apps, {f} fraud, {len(b['ents'])} shared things, "
              f"rings: {[r['typology'] + ':' + str(r['size']) for r in b['rings']]}")
    M = rep["metrics"]
    out = {
        "boards": boards,
        "kinds": KIND,
        "audit": {
            "n_apps": rep["dataset"]["n_targets"],
            "prevalence": rep["dataset"]["prevalence"],
            "best_baseline": best,
            "pr": {m: round(M[m]["pr_auc_mean"], 3) for m in M},
            "pr_std": {m: round(M[m]["pr_auc_std"], 3) for m in M},
            "lift": round(rep["lift"]["pr_auc_lift_mean"], 3),
            "lift_std": round(rep["lift"]["pr_auc_lift_std"], 3),
            "seeds": rep["settings"]["seeds"],
        },
    }
    with open(a.out, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print("wrote", a.out, os.path.getsize(a.out) // 1024, "KB")


if __name__ == "__main__":
    main()
