#!/usr/bin/env python3
"""
Group new sequences by WHERE THEY LAND ON THE FISH TREE (tree-based grouping).

Every other clustering run groups sequences in a k-mer or embedding space. Here the tree itself
decides: each test sequence is placed on the Fish Tree (same rules as models/phylo_placement.py),
and two sequences are close when their placements are close on the tree.

RULES, FIXED BEFORE RUNNING
  1. Sequences to group = those called "novel" by B4 (outputs/results_kmer[_extra].csv), the same
     detector as every other clustering file, so only the grouping step differs.
  2. Distance between two placed sequences = pendant_1 + pendant_2 + path length on the re-fitted
     tree between their two attachment points (0 if the two sequences are identical).
  3. Grouping = HDBSCAN(min_cluster_size=2, min_samples=1) on that distance (precomputed), the same
     settings as clustering/cluster.py.
  4. Naming = each group's most common phylo guess (the reference species nearest each member's
     attachment point); ties -> alphabetically first. Noise points are named one by one.
  Main comparison (clustering/score_clusters.py): ARI and B-cubed F1 vs kmer (B4's own
  clustering) and vs ft_taxonomy_withraw; naming outcomes reported too.

Teleo only (all other clustering runs are Teleo). Tools as in models/phylo_placement.py.
Usage (repo root):
    python clustering/tree_grouping.py --threads=8
    python clustering/score_clusters.py kmer ft_taxonomy_withraw phylo_tree ft_treeclade_withraw
Output: outputs/results/clusters_phylo_tree.csv (splits 1-10) and clusters_phylo_tree_extra.csv (11-30)
"""
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "models"))
import phylo_placement as pp  # noqa: E402

OUT = ROOT / "outputs" / "results"
HDBSCAN_ARGS = dict(min_cluster_size=2, min_samples=1)


def node_distances(root):
    """All-pairs path length between the nodes of the tree (small trees: a few hundred nodes)."""
    nodes, stack = [], [root]
    while stack:
        v = stack.pop(); nodes.append(v); stack.extend(v.kids)
    idx = {v: i for i, v in enumerate(nodes)}
    adj = [[] for _ in nodes]
    for v in nodes:
        for k in v.kids:
            adj[idx[v]].append((idx[k], k.length)); adj[idx[k]].append((idx[v], k.length))
    D = np.zeros((len(nodes), len(nodes)))
    for s in range(len(nodes)):
        seen, todo = {s}, [(s, 0.0)]
        while todo:
            u, d = todo.pop()
            D[s, u] = d
            for w, l in adj[u]:
                if w not in seen:
                    seen.add(w); todo.append((w, d + l))
    return idx, D


def placement_distance(ids, pos, idx, D):
    """pendant_i + pendant_j + path between the two attachment points."""
    n = len(ids)
    ends = []
    for sid in ids:
        v, a, pend, q = pos[sid]
        # attachment point sits on the edge (child v, parent v.parent): a from v, length - a from parent
        ends.append((idx[v], a, idx[v.parent], max(v.length - a, 0.0), pend, q, v))
    M = np.zeros((n, n))
    for i in range(n):
        ci, ai, pi, bi, ei, qi, vi = ends[i]
        for j in range(i + 1, n):
            cj, aj, pj, bj, ej, qj, vj = ends[j]
            if qi == qj:
                d = 0.0
            elif vi is vj:
                d = ei + ej + abs(ai - aj)
            else:
                d = ei + ej + min(ai + D[ci, cj] + aj, ai + D[ci, pj] + bj, bi + D[pi, cj] + aj, bi + D[pi, pj] + bj)
            M[i, j] = M[j, i] = d
    return M


def main():
    opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    pp.THREADS = opts.get("threads", pp.THREADS)
    if pp.MARKER != "teleo":
        sys.exit("tree_grouping.py runs on Teleo only (like the other clustering runs)")
    clean, seqs, splits, tree = pp.load()
    info = clean.drop_duplicates("species_name").set_index("species_name")
    OUT.mkdir(parents=True, exist_ok=True)
    for called_file, split_file, tag in [("results_kmer.csv", "splits.csv", ""),
                                         ("results_kmer_extra.csv", "splits_extra.csv", "_extra")]:
        called = pd.read_csv(ROOT / "outputs" / called_file, dtype={"seq_id": str})
        called = called[called.predicted_species == "novel"]
        sp_file = pd.read_csv(pp.DATA / split_file)
        rows = []
        for k in sorted(sp_file.split_id.unique()):
            t0 = time.time()
            s = splits[splits.split_id == k]
            ref, test = s[s.role == "reference"], s[s.role != "reference"]
            nov = called[called.split_id == k].seq_id.values
            if len(nov) == 0:
                continue
            with tempfile.TemporaryDirectory() as tmp:
                df, pos, root = pp.place_split(ref.seq_id.values, test.seq_id.values, clean, seqs, tree,
                                               Path(tmp), positions=True)
            guess_of = df.set_index("seq_id").phylo_species
            idx, D = node_distances(root)
            M = placement_distance(nov, pos, idx, D)
            lab = HDBSCAN(metric="precomputed", **HDBSCAN_ARGS).fit_predict(M) if len(nov) >= 2 else np.array([-1])
            guess = np.empty(len(nov), dtype=object)
            for c in np.unique(lab[lab >= 0]):
                m = lab == c
                vc = guess_of.loc[nov[m]].value_counts()
                guess[m] = sorted(vc.index[vc == vc.max()])[0]
            guess[lab < 0] = guess_of.loc[nov[lab < 0]].values
            rows.append(pd.DataFrame({"seq_id": nov, "split_id": k, "cluster_id": lab,
                                      "guess_genus": info.loc[guess, "Genus"].values,
                                      "guess_family": info.loc[guess, "Family"].values}))
            print(f"split {k:2d} | {len(nov)} novel seqs -> {lab.max() + 1} groups + {(lab < 0).sum()} noise "
                  f"({time.time() - t0:.0f}s)", flush=True)
        out = OUT / f"clusters_phylo_tree{tag}.csv"
        pd.concat(rows).to_csv(out, index=False)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
