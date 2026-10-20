"""
Patristic (tree) distances for many species at once.

The all-taxon Fish Tree of Life chronogram is ultrametric (every tip at the
same age), so the distance between two species = 2 x age of their most recent
common ancestor. We fill the whole matrix in one pass over the pruned tree,
which is fast even for thousands of species.

Usage (from the repo root):  python models/tree_distances.py
Output: data/global_tree_distances.npz   species (n,), D (n, n) float32, scale
        (D is divided by `scale` = max distance, so values are in 0-1)
"""
from pathlib import Path

import dendropy
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def patristic_matrix(tree_path, species):
    tree = dendropy.Tree.get(path=str(tree_path), schema="newick", preserve_underscores=True)
    labels = [s.replace(" ", "_") for s in species]
    sub = tree.extract_tree_with_taxa_labels(labels=labels)
    sub.calc_node_ages(is_force_max_age=True)
    idx = {l: i for i, l in enumerate(labels)}
    n = len(labels)
    D = np.zeros((n, n), dtype=np.float32)
    below = {}
    for node in sub.postorder_node_iter():
        if node.is_leaf():
            below[node] = [idx[node.taxon.label.strip("'")]]
            continue
        kids = [below.pop(c) for c in node.child_nodes()]
        d = 2 * node.age
        for a in range(len(kids)):
            for b in range(a + 1, len(kids)):
                ia, ib = np.array(kids[a]), np.array(kids[b])
                D[np.ix_(ia, ib)] = d
                D[np.ix_(ib, ia)] = d
        below[node] = [i for k in kids for i in k]
    return D


def main():
    g = pd.read_csv(DATA / "global_teleo.csv")
    fg = pd.read_csv(DATA / "guiana_teleo_clean.csv")
    species = sorted(set(g.loc[g.in_tree, "species_name"]) | set(fg.loc[fg.in_tree, "species_name"]))
    print("species:", len(species))
    D = patristic_matrix(DATA / "fish_tree_full.tre", species)
    scale = float(D.max())
    np.savez(DATA / "global_tree_distances.npz", species=np.array(species), D=D / scale, scale=scale)
    print(f"saved data/global_tree_distances.npz  shape {D.shape}  (max distance {scale:.1f} before scaling)")


if __name__ == "__main__":
    main()
