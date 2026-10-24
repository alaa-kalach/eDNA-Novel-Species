"""
Give the tree a fair second chance: try different ways of training on it, chosen on
GLOBAL data only (no French Guiana data), with the same held-out check as
select_taxtree.py / select_twogoal.py.

Tree variants (all use the same encoder, data and training steps):
    tree             plain tree distances
    tree2g_w10m0.4   current main model (tree + keep species apart)
    treecap60/120    deep pairs all count as "far": T = min(tree, cap), cap = 60 / 120 My
    treelog          log-stretched distances (close relatives spread out, deep pairs squeezed)
    treerank         rank (quantile) of the tree distance
    treew30/60       weight pairs by exp(-tree / tau), tau = 30 / 60 My (close pairs count more)
    treetrip         triplet loss: the tree only says which of two species is closer
    treeclade        taxonomy-like steps made from the tree (cut at the median genus / family /
                     order split ages of global species; no tuning)
References (not candidates): taxonomy (B6) and labels (B5).

Held-out check: 10% of global species hidden ('novel') + one held-out sequence of each
other species with >= 2 ('known'). Reported for novel sequences: family / genus of the
nearest training species centre, tree error (split age between the guessed and true
species, My, lower = better), plus AUROC and known accuracy.
Rule fixed in advance: the tree variant with the highest novel FAMILY placement wins.

Needs data/global_taxonomy.csv. Usage (repo root):
    python models/select_treeloss.py                 # all variants (~2 min each on a laptop)
    python models/select_treeloss.py treecap60 treelog
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.encoder import DistanceEncoder, novelty_scores  # noqa: E402
from models.kmers import kmer_profile  # noqa: E402
from models.pretrain import (HP, TRAIN, clade_ages, encoder_settings, global_target,  # noqa: E402
                             load_global, parse_args)

VARIANTS = ["tree", "tree2g_w10m0.4", "treecap60", "treecap120", "treelog", "treerank",
            "treew30", "treew60", "treetrip", "treeclade"]
REFERENCES = ["taxonomy", "labels"]


def main():
    args, _ = parse_args(sys.argv[1:])
    names = args or VARIANTS + REFERENCES
    g = load_global()
    X = np.array([kmer_profile(s) for s in g.teleo])
    tz = np.load(ROOT / "data" / "global_tree_distances.npz")
    r = g.species_name.map({s: i for i, s in enumerate(tz["species"])}).values
    age = tz["D"] * float(tz["scale"]) / 2
    tax = pd.read_csv(ROOT / "data" / "global_taxonomy.csv").set_index("species_name")
    genus = g.species_name.str.split().str[0].values
    family = tax.loc[g.species_name, "Family"].values
    if "treeclade" in names:
        print("tree-clade cut ages (My):", {k: round(v, 1) for k, v in clade_ages().items()})

    rng = np.random.default_rng(0)
    species = np.array(sorted(g.species_name.unique()))
    novel = set(rng.choice(species, size=len(species) // 10, replace=False))
    role = np.where(g.species_name.isin(novel), "novel", "ref").astype(object)
    for sp, grp in g[~g.species_name.isin(novel)].groupby("species_name"):
        seqs = sorted(grp.teleo.unique())
        if len(seqs) >= 2:
            role[grp.index[grp.teleo == seqs[rng.integers(len(seqs))]]] = "known"
    ref, test = role == "ref", role != "ref"
    y = role[test] == "novel"
    has_cong = np.array([x in set(genus[ref]) for x in genus[test]])
    fam_known = pd.notna(family[test])
    ref_family = dict(zip(g.species_name.values[ref], family[ref]))
    sp_row = dict(zip(g.species_name.values, r))

    results = []
    for name in names:
        t0 = time.time()
        T = global_target(name, tz)
        enc = DistanceEncoder(X.shape[1], seed=0, **encoder_settings(name, tz), **HP)
        enc.fit_minibatch(X[ref], r[ref], T, seed=0, verbose=False, **TRAIN)
        sc, pr = novelty_scores(enc.embed(X[ref]), g.species_name.values[ref], enc.embed(X[test]), scaled=False)
        ts = g.species_name.values[test]
        pf = np.array([ref_family.get(p) for p in pr], dtype=object)
        fam_ok = (pf == family[test]) & fam_known
        err = age[[sp_row[p] for p in pr], [sp_row[t] for t in ts]]
        row = dict(name=name, family=fam_ok[y & fam_known].mean(),
                   family_no_congener=fam_ok[y & fam_known & ~has_cong].mean(),
                   genus_congener=(np.array([p.split()[0] for p in pr]) == genus[test])[y & has_cong].mean(),
                   tree_error=err[y].mean(), auroc=roc_auc_score(y, sc), known_acc=(pr[~y] == ts[~y]).mean())
        results.append(row)
        print(f"{name:15s} | family {row['family']:.3f} | family (no congener) {row['family_no_congener']:.3f} "
              f"| genus (congener) {row['genus_congener']:.3f} | tree error {row['tree_error']:.1f} My "
              f"| AUROC {row['auroc']:.3f} | known acc {row['known_acc']:.3f} | {time.time()-t0:.0f}s", flush=True)

    df = pd.DataFrame(results).set_index("name")
    df.round(4).to_csv(ROOT / "outputs" / "results" / "select_treeloss.csv")
    cands = df.loc[[n for n in df.index if n not in REFERENCES]]
    if len(cands):
        best = cands["family"].idxmax()
        print(f"\nchosen tree variant: {best} (highest novel family placement on global held-out species)")
        if "taxonomy" in df.index:
            gap = df.at[best, "family"] - df.at["taxonomy", "family"]
            print(f"best tree variant vs taxonomy: family {gap:+.3f}, tree error "
                  f"{df.at[best, 'tree_error'] - df.at['taxonomy', 'tree_error']:+.1f} My")


if __name__ == "__main__":
    main()
