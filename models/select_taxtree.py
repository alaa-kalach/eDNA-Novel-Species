"""
Choose the tree weight a for the taxonomy + tree target on GLOBAL data only
(no French Guiana data involved), before any French Guiana test.

    target = (1 - a) * taxonomy distance + a * tree distance
    a = 0 is plain taxonomy (B6). If a = 0 wins here, the tree adds nothing.

Held-out check (same design as select_twogoal.py):
  - 10% of global species are hidden          -> 'novel'
  - one distinct sequence of each other species with >= 2 is held out -> 'known'
  - the encoder is trained on the rest (same settings as B6, no two-goal term)
Reported for the novel sequences: family / genus of the nearest training species
centre (placement), plus AUROC and known-species accuracy.
Rule fixed in advance: pick the a with the highest novel FAMILY placement
(ties -> smaller a).

Needs data/global_taxonomy.csv (python data/add_global_taxonomy.py).
Usage (from the repo root):  python models/select_taxtree.py [a values]   # default 0 0.1 0.25 0.5
Then train the chosen one like any target, e.g.
    python models/pretrain.py taxtree0.25 --seed=0      (and seeds 1-4)
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
from models.pretrain import HP, TRAIN, global_target, load_global, parse_args  # noqa: E402

ALPHAS = ["0", "0.1", "0.25", "0.5"]


def main():
    args, _ = parse_args(sys.argv[1:])
    alphas = args or ALPHAS
    g = load_global()
    X = np.array([kmer_profile(s) for s in g.teleo])
    tz = np.load(ROOT / "data" / "global_tree_distances.npz")
    r = g.species_name.map({s: i for i, s in enumerate(tz["species"])}).values
    tax = pd.read_csv(ROOT / "data" / "global_taxonomy.csv").set_index("species_name")
    genus = g.species_name.str.split().str[0].values
    family = tax.loc[g.species_name, "Family"].values

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
    ref_genera = set(genus[ref])
    has_cong = np.array([x in ref_genera for x in genus[test]])
    fam_known = pd.notna(family[test])                      # family label available
    ref_family = dict(zip(g.species_name.values[ref], family[ref]))

    results = []
    for a in alphas:
        t0 = time.time()
        T = global_target(f"taxtree{a}", tz)
        enc = DistanceEncoder(X.shape[1], seed=0, **HP)
        enc.fit_minibatch(X[ref], r[ref], T, seed=0, verbose=False, **TRAIN)
        sc, pr = novelty_scores(enc.embed(X[ref]), g.species_name.values[ref], enc.embed(X[test]), scaled=False)
        ts = g.species_name.values[test]
        pg = np.array([p.split()[0] for p in pr])
        pf = np.array([ref_family.get(p) for p in pr], dtype=object)
        fam_ok = (pf == family[test]) & fam_known
        m_f = y & fam_known
        row = dict(a=a, family=fam_ok[m_f].mean(), family_no_congener=fam_ok[m_f & ~has_cong].mean(),
                   genus_congener=(pg == genus[test])[y & has_cong].mean(),
                   auroc=roc_auc_score(y, sc), known_acc=(pr[~y] == ts[~y]).mean())
        results.append(row)
        print(f"a = {a:5s} | novel family {row['family']:.3f} | family (no congener) {row['family_no_congener']:.3f} "
              f"| genus (congener) {row['genus_congener']:.3f} | AUROC {row['auroc']:.3f} "
              f"| known acc {row['known_acc']:.3f} | {time.time()-t0:.0f}s", flush=True)

    best = max(results, key=lambda d: (round(d["family"], 3), -float(d["a"])))
    print(f"\nchosen a = {best['a']}  (highest novel family placement on global held-out species)")
    if float(best["a"]) == 0:
        print("a = 0 means plain taxonomy wins: on global data the tree term does not help placement.")


if __name__ == "__main__":
    main()
