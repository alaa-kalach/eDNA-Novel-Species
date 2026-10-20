"""
Choose the two-goal settings (sep_weight, margin) on GLOBAL data only.

Held-out check (no French Guiana data involved):
  - 10% of global species are hidden  -> 'novel'
  - one distinct sequence of each other species with >= 2 is held out -> 'known'
  - the encoder is pretrained on the rest
Reported: novelty AUROC, known-species accuracy, genus accuracy for novel
sequences whose genus is in training, and Spearman(embedding, tree) on novel.

The model trained here (on 90% of global species) is saved as the pretrained
model for each setting: models/pretrained_tree2g_<name>.npz

Usage (from the repo root):  python models/select_twogoal.py [setting names] [--seed=N]
Seed N > 0 saves models/pretrained_tree2g_<name>_s<N>.npz
"""
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.encoder import DistanceEncoder, novelty_scores, pairwise  # noqa: E402
from models.kmers import kmer_profile  # noqa: E402
from models.pretrain import HP, TRAIN, load_global, parse_args, save, seed_suffix  # noqa: E402

SETTINGS = {"none": (0.0, 0.0), "m0.1": (1.0, 0.1), "m0.2": (1.0, 0.2),
            "w10m0.2": (10.0, 0.2), "w10m0.4": (10.0, 0.4)}


def main():
    args, opts = parse_args(sys.argv[1:])
    seed = int(opts.get("seed", 0))
    g = load_global()
    X = np.array([kmer_profile(s) for s in g.teleo])
    tz = np.load(ROOT / "data" / "global_tree_distances.npz")
    row = {s: i for i, s in enumerate(tz["species"])}
    r = g.species_name.map(row).values
    T = tz["D"].astype(np.float64)
    genus = g.species_name.str.split().str[0].values

    rng = np.random.default_rng(0)               # same held-out check for every seed
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
    nov_one = g[role == "novel"].drop_duplicates("species_name").index.values

    names = args or list(SETTINGS)
    for name in names:
        w, m = SETTINGS[name]
        t0 = time.time()
        enc = DistanceEncoder(X.shape[1], seed=seed, sep_weight=w, margin=m, **HP)
        enc.fit_minibatch(X[ref], r[ref], T, seed=seed, verbose=False, **TRAIN)
        sc, pr = novelty_scores(enc.embed(X[ref]), g.species_name.values[ref], enc.embed(X[test]))
        ts = g.species_name.values[test]
        pg = np.array([p.split()[0] for p in pr])
        Zn = enc.embed(X[nov_one]); iu = np.triu_indices(len(Zn), 1)
        rho = spearmanr(pairwise(Zn)[iu], T[np.ix_(r[nov_one], r[nov_one])][iu])[0]
        print(f"{name:5s} seed {seed} (weight {w}, margin {m}) | AUROC {roc_auc_score(y, sc):.3f} "
              f"| known acc {(pr[~y] == ts[~y]).mean():.3f} "
              f"| novel genus {(pg == genus[test])[y & has_cong].mean():.3f} "
              f"| tree Spearman {rho:.3f} | {time.time()-t0:.0f}s", flush=True)
        save(enc, ROOT / "models" / f"pretrained_tree2g_{name}{seed_suffix(seed)}.npz")


if __name__ == "__main__":
    main()
