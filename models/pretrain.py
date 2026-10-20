"""
Pretrain the distance-matching encoder on GLOBAL fish Teleo sequences
(MitoFish via Mitohelper), excluding every French Guiana species and every
sequence identical to a French Guiana Teleo sequence -> no test leakage.

Targets
    tree      -> phylogenetic distance between species (main method)
    labels    -> 0 same species / 1 different species (baseline B5)
    taxonomy  -> 0 same species / .25 genus / .5 family / .75 order / 1 (baseline B6,
                 needs data/global_taxonomy.csv from data/add_global_taxonomy.py)

10% of global species are held out to check that the encoder generalises to
species it never saw (Spearman between embedding and tree distance).

Usage (from the repo root):
    python models/pretrain.py            # both targets
    python models/pretrain.py tree
    python models/pretrain.py labels --seed=1   # another random seed
Output: models/pretrained_<target>.npz      (seed 0)
        models/pretrained_<target>_s<N>.npz (seed N)
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.encoder import DistanceEncoder  # noqa: E402
from models.kmers import kmer_profile  # noqa: E402

DATA = ROOT / "data"
HP = dict(hidden=256, out_dim=64, lr=1e-3, weight_decay=1e-4)
TRAIN = dict(steps=4000, batch=512)


def load_global():
    g = pd.read_csv(DATA / "global_teleo.csv")
    fg = pd.read_csv(DATA / "guiana_teleo_clean.csv")
    fg_seqs = set(fg["teleo"].str.strip().str.lower())
    g = g[g.in_tree & ~g.in_guiana & ~g.teleo.isin(fg_seqs)].reset_index(drop=True)
    print(f"global training data: {len(g)} sequences, {g.species_name.nunique()} species")
    return g


def parse_args(argv):
    """Split argv into plain arguments and --key=value options."""
    opts = dict(a[2:].split("=", 1) for a in argv if a.startswith("--"))
    return [a for a in argv if not a.startswith("--")], opts


def seed_suffix(seed):
    """Seed 0 keeps the original file names; other seeds get _s<N>."""
    return "" if int(seed) == 0 else f"_s{int(seed)}"


def taxonomy_matrix(species):
    """Taxonomy distance between global species (same scale as the regional D_tax).
    A missing family/order counts as different from every other species."""
    tax = pd.read_csv(DATA / "global_taxonomy.csv").set_index("species_name").loc[list(species)]
    D = np.ones((len(species), len(species)))
    for level, d in [("Order", 0.75), ("Family", 0.5), ("Genus", 0.25)]:
        v = tax[level].astype(object).values
        same = (v[:, None] == v[None, :]) & pd.notna(v)[:, None]
        D[same] = d
    np.fill_diagonal(D, 0.0)
    return D


def global_target(target, tz):
    """Target distance matrix over all species in global_tree_distances.npz."""
    if target.startswith("labels"):
        return 1.0 - np.eye(len(tz["species"]))
    if target.startswith("taxonomy"):
        return taxonomy_matrix(tz["species"])
    return tz["D"].astype(np.float64)                   # tree (main method, incl. tree2g_*)


def save(enc, path):
    np.savez(path, **{p: getattr(enc, p) for p in enc.params}, mu=enc.mu, sd=enc.sd,
             linear=enc.linear, sep_weight=enc.sep_weight, margin=enc.margin)


def load_encoder(path):
    z = np.load(path)
    enc = DistanceEncoder(z["W1"].shape[0], hidden=0 if z["linear"] else z["W1"].shape[1],
                          out_dim=(z["W1"] if z["linear"] else z["W2"]).shape[1])
    for p in enc.params:
        setattr(enc, p, z[p])
    enc.mu, enc.sd = z["mu"], z["sd"]
    if "sep_weight" in z.files:                      # two-goal settings (if any)
        enc.sep_weight, enc.margin = float(z["sep_weight"]), float(z["margin"])
    return enc


def main():
    args, opts = parse_args(sys.argv[1:])
    seed = int(opts.get("seed", 0))
    targets = args or ["tree", "labels"]
    g = load_global()
    t0 = time.time()
    X = np.array([kmer_profile(s) for s in g.teleo])
    print(f"k-mer features {X.shape} in {time.time()-t0:.0f}s")

    tz = np.load(DATA / "global_tree_distances.npz")
    sp_row = {s: i for i, s in enumerate(tz["species"])}
    rows = g.species_name.map(sp_row).values

    rng = np.random.default_rng(0)               # held-out species stay the same for every seed
    species = np.array(sorted(g.species_name.unique()))
    val_sp = set(rng.choice(species, size=len(species) // 10, replace=False))
    is_val = g.species_name.isin(val_sp).values
    # one sequence per held-out species keeps the validation matrix small
    val_rows = g[is_val].drop_duplicates("species_name").index.values

    for target in targets:
        if target not in ("tree", "labels", "taxonomy"):
            raise ValueError(f"unknown target {target!r} (use tree, labels or taxonomy)")
        T = global_target(target, tz)
        print(f"\n=== pretraining target: {target} (seed {seed}) ===")
        enc = DistanceEncoder(X.shape[1], seed=seed, **HP)
        enc.fit_minibatch(X[~is_val], rows[~is_val], T, seed=seed,
                          X_val=X[val_rows], val_idx=rows[val_rows], **TRAIN)
        out = ROOT / "models" / f"pretrained_{target}{seed_suffix(seed)}.npz"
        save(enc, out)
        print(f"saved {out}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
