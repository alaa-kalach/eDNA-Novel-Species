"""
Pretrain the distance-matching encoder on GLOBAL fish Teleo sequences
(MitoFish via Mitohelper), excluding every French Guiana species and every
sequence identical to a French Guiana Teleo sequence -> no test leakage.

Targets
    tree      -> phylogenetic distance between species (main method)
    labels    -> 0 same species / 1 different species (baseline B5)
    taxonomy  -> 0 same species / .25 genus / .5 family / .75 order / 1 (baseline B6,
                 needs data/global_taxonomy.csv from data/add_global_taxonomy.py)
    taxtree<a> -> (1 - a) * taxonomy + a * tree, e.g. taxtree0.25 (main method candidate:
                 clean taxonomic steps, the tree orders species inside each step;
                 a is chosen on global data only by models/select_taxtree.py)

10% of global species are held out to check that the encoder generalises to
species it never saw (Spearman between embedding and tree distance).

Usage (from the repo root):
    python models/pretrain.py            # both targets
    python models/pretrain.py tree
    python models/pretrain.py labels --seed=1   # another random seed
    python models/pretrain.py taxonomy --marker=full   # long 12S region (data/make_full12s.py)
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


def load_global(marker="teleo"):
    """Global training sequences in column 'teleo' (for marker='full' that column holds
    the long 12S region from data/make_full12s.py). No French Guiana species and no
    sequence identical to a French Guiana sequence -> no test leakage."""
    if marker == "full":
        g = pd.read_csv(DATA / "global_full12s.csv").rename(columns={"seq": "teleo"})
        fg_seqs = set(pd.read_csv(DATA / "guiana_full12s.csv")["seq"])
    else:
        g = pd.read_csv(DATA / "global_teleo.csv")
        fg_seqs = set(pd.read_csv(DATA / "guiana_teleo_clean.csv")["teleo"].str.strip().str.lower())
    g = g[g.in_tree & ~g.in_guiana & ~g.teleo.isin(fg_seqs)].reset_index(drop=True)
    print(f"global training data ({marker}): {len(g)} sequences, {g.species_name.nunique()} species")
    return g


def two_goal_settings(target):
    """'tree2g_w10m0.4' -> (10.0, 0.4); any other target -> (0, 0) (no two-goal term)."""
    import re
    m = re.fullmatch(r"tree2g_w([\d.]+)m([\d.]+)", target)
    return (float(m.group(1)), float(m.group(2))) if m else (0.0, 0.0)


def my_to_d(my, tz):
    """split age in million years -> scaled tree distance (tree distance = 2 x age / scale)"""
    return 2 * my / float(tz["scale"])


def encoder_settings(target, tz):
    """Extra DistanceEncoder settings implied by the target name.
    tree2g_w<W>m<M> : two-goal term          treew<My> : weight pairs by exp(-tree / tau), tau = <My>
    treetrip        : triplet loss on the tree (only 'who is closer' is used)"""
    w, m = two_goal_settings(target)
    out = dict(sep_weight=w, margin=m)
    if target.startswith("treew"):
        out["pair_tau"] = my_to_d(float(target[len("treew"):]), tz)
    if target == "treetrip":
        out["loss_type"] = "triplet"
    return out


def clade_ages():
    """Cut ages for tree-made clades, fixed in advance: the median split age of
    same-genus, same-family and same-order species pairs among GLOBAL training species
    (data/global_taxonomy.csv; French Guiana species are not used)."""
    tz = np.load(DATA / "global_tree_distances.npz")
    age = tz["D"] * float(tz["scale"]) / 2
    tax = pd.read_csv(DATA / "global_taxonomy.csv").set_index("species_name").loc[list(tz["species"])]
    fg = set(pd.read_csv(DATA / "guiana_teleo_clean.csv")["species_name"])
    keep = ~tax.index.isin(fg)
    rng = np.random.default_rng(0)
    idx = np.where(keep)[0]
    i, j = rng.choice(idx, 400_000), rng.choice(idx, 400_000)
    ok = i != j
    i, j = i[ok], j[ok]
    ages = {}
    for level in ["Genus", "Family", "Order"]:
        v = tax[level].astype(object).values
        same = (v[i] == v[j]) & pd.notna(v[i])
        if level == "Family":
            same &= tax["Genus"].values[i] != tax["Genus"].values[j]
        if level == "Order":
            same &= tax["Family"].astype(object).values[i] != tax["Family"].astype(object).values[j]
        ages[level] = float(np.median(age[i[same], j[same]]))
    return ages


def global_target(target, tz):
    """Target distance matrix over all species in global_tree_distances.npz."""
    D = tz["D"].astype(np.float64)
    if target.startswith("labels"):
        return 1.0 - np.eye(len(tz["species"]))
    if target.startswith("taxonomy"):
        return taxonomy_matrix(tz["species"])
    if target.startswith("taxtree"):
        a = taxtree_alpha(target)
        return (1 - a) * taxonomy_matrix(tz["species"]) + a * D
    if target.startswith("treecap"):                  # deep pairs all count as "far"
        c = my_to_d(float(target[len("treecap"):]), tz)
        return np.minimum(D, c) / c
    if target == "treelog":                           # stretch close distances, squeeze deep ones
        e = my_to_d(10, tz)
        return np.log1p(D / e) / np.log1p(1 / e)
    if target == "treerank":                          # rank (quantile) of each tree distance
        sample = np.sort(D[np.triu_indices(len(D), 1)][::97])
        T = np.searchsorted(sample, D) / len(sample)
        np.fill_diagonal(T, 0)
        return T
    if target == "treeclade":                         # taxonomy-like steps made from the tree
        ages = clade_ages()
        age = D * float(tz["scale"]) / 2
        T = 1.0 - 0.25 * sum((age <= ages[lv]).astype(float) for lv in ["Genus", "Family", "Order"])
        np.fill_diagonal(T, 0)
        return T
    return D                                          # tree (main method, incl. tree2g_*, treew*, treetrip)


def marker_suffix(marker):
    """Teleo keeps the original file names; the long 12S region adds _full."""
    return "_full" if marker == "full" else ""


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


def taxtree_alpha(target):
    """'taxtree0.25' -> 0.25"""
    return float(target[len("taxtree"):])


def save(enc, path):
    np.savez(path, **{p: getattr(enc, p) for p in enc.params}, mu=enc.mu, sd=enc.sd,
             linear=enc.linear, sep_weight=enc.sep_weight, margin=enc.margin,
             pair_tau=enc.pair_tau, loss_type=enc.loss_type, triplet_margin=enc.triplet_margin)


def load_encoder(path):
    z = np.load(path)
    enc = DistanceEncoder(z["W1"].shape[0], hidden=0 if z["linear"] else z["W1"].shape[1],
                          out_dim=(z["W1"] if z["linear"] else z["W2"]).shape[1])
    for p in enc.params:
        setattr(enc, p, z[p])
    enc.mu, enc.sd = z["mu"], z["sd"]
    if "sep_weight" in z.files:                      # two-goal settings (if any)
        enc.sep_weight, enc.margin = float(z["sep_weight"]), float(z["margin"])
    if "loss_type" in z.files:                       # loss variants (if any)
        enc.pair_tau, enc.loss_type = float(z["pair_tau"]), str(z["loss_type"])
        enc.triplet_margin = float(z["triplet_margin"])
    return enc


def main():
    args, opts = parse_args(sys.argv[1:])
    seed = int(opts.get("seed", 0))
    targets = args or ["tree", "labels"]
    marker = opts.get("marker", "teleo")
    g = load_global(marker)
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
        if target not in ("tree", "labels", "taxonomy", "treelog", "treerank", "treetrip", "treeclade") \
                and not target.startswith(("taxtree", "tree2g_", "treecap", "treew")):
            raise ValueError(f"unknown target {target!r}")
        T = global_target(target, tz)
        print(f"\n=== pretraining target: {target} (seed {seed}) ===")
        enc = DistanceEncoder(X.shape[1], seed=seed, **encoder_settings(target, tz), **HP)
        enc.fit_minibatch(X[~is_val], rows[~is_val], T, seed=seed,
                          X_val=X[val_rows], val_idx=rows[val_rows], **TRAIN)
        out = ROOT / "models" / f"pretrained_{target}{marker_suffix(marker)}{seed_suffix(seed)}.npz"
        save(enc, out)
        print(f"saved {out}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
