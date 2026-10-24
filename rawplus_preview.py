"""
Quick preview (not the official version): raw k-mers + model distance for placement.

For each test sequence and reference species centre:
    D = D_raw / median_raw + D_model / median_model        (equal weight, fixed in advance)
medians = median distance between reference species centres (reference data only).
Placement = nearest reference species under D -> its genus / family.
Outcomes: family / genus placement, and the tree outcome = age in million years of the
split between the guessed and the true species (lower = better; no taxonomic ranks involved).
Models: tree (ft_tree2g_w10m0.4), taxonomy (ft_taxonomy), labels (ft_labels); raw alone too.
Extra check: raw + taxonomy + tree = D_raw + D_taxonomy + D_tree (all equal weight).
Each split's score is averaged over the seeds, then paired Wilcoxon over the 30 splits.

Needs outputs/embeddings/embeddings_ft_<model>[_s<N>]_split<k>.npz (from finetune_eval.py)
and data/kmer_features_k3to6.npz (python models/kmers.py).
Usage (repo root):  python rawplus_preview.py                  # Teleo
                    python rawplus_preview.py --marker=full    # long 12S region
                    python rawplus_preview.py --add=treecap60  # also a new model
"""
from pathlib import Path

import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent
EMB = ROOT / "outputs" / "embeddings"
MARKER = "full" if "--marker=full" in sys.argv[1:] else "teleo"     # --marker=full -> long 12S region
MK = "_full" if MARKER == "full" else ""
MODELS = {"tree": f"ft_tree2g_w10m0.4{MK}", "taxonomy": f"ft_taxonomy{MK}", "labels": f"ft_labels{MK}"}
SEEDS = [0, 1, 2, 3, 4]
# model distances added to the raw distance (equal weights, same seed for both models)
COMBOS = ["tree", "taxonomy", "labels", "taxonomy+tree"]
# --add=<model> adds another fine-tuned model, e.g. --add=treecap60 (files embeddings_ft_treecap60...)
ADDED = [a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--add=")]
for a in ADDED:
    MODELS[a] = f"ft_{a}{MK}"
    COMBOS.append(a)

clean = pd.read_csv(ROOT / "data/guiana_teleo_clean.csv")
clean["seq_id"] = clean["Specimen identification code"].astype(str)
clean = clean[clean.in_tree].drop_duplicates("seq_id").set_index("seq_id")
info = clean.drop_duplicates("species_name").set_index("species_name")
splits = pd.concat([pd.read_csv(ROOT / "data/splits.csv"), pd.read_csv(ROOT / "data/splits_extra.csv")])
splits["seq_id"] = splits.seq_id.astype(str)
k = np.load(ROOT / ("data/kmer_features_full12S.npz" if MARKER == "full" else "data/kmer_features_k3to6.npz"),
            allow_pickle=True)
RAW = pd.DataFrame(k["X"], index=k["seq_id"].astype(str))


def dist(A, B):
    return np.sqrt(np.maximum((A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None] - 2 * A @ B.T, 0))


def centre_dist(V, ref, rsp, test):
    """Distances test -> species centres, scaled by the median distance between centres."""
    sps = np.unique(rsp)
    C = np.array([V.loc[ref[rsp == s]].values.mean(0) for s in sps])
    m = np.median(dist(C, C)[np.triu_indices(len(C), 1)])
    return sps, dist(V.loc[test].values, C) / m


TZ = np.load(ROOT / "data/global_tree_distances.npz")
TROW = {s: i for i, s in enumerate(TZ["species"])}
TSCALE = float(TZ["scale"])


def scores(pred, true, diff):
    fam = info.loc[pred, "Family"].values == info.loc[true, "Family"].values
    gen = info.loc[pred, "Genus"].values == info.loc[true, "Genus"].values
    # tree outcome: age (million years) of the split between guessed and true species
    age = TZ["D"][[TROW[x] for x in pred], [TROW[x] for x in true]] * TSCALE / 2
    return dict(family=fam.mean(), family_easy=fam[diff == "easy"].mean(), genus_hard=gen[diff == "hard"].mean(),
                tree_error=age.mean(), tree_error_easy=age[diff == "easy"].mean())


def emb_file(model, seed, sid):
    return EMB / f"embeddings_{model}{'' if seed == 0 else f'_s{seed}'}_split{sid}.npz"


rows = []
for sid in sorted(splits.split_id.unique()):
    s = splits[splits.split_id == sid]
    ref = s[s.role == "reference"].seq_id.values
    nov = s[s.role == "test_novel"]
    test, diff = nov.seq_id.values, nov.difficulty.values
    rsp, true = clean.loc[ref, "species_name"].values, clean.loc[test, "species_name"].values
    sps, Draw = centre_dist(RAW, ref, rsp, test)
    r = {"split": sid, **{f"raw|{o}": v for o, v in scores(sps[Draw.argmin(1)], true, diff).items()}}
    D = {}                                     # (model, seed) -> scaled model distance
    for name, model in MODELS.items():
        for seed in SEEDS:
            f = emb_file(model, seed, sid)
            if f.exists():
                z = np.load(f, allow_pickle=True)
                V = pd.DataFrame(z["emb"], index=z["seq_id"].astype(str))
                D[name, seed] = centre_dist(V, ref, rsp, test)[1]
    for combo in COMBOS:
        parts = combo.split("+")
        per_seed = [scores(sps[(Draw + sum(D[m, seed] for m in parts)).argmin(1)], true, diff)
                    for seed in SEEDS if all((m, seed) in D for m in parts)]
        if per_seed:
            for o in per_seed[0]:
                r[f"raw+{combo}|{o}"] = np.mean([p[o] for p in per_seed])
            r[f"raw+{combo}|n_seeds"] = len(per_seed)
    rows.append(r)
df = pd.DataFrame(rows).set_index("split")


def compare(a, b):
    if f"{a}|family" not in df or f"{b}|family" not in df:
        print(f"  (skipped {a} vs {b}: embeddings missing)")
        return
    print(f"\n{a}  vs  {b}")
    for o in ["family", "family_easy", "genus_hard", "tree_error", "tree_error_easy"]:
        x, y = df[f"{a}|{o}"], df[f"{b}|{o}"]
        d = x - y
        p = wilcoxon(x, y).pvalue if (d != 0).sum() > 1 else 1.0
        better = (d < 0) if o.startswith("tree_error") else (d > 0)     # tree error: lower = better
        unit = " My (lower = better)" if o.startswith("tree_error") else ""
        print(f"  {o:15s} {x.mean():.3f} vs {y.mean():.3f} | diff {d.mean():+.3f} "
              f"| better in {better.sum()}/{len(d)} | p = {p:.3f}{unit}")


seeds_used = {n: int(df[f"raw+{n}|n_seeds"].min()) for n in COMBOS if f"raw+{n}|n_seeds" in df}
print(f"marker: {MARKER} | {len(df)} splits | seeds per model: {seeds_used}")
compare("raw+tree", "raw+taxonomy")      # the key question: does the tree beat taxonomy?
compare("raw+tree", "raw+labels")
compare("raw+taxonomy", "raw")
compare("raw+tree", "raw")
print("\n--- extra check: all three together ---")
compare("raw+taxonomy+tree", "raw+taxonomy")
compare("raw+taxonomy+tree", "raw")
for a in ADDED:
    print(f"\n--- added model: {a} ---")
    compare(f"raw+{a}", "raw+taxonomy")
    compare(f"raw+{a}", "raw+tree")
    compare(f"raw+{a}", "raw")
