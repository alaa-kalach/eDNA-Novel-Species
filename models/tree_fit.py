"""
Diagnostic: where does each model's space follow the tree, and where does it fail?

For every split, species centres are computed in each space (model embeddings or raw
k-mers). Species pairs are grouped by the age of their split on the tree
(LCA age = half the tree distance, million years):
    close  < 60 My   (about within a family; same-family median is 59 My)
    mid    60-120 My
    deep   >= 120 My (mostly different orders)
For each group we report Spearman(distance in the space, tree distance), separately for
  ref-ref   : pairs of reference species (the model was trained on them)
  novel-ref : pairs of one novel species and one reference species (never trained on)
plus  nn-tree : how often a novel species' nearest reference species in the space is
                among its 5 nearest reference species on the tree (local neighbourhood).
Scores are averaged over splits and seeds.

Usage (repo root):  python models/tree_fit.py                 # Teleo
                    python models/tree_fit.py --marker=full   # long 12S
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
DATA, EMB = ROOT / "data", ROOT / "outputs" / "embeddings"
MARKER = "full" if "--marker=full" in sys.argv[1:] else "teleo"
MK = "_full" if MARKER == "full" else ""
MODELS = {"tree": f"ft_tree2g_w10m0.4{MK}", "taxonomy": f"ft_taxonomy{MK}", "labels": f"ft_labels{MK}"}
EXTRA = [a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--add=")]   # e.g. --add=ft_treecap0.3
for e in EXTRA:
    MODELS[e] = e + MK
SEEDS = [0, 1, 2, 3, 4]
BINS = [("close <60My", 0, 60), ("mid 60-120My", 60, 120), ("deep >=120My", 120, 1e9)]


def dist(A, B):
    return np.sqrt(np.maximum((A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None] - 2 * A @ B.T, 0))


def main():
    clean = pd.read_csv(DATA / "guiana_teleo_clean.csv")
    clean["seq_id"] = clean["Specimen identification code"].astype(str)
    sp_of = clean[clean.in_tree].drop_duplicates("seq_id").set_index("seq_id")["species_name"]
    splits = pd.concat([pd.read_csv(DATA / "splits.csv"), pd.read_csv(DATA / "splits_extra.csv")])
    splits["seq_id"] = splits.seq_id.astype(str)
    tz = np.load(DATA / "global_tree_distances.npz")
    row = {s: i for i, s in enumerate(tz["species"])}
    AGE = tz["D"] * float(tz["scale"]) / 2
    k = np.load(DATA / ("kmer_features_full12S.npz" if MARKER == "full" else "kmer_features_k3to6.npz"),
                allow_pickle=True)
    RAW = pd.DataFrame(k["X"], index=k["seq_id"].astype(str))

    def measure(V, s):
        ref_ids = s[s.role == "reference"].seq_id.values
        nov_ids = s[s.role == "test_novel"].seq_id.values
        V = V.loc[np.concatenate([ref_ids, nov_ids])]
        sp = sp_of.loc[V.index].values
        C = V.groupby(sp).mean()
        ref_sp = np.unique(sp_of.loc[ref_ids].values)
        nov_sp = np.unique(sp_of.loc[nov_ids].values)
        out = {}
        for name, A, B in [("ref-ref", ref_sp, ref_sp), ("novel-ref", nov_sp, ref_sp)]:
            D = dist(C.loc[A].values, C.loc[B].values)
            G = AGE[np.ix_([row[x] for x in A], [row[x] for x in B])]
            mask = np.triu(np.ones_like(D, bool), 1) if name == "ref-ref" else np.ones_like(D, bool)
            for b, lo, hi in BINS:
                m = mask & (G >= lo) & (G < hi)
                out[f"{name} | {b}"] = spearmanr(D[m], G[m])[0] if m.sum() > 10 else np.nan
            if name == "novel-ref":
                nn = D.argmin(1)
                top5 = np.argsort(G, axis=1)[:, :5]
                out["nn-tree top5"] = np.mean([nn[i] in top5[i] for i in range(len(A))])
        return out

    res = {}
    for sid in sorted(splits.split_id.unique()):
        s = splits[splits.split_id == sid]
        res.setdefault("raw", []).append(measure(RAW, s))
        for name, model in MODELS.items():
            for seed in SEEDS:
                f = EMB / f"embeddings_{model}{'' if seed == 0 else f'_s{seed}'}_split{sid}.npz"
                if f.exists():
                    z = np.load(f, allow_pickle=True)
                    res.setdefault(name, []).append(measure(pd.DataFrame(z["emb"], index=z["seq_id"].astype(str)), s))
    table = pd.DataFrame({m: pd.DataFrame(v).mean() for m, v in res.items()})
    n = {m: len(v) for m, v in res.items()}
    print(f"marker: {MARKER} | runs averaged (splits x seeds): {n}")
    print("Spearman(distance in the space, tree distance) inside each age group; nn-tree top5 = share of novel")
    print("species whose nearest reference species in the space is among its 5 nearest on the tree\n")
    print(table.round(3).to_string())
    table.round(4).to_csv(ROOT / "outputs" / "results" / f"tree_fit{MK}.csv")
    print(f"\nsaved outputs/results/tree_fit{MK}.csv")


if __name__ == "__main__":
    main()
