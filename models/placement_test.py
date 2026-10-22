"""
Follow-up test (decided BEFORE looking at the 20 extra splits):
does the tree help PLACE new species, compared with the same model trained
with species labels only?

Primary outcome   : family placement of novel species
                    (predicted nearest known species is in the true family)
Secondary outcomes: family placement when no congener is in the reference
                    ('easy' novel), genus placement when a congener exists
                    ('hard' novel), novelty AUROC.
Comparison        : ft_tree2g_w10m0.4 vs ft_labels, paired over 30 splits
                    (splits 1-10 from splits.csv + 11-30 from splits_extra.csv),
                    Wilcoxon signed-rank test + bootstrap 95% CI of the mean difference.
Tree outcome (added before the long-12S run): placement error on the tree = age in
million years of the split between the guessed species and the true species
(tree_error; tree_error_easy = only novel species without a congener). Lower is better.
It does not depend on taxonomic ranks, so it does not favour the taxonomy model.
Raw k-mers are shown for reference.

Seeds: with --seeds=0,1,2,3,4 each model's per-split scores are averaged over
the training seeds first, so the test is not decided by one lucky/unlucky run.
The per-seed differences are printed too.

Usage (from the repo root):
    python models/placement_test.py                    # seed 0 only
    python models/placement_test.py --seeds=0,1,2,3,4  # average over seeds
    python models/placement_test.py --b=ft_taxonomy    # compare against B6 instead of B5
    python models/placement_test.py --a=ft_taxtree0.25 --b=ft_taxonomy --seeds=0,1,2,3,4
    python models/placement_test.py --a=ft_tree2g_w10m0.4_full --b=ft_taxonomy_full --marker=full --seeds=0,1,2,3,4
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.encoder import novelty_scores  # noqa: E402
from models.pretrain import parse_args, seed_suffix  # noqa: E402

DATA, RES = ROOT / "data", ROOT / "outputs" / "results"
A, B = "ft_tree2g_w10m0.4", "ft_labels"   # proposal (scaled) score; placement is identical for both scores


def load(marker="teleo"):
    clean = pd.read_csv(DATA / "guiana_teleo_clean.csv")
    clean["seq_id"] = clean["Specimen identification code"].astype(str)
    clean = clean[clean.in_tree].set_index("seq_id")
    splits = pd.concat([pd.read_csv(DATA / "splits.csv"), pd.read_csv(DATA / "splits_extra.csv")])
    splits["seq_id"] = splits["seq_id"].astype(str)
    k = np.load(DATA / ("kmer_features_full12S.npz" if marker == "full" else "kmer_features_k3to6.npz"),
                allow_pickle=True)
    X = pd.DataFrame(k["X"], index=k["seq_id"].astype(str))
    return clean, splits, X


def results(name, seed=0):
    sfx = seed_suffix(seed)
    parts = [pd.read_csv(RES / f"results_{name}{sfx}.csv"), pd.read_csv(RES / f"results_{name}{sfx}_extra.csv")]
    r = pd.concat(parts)
    r["seq_id"] = r["seq_id"].astype(str)
    return r


def raw_results(splits, clean, X):
    out = []
    for k in sorted(splits.split_id.unique()):
        s = splits[splits.split_id == k]
        ref, te = s[s.role == "reference"], s[s.role != "reference"]
        sc, pr = novelty_scores(X.loc[ref.seq_id].values, clean.loc[ref.seq_id, "species_name"].values,
                                X.loc[te.seq_id].values, scaled=False)
        out.append(pd.DataFrame({"seq_id": te.seq_id.values, "split_id": k,
                                 "novelty_score": sc, "predicted_species": pr}))
    return pd.concat(out)


_TREE = {}


def split_age(pred, true):
    """Placement error on the tree: age (million years) of the split between the guessed
    species and the true species = half the patristic distance. Lower is better."""
    if not _TREE:
        z = np.load(DATA / "global_tree_distances.npz")
        _TREE.update(row={s: i for i, s in enumerate(z["species"])}, D=z["D"], scale=float(z["scale"]))
    i = np.array([_TREE["row"][s] for s in pred]); j = np.array([_TREE["row"][s] for s in true])
    return _TREE["D"][i, j] * _TREE["scale"] / 2


def per_split(r, splits, clean):
    info = clean.drop_duplicates("species_name").set_index("species_name")
    rows = []
    for k in sorted(splits.split_id.unique()):
        x = r[r.split_id == k].merge(splits[splits.split_id == k][["seq_id", "role", "difficulty"]], on="seq_id")
        y = (x.role == "test_novel").values
        t, p = clean.loc[x.seq_id, "species_name"].values, x.predicted_species.values
        fam = info.loc[p, "Family"].values == info.loc[t, "Family"].values
        gen = info.loc[p, "Genus"].values == info.loc[t, "Genus"].values
        easy, hard = y & (x.difficulty == "easy").values, y & (x.difficulty == "hard").values
        age = split_age(p, t)
        rows.append(dict(split=k, family=fam[y].mean(), family_easy=fam[easy].mean(),
                         genus_hard=gen[hard].mean(), auroc=roc_auc_score(y, x.novelty_score),
                         tree_error=age[y].mean(), tree_error_easy=age[easy].mean()))
    return pd.DataFrame(rows).set_index("split")


OUTCOMES = ["family", "family_easy", "genus_hard", "auroc", "tree_error", "tree_error_easy"]
LOWER_BETTER = {"tree_error", "tree_error_easy"}


def main():
    _, opts = parse_args(sys.argv[1:])
    seeds = [int(x) for x in opts.get("seeds", "0").split(",")]
    global A, B
    A, B = opts.get("a", A), opts.get("b", B)
    clean, splits, X = load(opts.get("marker", "teleo"))     # raw k-mers of the same marker
    runs_a = [per_split(results(A, s), splits, clean) for s in seeds]
    runs_b = [per_split(results(B, s), splits, clean) for s in seeds]
    a = sum(runs_a) / len(seeds)          # average over seeds, split by split
    b = sum(runs_b) / len(seeds)
    raw = per_split(raw_results(splits, clean, X), splits, clean)
    rng = np.random.default_rng(0)
    print(f"{len(a)} splits | seeds {seeds} | main = {A} | baseline = {B}\n")
    for c in OUTCOMES:
        d = (a[c] - b[c]).values
        boots = [rng.choice(d, len(d)).mean() for _ in range(5000)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        p = wilcoxon(a[c], b[c]).pvalue if (d != 0).any() else 1.0
        tag = {"family": "  <- primary outcome",
               "tree_error": "  <- tree outcome (million years, lower = better)"}.get(c, "")
        print(f"{c:12s} main {a[c].mean():.3f} | base {b[c].mean():.3f} | raw {raw[c].mean():.3f} | "
              f"diff {d.mean():+.3f} [95% CI {lo:+.3f}, {hi:+.3f}] | main better in "
              f"{((d < 0) if c in LOWER_BETTER else (d > 0)).sum()}/{len(d)} "
              f"| Wilcoxon p = {p:.3f}{tag}")
    if len(seeds) > 1:
        print("\nmean difference (main - baseline) for each seed on its own:")
        for c in OUTCOMES:
            per = [(ra[c] - rb[c]).mean() for ra, rb in zip(runs_a, runs_b)]
            print(f"{c:12s} " + "  ".join(f"s{s}: {d:+.3f}" for s, d in zip(seeds, per)))


if __name__ == "__main__":
    main()
