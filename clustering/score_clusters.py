#!/usr/bin/env python3
"""
Score the clustering files (outputs/results/clusters_<name>.csv + clusters_<name>_extra.csv).

Per split, on the sequences called "novel" (same set for every method when they all use the
same detector, e.g. B4):
  Grouping (true labels = species; noise points (-1) count as groups of one)
    ARI, AMI, bcubed_f1 (B-cubed F1)
  Naming (truly novel sequences only)
    family       guessed family correct
    family_easy  same, novel species with no same-genus species in the reference
    genus_hard   guessed genus correct, a same-genus species is in the reference
Each split = one number per outcome; methods compared with a paired Wilcoxon test over the
splits + bootstrap 95% CI of the mean difference.

Usage (repo root):
    python clustering/score_clusters.py kmer ft_taxonomy ft_treecap90_withraw ...
    python clustering/score_clusters.py            # every clusters_*.csv found (B4 novel calls)
Compares every method with the first two names given (default: kmer and ft_taxonomy).
Options:
    --out=_vs      write cluster_scores_vs_per_split.csv / _tests.csv (no overwriting other runs)
    --avg-seeds    also load each method's model seeds 1-9 (<name>_sN, before any _vs suffix) and
                   average each split over the seeds BEFORE the tests (as placement_kmer_tree.py)
Example (VSEARCH-called sequences, gap_grouping.py --detector=vsearch --emb=...):
    python clustering/score_clusters.py --out=_vs --avg-seeds gap_align_vs gap_align_ft_taxonomy_vs \
        gap_align_ft_treecap90_vs gap_kmer_vs gap_kmer_ft_taxonomy_vs gap_kmer_ft_treecap90_vs kmer_vs
Output: outputs/results/cluster_scores[<out>]_per_split.csv, cluster_scores[<out>]_tests.csv
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

ROOT = Path(__file__).resolve().parents[1]
DATA, OUT = ROOT / "data", ROOT / "outputs" / "results"
OUTCOMES = ["ARI", "AMI", "bcubed_f1", "family", "family_easy", "genus_hard"]


def bcubed_f1(lab, truth):
    lab, truth = np.asarray(lab), np.asarray(truth)
    same_c = lab[:, None] == lab[None, :]
    same_t = truth[:, None] == truth[None, :]
    both = (same_c & same_t).sum(1)
    p, r = (both / same_c.sum(1)).mean(), (both / same_t.sum(1)).mean()
    return 2 * p * r / (p + r)


def load_method(name):
    files = [OUT / f"clusters_{name}.csv", OUT / f"clusters_{name}_extra.csv"]
    found = [pd.read_csv(f, dtype={"seq_id": str}) for f in files if f.exists()]
    return pd.concat(found) if found else None


def seed_variants(name):
    """name, then the same method for model seeds 1-9 (gap_grouping.py puts _sN before _vs)."""
    return [(0, name)] + [(n, re.sub(r"(_vs)?$", rf"_s{n}\1", name, count=1)) for n in range(1, 10)]


def main():
    opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    out_tag = opts.get("out", "")                       # e.g. --out=_vs -> cluster_scores_vs_*.csv
    avg_seeds = "--avg-seeds" in sys.argv[1:]           # average each split over model seeds first
    names = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not names:
        names = sorted({p.stem[len("clusters_"):].removesuffix("_extra") for p in OUT.glob("clusters_*.csv")
                        if "truenovel" not in p.stem})
        names = ["kmer", "ft_taxonomy"] + [n for n in names if n not in ("kmer", "ft_taxonomy")]
    clean = pd.read_csv(DATA / "guiana_teleo_clean.csv")
    clean["seq_id"] = clean["Specimen identification code"].astype(str)
    clean = clean[clean.in_tree].drop_duplicates("seq_id").set_index("seq_id")
    splits = pd.concat([pd.read_csv(DATA / "splits.csv"), pd.read_csv(DATA / "splits_extra.csv")])
    splits["seq_id"] = splits.seq_id.astype(str)
    role = splits.set_index(["split_id", "seq_id"])[["role", "difficulty"]]

    rows = []
    for name, seed, fname in [(n, s, f) for n in names
                              for s, f in (seed_variants(n) if avg_seeds else [(0, n)])]:
        df = load_method(fname)
        if df is None:
            if seed == 0:
                print(f"skipped {name}: no clusters_{name}*.csv")
            continue
        for sid, g in df.groupby("split_id"):
            g = g.reset_index(drop=True)
            lab = g.cluster_id.values.copy()
            noise = lab < 0
            lab[noise] = lab.max() + 1 + np.arange(noise.sum())          # noise = groups of one
            truth = clean.loc[g.seq_id, "species_name"].values
            r = role.loc[[(sid, q) for q in g.seq_id]]
            nov = (r.role == "test_novel").values
            easy, hard = nov & (r.difficulty == "easy").values, nov & (r.difficulty == "hard").values
            fam = g.guess_family.values == clean.loc[g.seq_id, "Family"].values
            gen = g.guess_genus.values == clean.loc[g.seq_id, "Genus"].values
            m = lambda x: x.mean() if len(x) else np.nan  # noqa: E731
            rows.append(dict(method=name, seed=seed, split=sid, n_called=len(g), n_true_novel=int(nov.sum()),
                             ARI=adjusted_rand_score(truth, lab), AMI=adjusted_mutual_info_score(truth, lab),
                             bcubed_f1=bcubed_f1(lab, truth), family=m(fam[nov]), family_easy=m(fam[easy]),
                             genus_hard=m(gen[hard])))
    per = pd.DataFrame(rows)
    n_seeds = per.groupby("method").seed.nunique()
    per = per.groupby(["method", "split"], as_index=False, sort=False).mean(numeric_only=True).drop(columns="seed")
    per["n_seeds"] = per.method.map(n_seeds)
    OUT.mkdir(parents=True, exist_ok=True)
    per.round(4).to_csv(OUT / f"cluster_scores{out_tag}_per_split.csv", index=False)
    if avg_seeds:
        print("model seeds per method: " + ", ".join(f"{k} {v}" for k, v in n_seeds.items()))

    tab = per.groupby("method")[OUTCOMES + ["n_called"]].mean().loc[[n for n in names if n in set(per.method)]]
    n_splits = per.groupby("method").split.nunique()
    print("mean over splits (splits per method: " + ", ".join(f"{k} {v}" for k, v in n_splits.items()) + ")")
    print(tab.round(3).to_string())

    rng = np.random.default_rng(0)
    tests = []
    bases = [b for b in names[:2] if b in set(per.method)]
    print("\npaired comparisons (Wilcoxon over splits; diff = method - baseline)")
    for b in bases:
        B = per[per.method == b].set_index("split")
        for a in tab.index:
            if a == b or (a in bases and bases.index(a) < bases.index(b)):
                continue                                 # each pair once
            A = per[per.method == a].set_index("split")
            cells = []
            for o in OUTCOMES:
                d = (A[o] - B[o]).dropna()
                if len(d) < 2:
                    continue
                boots = [rng.choice(d.values, len(d)).mean() for _ in range(5000)]
                lo, hi = np.percentile(boots, [2.5, 97.5])
                p = wilcoxon(d.values).pvalue if (d != 0).any() else 1.0
                tests.append(dict(a=a, b=b, outcome=o, mean_a=A[o].mean(), mean_b=B[o].mean(), diff=d.mean(),
                                  ci_lo=lo, ci_hi=hi, a_better=int((d > 0).sum()), b_better=int((d < 0).sum()),
                                  n=len(d), wilcoxon_p=p))
                cells.append(f"{o} {d.mean():+.3f} (p={p:.3f})")
            print(f"  {a} vs {b}: " + " | ".join(cells))
    pd.DataFrame(tests).to_csv(OUT / f"cluster_scores{out_tag}_tests.csv", index=False)
    print(f"\nwrote outputs/results/cluster_scores{out_tag}_per_split.csv and cluster_scores{out_tag}_tests.csv")


if __name__ == "__main__":
    main()
