#!/usr/bin/env python3
"""
Taxonomy + tree together for PLACING new species, with no new training: combine the two
fine-tuned models that already exist (B6 taxonomy and the tree-clade model) on top of raw k-mers.

Why: the tree-clade model wins on family for new species with no same-genus relative in the
reference (family_easy), taxonomy wins on genus when a same-genus relative is there
(genus_hard). If the two are complementary, using both should keep both strengths.

Method (per split, per training seed; same code, centres and scaling as
models/placement_kmer_tree.py, whose helpers are imported):
  Rn, Tn, Cn = distance to each reference species centre in raw k-mer, taxonomy-model and
               tree-clade-model space, each divided by the median distance between reference
               species centres in its own space (reference only)
  raw+taxonomy             Rn + Tn                    (current comparison method)
  raw+treeclade            Rn + Cn                    (current main model)
  raw+taxonomy+treeclade   Rn + 0.5 Tn + 0.5 Cn       <- MAIN, fixed in advance: the model
                           share stays 1 as in every raw+model method; it is only split
                           equally between the two models (no weight is tuned)
  sensitivity (reported, never used to choose):  Rn + Tn + Cn  (each model at full weight)
  The taxonomy and tree-clade embeddings of the SAME training seed are combined; each split's
  score is averaged over the seeds first, as in placement_kmer_tree.py.

Outcomes (truly novel sequences unless stated; from placement_kmer_tree.outcomes):
  family, family_easy, genus_hard, known_sp, tree_error, tree_error_easy (My, lower = better)
Tests: paired Wilcoxon over the 30 splits + bootstrap 95% CI.
  primary (Holm within):  taxtree vs raw+taxonomy, taxtree vs raw+treeclade
  secondary:              taxtree vs raw, sensitivity version vs raw+taxonomy

Usage (repo root):  python models/placement_taxtree.py                 # Teleo, seeds 0-4
                    python models/placement_taxtree.py --marker=full   # long 12S
                    python models/placement_taxtree.py --seeds=0       # quick check
Needs the same inputs as placement_kmer_tree.py for the taxonomy and treeclade targets.
Output (outputs/results/):  placement_taxtree[_full]_per_seed.csv, _per_split.csv, _tests.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "models"))
import placement_kmer_tree as pk  # noqa: E402  (reads --marker=full from sys.argv too)

OUT = ROOT / "outputs" / "results"
KNOWN_OPTS = {"marker", "seeds"}
MAIN = "raw+taxonomy+treeclade"
SENS = "raw+taxonomy+treeclade (each w=1)"
METHODS = ["raw", "raw+taxonomy", "raw+treeclade", MAIN, SENS]
PRIMARY = [(MAIN, "raw+taxonomy"), (MAIN, "raw+treeclade")]
SECONDARY = [(MAIN, "raw"), (SENS, "raw+taxonomy"), ("raw+treeclade", "raw+taxonomy")]


def parse():
    opts = {}
    for a in sys.argv[1:]:
        if not (a.startswith("--") and "=" in a) or a[2:].split("=", 1)[0] not in KNOWN_OPTS:
            sys.exit(f"unknown option {a!r} - allowed: --marker=full, --seeds=0,1,2,3,4")
        k, v = a[2:].split("=", 1)
        opts[k] = v
    if opts.get("marker", "teleo") not in ("teleo", "full"):
        sys.exit("--marker must be teleo or full")
    return [int(x) for x in opts.get("seeds", "0,1,2,3,4").split(",")]


def model_dist(target, seed, k, ref, test, rsp, sp):
    z = np.load(pk.emb_path(target, seed, k), allow_pickle=True)
    V = pd.DataFrame(z["emb"], index=z["seq_id"].astype(str))
    sp_m, D, med = pk.centre_distances(V.loc[ref.seq_id].values, rsp, V.loc[test.seq_id].values)
    assert (sp_m == sp).all()
    return D / med


def run(seeds):
    clean, splits, X = pk.load()
    info = clean.drop_duplicates("species_name").set_index("species_name")
    missing = [pk.emb_path(t, s, k) for t in ("taxonomy", "treeclade") for s in seeds
               for k in sorted(splits.split_id.unique()) if not pk.emb_path(t, s, k).exists()]
    if missing:
        sys.exit(f"{len(missing)} embedding files missing, e.g. {missing[0].relative_to(ROOT)}")
    rows = []
    for k in sorted(splits.split_id.unique()):
        s = splits[splits.split_id == k]
        ref, test = s[s.role == "reference"], s[s.role != "reference"].reset_index(drop=True)
        rsp = clean.loc[ref.seq_id, "species_name"].values
        sp, D_raw, m_raw = pk.centre_distances(X.loc[ref.seq_id].values, rsp, X.loc[test.seq_id].values)
        Rn = D_raw / m_raw
        rows.append(dict(split=k, seed=-1, method="raw", **pk.outcomes(sp[Rn.argmin(1)], test, clean, info)))
        for seed in seeds:
            Tn = model_dist("taxonomy", seed, k, ref, test, rsp, sp)
            Cn = model_dist("treeclade", seed, k, ref, test, rsp, sp)
            for name, D in [("raw+taxonomy", Rn + Tn), ("raw+treeclade", Rn + Cn),
                            (MAIN, Rn + 0.5 * Tn + 0.5 * Cn), (SENS, Rn + Tn + Cn)]:
                rows.append(dict(split=k, seed=seed, method=name, **pk.outcomes(sp[D.argmin(1)], test, clean, info)))
        print(f"[{pk.MARKER}] split {k:2d} done", flush=True)
    return pd.DataFrame(rows)


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj, run_ = np.empty_like(p), 0.0
    for r, i in enumerate(o):
        run_ = max(run_, (len(p) - r) * p[i])
        adj[i] = min(run_, 1.0)
    return adj


def compare(per, pairs, family, rng):
    out = []
    for a, b in pairs:
        A, B = per[per.method == a].set_index("split"), per[per.method == b].set_index("split")
        for c in pk.OUTCOMES:
            d = (A[c] - B[c]).dropna()
            boots = [rng.choice(d.values, len(d)).mean() for _ in range(5000)]
            p = wilcoxon(d.values).pvalue if (d != 0).sum() > 1 else 1.0
            a_w, b_w = ((d < 0), (d > 0)) if c in pk.LOWER_BETTER else ((d > 0), (d < 0))
            out.append(dict(family=family, a=a, b=b, outcome=c, mean_a=A[c].mean(), mean_b=B[c].mean(),
                            diff=d.mean(), ci_lo=np.percentile(boots, 2.5), ci_hi=np.percentile(boots, 97.5),
                            a_better=int(a_w.sum()), b_better=int(b_w.sum()), n=len(d), wilcoxon_p=p))
    t = pd.DataFrame(out)
    t["holm_p"] = holm(t.wilcoxon_p.values)
    return t


def main():
    seeds = parse()
    per_seed = run(seeds)
    per = per_seed.groupby(["split", "method"], as_index=False)[pk.OUTCOMES].mean()
    sfx = pk.MK
    OUT.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(OUT / f"placement_taxtree{sfx}_per_seed.csv", index=False)
    per.to_csv(OUT / f"placement_taxtree{sfx}_per_split.csv", index=False)
    rng = np.random.default_rng(0)
    tests = pd.concat([compare(per, PRIMARY, "primary", rng), compare(per, SECONDARY, "secondary", rng)])
    tests.to_csv(OUT / f"placement_taxtree{sfx}_tests.csv", index=False)

    print(f"\n[{pk.MARKER}] {per.split.nunique()} splits | seeds {seeds} | mean over splits "
          "(tree_error in million years, lower = better)")
    print("CHECK: raw+taxonomy and raw+treeclade should match placement_kmer_tree at w = 1")
    print(per.groupby("method")[pk.OUTCOMES].mean().loc[METHODS].round(4).to_string())
    for fam in ["primary", "secondary"]:
        t = tests[tests.family == fam]
        print(f"\n{fam} comparisons (Holm within this family)")
        print(t[["a", "b", "outcome", "mean_a", "mean_b", "diff", "ci_lo", "ci_hi", "a_better", "b_better",
                 "wilcoxon_p", "holm_p"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
