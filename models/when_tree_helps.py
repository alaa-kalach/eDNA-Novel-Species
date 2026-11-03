#!/usr/bin/env python3
"""
WHEN does the tree help placement? Gain of tree-based placement split by how ISOLATED each new
species is from the reference library.

Expectation (stated before running): the tree's gain over sequence similarity grows with
isolation, i.e. the tree adds information where the sequence has no close match in the library
(this would explain the earlier pattern: better family_easy, not better genus_hard).
Analysis only: it reads per-sequence guesses that are already on disk; nothing is re-trained.

RULES, FIXED BEFORE RUNNING
  Sequences: truly novel test sequences (role == test_novel), all 30 splits.
  Isolation, two measures:
    tree  (main)      age (million years) of the split between the new species and its CLOSEST
                      reference species in that split, on the Fish Tree (global_tree_distances.npz,
                      age = tree distance / 2). Bins fixed in advance by biology:
                      < 30 My, 30-60 My, >= 60 My (~ the median genus / family split ages here,
                      the same cut points as models/detect_features.py).
    sequence (secondary, observable without knowing the species, so usable by a later hybrid
                      rule) 1 - identity to the nearest reference sequence on the placement
                      alignment (align_novelty in phylo_placement_per_seq.csv). Bins = quartiles
                      of that value over all novel sequences of the marker (Q1 = least isolated).
  Methods (guesses already saved):
    phylo               placement on the Fish Tree      (outputs/results/phylo_placement[_full]_per_seq.csv)
    alignment nearest   same alignment, no tree         (same file)
    raw, raw+taxonomy, raw+treecap90 (w = 1)           (outputs/placement_kmer_tree[_full]_per_seq.csv;
                        model methods: each sequence's score is averaged over the 5 seeds)
  Outcomes per sequence: family correct (0/1), tree_error = split age between guessed and true
  species (My, lower = better).
  Unit = split: per split and bin, the mean over its novel sequences; methods compared with a
  paired Wilcoxon test over splits + bootstrap 95% CI of the mean difference (as elsewhere).
  PRIMARY TEST (one per marker): phylo minus alignment nearest, family, tree isolation;
    per split  trend = gain(>= 60 My) - gain(< 30 My); Wilcoxon over splits that have both bins.
    Supported if the trend is > 0 with a 95% CI above 0 on Teleo.
  Secondary (reported, not used to decide): the same for tree_error and for sequence isolation;
    per-bin gains; raw+treecap90 vs raw+taxonomy and vs raw (tree model vs taxonomy model).

Usage (repo root):  python models/when_tree_helps.py          # Teleo and long 12S if present
Needs:  outputs/results/phylo_placement[_full]_per_seq.csv   (models/phylo_placement.py)
        outputs/placement_kmer_tree[_full]_per_seq.csv       (models/placement_kmer_tree.py)
        data/global_tree_distances.npz                       (models/tree_distances.py)
Output: outputs/results/when_tree_helps[_full]_per_seq.csv    (isolation + correctness per sequence)
        outputs/results/when_tree_helps[_full]_per_split.csv  (split x isolation x bin x method)
        outputs/results/when_tree_helps[_full]_tests.csv      (per-bin gains and trend tests)
        figures/when_tree_helps.png                           (family accuracy by tree isolation)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
DATA, OUT, FIG = ROOT / "data", ROOT / "outputs" / "results", ROOT / "figures"
TREE_BINS = [0, 30, 60, np.inf]
TREE_LABELS = ["< 30 My", "30-60 My", ">= 60 My"]
SEQ_LABELS = ["Q1 (closest)", "Q2", "Q3", "Q4 (most isolated)"]
METHODS = ["alignment nearest", "phylo", "raw", "raw+taxonomy", "raw+treecap90"]
KT_METHODS = ["raw", "raw+taxonomy", "raw+treecap90"]
PAIRS = [("phylo", "alignment nearest"),          # primary: the tree's own effect, same alignment
         ("raw+treecap90", "raw+taxonomy"),        # tree model vs taxonomy model
         ("raw+treecap90", "raw"),
         ("phylo", "raw+taxonomy")]
OUTCOMES = ["family", "tree_error"]               # tree_error: lower = better
MARKERS = {"teleo": "", "full": "_full"}


def load():
    clean = pd.read_csv(DATA / "guiana_teleo_clean.csv")
    clean["seq_id"] = clean["Specimen identification code"].astype(str)
    clean = clean[clean.in_tree].drop_duplicates("seq_id").set_index("seq_id")
    splits = pd.concat([pd.read_csv(DATA / "splits.csv"), pd.read_csv(DATA / "splits_extra.csv")])
    splits["seq_id"] = splits.seq_id.astype(str)
    tf = DATA / "global_tree_distances.npz"
    if not tf.exists():
        sys.exit(f"missing {tf} - run: python models/tree_distances.py")
    z = np.load(tf)
    row = {s: i for i, s in enumerate(z["species"])}
    age = z["D"] * float(z["scale"]) / 2                      # split age in million years
    return clean, splits, row, age


def per_sequence(mk, clean, splits, row, age):
    """One row per (split, novel sequence): isolation measures + each method's outcomes."""
    pf = OUT / f"phylo_placement{mk}_per_seq.csv"
    kf = ROOT / "outputs" / f"placement_kmer_tree{mk}_per_seq.csv"
    for f in (pf, kf):
        if not f.exists():
            print(f"  skipped: missing {f.relative_to(ROOT)}")
            return None
    ph = pd.read_csv(pf, dtype={"seq_id": str}).set_index(["split_id", "seq_id"])
    kt = pd.read_csv(kf, dtype={"seq_id": str})
    kt = kt[kt.method.isin(KT_METHODS)]
    info = clean.drop_duplicates("species_name").set_index("species_name")

    nov = splits[splits.role == "test_novel"][["split_id", "seq_id", "difficulty"]].copy()
    nov["true"] = clean.loc[nov.seq_id, "species_name"].values
    # tree isolation: closest reference species in the same split
    iso = {}
    for k, s in splits[splits.role == "reference"].groupby("split_id"):
        ref_rows = [row[x] for x in np.unique(clean.loc[s.seq_id, "species_name"])]
        for sp in nov.loc[nov.split_id == k, "true"].unique():
            iso[(k, sp)] = age[row[sp], ref_rows].min()
    nov["iso_tree"] = [iso[(k, sp)] for k, sp in zip(nov.split_id, nov.true)]
    key = list(zip(nov.split_id, nov.seq_id))
    nov["iso_seq"] = ph.loc[key, "align_novelty"].values

    def score(guess, true):
        fam = (info.loc[guess, "Family"].values == info.loc[true, "Family"].values).astype(float)
        err = age[[row[g] for g in guess], [row[t] for t in true]]
        return fam, err

    for m, col in [("phylo", "phylo_species"), ("alignment nearest", "align_species")]:
        nov[f"{m}|family"], nov[f"{m}|tree_error"] = score(ph.loc[key, col].values, nov.true.values)
    kt = kt.merge(nov[["split_id", "seq_id", "true"]], on=["split_id", "seq_id"])
    kt["family"], kt["tree_error"] = score(kt.guess.values, kt.true.values)
    kt = kt.groupby(["method", "split_id", "seq_id"])[OUTCOMES].mean()   # average over seeds
    for m in KT_METHODS:
        got = kt.loc[m].reindex(key)
        if got.family.isna().any():
            sys.exit(f"{kf.name}: {m} has no guess for some novel sequences - rerun placement_kmer_tree.py")
        for o in OUTCOMES:
            nov[f"{m}|{o}"] = got[o].values

    nov["bin_tree"] = pd.cut(nov.iso_tree, TREE_BINS, right=False, labels=TREE_LABELS)
    nov["bin_seq"] = pd.qcut(nov.iso_seq.rank(method="first"), 4, labels=SEQ_LABELS)
    return nov


def per_split(nov):
    rows = []
    for iso in ["tree", "seq"]:
        for (k, b), g in nov.groupby(["split_id", f"bin_{iso}"], observed=True):
            for m in METHODS:
                rows.append(dict(isolation=iso, split=k, bin=b, method=m, n=len(g),
                                 **{o: g[f"{m}|{o}"].mean() for o in OUTCOMES}))
    return pd.DataFrame(rows)


def paired(d, rng):
    d = d.dropna()
    if len(d) < 2:
        return dict(diff=np.nan, ci_lo=np.nan, ci_hi=np.nan, n=len(d), wilcoxon_p=np.nan)
    boots = [rng.choice(d.values, len(d)).mean() for _ in range(5000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p = wilcoxon(d.values).pvalue if (d != 0).any() else 1.0
    return dict(diff=d.mean(), ci_lo=lo, ci_hi=hi, n=len(d), wilcoxon_p=p)


def tests(ps, rng):
    out = []
    for iso, labels in [("tree", TREE_LABELS), ("seq", SEQ_LABELS)]:
        sub = ps[ps.isolation == iso]
        for a, b in PAIRS:
            for o in OUTCOMES:
                A = sub[sub.method == a].pivot(index="split", columns="bin", values=o)
                B = sub[sub.method == b].pivot(index="split", columns="bin", values=o)
                gain = (A - B).reindex(columns=labels)
                for lab in labels:
                    out.append(dict(isolation=iso, a=a, b=b, outcome=o, bin=lab,
                                    mean_a=A[lab].mean(), mean_b=B[lab].mean(), **paired(gain[lab], rng)))
                trend = gain[labels[-1]] - gain[labels[0]]
                primary = iso == "tree" and (a, b) == PAIRS[0] and o == "family"
                out.append(dict(isolation=iso, a=a, b=b, outcome=o, bin=f"trend: {labels[-1]} minus {labels[0]}",
                                primary=primary, **paired(trend, rng)))
    t = pd.DataFrame(out)
    t["primary"] = t["primary"].fillna(False).astype(bool)
    return t


def report(name, nov, t):
    print(f"\n=== {name}: {len(nov)} novel sequences, {nov.split_id.nunique()} splits")
    for iso, labels in [("tree", TREE_LABELS), ("seq", SEQ_LABELS)]:
        what = ("tree isolation (age of the split to the closest reference species)" if iso == "tree"
                else "sequence isolation (1 - identity to the nearest reference sequence)")
        print(f"\n{what}\n" + "-" * len(what))
        lo_hi = nov.groupby(f"bin_{iso}", observed=True)[f"iso_{iso}"].agg(["min", "max"])
        n = nov.groupby(f"bin_{iso}", observed=True).size()
        tab = pd.DataFrame({m: nov.groupby(f"bin_{iso}", observed=True)[f"{m}|family"].mean() for m in METHODS})
        tab.insert(0, "range", [f"{a:.3g}-{b:.3g}" for a, b in zip(lo_hi["min"], lo_hi["max"])])
        tab.insert(0, "n_seqs", n)
        print("family correct (pooled over splits, for reading only; the tests below are per split):")
        print(tab.round(3).to_string())
        for a, b in PAIRS[:2]:
            for o in OUTCOMES:
                r = t[(t.isolation == iso) & (t.a == a) & (t.b == b) & (t.outcome == o)]
                cells = " | ".join(f"{x.bin.replace('trend: ', 'TREND ')} {x.diff:+.3f} [{x.ci_lo:+.3f}, {x.ci_hi:+.3f}] "
                                   f"p={x.wilcoxon_p:.3g}" for x in r.itertuples())
                tag = "  <- PRIMARY (trend)" if r.primary.any() else ""
                print(f"  {a} minus {b}, {o}{' (lower = better)' if o == 'tree_error' else ''}:{tag}\n    {cells}")


def figure(results):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed - figure skipped (pip install matplotlib)")
        return
    # palette: validated reference categorical slots 1-3 (light mode); legend + distinct markers,
    # so identity is never colour alone (lines meet at >= 60 My, so no end-of-line labels)
    style = {"alignment nearest": ("#2a78d6", "o", "alignment (no tree)"),
             "phylo": ("#eb6834", "s", "tree placement"),
             "raw+treecap90": ("#1baf7a", "^", "k-mers + tree model")}
    ink, ink2, grid, surface = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    names = {"teleo": "Teleo (short marker, used in eDNA)", "full": "Long 12S"}
    fig, axes = plt.subplots(1, len(results), figsize=(5.2 * len(results), 4.2), sharey=True,
                             facecolor=surface, squeeze=False)
    for ax, (marker, (nov, ps)) in zip(axes[0], results.items()):
        sub = ps[ps.isolation == "tree"]
        x = np.arange(len(TREE_LABELS))
        for m, (col, mk, lab) in style.items():
            g = sub[sub.method == m].groupby("bin", observed=True).family
            mean = g.mean().reindex(TREE_LABELS)
            se = (g.std() / np.sqrt(g.count())).reindex(TREE_LABELS)
            ax.errorbar(x, mean * 100, yerr=1.96 * se * 100, color=col, marker=mk, ms=7, lw=2,
                        capsize=3, elinewidth=1, label=lab, mec=surface, mew=1.5)
        n = nov.groupby("bin_tree", observed=True).size().reindex(TREE_LABELS)
        ax.set_xticks(x, [f"{lab}\n(n = {v})" for lab, v in zip(TREE_LABELS, n)], fontsize=9, color=ink2)
        ax.set_xlim(-0.3, len(x) - 0.3)
        ax.set_title(names[marker], fontsize=11, color=ink, loc="left")
        ax.set_xlabel("Age of split to the closest species in the library", fontsize=9, color=ink2)
        ax.set_facecolor(surface)
        ax.grid(axis="y", color=grid, lw=0.8)
        ax.set_axisbelow(True)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
        for s in ["left", "bottom"]:
            ax.spines[s].set_color(grid)
        ax.tick_params(colors=ink2, labelsize=9)
    axes[0][0].set_ylabel("New species placed in the correct family (%)", fontsize=9, color=ink2)
    axes[0][0].set_ylim(0, 100)
    axes[0][0].legend(frameon=False, fontsize=8.5, loc="lower left", labelcolor=ink2)
    fig.suptitle("Family placement of new species, by how far their closest relative in the library is",
                 fontsize=11.5, color=ink, x=0.01, ha="left")
    fig.text(0.01, 0.005, "Mean over 30 splits; bars = 95% interval over splits. Model methods averaged over 5 seeds.",
             fontsize=7.5, color=ink2)
    fig.tight_layout(rect=(0, 0.03, 0.97, 0.95))
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / "when_tree_helps.png", dpi=200, facecolor=surface)
    print(f"\nwrote {FIG / 'when_tree_helps.png'}")


def main():
    clean, splits, row, age = load()
    rng = np.random.default_rng(0)
    OUT.mkdir(parents=True, exist_ok=True)
    kw = dict(index=False, lineterminator="\n")
    results = {}
    for marker, mk in MARKERS.items():
        nov = per_sequence(mk, clean, splits, row, age)
        if nov is None:
            continue
        ps = per_split(nov)
        t = tests(ps, rng)
        report(marker, nov, t)
        nov.to_csv(OUT / f"when_tree_helps{mk}_per_seq.csv", **kw)
        ps.to_csv(OUT / f"when_tree_helps{mk}_per_split.csv", **kw)
        t.to_csv(OUT / f"when_tree_helps{mk}_tests.csv", **kw)
        print(f"\nwrote outputs/results/when_tree_helps{mk}_per_seq.csv, _per_split.csv, _tests.csv")
        results[marker] = (nov, ps)
    if results:
        figure(results)


if __name__ == "__main__":
    main()
