#!/usr/bin/env python3
"""
Hybrid placement: tree placement by default, switch to a model method when an OBSERVABLE signal
says the tree placement is unreliable. Follows models/when_tree_helps.py, which found that the
tree's gain depends on how divergent the new sequence is.

Feasibility (checked before writing, on the existing per-sequence files): even the best rule
chosen IN HINDSIGHT only lifts Teleo family placement from 0.748 to ~0.762. Expect a small gain.

RULES, FIXED BEFORE RUNNING
  Sequences: truly novel test sequences (role == test_novel), all 30 splits (as in every
    placement analysis; in the full pipeline the same rule is applied to the VSEARCH-novel calls).
  Default method: phylo (placement on the Fish Tree, models/phylo_placement.py).
  Switch-to candidates: raw+taxonomy, raw+treecap90 (w = 1, models/placement_kmer_tree.py;
    seed-averaged: a sequence's score is the mean over the 5 seeds of the switched-to method).
  Signals (all observable for a new sequence, high = less trust in the tree placement):
    divergence   1 - identity to the nearest reference sequence (align_novelty)
    pendant      length of the new branch EPA-ng adds for the sequence
    uncertainty  1 - LWR of the chosen placement
  Rule = (switch-to method, signal, cut-off t): use the switch-to method when signal >= t,
    else phylo. Cut-offs = 50%, 51%, ..., 99% quantiles of the signal on the TUNING sequences,
    plus "never switch" (always phylo).
  Tuning, leave-one-split-out: for test split k, the rule is chosen on the novel sequences of all
    OTHER splits, excluding every sequence whose species is novel in split k (no test species is
    ever used to choose its own rule). Criterion = mean family correct, averaged per split then
    over splits (the same way the outcome is scored). Ties -> never switch, then the earlier
    candidate in the order above, then the higher (more conservative) cut-off.
  Outcomes, per split (mean over its novel sequences): family (PRIMARY), family_easy (no same-
    genus species in the reference), genus_hard (one is), tree_error (My, lower = better).
  PRIMARY TEST (one per marker): hybrid minus phylo, family, paired Wilcoxon over the 30 splits
    + bootstrap 95% CI. Secondary: hybrid vs raw+taxonomy, raw+treecap90, alignment nearest, on
    every outcome. Also reported: which rule each split chose, and the hindsight ceiling (the
    best single rule chosen on ALL data, NOT a result, only to show how much room there was).

Usage (repo root):  python models/hybrid_placement.py        # Teleo and long 12S if present
Needs:  outputs/results/phylo_placement[_full]_per_seq.csv   (models/phylo_placement.py)
        outputs/placement_kmer_tree[_full]_per_seq.csv       (models/placement_kmer_tree.py)
        data/global_tree_distances.npz                       (models/tree_distances.py)
Output: outputs/results/hybrid_placement[_full]_per_split.csv (split x method x outcome)
        outputs/results/hybrid_placement[_full]_rules.csv     (rule chosen for each split)
        outputs/results/hybrid_placement[_full]_tests.csv     (paired comparisons)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
DATA, OUT = ROOT / "data", ROOT / "outputs" / "results"
MARKERS = {"teleo": "", "full": "_full"}
DEFAULT = "phylo"
SWITCH_TO = ["raw+taxonomy", "raw+treecap90"]
SIGNALS = ["divergence", "pendant", "uncertainty"]
QUANTILES = np.round(np.arange(0.50, 0.995, 0.01), 2)
KT_METHODS = ["raw", "raw+taxonomy", "raw+treecap90"]
METHODS = ["phylo", "alignment nearest"] + KT_METHODS
OUTCOMES = ["family", "family_easy", "genus_hard", "tree_error"]   # tree_error: lower = better
COMPARE = ["phylo", "raw+taxonomy", "raw+treecap90", "alignment nearest"]


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
    return clean, splits, row, z["D"] * float(z["scale"]) / 2     # split age in million years


def per_sequence(mk, clean, splits, row, age):
    """One row per (split, novel sequence): signals + each method's family / genus / tree error."""
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

    nov = splits[splits.role == "test_novel"][["split_id", "seq_id", "difficulty"]].reset_index(drop=True)
    nov["true"] = clean.loc[nov.seq_id, "species_name"].values
    key = list(zip(nov.split_id, nov.seq_id))
    nov["divergence"] = ph.loc[key, "align_novelty"].values
    nov["pendant"] = ph.loc[key, "pendant"].values
    nov["uncertainty"] = 1 - ph.loc[key, "lwr"].values

    def score(guess, true):
        g, t = info.loc[guess], info.loc[true]
        return ((g.Family.values == t.Family.values).astype(float),
                (g.Genus.values == t.Genus.values).astype(float),
                age[[row[x] for x in guess], [row[x] for x in true]])

    for m, col in [("phylo", "phylo_species"), ("alignment nearest", "align_species")]:
        nov[f"{m}|family"], nov[f"{m}|genus"], nov[f"{m}|tree_error"] = score(ph.loc[key, col].values,
                                                                              nov.true.values)
    kt = kt.merge(nov[["split_id", "seq_id", "true"]], on=["split_id", "seq_id"])
    kt["family"], kt["genus"], kt["tree_error"] = score(kt.guess.values, kt.true.values)
    kt = kt.groupby(["method", "split_id", "seq_id"])[["family", "genus", "tree_error"]].mean()
    for m in KT_METHODS:
        got = kt.loc[m].reindex(key)
        if got.family.isna().any():
            sys.exit(f"{kf.name}: {m} has no guess for some novel sequences - rerun placement_kmer_tree.py")
        for o in ["family", "genus", "tree_error"]:
            nov[f"{m}|{o}"] = got[o].values
    return nov


def split_mean_family(df, values):
    """Family correct averaged per split, then over splits (the tuning criterion)."""
    return pd.Series(values, index=df.index).groupby(df.split_id.values).mean().mean()


def choose_rule(tune):
    """Best (switch_to, signal, cut-off) on the tuning sequences; ties keep the earlier option."""
    base = tune[f"{DEFAULT}|family"].values
    best = dict(switch_to="never", signal="-", quantile=np.nan, cutoff=np.inf,
                score=split_mean_family(tune, base))
    for alt in SWITCH_TO:
        alt_v = tune[f"{alt}|family"].values
        for sig in SIGNALS:
            s = tune[sig].values
            for q in QUANTILES[::-1]:                    # higher cut-off first (tie -> conservative)
                t = np.quantile(s, q)
                sc = split_mean_family(tune, np.where(s >= t, alt_v, base))
                if sc > best["score"] + 1e-12:
                    best = dict(switch_to=alt, signal=sig, quantile=q, cutoff=t, score=sc)
    return best


def apply_rule(df, rule):
    """Outcome columns of the hybrid for the rows of df."""
    use = (df[rule["signal"]].values >= rule["cutoff"]) if rule["switch_to"] != "never" \
        else np.zeros(len(df), bool)
    out = {}
    for o in ["family", "genus", "tree_error"]:
        out[o] = df[f"{DEFAULT}|{o}"].values if rule["switch_to"] == "never" \
            else np.where(use, df[f"{rule['switch_to']}|{o}"].values, df[f"{DEFAULT}|{o}"].values)
    return out, use


def split_outcomes(g, fam, gen, err):
    easy, hard = (g.difficulty == "easy").values, (g.difficulty == "hard").values
    m = lambda x: x.mean() if len(x) else np.nan  # noqa: E731
    return dict(family=m(fam), family_easy=m(fam[easy]), genus_hard=m(gen[hard]), tree_error=m(err))


def run(nov):
    rules, rows = [], []
    for k in sorted(nov.split_id.unique()):
        test = nov[nov.split_id == k]
        tune = nov[(nov.split_id != k) & ~nov.true.isin(set(test.true))]
        rule = choose_rule(tune)
        out, use = apply_rule(test, rule)
        rules.append(dict(split=k, **{x: rule[x] for x in ["switch_to", "signal", "quantile", "cutoff"]},
                          tuning_family=rule["score"], n_tuning=len(tune), n_test=len(test),
                          share_switched=use.mean()))
        rows.append(dict(split=k, method="hybrid", **split_outcomes(test, out["family"], out["genus"],
                                                                     out["tree_error"])))
        for m in METHODS:
            rows.append(dict(split=k, method=m, **split_outcomes(test, test[f"{m}|family"].values,
                                                                 test[f"{m}|genus"].values,
                                                                 test[f"{m}|tree_error"].values)))
    return pd.DataFrame(rules), pd.DataFrame(rows)


def tests(ps, rng):
    out = []
    A = ps[ps.method == "hybrid"].set_index("split")
    for b in COMPARE:
        B = ps[ps.method == b].set_index("split")
        for o in OUTCOMES:
            d = (A[o] - B[o]).dropna()
            boots = [rng.choice(d.values, len(d)).mean() for _ in range(5000)]
            lo, hi = np.percentile(boots, [2.5, 97.5])
            p = wilcoxon(d.values).pvalue if (d != 0).any() else 1.0
            better = (d < 0) if o == "tree_error" else (d > 0)
            worse = (d > 0) if o == "tree_error" else (d < 0)
            out.append(dict(a="hybrid", b=b, outcome=o, primary=(b == DEFAULT and o == "family"),
                            mean_a=A[o].mean(), mean_b=B[o].mean(), diff=d.mean(), ci_lo=lo, ci_hi=hi,
                            a_better=int(better.sum()), b_better=int(worse.sum()), n=len(d), wilcoxon_p=p))
    return pd.DataFrame(out)


def hindsight(nov):
    """Best single rule chosen on ALL sequences (test included) = the ceiling. NOT a result."""
    r = choose_rule(nov)
    return r


def report(name, nov, rules, ps, t, ceiling):
    print(f"\n=== {name}: {len(nov)} novel sequences, {nov.split_id.nunique()} splits")
    print("\nrule chosen per split (tuned on the other splits, test species left out):")
    rc = rules.groupby(["switch_to", "signal"]).agg(splits=("split", "size"),
                                                    median_quantile=("quantile", "median"),
                                                    mean_share_switched=("share_switched", "mean"))
    print(rc.round(3).to_string())
    print("\nmean over splits:")
    tab = ps.groupby("method")[OUTCOMES].mean().loc[["hybrid"] + METHODS]
    print(tab.round(3).to_string())
    print(f"\nhindsight ceiling (rule chosen on ALL data, not a result): {ceiling['switch_to']} when "
          f"{ceiling['signal']} >= {ceiling['cutoff']:.3g} -> family {ceiling['score']:.3f}")
    print("\npaired comparisons (Wilcoxon over splits; diff = hybrid - other; tree_error lower = better)")
    for r in t.itertuples():
        tag = "  <- PRIMARY" if r.primary else ""
        print(f"  hybrid vs {r.b:18s} {r.outcome:11s} {r.mean_a:7.3f} vs {r.mean_b:7.3f} diff {r.diff:+.3f} "
              f"[{r.ci_lo:+.3f}, {r.ci_hi:+.3f}] better/worse {r.a_better:2d}/{r.b_better:2d} "
              f"p = {r.wilcoxon_p:.3g}{tag}")


def main():
    clean, splits, row, age = load()
    rng = np.random.default_rng(0)
    OUT.mkdir(parents=True, exist_ok=True)
    kw = dict(index=False, lineterminator="\n")
    for marker, mk in MARKERS.items():
        nov = per_sequence(mk, clean, splits, row, age)
        if nov is None:
            continue
        rules, ps = run(nov)
        t = tests(ps, rng)
        report(marker, nov, rules, ps, t, hindsight(nov))
        rules.to_csv(OUT / f"hybrid_placement{mk}_rules.csv", **kw)
        ps.to_csv(OUT / f"hybrid_placement{mk}_per_split.csv", **kw)
        t.to_csv(OUT / f"hybrid_placement{mk}_tests.csv", **kw)
        print(f"\nwrote outputs/results/hybrid_placement{mk}_rules.csv, _per_split.csv, _tests.csv")


if __name__ == "__main__":
    main()
