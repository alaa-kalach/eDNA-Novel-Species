#!/usr/bin/env python3
"""
Conformal novelty detection: the B2 / VSEARCH score with a cut-off that CONTROLS how often a
known species is wrongly called novel, instead of a cut-off tuned for F1.

Score (unchanged from B2):  d_q = 1 - best identity of query q to the reference
                            (data/pairwise_identity_<marker>.npz, same tie rule as B2).
Only the cut-off changes. Null hypothesis for a query: "it belongs to a species in the reference".
  calibration scores c_i = scores of KNOWN-species sequences held out exactly like test_known
                           (one distinct Teleo sequence, all its copies, of a species that keeps
                           at least one other distinct sequence in the reference)
  conformal p-value       p_q = (1 + sum_i w_i [c_i >= d_q]) / (1 + sum_i w_i)
  call novel              p_q <= alpha
  weights                 each calibration SPECIES counts once in total (w = 1 / its held-out rows),
                          so species with many haplotypes do not dominate; the test point counts 1.
Guarantee (split-conformal, under exchangeability of the held-out known species with the test known
species): expected share of known sequences called novel <= alpha. The test below checks it.

RULES, FIXED BEFORE RUNNING
  Calibration (MAIN = loo): within each split's reference only, every species with >= 2 distinct
    Teleo sequences; each distinct sequence in turn is held out (all copies) and scored against the
    rest of the reference. No test sequence is ever used.
  Calibration (secondary = pseudo): the known queries of the 5 pseudo-splits B2 already uses
    (baselines/common.py pseudo_split, same seeds), scored against their pseudo-reference.
  alpha = 0.01, 0.05, 0.10.  PRIMARY alpha = 0.05.
  Outcomes per split (as models/group_cutoffs.py): f1, novel_recall, close_recall, known_acc,
    known_fp; plus the identity cut-off implied (lowest score called novel).
  Comparison method: B2 as already run ("global" in group_cutoffs[_full]_per_split.csv, F1-tuned
    on the pseudo-splits); the test sets are checked to be identical.
  PRIMARY (per marker), both required:
    (1) validity   loo, alpha 0.05: mean known_fp over the 30 splits; NOT rejected if the bootstrap
                   95% CI lower bound <= 0.05 (the same check is reported for B2, for contrast)
    (2) control    loo alpha 0.05 known_fp < B2 known_fp: paired Wilcoxon over splits, 95% CI of the
                   difference below 0, Holm over the two markers' p-values (holm_p filled in when
                   both markers have run; until then p is reported raw)
  Secondary (reported, not used to decide): every (calibration, alpha) vs B2 on every outcome; the
    COST of control = novel_recall and close_recall lost vs B2; share of splits with known_fp > alpha.

Usage (repo root):  python models/conformal_detect.py                 # Teleo
                    python models/conformal_detect.py --marker=full   # long 12S
Needs:  data/pairwise_identity_<marker>.npz, outputs/results/group_cutoffs[_full]_per_split.csv
Output (outputs/results/):
  conformal_detect[_full]_per_split.csv   split x calibration x alpha
  conformal_detect[_full]_validity.csv    mean known_fp + CI per (calibration, alpha), and B2
  conformal_detect[_full]_tests.csv       paired comparisons vs B2 (+ holm_p across markers)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
DATA, OUT = ROOT / "data", ROOT / "outputs" / "results"
sys.path.insert(0, str(ROOT / "baselines"))
sys.path.insert(0, str(ROOT / "models"))
from common import PSEUDO_REPEATS, pseudo_rng, pseudo_split  # noqa: E402
from group_cutoffs import CLOSE_DIFFS, Context, holm  # noqa: E402

ALPHAS = [0.01, 0.05, 0.10]
PRIMARY_ALPHA = 0.05
CALIBRATIONS = ["loo", "pseudo"]
OUTCOMES = ["f1", "novel_recall", "close_recall", "known_acc", "known_fp"]


# ------------------------------------------------------------------ calibration
def loo_calibration(ctx, ref):
    ri = ctx.pos[ref.index.values].values
    M = ctx.ID[np.ix_(ri, ri)]
    sp, tel = ref.species_name.to_numpy(dtype=object), ref.teleo.to_numpy(dtype=object)
    scores, weights = [], []
    for s in np.unique(sp):
        idx = np.where(sp == s)[0]
        u = np.unique(tel[idx])
        if len(u) < 2:
            continue
        sc = []
        for t in u:
            held = idx[tel[idx] == t]
            keep = np.ones(len(ri), bool)
            keep[held] = False
            sc.append(1 - M[np.ix_(held, np.where(keep)[0])].max(1))
        sc = np.concatenate(sc)
        scores.append(sc)
        weights.append(np.full(len(sc), 1 / len(sc)))
    return np.concatenate(scores), np.concatenate(weights)


def pseudo_calibration(ctx, ref, sid):
    scores, weights = [], []
    for r in range(PSEUDO_REPEATS):
        pref, pq = pseudo_split(ref, pseudo_rng(sid, r))
        kn = pq[~pq.is_novel]
        if len(kn) == 0:
            continue
        d, _ = ctx.best_hits(kn.index.values, pref.index.values)
        cnt = kn.species_name.map(kn.species_name.value_counts()).to_numpy(float)
        scores.append(d)
        weights.append(1 / cnt / PSEUDO_REPEATS)       # each species once per repeat, repeats averaged
    return np.concatenate(scores), np.concatenate(weights)


def conformal_p(cal, w, s):
    """p = (1 + sum w [cal >= s]) / (1 + sum w), vectorised over the queries s."""
    o = np.argsort(cal, kind="stable")
    c, ww = cal[o], w[o]
    tail = np.r_[np.cumsum(ww[::-1])[::-1], 0.0]       # tail[i] = sum of w for c[i:], tail[n] = 0
    i = np.searchsorted(c, s - 1e-12, side="left")     # first calibration score >= s
    return (1 + tail[i]) / (1 + w.sum())


# ------------------------------------------------------------------ one split
def run_split(ctx, ref, test, sid):
    ref_ids = ref.index.values
    y = (test.role == "test_novel").to_numpy(bool)
    d, near = ctx.best_hits(test.index.values, ref_ids)
    known = ~y
    true_sp = test.species_name.to_numpy(dtype=object)
    ti, ri = ctx.pos[test.index.values].values, ctx.pos[ref_ids].values
    close = y & (ctx.n_diff_teleo[np.ix_(ti, ri)].min(1) <= CLOSE_DIFFS + 1e-6)
    cals = {"loo": loo_calibration(ctx, ref), "pseudo": pseudo_calibration(ctx, ref, sid)}
    rows = []
    for cname, (cal, w) in cals.items():
        p = conformal_p(cal, w, d)
        for a in ALPHAS:
            call = p <= a
            rows.append(dict(split=sid, calibration=cname, alpha=a, n_test=len(y), n_novel=int(y.sum()),
                             n_close=int(close.sum()), n_calib=len(cal), n_calib_species=round(float(w.sum()), 2),
                             f1=2 * (call & y).sum() / max(call.sum() + y.sum(), 1),
                             novel_recall=call[y].mean(),
                             close_recall=call[close].mean() if close.any() else np.nan,
                             known_acc=(~call[known] & (near[known] == true_sp[known])).mean(),
                             known_fp=call[known].mean(),
                             identity_cutoff=1 - d[call].min() if call.any() else np.nan))
    return rows


# ------------------------------------------------------------------ statistics
def boot_ci(x, rng):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    b = [rng.choice(x, len(x)).mean() for _ in range(5000)]
    return x.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)


def main():
    opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    marker = "teleo" if opts.get("marker", "teleo") == "teleo" else "full"
    suffix = "" if marker == "teleo" else "_full"
    ctx = Context(marker)
    splits = pd.concat([pd.read_csv(DATA / f, dtype={"seq_id": str, "difficulty": str})
                        for f in ["splits.csv", "splits_extra.csv"]])
    rows = []
    for sid in sorted(splits.split_id.unique()):
        s = splits[splits.split_id == sid].set_index("seq_id")
        m = ctx.seqs[["species_name", "Genus", "teleo"]].join(s[["role", "difficulty"]], how="inner")
        ref, test = m[m.role == "reference"].copy(), m[m.role != "reference"].copy()
        r = run_split(ctx, ref, test, sid)
        rows += r
        main_row = [x for x in r if x["calibration"] == "loo" and x["alpha"] == PRIMARY_ALPHA][0]
        print(f"[{marker}] split {sid:2d} | loo a=0.05: known_fp {main_row['known_fp']:.3f} "
              f"novel_recall {main_row['novel_recall']:.3f} cut-off {main_row['identity_cutoff']:.4f} "
              f"| {main_row['n_calib_species']:.0f} calibration species", flush=True)
    per = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    per.round(5).to_csv(OUT / f"conformal_detect{suffix}_per_split.csv", index=False)

    b2 = pd.read_csv(OUT / f"group_cutoffs{suffix}_per_split.csv")
    b2 = b2[b2.method == "global"].set_index("split")
    chk = per[(per.calibration == "loo") & (per.alpha == PRIMARY_ALPHA)].set_index("split")
    for c in ["n_test", "n_novel", "n_close"]:
        if not chk[c].equals(b2.loc[chk.index, c]):
            sys.exit(f"{c} differs from group_cutoffs{suffix}_per_split.csv - not the same test sets")

    rng = np.random.default_rng(0)
    val = []
    for (cname, a), g in per.groupby(["calibration", "alpha"]):
        mu, lo, hi = boot_ci(g.known_fp, rng)
        val.append(dict(method=f"conformal {cname}", alpha=a, mean_known_fp=mu, ci_lo=lo, ci_hi=hi,
                        splits_over_alpha=int((g.known_fp > a).sum()), valid=bool(lo <= a),
                        primary=bool(cname == "loo" and a == PRIMARY_ALPHA)))
    mu, lo, hi = boot_ci(b2.known_fp, rng)
    for a in ALPHAS:
        val.append(dict(method="B2 (F1-tuned)", alpha=a, mean_known_fp=mu, ci_lo=lo, ci_hi=hi,
                        splits_over_alpha=int((b2.known_fp > a).sum()), valid=bool(lo <= a), primary=False))
    val = pd.DataFrame(val)
    val.to_csv(OUT / f"conformal_detect{suffix}_validity.csv", index=False)

    tests = []
    for (cname, a), g in per.groupby(["calibration", "alpha"]):
        A = g.set_index("split")
        for o in OUTCOMES:
            d = (A[o] - b2.loc[A.index, o]).dropna()
            boots = [rng.choice(d.values, len(d)).mean() for _ in range(5000)]
            p = wilcoxon(d.values).pvalue if (d != 0).sum() > 1 else 1.0
            tests.append(dict(marker=marker, calibration=cname, alpha=a, outcome=o,
                              primary=bool(cname == "loo" and a == PRIMARY_ALPHA and o == "known_fp"),
                              mean_conformal=A[o].mean(), mean_b2=b2.loc[A.index, o].mean(), diff=d.mean(),
                              ci_lo=np.percentile(boots, 2.5), ci_hi=np.percentile(boots, 97.5),
                              conformal_better=int(((d < 0) if o == "known_fp" else (d > 0)).sum()),
                              b2_better=int(((d > 0) if o == "known_fp" else (d < 0)).sum()),
                              n=len(d), wilcoxon_p=p))
    tests = pd.DataFrame(tests)
    tests.to_csv(OUT / f"conformal_detect{suffix}_tests.csv", index=False)

    # Holm for the primary control test across the two markers, once both have run
    files = [OUT / "conformal_detect_tests.csv", OUT / "conformal_detect_full_tests.csv"]
    if all(f.exists() for f in files):
        both = [pd.read_csv(f) for f in files]
        ps = [t.loc[t.primary, "wilcoxon_p"].iloc[0] for t in both]
        adj = holm(ps)
        for f, t, h in zip(files, both, adj):
            t["holm_p"] = np.where(t.primary, h, np.nan)
            t.to_csv(f, index=False)
        tests = pd.read_csv(OUT / f"conformal_detect{suffix}_tests.csv")

    print(f"\n[{marker}] VALIDITY: share of known sequences called novel (mean over splits, 95% CI)")
    print(val.round(4).to_string(index=False))
    print(f"\n[{marker}] conformal vs B2 (diff = conformal - B2)")
    cols = ["calibration", "alpha", "outcome", "mean_conformal", "mean_b2", "diff", "ci_lo", "ci_hi", "wilcoxon_p"]
    cols += ["holm_p"] if "holm_p" in tests else []
    print(tests[cols].round(4).to_string(index=False))

    v = val[val.primary].iloc[0]
    t = tests[tests.primary].iloc[0]
    p_dec = t["holm_p"] if "holm_p" in tests and not pd.isna(t.get("holm_p", np.nan)) else t["wilcoxon_p"]
    ok = bool(v.valid) and t.ci_hi < 0 and p_dec < 0.05
    note = "" if "holm_p" in tests else " (raw p; Holm across markers after both runs)"
    print(f"\n[{marker}] PRIMARY: validity {'held' if v.valid else 'REJECTED'} "
          f"(mean known_fp {v.mean_known_fp:.3f}, CI {v.ci_lo:.3f}-{v.ci_hi:.3f}); control vs B2 "
          f"diff {t['diff']:+.3f} (CI {t.ci_lo:+.3f} to {t.ci_hi:+.3f}, p {p_dec:.2g}){note} "
          f"-> {'CONFIRMED' if ok else 'NOT CONFIRMED'}")


if __name__ == "__main__":
    main()
