#!/usr/bin/env python3
"""
End-to-end pipeline: what a user actually gets when detection, placement and grouping run together.

Same 30 splits, ALL test sequences (known + novel). Three pipelines:
  A  forced        no detection: every sequence gets its nearest local species (B1 / Waggoner-style)
  B  current B2    VSEARCH, F1-tuned cut-off (outputs/results/group_cutoffs[_full]_calls.csv.gz,
                   method "global") -> novel calls placed on the LOCAL nearest species
                   -> novel calls grouped (calibrated gap grouping)
  C  proposed      conformal detection (models/conformal_detect.py, leave-one-haplotype-out
                   calibration), alpha 0.05 and 0.10 -> novel calls placed on LOCAL + GLOBAL
                   (models/global_library.py rule and cache) -> novel calls grouped (same grouping)
Nothing is re-tuned: every piece uses the rules already fixed in its own script.

Grouping (B and C alike): calibrated barcode gap (clustering/gap_grouping.calibrate / group) on
  D = 1 - VSEARCH identity (data/pairwise_identity_<marker>.npz), so both markers use the same
  distance; linkage + cut-off chosen on the reference by ARI, grid 0-0.15 in 0.0025 steps.
  (The confirmed Teleo grouping result used alignment identity; here the distance is the same for
  every pipeline, so only WHICH sequences are called novel differs between B and C.)

Outcomes per split
  known_correct      known sequences not called novel AND given their right species
  known_called_novel known sequences wrongly called novel
  novel_called       novel sequences called novel (1 - novel_named_known)
  novel_named_known  novel sequences given a known species name (the original problem)
  novel_family_e2e   novel sequences called novel AND placed in the right family (missed = wrong)
  novel_genus_e2e    same for genus
  family_if_called   family right among novel sequences that were called novel
  balanced_correct   (known_correct + novel_called) / 2
  ARI                grouping of the novel-called sequences vs true species (noise = singletons)
  novel_count_error  |number of groups - true number of novel species in the test set|
                     (forced pipeline: estimate 0 new species)
Comparisons (paired over splits, Wilcoxon + bootstrap 95% CI), every outcome:
  C(alpha 0.10) vs B, C(alpha 0.10) vs A, C(alpha 0.05) vs B.
  No single pass/fail rule: C vs B is EXPECTED to be a trade-off (fewer false alarms and better
  placement of what is caught, fewer novel caught). The table reports it as such.

Usage (repo root):  python models/end_to_end.py                       # Teleo
                    python models/end_to_end.py --marker=full
                    [--global-tag=_cov90]   use the coverage-filtered global cache instead (exploratory)
Needs: group_cutoffs[_full]_calls.csv.gz, global_library_hits[_full][<tag>].csv (run global_library.py)
Output: outputs/results/end_to_end[_full][<tag>]_per_split.csv and _tests.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).resolve().parents[1]
DATA, OUT = ROOT / "data", ROOT / "outputs" / "results"
for d in ["baselines", "models", "clustering"]:
    sys.path.insert(0, str(ROOT / d))
from conformal_detect import conformal_p, loo_calibration  # noqa: E402
from gap_grouping import T_ALIGN, calibrate, group  # noqa: E402
from global_library import taxonomy_maps  # noqa: E402
from group_cutoffs import Context  # noqa: E402

PIPELINES = ["A forced", "B current (B2)", "C proposed a=0.05", "C proposed a=0.10"]
OUTCOMES = ["known_correct", "known_called_novel", "novel_called", "novel_named_known", "novel_family_e2e",
            "novel_genus_e2e", "family_if_called", "balanced_correct", "ARI", "novel_count_error"]
LOWER_BETTER = {"known_called_novel", "novel_named_known", "novel_count_error"}
COMPARISONS = [("C proposed a=0.10", "B current (B2)"), ("C proposed a=0.10", "A forced"),
               ("C proposed a=0.05", "B current (B2)")]


def grouping(ctx, ref, called_ids):
    """Calibrated gap grouping of the called sequences on 1 - VSEARCH identity."""
    if len(called_ids) == 0:
        return np.array([], int)
    ri, ci = ctx.pos[ref.index.values].values, ctx.pos[called_ids].values
    Dref = 1 - ctx.ID[np.ix_(ri, ri)]
    Dref = (Dref + Dref.T) / 2
    np.fill_diagonal(Dref, 0)
    method, t, _ = calibrate(Dref, ref.species_name.values, T_ALIGN)
    Dn = 1 - ctx.ID[np.ix_(ci, ci)]
    Dn = (Dn + Dn.T) / 2
    np.fill_diagonal(Dn, 0)
    return group(Dn, method, t)


def score(name, test, called, s_place, tax, labels):
    nov = (test.role == "test_novel").to_numpy(bool)
    known = ~nov
    ts, tg, tf = (test[c].to_numpy(object) for c in ["species_name", "Genus", "Family"])
    pg = tax.Genus.reindex(s_place).to_numpy(object)
    pf = tax.Family.reindex(s_place).to_numpy(object)
    fam_ok, gen_ok = (pf == tf), (pg == tg)
    n_true_novel_sp = test.species_name[nov].nunique()
    r = dict(pipeline=name,
             known_correct=float((~called[known] & (s_place[known] == ts[known])).mean()),
             known_called_novel=float(called[known].mean()),
             novel_called=float(called[nov].mean()),
             novel_named_known=float(1 - called[nov].mean()),
             novel_family_e2e=float((called & fam_ok)[nov].mean()),
             novel_genus_e2e=float((called & gen_ok)[nov].mean()),
             family_if_called=float(fam_ok[nov & called].mean()) if (nov & called).any() else np.nan)
    r["balanced_correct"] = (r["known_correct"] + r["novel_called"]) / 2
    if labels is None or len(labels) == 0:
        r["ARI"], r["novel_count_error"], r["n_groups"] = np.nan, float(n_true_novel_sp), 0
    else:
        r["ARI"] = adjusted_rand_score(ts[called], labels)
        r["n_groups"] = len(np.unique(labels))
        r["novel_count_error"] = float(abs(r["n_groups"] - n_true_novel_sp))
    r["n_true_novel_species"] = n_true_novel_sp
    return r


def main():
    opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    marker = "teleo" if opts.get("marker", "teleo") == "teleo" else "full"
    suffix = "" if marker == "teleo" else "_full"
    gtag = opts.get("global-tag", "")
    ctx = Context(marker)
    seqs = ctx.seqs.copy()

    calls = pd.read_csv(OUT / f"group_cutoffs{suffix}_calls.csv.gz", dtype={"seq_id": str})
    calls = calls[calls.method == "global"].set_index(["split_id", "seq_id"])
    gh_file = OUT / f"global_library_hits{suffix}{gtag}.csv"
    if not gh_file.exists():
        sys.exit(f"missing {gh_file.name}: run models/global_library.py first")
    gh = pd.read_csv(gh_file, dtype={"seq_id": str}).set_index("seq_id")
    fam_of_genus = taxonomy_maps(seqs)
    sp_info = seqs.drop_duplicates("species_name").set_index("species_name")[["Genus", "Family"]]
    gsp = pd.Series(gh.best_species.dropna().unique())
    gtax = pd.DataFrame({"Genus": gsp.str.split().str[0].values}, index=gsp.values)
    gtax["Family"] = gtax.Genus.map(fam_of_genus)
    tax = pd.concat([sp_info, gtax[~gtax.index.isin(sp_info.index)]])

    splits = pd.concat([pd.read_csv(DATA / f, dtype={"seq_id": str, "difficulty": str})
                        for f in ["splits.csv", "splits_extra.csv"]])
    rows = []
    for sid in sorted(splits.split_id.unique()):
        s = splits[splits.split_id == sid].set_index("seq_id")
        m = seqs[["species_name", "Genus", "Family", "teleo"]].join(s[["role", "difficulty"]], how="inner")
        ref, test = m[m.role == "reference"].copy(), m[m.role != "reference"].copy()
        d_loc, s_loc = ctx.best_hits(test.index.values, ref.index.values)
        gid = gh.best_id.reindex(test.index).fillna(0).to_numpy(float)
        gs = gh.best_species.reindex(test.index).to_numpy(object)
        s_glob = np.where(gid > (1 - d_loc) + 1e-9, gs, s_loc)

        b2 = calls.loc[[(sid, q) for q in test.index]]
        if not (b2.nearest_species.values == s_loc).all():
            print(f"  note split {sid}: {(b2.nearest_species.values != s_loc).sum()} B2 nearest species differ "
                  f"from the matrix best hit (tie handling); B2 placement uses the matrix best hit")
        b2_called = b2.called_novel.to_numpy(int).astype(bool)
        cal, w = loo_calibration(ctx, ref)
        p = conformal_p(cal, w, d_loc)

        pipes = {"A forced": (np.zeros(len(test), bool), s_loc, False),
                 "B current (B2)": (b2_called, s_loc, True),
                 "C proposed a=0.05": (p <= 0.05, s_glob, True),
                 "C proposed a=0.10": (p <= 0.10, s_glob, True)}
        for name, (called, place, do_group) in pipes.items():
            # known-called sequences are always named with the LOCAL nearest species
            s_out = np.where(called, place, s_loc)
            labels = grouping(ctx, ref, test.index.values[called]) if do_group else None
            r = score(name, test, called, s_out, tax, labels)
            r["split"] = sid
            rows.append(r)
        x = {r["pipeline"]: r for r in rows if r["split"] == sid}
        print(f"[{marker}] split {sid:2d} | balanced: " + " | ".join(
            f"{k.split()[0]}{'' if 'a=' not in k else k[-5:]} {v['balanced_correct']:.3f}" for k, v in x.items()),
            flush=True)

    per = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    per.round(5).to_csv(OUT / f"end_to_end{suffix}{gtag}_per_split.csv", index=False)

    rng = np.random.default_rng(0)
    P = {k: g.set_index("split") for k, g in per.groupby("pipeline")}
    tests = []
    for a, b in COMPARISONS:
        for o in OUTCOMES:
            d = (P[a][o] - P[b][o]).dropna()
            if len(d) < 2:
                continue
            boots = [rng.choice(d.values, len(d)).mean() for _ in range(5000)]
            pv = wilcoxon(d.values).pvalue if (d != 0).sum() > 1 else 1.0
            sign = -1 if o in LOWER_BETTER else 1
            tests.append(dict(marker=marker, a=a, b=b, outcome=o, mean_a=P[a][o].loc[d.index].mean(),
                              mean_b=P[b][o].loc[d.index].mean(), diff=d.mean(), ci_lo=np.percentile(boots, 2.5),
                              ci_hi=np.percentile(boots, 97.5), a_better=int((sign * d > 0).sum()),
                              b_better=int((sign * d < 0).sum()), n=len(d), wilcoxon_p=pv))
    tests = pd.DataFrame(tests)
    tests.to_csv(OUT / f"end_to_end{suffix}{gtag}_tests.csv", index=False)

    print(f"\n[{marker}] HEADLINE TABLE (mean over {per.split.nunique()} splits)")
    print(per.groupby("pipeline")[OUTCOMES + ["n_groups", "n_true_novel_species"]].mean()
          .loc[PIPELINES].T.round(3).to_string())
    print(f"\n[{marker}] paired comparisons (diff = a - b)")
    print(tests[["a", "b", "outcome", "mean_a", "mean_b", "diff", "ci_lo", "ci_hi", "wilcoxon_p"]]
          .round(4).to_string(index=False))
    print(f"\nwrote outputs/results/end_to_end{suffix}{gtag}_per_split.csv and _tests.csv")


if __name__ == "__main__":
    main()
