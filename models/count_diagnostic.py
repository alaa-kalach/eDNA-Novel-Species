#!/usr/bin/env python3
"""
Novel-species COUNT diagnostic for the end-to-end pipelines (models/end_to_end.py).

Why: at alpha 0.10 the proposed pipeline's estimated number of new species is close to the truth
(error 4.7 vs 13.2 for B2, Teleo), but at alpha 0.05 it is not (13.6). That pattern suggests two
errors cancelling. This splits the count error into its parts, which add up EXACTLY:

  n_groups - n_true_novel_species = (fp_groups + oversplit) - (missed + lumped)
    fp_groups  groups that contain ONLY known sequences (false alarms)              -> count too high
    oversplit  sum over caught novel species of (groups it is spread over - 1)      -> count too high
    missed     novel species with NO sequence called novel                          -> count too low
    lumped     caught novel species that share a group with another novel species
               (= caught + oversplit - groups containing >= 1 novel sequence)       -> count too low
  gross_error = fp_groups + oversplit + missed + lumped   (no cancellation possible)

Same calls, placement-free, same grouping as models/end_to_end.py (imported, nothing re-tuned):
B = B2 F1-tuned calls; C = conformal (leave-one-haplotype-out) at alpha 0.05 and 0.10;
calibrated gap grouping on 1 - VSEARCH identity. Also reported: mixed_groups (known + novel members).

RULE, FIXED BEFORE RUNNING
  The claim "the proposed pipeline estimates the number of new species better" is SUPPORTED only if
  gross_error(C, alpha 0.10) < gross_error(B): paired Wilcoxon over the 30 splits, bootstrap 95% CI
  of the difference below 0, p < 0.05 - on Teleo (long 12S = replication). Otherwise the smaller net
  count error is reported as cancelling errors, not as a result.

Usage (repo root):  python models/count_diagnostic.py                 # Teleo
                    python models/count_diagnostic.py --marker=full
Needs: outputs/results/group_cutoffs[_full]_calls.csv.gz
Output: outputs/results/count_diagnostic[_full]_per_split.csv and _tests.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
DATA, OUT = ROOT / "data", ROOT / "outputs" / "results"
for d in ["baselines", "models", "clustering"]:
    sys.path.insert(0, str(ROOT / d))
from conformal_detect import conformal_p, loo_calibration  # noqa: E402
from end_to_end import grouping  # noqa: E402
from group_cutoffs import Context  # noqa: E402

PIPELINES = ["B current (B2)", "C proposed a=0.05", "C proposed a=0.10"]
PARTS = ["n_groups", "n_true_novel_species", "net_error", "fp_groups", "oversplit", "missed", "lumped",
         "gross_error", "mixed_groups", "caught_species"]


def decompose(test, called, labels):
    nov = (test.role == "test_novel").to_numpy(bool)
    sp = test.species_name.to_numpy(object)
    n_true = len(set(sp[nov]))
    if len(labels) == 0:
        return dict(n_groups=0, n_true_novel_species=n_true, net_error=-n_true, fp_groups=0, oversplit=0,
                    missed=n_true, lumped=0, gross_error=n_true, mixed_groups=0, caught_species=0)
    c_sp, c_nov = sp[called], nov[called]
    g = pd.DataFrame({"group": labels, "sp": c_sp, "nov": c_nov})
    per_group = g.groupby("group").nov.agg(["any", "all"])
    fp_groups = int((~per_group["any"]).sum())
    mixed = int((per_group["any"] & ~per_group["all"]).sum())
    gn = g[g.nov]
    caught = gn.sp.nunique()
    oversplit = int((gn.groupby("sp").group.nunique() - 1).sum())
    groups_with_novel = gn.group.nunique()
    lumped = caught + oversplit - groups_with_novel
    missed = n_true - caught
    n_groups = len(per_group)
    r = dict(n_groups=n_groups, n_true_novel_species=n_true, net_error=n_groups - n_true, fp_groups=fp_groups,
             oversplit=oversplit, missed=missed, lumped=lumped,
             gross_error=fp_groups + oversplit + missed + lumped, mixed_groups=mixed, caught_species=caught)
    assert r["net_error"] == (fp_groups + oversplit) - (missed + lumped), r
    return r


def main():
    opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    marker = "teleo" if opts.get("marker", "teleo") == "teleo" else "full"
    suffix = "" if marker == "teleo" else "_full"
    ctx = Context(marker)
    calls = pd.read_csv(OUT / f"group_cutoffs{suffix}_calls.csv.gz", dtype={"seq_id": str})
    calls = calls[calls.method == "global"].set_index(["split_id", "seq_id"])
    splits = pd.concat([pd.read_csv(DATA / f, dtype={"seq_id": str, "difficulty": str})
                        for f in ["splits.csv", "splits_extra.csv"]])
    rows = []
    for sid in sorted(splits.split_id.unique()):
        s = splits[splits.split_id == sid].set_index("seq_id")
        m = ctx.seqs[["species_name", "Genus", "teleo"]].join(s[["role", "difficulty"]], how="inner")
        ref, test = m[m.role == "reference"].copy(), m[m.role != "reference"].copy()
        d_loc, _ = ctx.best_hits(test.index.values, ref.index.values)
        b2 = calls.loc[[(sid, q) for q in test.index]].called_novel.to_numpy(int).astype(bool)
        cal, w = loo_calibration(ctx, ref)
        p = conformal_p(cal, w, d_loc)
        for name, called in [("B current (B2)", b2), ("C proposed a=0.05", p <= 0.05),
                             ("C proposed a=0.10", p <= 0.10)]:
            labels = grouping(ctx, ref, test.index.values[called])
            r = decompose(test, called, labels)
            r.update(split=sid, pipeline=name)
            rows.append(r)
        x = {r["pipeline"]: r for r in rows if r["split"] == sid}
        print(f"[{marker}] split {sid:2d} | " + " | ".join(
            f"{k.split()[0]}{'' if 'a=' not in k else k[-5:]}: net {v['net_error']:+d} "
            f"(fp {v['fp_groups']} split {v['oversplit']} missed {v['missed']} lumped {v['lumped']})"
            for k, v in x.items()), flush=True)

    per = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    per.to_csv(OUT / f"count_diagnostic{suffix}_per_split.csv", index=False)

    rng = np.random.default_rng(0)
    P = {k: g.set_index("split") for k, g in per.groupby("pipeline")}
    tests = []
    for a in ["C proposed a=0.10", "C proposed a=0.05"]:
        for o in ["gross_error", "net_error", "fp_groups", "oversplit", "missed", "lumped"]:
            d = (P[a][o] - P["B current (B2)"][o]).astype(float)
            boots = [rng.choice(d.values, len(d)).mean() for _ in range(5000)]
            pv = wilcoxon(d.values).pvalue if (d != 0).sum() > 1 else 1.0
            tests.append(dict(marker=marker, a=a, b="B current (B2)", outcome=o,
                              primary=bool(a == "C proposed a=0.10" and o == "gross_error"),
                              mean_a=P[a][o].mean(), mean_b=P["B current (B2)"][o].mean(), diff=d.mean(),
                              ci_lo=np.percentile(boots, 2.5), ci_hi=np.percentile(boots, 97.5),
                              n=len(d), wilcoxon_p=pv))
    tests = pd.DataFrame(tests)
    tests.to_csv(OUT / f"count_diagnostic{suffix}_tests.csv", index=False)

    print(f"\n[{marker}] COUNT ERROR PARTS (mean over {per.split.nunique()} splits)")
    print("   n_groups - n_true = (fp_groups + oversplit) - (missed + lumped);  gross = sum of the four")
    print(per.groupby("pipeline")[PARTS].mean().loc[PIPELINES].T.round(2).to_string())
    print(f"\n[{marker}] vs B2 (diff = C - B2; negative = fewer errors)")
    print(tests[["a", "outcome", "mean_a", "mean_b", "diff", "ci_lo", "ci_hi", "wilcoxon_p"]]
          .round(3).to_string(index=False))
    t = tests[tests.primary].iloc[0]
    ok = t.ci_hi < 0 and t.wilcoxon_p < 0.05
    print(f"\n[{marker}] RULE (gross_error, C a=0.10 - B2): {t['diff']:+.2f} (CI {t.ci_lo:+.2f} to {t.ci_hi:+.2f}, "
          f"p {t.wilcoxon_p:.2g}) -> count claim {'SUPPORTED' if ok else 'NOT SUPPORTED (cancelling errors)'}")


if __name__ == "__main__":
    main()
