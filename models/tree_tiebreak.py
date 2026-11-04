#!/usr/bin/env python3
"""
Tree as a TIE-BREAKER for VSEARCH: use the tree only where sequence similarity cannot decide.

VSEARCH's nearest match places new species as well as tree placement, also on the same one
sequence per species (models/placement_vsearch_check.py), and a better alignment for EPA-ng did
not help. Last principled attempt: keep VSEARCH, but when its best matches are nearly tied between
species of DIFFERENT genera, let the tree decide. If the tree carries information the sequence
does not, it should show here.

RULES, FIXED BEFORE RUNNING
  Sequences: truly novel test sequences, Teleo, all 30 splits (placement outcomes, as elsewhere).
  VSEARCH: same settings as baselines/common.py vsearch_best_hits (exhaustive global alignment of
    each test sequence against EVERY reference sequence), but every hit is kept. Per reference
    species its best identity. VSEARCH's answer = exactly the B2 rule (best identity; ties -> species
    with most tied hits, then alphabetical); checked against outputs/results_vsearch*.csv.
  Near-tied candidates at margin m = reference species whose best identity >= best - m.
  Ambiguous = the candidates belong to more than one GENUS. Only ambiguous sequences can change.
  Tree guess = phylo_species in outputs/results/phylo_placement_per_seq.csv (tree placement).
  Two variants:
    free         ambiguous -> the tree's guess
    constrained  ambiguous -> the candidate species closest ON THE FISH TREE to the tree's guess
                 (ties -> higher identity, then alphabetical), i.e. the tree only chooses among
                 species the sequence also supports
  Margins m: 0 (exact ties only), 0.005, 0.01, ..., 0.05 (identity as a fraction).
  Tuning, leave-one-split-out: for test split k, (variant, m) is chosen on the novel sequences of
    the OTHER splits, excluding every sequence whose species is novel in split k. Criterion = mean
    family correct (per split, then over splits). "Never" (plain VSEARCH) is always a candidate;
    a rule must beat it strictly. Ties -> never, then free before constrained, then smaller m.
  Outcomes per split: family (PRIMARY), family_easy, genus_hard, tree_error (My, lower = better).
  PRIMARY TEST: tie-break rule minus VSEARCH, family, paired Wilcoxon over the 30 splits +
    bootstrap 95% CI. DECISION, fixed before running: the tree adds value over VSEARCH only if the
    95% CI is above 0. Secondary: vs tree placement; share of sequences switched; on the switched
    sequences only, VSEARCH vs tree accuracy; every (variant, m) on all data (sensitivity only,
    NOT a result).

Usage (repo root):  python models/tree_tiebreak.py [--threads=8]
Needs:  VSEARCH (tools/vsearch.exe, PATH, or $VSEARCH), outputs/results/phylo_placement_per_seq.csv,
        outputs/results_vsearch[_extra].csv (check only), data/global_tree_distances.npz
Output: outputs/results/tree_tiebreak_candidates.csv  (near-tied species per novel sequence, cache;
                                                       delete it to rerun VSEARCH)
        outputs/results/tree_tiebreak_per_seq.csv, _per_split.csv, _rules.csv, _tests.csv
"""
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "baselines"))
from common import _run, find_vsearch, load_data, write_fasta  # noqa: E402

DATA, OUT = ROOT / "data", ROOT / "outputs" / "results"
MARGINS = np.round(np.arange(0, 0.0501, 0.005), 3)
MAX_MARGIN = MARGINS.max()
VARIANTS = ["free", "constrained"]
OUTCOMES = ["family", "family_easy", "genus_hard", "tree_error"]
CACHE = OUT / "tree_tiebreak_candidates.csv"


# ------------------------------------------------------------------ VSEARCH, all hits
def all_hits(query, ref, vsearch, threads):
    with tempfile.TemporaryDirectory() as tmp:
        qf, rf, of = (os.path.join(tmp, x) for x in ("q.fa", "r.fa", "hits.tsv"))
        write_fasta(query, qf)
        write_fasta(ref, rf)
        _run([vsearch, "--usearch_global", qf, "--db", rf, "--id", "0.0",
              "--maxaccepts", "0", "--maxrejects", "0", "--minwordmatches", "0",
              "--strand", "plus", "--qmask", "none", "--dbmask", "none",
              "--userout", of, "--userfields", "query+target+id",
              "--output_no_hits", "--threads", str(threads), "--quiet"])
        h = pd.read_csv(of, sep="\t", header=None, names=["q", "t", "id"], dtype={"q": str, "t": str})
    h = h[h.t != "*"].copy()
    h["sp"] = ref.species_name.reindex(h.t).values
    h["id"] = h["id"] / 100.0
    return h


def candidates(threads):
    """Per novel sequence: VSEARCH answer + every species within MAX_MARGIN of the best identity."""
    if CACHE.exists():
        print(f"using cached {CACHE.relative_to(ROOT)} (delete it to rerun VSEARCH)")
        return pd.read_csv(CACHE, dtype={"seq_id": str})
    vs, rows = find_vsearch(), []
    for split_file in ["splits.csv", "splits_extra.csv"]:
        seqs, splits = load_data(split_file)
        for k in sorted(splits.split_id.unique()):
            s = splits[splits.split_id == k].set_index("seq_id")
            ref = seqs.loc[s.index[s.role == "reference"]]
            nov = seqs.loc[s.index[s.role == "test_novel"]]
            h = all_hits(nov, ref, vs, threads)
            for q, g in h.groupby("q", sort=False):
                best = g.id.max()
                top = g[g.id == best].sp.value_counts()                 # B2 tie rule
                answer = sorted(top.index[top == top.max()])[0]
                per_sp = g.groupby("sp").id.max()
                near = per_sp[per_sp >= best - MAX_MARGIN - 1e-9]
                for sp, ident in near.items():
                    rows.append(dict(split_id=k, seq_id=q, species=sp, identity=ident, best=best,
                                     vsearch=answer))
            print(f"split {k:2d} | {len(nov)} novel seqs | {len(ref)} reference seqs", flush=True)
    c = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    c.to_csv(CACHE, index=False, lineterminator="\n")
    return c


# ------------------------------------------------------------------ rule
def build(c, ph, info, row, age, clean):
    """One row per novel sequence with the guess of every (variant, margin) and the outcomes."""
    splits = pd.concat([pd.read_csv(DATA / "splits.csv"), pd.read_csv(DATA / "splits_extra.csv")])
    splits["seq_id"] = splits.seq_id.astype(str)
    nov = splits[splits.role == "test_novel"][["split_id", "seq_id", "difficulty"]].reset_index(drop=True)
    nov["true"] = clean.loc[nov.seq_id, "species_name"].values
    first = c.drop_duplicates(["split_id", "seq_id"]).set_index(["split_id", "seq_id"])
    key = list(zip(nov.split_id, nov.seq_id))
    if first.reindex(key).vsearch.isna().any():
        sys.exit("some novel sequences have no VSEARCH hit - delete the cache and rerun")
    nov["vsearch"] = first.loc[key, "vsearch"].values
    nov["tree"] = ph.set_index(["split_id", "seq_id"]).loc[key, "phylo_species"].values
    genus = info.Genus
    groups = {k: (g.species.values, g.identity.values, g.best.values[0], genus.loc[g.species].values)
              for k, g in c.groupby(["split_id", "seq_id"], sort=False)}
    cols = {(v, m): [] for v in VARIANTS + ["amb"] for m in MARGINS}
    for k, t, vs in zip(key, nov.tree.values, nov.vsearch.values):
        sp, ident, best, gen = groups[k]
        d_tree = age[row[t], [row[s] for s in sp]]
        for m in MARGINS:
            sel = ident >= best - m - 1e-9
            amb = len(set(gen[sel])) > 1
            cols[("amb", m)].append(amb)
            if not amb:
                cols[("free", m)].append(vs)
                cols[("constrained", m)].append(vs)
                continue
            cols[("free", m)].append(t)
            cs, ci, cd = sp[sel], ident[sel], d_tree[sel]
            order = np.lexsort((cs, -ci, cd))             # closest on the tree, then identity, then name
            cols[("constrained", m)].append(cs[order[0]])
    for (v, m), vals in cols.items():
        nov[f"{v}|{m}"] = vals
    return nov


def correct(nov, guess_col, info, row, age):
    g, t = nov[guess_col].values, nov.true.values
    fam = (info.loc[g, "Family"].values == info.loc[t, "Family"].values).astype(float)
    gen = (info.loc[g, "Genus"].values == info.loc[t, "Genus"].values).astype(float)
    err = age[[row[x] for x in g], [row[x] for x in t]]
    return fam, gen, err


def split_scores(nov, fam, gen, err):
    df = pd.DataFrame({"split": nov.split_id.values, "family": fam, "genus": gen, "err": err,
                       "easy": (nov.difficulty == "easy").values, "hard": (nov.difficulty == "hard").values})
    rows = []
    for k, d in df.groupby("split"):
        rows.append(dict(split=k, family=d.family.mean(), family_easy=d.family[d.easy].mean(),
                         genus_hard=d.genus[d.hard].mean(), tree_error=d.err.mean()))
    return pd.DataFrame(rows)


def per_split_family(nov, fam):
    return pd.Series(fam).groupby(nov.split_id.values).mean().mean()


def run(nov, info, row, age):
    F = {col: correct(nov, col, info, row, age) for col in
         ["vsearch", "tree"] + [f"{v}|{m}" for v in VARIANTS for m in MARGINS]}
    rules, hyb = [], {o: np.empty(len(nov)) for o in ["family", "genus", "err"]}
    chosen = np.empty(len(nov), dtype=object)
    for k in sorted(nov.split_id.unique()):
        te = (nov.split_id == k).values
        tu = (~te) & ~nov.true.isin(set(nov.true[te])).values
        sub = nov[tu]
        best = ("never", np.nan, per_split_family(sub, F["vsearch"][0][tu]))
        for v in VARIANTS:
            for m in MARGINS:
                sc = per_split_family(sub, F[f"{v}|{m}"][0][tu])
                if sc > best[2] + 1e-12:
                    best = (v, m, sc)
        col = "vsearch" if best[0] == "never" else f"{best[0]}|{best[1]}"
        for i, o in enumerate(["family", "genus", "err"]):
            hyb[o][te] = F[col][i][te]
        chosen[te] = nov[col].values[te]
        if best[0] == "never":
            switched = np.zeros(te.sum(), bool)
        else:
            switched = nov[f"amb|{best[1]}"].values[te].astype(bool)
        rules.append(dict(split=k, variant=best[0], margin=best[1], tuning_family=best[2],
                          n_tuning=int(tu.sum()), n_test=int(te.sum()), share_ambiguous=switched.mean()))
    nov = nov.assign(tiebreak=chosen)
    ps = []
    for name, (fam, gen, err) in [("tie-break rule", (hyb["family"], hyb["genus"], hyb["err"])),
                                  ("VSEARCH", F["vsearch"]), ("tree placement", F["tree"])]:
        ps.append(split_scores(nov, fam, gen, err).assign(method=name))
    sens = []
    for v in VARIANTS:
        for m in MARGINS:
            amb = nov[f"amb|{m}"].values.astype(bool)
            sens.append(dict(variant=v, margin=m, share_ambiguous=amb.mean(),
                             family=per_split_family(nov, F[f"{v}|{m}"][0]),
                             vsearch_on_ambiguous=F["vsearch"][0][amb].mean() if amb.any() else np.nan,
                             rule_on_ambiguous=F[f"{v}|{m}"][0][amb].mean() if amb.any() else np.nan))
    return nov, pd.DataFrame(rules), pd.concat(ps, ignore_index=True), pd.DataFrame(sens), F


def tests(ps, rng):
    out = []
    A = ps[ps.method == "tie-break rule"].set_index("split")
    for b in ["VSEARCH", "tree placement"]:
        B = ps[ps.method == b].set_index("split")
        for o in OUTCOMES:
            d = (A[o] - B[o]).dropna()
            boots = [rng.choice(d.values, len(d)).mean() for _ in range(5000)]
            lo, hi = np.percentile(boots, [2.5, 97.5])
            p = wilcoxon(d.values).pvalue if (d != 0).any() else 1.0
            better, worse = ((d < 0), (d > 0)) if o == "tree_error" else ((d > 0), (d < 0))
            out.append(dict(a="tie-break rule", b=b, outcome=o, primary=(b == "VSEARCH" and o == "family"),
                            mean_a=A[o].mean(), mean_b=B[o].mean(), diff=d.mean(), ci_lo=lo, ci_hi=hi,
                            a_better=int(better.sum()), b_better=int(worse.sum()), n=len(d), wilcoxon_p=p))
    return pd.DataFrame(out)


def main():
    opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    threads = int(opts.get("threads", 0))
    pf, tf = OUT / "phylo_placement_per_seq.csv", DATA / "global_tree_distances.npz"
    for f in (pf, tf):
        if not f.exists():
            sys.exit(f"missing {f.relative_to(ROOT)}")
    clean = pd.read_csv(DATA / "guiana_teleo_clean.csv")
    clean["seq_id"] = clean["Specimen identification code"].astype(str)
    clean = clean[clean.in_tree].drop_duplicates("seq_id").set_index("seq_id")
    info = clean.drop_duplicates("species_name").set_index("species_name")
    z = np.load(tf)
    row, age = {s: i for i, s in enumerate(z["species"])}, z["D"] * float(z["scale"]) / 2

    c = candidates(threads)
    vf = [ROOT / "outputs" / f for f in ["results_vsearch.csv", "results_vsearch_extra.csv"]]
    if all(f.exists() for f in vf):
        b2 = pd.concat([pd.read_csv(f, dtype={"seq_id": str}) for f in vf])
        m = c.drop_duplicates(["split_id", "seq_id"]).merge(b2, on=["split_id", "seq_id"])
        print(f"check: VSEARCH answer = B2 nearest_species for {(m.vsearch == m.nearest_species).mean():.2%} "
              f"of {len(m)} novel sequences")
    ph = pd.read_csv(pf, dtype={"seq_id": str})
    nov = build(c, ph, info, row, age, clean)
    nov, rules, ps, sens, F = run(nov, info, row, age)
    t = tests(ps, np.random.default_rng(0))

    print(f"\nTeleo, {ps.split.nunique()} splits, {len(nov)} novel sequences")
    print("\nrule chosen per split (tuned on the other splits, test species left out):")
    print(rules.groupby(["variant", "margin"], dropna=False).agg(
        splits=("split", "size"), share_switched=("share_ambiguous", "mean")).round(3).to_string())
    print("\nmean over splits (tree_error in My, lower = better):")
    print(ps.groupby("method")[OUTCOMES].mean().loc[["tie-break rule", "VSEARCH", "tree placement"]]
          .round(3).to_string())
    print("\npaired comparisons (Wilcoxon over splits; diff = tie-break rule - other)")
    for r in t.itertuples():
        tag = "  <- PRIMARY" if r.primary else ""
        print(f"  vs {r.b:15s} {r.outcome:11s} {r.mean_a:.3f} vs {r.mean_b:.3f} diff {r.diff:+.3f} "
              f"[{r.ci_lo:+.3f}, {r.ci_hi:+.3f}] better/worse {r.a_better}/{r.b_better} p = {r.wilcoxon_p:.3g}{tag}")
    print("\nsensitivity, every rule on ALL data (NOT a result; family = mean over splits;")
    print("'on ambiguous' = family correct on the ambiguous sequences only):")
    print(sens.round(3).to_string(index=False))
    p = t[t.primary].iloc[0]
    verdict = "the tree adds value as a tie-breaker" if p.ci_lo > 0 else \
        "the tree does NOT add value over VSEARCH, even as a tie-breaker"
    print(f"\nDECISION (fixed in advance): rule - VSEARCH, family = {p['diff']:+.3f} "
          f"[{p.ci_lo:+.3f}, {p.ci_hi:+.3f}] -> {verdict}")

    kw = dict(index=False, lineterminator="\n")
    keep = ["split_id", "seq_id", "difficulty", "true", "vsearch", "tree", "tiebreak"] + \
        [f"amb|{m}" for m in MARGINS]
    nov[keep].to_csv(OUT / "tree_tiebreak_per_seq.csv", **kw)
    ps.to_csv(OUT / "tree_tiebreak_per_split.csv", **kw)
    rules.to_csv(OUT / "tree_tiebreak_rules.csv", **kw)
    t.to_csv(OUT / "tree_tiebreak_tests.csv", **kw)
    sens.to_csv(OUT / "tree_tiebreak_sensitivity.csv", **kw)
    print("\nwrote outputs/results/tree_tiebreak_per_seq.csv, _per_split.csv, _rules.csv, _tests.csv, "
          "_sensitivity.csv")


if __name__ == "__main__":
    main()
