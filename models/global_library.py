#!/usr/bin/env python3
"""
Placement with a GLOBAL reference library: does adding MitoFish sequences from outside French
Guiana place new species better, by giving them relatives the local library lacks?

Design (two-stage, so detection and known species are untouched):
  stage 1  detection as now, against the LOCAL (Guiana) reference only
  stage 2  a sequence called novel is placed on its nearest sequence in LOCAL + GLOBAL
Here, as in every placement analysis, stage 2 is scored on the truly novel test sequences.

RULES, FIXED BEFORE RUNNING
  Global library (data/global_teleo.csv | data/global_full12s.csv), kept rows:
    1. in_guiana == False (no French Guiana species at all -> the hidden species cannot leak)
    2. family known: genus = first word of the name, family from data/global_taxonomy.csv by genus
       (Guiana genera from data/guiana_teleo_clean.csv); rows with no family are dropped
    3. sequence NOT identical to any French Guiana sequence of that marker (identical = likely the
       same species under another name)
  Sensitivity (reported, not decisive): also drop global species whose specific epithet equals that
    of a Guiana species in the SAME family (possible synonyms / genus moves).
  Nearest sequence: VSEARCH, same settings and tie rule as B2 (baselines/common.vsearch_best_hits),
    run ONCE per marker for every Guiana sequence against the global library (cached), and the local
    best hit from data/pairwise_identity_<marker>.npz (same tie rule, models/group_cutoffs.Context).
    Combined rule: the global hit is used only if its identity is STRICTLY higher than the local
    best; ties -> local.
  Outcomes per split, truly novel sequences: family, family_easy (no same-genus species in the local
    reference), genus_hard (one is), genus_easy (genus correct when NO local congener exists; only
    reachable through the global library), share_global (placed on a global hit).
    Secondary: known_sp_if_global = known test sequences given their right species if stage 2 were
    used for them too (shows why detection stays local).
  PRIMARY (Teleo): local+global vs local, family_easy. Confirmed if the 95% CI of the paired mean
    difference is above 0 and Wilcoxon p < 0.05.
  Secondary: same on every outcome; long 12S = replication; local+global vs tree placement (phylo,
    outputs/results/phylo_placement[_full]_per_split.csv); sensitivity library vs local.
  Check printed: local family should reproduce placement_vsearch_check "VSEARCH all" (~0.758, Teleo).

EXPLORATORY OPTION (added after seeing the long-12S result; not a confirmatory test)
  --min-cov=0.9   a global hit counts only if VSEARCH aligns >= 90% of the Guiana query to it
                  (--query_cov). Question: did short global sequences (16% of global long 12S are
                  shorter than any Guiana sequence) cause the long-12S drop? Runs local, local+global
                  (as before) and local+global with the filter; nothing else changes. Outputs carry
                  a _cov<NN> suffix so the original files are not overwritten.
                  Reading, fixed before running: supported if the filter removes the long-12S family
                  loss (local+global(cov) - local family_easy no longer negative in splits 11, 13, 15
                  and the mean difference is >= the unfiltered one); else the explanation is dropped.

Usage (repo root):  python models/global_library.py [--marker=full] [--threads=8] [--min-cov=0.9]
Needs:  VSEARCH (tools/vsearch.exe, PATH or $VSEARCH), data/pairwise_identity_<marker>.npz
Output (outputs/results/):
  global_library_hits[_full][_sens].csv   cache: best global hit of every Guiana sequence
  global_library[_full]_per_split.csv     split x method x outcome
  global_library[_full]_tests.csv         paired comparisons
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
import os  # noqa: E402
import tempfile  # noqa: E402
from common import _run, find_vsearch, vsearch_best_hits, write_fasta  # noqa: E402
from group_cutoffs import Context  # noqa: E402

OUTCOMES = ["family", "family_easy", "genus_hard", "genus_easy", "share_global", "known_sp_if_global"]


def taxonomy_maps(clean):
    gt = pd.read_csv(DATA / "global_taxonomy.csv")
    fam_of_genus = pd.concat([gt[["Genus", "Family"]], clean[["Genus", "Family"]]]).dropna() \
        .drop_duplicates("Genus").set_index("Genus").Family
    return fam_of_genus


def global_library(marker, clean, guiana_seqs, fam_of_genus, sensitivity):
    f, col = ("global_teleo.csv", "teleo") if marker == "teleo" else ("global_full12s.csv", "seq")
    g = pd.read_csv(DATA / f)
    n0 = len(g)
    g = g[~g.in_guiana].copy()
    g["seqn"] = g[col].astype(str).str.strip().str.lower()
    g["Genus"] = g.species_name.str.split().str[0]
    g["Family"] = g.Genus.map(fam_of_genus)
    n1 = len(g)
    g = g[g.Family.notna()]
    n2 = len(g)
    g = g[~g.seqn.isin(set(guiana_seqs))]
    n3 = len(g)
    if sensitivity:
        cl = clean.drop_duplicates("species_name")
        pairs = set(zip(cl.species_name.str.split().str[-1], cl.Family))
        g = g[[(s.split()[-1], fa) not in pairs for s, fa in zip(g.species_name, g.Family)]]
    print(f"[{marker}] global library: {n0} rows -> not Guiana {n1} -> family known {n2} -> "
          f"not identical to a Guiana sequence {n3}" + (f" -> no shared epithet {len(g)}" if sensitivity else "")
          + f" | {g.species_name.nunique()} species, {g.Genus.nunique()} genera")
    g = g.reset_index(drop=True)
    g.index = [f"g{i}" for i in range(len(g))]
    return g


def vsearch_best_hits_cov(query, ref, vsearch, threads, min_cov):
    """common.vsearch_best_hits with the same settings + --query_cov min_cov (same tie rule)."""
    with tempfile.TemporaryDirectory() as tmp:
        qf, rf, of = (os.path.join(tmp, x) for x in ("q.fa", "r.fa", "hits.tsv"))
        write_fasta(query, qf)
        write_fasta(ref, rf)
        _run([vsearch, "--usearch_global", qf, "--db", rf, "--id", "0.0",
              "--maxaccepts", "0", "--maxrejects", "0", "--top_hits_only", "--minwordmatches", "0",
              "--query_cov", str(min_cov),
              "--strand", "plus", "--qmask", "none", "--dbmask", "none",
              "--userout", of, "--userfields", "query+target+id",
              "--output_no_hits", "--threads", str(threads), "--quiet"])
        hits = pd.read_csv(of, sep="\t", header=None, names=["q", "t", "id"], dtype={"q": str, "t": str})
    hits = hits[hits["t"] != "*"]
    hits["sp"] = ref["species_name"].reindex(hits["t"]).values
    best = {}
    for q, g in hits.groupby("q", sort=False):
        counts = g["sp"].value_counts()
        best[q] = (g["id"].max() / 100.0, sorted(counts.index[counts == counts.max()])[0])
    out = pd.DataFrame(index=query.index)
    out["best_id"] = [best.get(q, (0.0, None))[0] for q in query.index]
    out["best_species"] = [best.get(q, (0.0, None))[1] for q in query.index]
    return out


def global_hits(marker, query, glib, threads, tag, min_cov=None):
    cache = OUT / f"global_library_hits{'' if marker == 'teleo' else '_full'}{tag}.csv"
    if cache.exists():
        h = pd.read_csv(cache, dtype={"seq_id": str}).set_index("seq_id")
        if set(h.index) == set(query.index):
            print(f"[{marker}] using cached {cache.name}")
            return h
    print(f"[{marker}] VSEARCH: {len(query)} Guiana sequences vs {len(glib)} global sequences ...", flush=True)
    ref = pd.DataFrame({"teleo": glib.seqn.str.upper(), "species_name": glib.species_name}, index=glib.index)
    if min_cov:
        h = vsearch_best_hits_cov(query, ref, find_vsearch(), threads, min_cov)
    else:
        h = vsearch_best_hits(query, ref, find_vsearch(), threads=threads)
    h.index.name = "seq_id"
    OUT.mkdir(parents=True, exist_ok=True)
    h.to_csv(cache)
    return h


def paired(A, B, o, rng):
    d = (A[o] - B[o]).dropna()
    if len(d) < 2:
        return None
    boots = [rng.choice(d.values, len(d)).mean() for _ in range(5000)]
    p = wilcoxon(d.values).pvalue if (d != 0).sum() > 1 else 1.0
    return dict(mean_a=A[o].loc[d.index].mean(), mean_b=B[o].loc[d.index].mean(), diff=d.mean(),
                ci_lo=np.percentile(boots, 2.5), ci_hi=np.percentile(boots, 97.5),
                a_better=int((d > 0).sum()), b_better=int((d < 0).sum()), n=len(d), wilcoxon_p=p)


def main():
    opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    marker = "teleo" if opts.get("marker", "teleo") == "teleo" else "full"
    suffix = "" if marker == "teleo" else "_full"
    threads = int(opts.get("threads", 0))
    min_cov = float(opts.get("min-cov", 0))
    cov_tag = f"_cov{int(round(min_cov * 100))}" if min_cov > 0 else ""
    cov_name = f"local+global (cov>={min_cov:g})"
    ctx = Context(marker)
    clean = ctx.seqs.copy()
    clean["Family"] = pd.read_csv(DATA / "guiana_teleo_clean.csv", dtype={"Specimen identification code": str}) \
        .drop_duplicates("Specimen identification code").set_index("Specimen identification code") \
        .Family.reindex(clean.index).values
    if marker == "teleo":
        qseq = clean.teleo.str.upper()
    else:
        f = pd.read_csv(DATA / "guiana_full12s.csv", dtype={"seq_id": str}).drop_duplicates("seq_id")
        qseq = f.set_index("seq_id").seq.reindex(clean.index).str.strip().str.upper()
        if qseq.isna().any():
            sys.exit("some Guiana sequences have no long-12S sequence")
    query = pd.DataFrame({"teleo": qseq, "species_name": clean.species_name}, index=clean.index)
    all_guiana = set(qseq.str.lower())
    if marker == "teleo":
        all_guiana |= set(pd.read_csv(DATA / "guiana_teleo_clean.csv").teleo.str.strip().str.lower())
    fam_of_genus = taxonomy_maps(clean)
    sp_info = clean.drop_duplicates("species_name").set_index("species_name")[["Genus", "Family"]]

    libs = {}
    lib_specs = ([("local+global", False, "", None), (cov_name, False, cov_tag, min_cov)] if min_cov > 0 else
                 [("local+global", False, "", None), ("local+global (sensitivity)", True, "_sens", None)])
    for name, sens, tag, mc in lib_specs:
        glib = global_library(marker, clean, all_guiana, fam_of_genus, sens)
        h = global_hits(marker, query, glib, threads, tag, mc)
        gtax = glib.drop_duplicates("species_name").set_index("species_name")[["Genus", "Family"]]
        libs[name] = (h, pd.concat([sp_info, gtax[~gtax.index.isin(sp_info.index)]]))

    splits = pd.concat([pd.read_csv(DATA / f, dtype={"seq_id": str, "difficulty": str})
                        for f in ["splits.csv", "splits_extra.csv"]])
    rows = []
    for sid in sorted(splits.split_id.unique()):
        s = splits[splits.split_id == sid].set_index("seq_id")
        m = clean[["species_name", "Genus", "Family"]].join(s[["role", "difficulty"]], how="inner")
        ref, test = m[m.role == "reference"], m[m.role != "reference"]
        d_loc, s_loc = ctx.best_hits(test.index.values, ref.index.values)
        id_loc = 1 - d_loc
        nov = (test.role == "test_novel").to_numpy(bool)
        easy = nov & (test.difficulty == "easy").to_numpy(bool, na_value=False)
        hard = nov & (test.difficulty == "hard").to_numpy(bool, na_value=False)
        known = ~nov
        tg, tf, ts = test.Genus.to_numpy(object), test.Family.to_numpy(object), test.species_name.to_numpy(object)
        methods = {"local": (s_loc, np.zeros(len(test), bool), sp_info)}
        for name, (h, tax) in libs.items():
            gid = h.best_id.reindex(test.index).fillna(0).to_numpy(float)
            gsp = h.best_species.reindex(test.index).to_numpy(object)
            use = gid > id_loc + 1e-9
            methods[name] = (np.where(use, gsp, s_loc), use, tax)
        for name, (guess, use, tax) in methods.items():
            gg = tax.Genus.reindex(guess).to_numpy(object)
            gf = tax.Family.reindex(guess).to_numpy(object)
            fam, gen = gf == tf, gg == tg
            mean = lambda x: float(x.mean()) if len(x) else np.nan  # noqa: E731
            rows.append(dict(split=sid, method=name, n_novel=int(nov.sum()), family=mean(fam[nov]),
                             family_easy=mean(fam[easy]), genus_hard=mean(gen[hard]), genus_easy=mean(gen[easy]),
                             share_global=mean(use[nov]), known_sp_if_global=mean((guess == ts)[known])))
        r = {x["method"]: x for x in rows if x["split"] == sid}
        print(f"[{marker}] split {sid:2d} | family_easy local {r['local']['family_easy']:.3f} -> "
              f"global {r['local+global']['family_easy']:.3f} | placed on global "
              f"{r['local+global']['share_global']:.2f}", flush=True)
    per = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    per.round(5).to_csv(OUT / f"global_library{suffix}{cov_tag}_per_split.csv", index=False)

    rng = np.random.default_rng(0)
    tests = []
    P = {k: g.set_index("split") for k, g in per.groupby("method")}
    pairs = [(n, "local") for n, *_ in lib_specs] + ([(cov_name, "local+global")] if min_cov > 0 else [])
    for a, b in pairs:
        for o in OUTCOMES:
            r = paired(P[a], P[b], o, rng)
            if r:
                tests.append(dict(a=a, b=b, outcome=o, marker=marker,
                                  primary=bool(marker == "teleo" and not min_cov and a == "local+global"
                                               and b == "local" and o == "family_easy"), **r))
    ph = pd.read_csv(OUT / f"phylo_placement{suffix}_per_split.csv")
    ph = ph[ph.method == "phylo"].set_index("split")
    for o in ["family", "family_easy", "genus_hard"]:
        r = paired(P["local+global"], ph, o, rng)
        if r:
            tests.append(dict(a="local+global", b="phylo (tree placement)", outcome=o, marker=marker,
                              primary=False, **r))
    tests = pd.DataFrame(tests)
    tests.to_csv(OUT / f"global_library{suffix}{cov_tag}_tests.csv", index=False)

    print(f"\n[{marker}] mean over {per.split.nunique()} splits")
    print(per.groupby("method")[OUTCOMES].mean().round(4).to_string())
    if marker == "teleo":
        print("check: local family should match placement_vsearch_check 'VSEARCH all' family (~0.758)")
    print(f"\n[{marker}] paired comparisons (diff = a - b)")
    print(tests[["a", "b", "outcome", "mean_a", "mean_b", "diff", "ci_lo", "ci_hi", "wilcoxon_p"]]
          .round(4).to_string(index=False))
    if min_cov > 0:
        P2 = {k: g.set_index("split") for k, g in per.groupby("method")}
        d_cov = (P2[cov_name].family_easy - P2["local"].family_easy)
        d_raw = (P2["local+global"].family_easy - P2["local"].family_easy)
        bad = [k for k in [11, 13, 15] if k in d_cov.index]
        print(f"\n[{marker}] EXPLORATORY coverage check (family_easy, minus local):")
        print("  splits " + ", ".join(f"{k}: unfiltered {d_raw[k]:+.3f} -> filtered {d_cov[k]:+.3f}" for k in bad))
        ok = all(d_cov[k] >= 0 for k in bad) and d_cov.mean() >= d_raw.mean()
        print(f"  mean: unfiltered {d_raw.mean():+.4f}, filtered {d_cov.mean():+.4f} -> short-sequence explanation "
              f"{'SUPPORTED' if ok else 'NOT SUPPORTED'} (exploratory)")
    elif marker == "teleo":
        t = tests[tests.primary].iloc[0]
        ok = t.ci_lo > 0 and t.wilcoxon_p < 0.05
        print(f"\nPRIMARY (Teleo, family_easy, local+global - local): {t['diff']:+.3f} "
              f"(CI {t.ci_lo:+.3f} to {t.ci_hi:+.3f}, p {t.wilcoxon_p:.2g}) -> {'CONFIRMED' if ok else 'NOT CONFIRMED'}")


if __name__ == "__main__":
    main()
