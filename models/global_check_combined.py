#!/usr/bin/env python3
"""
Confirm the taxonomy + tree combination on GLOBAL data only (no French Guiana data), with the
SAME held-out check that chose the tree-clade model (models/select_treeloss.py).

Why: on French Guiana, raw + 1/2 taxonomy + 1/2 treeclade beat raw + taxonomy and raw + treeclade
(models/placement_taxtree.py). The weights were fixed in advance, but the idea came from looking
at French Guiana results. If the combination also wins here, where nothing from French Guiana is
used, the claim does not rest on the test data.

Held-out check (copied from select_treeloss.py, same random choices):
  10% of global species hidden ('novel') + one held-out sequence of each other species with
  >= 2 distinct sequences ('known'); encoders trained on the rest ('ref'), same settings as
  select_treeloss.py (seed 0 there; here seeds 0-4, held-out set identical for every seed).
Placement (same as placement_kmer_tree.py / placement_taxtree.py): distance to each reference
species centre in raw k-mer, taxonomy-model and tree-clade-model space, each divided by the
median distance between reference species centres in its own space (reference only):
  taxonomy alone / treeclade alone   model space only  (CHECK: seed 0 must reproduce
                                     select_treeloss.csv: family 0.6865 / 0.6810 on Teleo)
  raw                                raw k-mers only
  raw+taxonomy                       Rn + Tn
  raw+treeclade                      Rn + Cn
  raw+taxonomy+treeclade             Rn + 0.5 Tn + 0.5 Cn   <- the method being confirmed
  raw+taxonomy+treeclade (each w=1)  Rn + Tn + Cn           (sensitivity, reported only)
Outcomes, novel sequences (as select_treeloss.py):
  family, family_no_congener (no same-genus species in training), genus_congener (a same-genus
  species is in training), tree_error and tree_error_no_congener (split age, My, lower = better);
  known_acc = species correct for the held-out known sequences.

Decision rule, fixed before running (same primary outcome as the rule that chose treeclade):
  CONFIRMED if the combination's novel FAMILY placement is higher than raw+taxonomy with the
  95% CI of the difference above 0. Tree error and the other outcomes are reported alongside.
There is one held-out split, so uncertainty comes from a paired bootstrap over NOVEL SPECIES
(5000 resamples; each sequence's result first averaged over the training seeds), plus the
difference for every seed separately.

Needs data/global_taxonomy.csv (see run_global_check.sh) and data/global_tree_distances.npz.
Usage (repo root):  python models/global_check_combined.py                # Teleo, seeds 0-4
                    python models/global_check_combined.py --marker=full  # long 12S
                    python models/global_check_combined.py --seeds=0      # quick check
Time: two encoders per seed, ~2 min each on a laptop (Teleo) -> ~20 min for 5 seeds.
Output (outputs/results/): global_check_combined[_full]_per_seed.csv, _tests.csv
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.encoder import DistanceEncoder  # noqa: E402
from models.kmers import kmer_profile  # noqa: E402
from models.pretrain import HP, TRAIN, encoder_settings, global_target, load_global  # noqa: E402

OUT = ROOT / "outputs" / "results"
KNOWN_OPTS = {"marker", "seeds"}
MAIN = "raw+taxonomy+treeclade"
SENS = "raw+taxonomy+treeclade (each w=1)"
METHODS = ["taxonomy alone", "treeclade alone", "raw", "raw+taxonomy", "raw+treeclade", MAIN, SENS]
OUTCOMES = ["family", "family_no_congener", "genus_congener", "tree_error", "tree_error_no_congener"]
LOWER_BETTER = {"tree_error", "tree_error_no_congener"}
PAIRS = [(MAIN, "raw+taxonomy"), (MAIN, "raw+treeclade"), (MAIN, "raw"), (SENS, "raw+taxonomy"),
         ("raw+treeclade", "raw+taxonomy"), ("treeclade alone", "taxonomy alone")]


def parse():
    opts = {}
    for a in sys.argv[1:]:
        if not (a.startswith("--") and "=" in a) or a[2:].split("=", 1)[0] not in KNOWN_OPTS:
            sys.exit(f"unknown option {a!r} - allowed: --marker=full, --seeds=0,1,2,3,4")
        k, v = a[2:].split("=", 1)
        opts[k] = v
    marker = opts.get("marker", "teleo")
    if marker not in ("teleo", "full"):
        sys.exit("--marker must be teleo or full")
    return marker, [int(x) for x in opts.get("seeds", "0,1,2,3,4").split(",")]


def centre_dist(Z_ref, ref_sp, Z_test, sp):
    """test -> reference species centre distances / median distance between centres."""
    C = np.stack([Z_ref[ref_sp == s].mean(0) for s in sp])

    def dist(A, B):
        return np.sqrt(np.maximum((A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None] - 2 * A @ B.T, 0))

    CC = dist(C, C)
    return dist(Z_test, C) / np.median(CC[np.triu_indices(len(sp), 1)])


def main():
    marker, seeds = parse()
    sfx = "_full" if marker == "full" else ""
    tax_file = ROOT / "data" / "global_taxonomy.csv"
    if not tax_file.exists():
        sys.exit(f"missing {tax_file} - see run_global_check.sh (download MitoFish, then "
                 "python data/add_global_taxonomy.py)")
    g = load_global(marker)
    X = np.array([kmer_profile(s) for s in g.teleo])
    tz = np.load(ROOT / "data" / "global_tree_distances.npz")
    r = g.species_name.map({s: i for i, s in enumerate(tz["species"])}).values
    age = tz["D"] * float(tz["scale"]) / 2
    tax = pd.read_csv(tax_file).set_index("species_name")
    genus = g.species_name.str.split().str[0].values
    family = tax.loc[g.species_name, "Family"].values

    # ---- held-out check: identical to select_treeloss.py
    rng = np.random.default_rng(0)
    species = np.array(sorted(g.species_name.unique()))
    novel = set(rng.choice(species, size=len(species) // 10, replace=False))
    role = np.where(g.species_name.isin(novel), "novel", "ref").astype(object)
    for sp_, grp in g[~g.species_name.isin(novel)].groupby("species_name"):
        seqs = sorted(grp.teleo.unique())
        if len(seqs) >= 2:
            role[grp.index[grp.teleo == seqs[rng.integers(len(seqs))]]] = "known"
    ref, test = role == "ref", role != "ref"
    y = role[test] == "novel"
    has_cong = np.array([x in set(genus[ref]) for x in genus[test]])
    fam_known = pd.notna(family[test])
    ref_sp, ts = g.species_name.values[ref], g.species_name.values[test]
    ref_family = dict(zip(ref_sp, family[ref]))
    sp_row = dict(zip(g.species_name.values, r))
    sp = np.unique(ref_sp)
    print(f"held-out check ({marker}): {len(sp)} training species, {int(y.sum())} novel sequences "
          f"({len(set(ts[y]))} species), {int((~y).sum())} known sequences")

    Rn = centre_dist(X[ref], ref_sp, X[test], sp)

    def per_seq(pred):
        """per test sequence: family ok, genus ok, split age, species ok."""
        pf = np.array([ref_family.get(p) for p in pred], dtype=object)
        return dict(fam=((pf == family[test]) & fam_known).astype(float),
                    gen=(np.array([p.split()[0] for p in pred]) == genus[test]).astype(float),
                    err=age[[sp_row[p] for p in pred], [sp_row[t] for t in ts]],
                    spc=(pred == ts).astype(float))

    def summary(s):
        return dict(family=s["fam"][y & fam_known].mean(),
                    family_no_congener=s["fam"][y & fam_known & ~has_cong].mean(),
                    genus_congener=s["gen"][y & has_cong].mean(),
                    tree_error=s["err"][y].mean(), tree_error_no_congener=s["err"][y & ~has_cong].mean(),
                    known_acc=s["spc"][~y].mean())

    seq_res = {m: [] for m in METHODS}          # per method: list over seeds of per-seq dicts
    rows = []
    raw_s = per_seq(sp[Rn.argmin(1)])
    for seed in seeds:
        Z = {}
        for name in ["taxonomy", "treeclade"]:
            t0 = time.time()
            enc = DistanceEncoder(X.shape[1], seed=seed, **encoder_settings(name, tz), **HP)
            enc.fit_minibatch(X[ref], r[ref], global_target(name, tz), seed=seed, verbose=False, **TRAIN)
            Z[name] = centre_dist(enc.embed(X[ref]), ref_sp, enc.embed(X[test]), sp)
            print(f"seed {seed} | trained {name} ({time.time() - t0:.0f}s)", flush=True)
        Tn, Cn = Z["taxonomy"], Z["treeclade"]
        for m, D in [("taxonomy alone", Tn), ("treeclade alone", Cn), ("raw", None),
                     ("raw+taxonomy", Rn + Tn), ("raw+treeclade", Rn + Cn),
                     (MAIN, Rn + 0.5 * Tn + 0.5 * Cn), (SENS, Rn + Tn + Cn)]:
            s = raw_s if D is None else per_seq(sp[D.argmin(1)])
            seq_res[m].append(s)
            rows.append(dict(seed=seed, method=m, **summary(s)))
        cur = pd.DataFrame([x for x in rows if x["seed"] == seed]).set_index("method")
        print(f"seed {seed} | family: " + " | ".join(f"{m} {cur.at[m, 'family']:.3f}" for m in METHODS), flush=True)

    per_seed = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    per_seed.round(5).to_csv(OUT / f"global_check_combined{sfx}_per_seed.csv", index=False)

    # ---- seed-averaged per-sequence results, paired bootstrap over novel species
    avg = {m: {k: np.mean([s[k] for s in seq_res[m]], 0) for k in ["fam", "gen", "err", "spc"]} for m in METHODS}
    masks = {"family": ("fam", y & fam_known), "family_no_congener": ("fam", y & fam_known & ~has_cong),
             "genus_congener": ("gen", y & has_cong), "tree_error": ("err", y),
             "tree_error_no_congener": ("err", y & ~has_cong)}
    nov_species = np.array(sorted(set(ts[y])))
    members = {s: np.where(ts == s)[0] for s in nov_species}
    brng = np.random.default_rng(0)
    boot_idx = [np.concatenate([members[s] for s in brng.choice(nov_species, len(nov_species))])
                for _ in range(5000)]
    tests = []
    for a, b in PAIRS:
        for o in OUTCOMES:
            key, mask = masks[o]
            d = avg[a][key] - avg[b][key]
            diff = d[mask].mean()
            boots = [d[i][mask[i]].mean() for i in boot_idx]
            lo, hi = np.percentile(boots, [2.5, 97.5])
            per_s = [per_seed[(per_seed.seed == s) & (per_seed.method == a)][o].iloc[0] -
                     per_seed[(per_seed.seed == s) & (per_seed.method == b)][o].iloc[0] for s in seeds]
            better = (np.array(per_s) < 0) if o in LOWER_BETTER else (np.array(per_s) > 0)
            tests.append(dict(a=a, b=b, outcome=o, mean_a=avg[a][key][mask].mean(), mean_b=avg[b][key][mask].mean(),
                              diff=diff, ci_lo=lo, ci_hi=hi, seeds_a_better=int(better.sum()), n_seeds=len(seeds),
                              per_seed_diff=" ".join(f"{x:+.4f}" for x in per_s)))
    tests = pd.DataFrame(tests)
    tests.to_csv(OUT / f"global_check_combined{sfx}_tests.csv", index=False)

    print(f"\n[{marker}] global held-out check | seeds {seeds} | mean over seeds")
    print("CHECK (Teleo, seed 0): 'taxonomy alone' / 'treeclade alone' family should be 0.6865 / 0.6810 "
          "as in select_treeloss.csv")
    print(per_seed.groupby("method")[OUTCOMES + ["known_acc"]].mean().loc[METHODS].round(4).to_string())
    print("\npaired differences (a - b), 95% CI = bootstrap over novel species; tree_error: lower = better")
    print(tests[["a", "b", "outcome", "mean_a", "mean_b", "diff", "ci_lo", "ci_hi", "seeds_a_better",
                 "n_seeds"]].round(4).to_string(index=False))
    t = tests[(tests.a == MAIN) & (tests.b == "raw+taxonomy") & (tests.outcome == "family")].iloc[0]
    verdict = "CONFIRMED" if t.ci_lo > 0 else "NOT CONFIRMED"
    print(f"\nDECISION (rule fixed in advance): combination vs raw+taxonomy, novel family "
          f"{t['diff']:+.4f} [{t.ci_lo:+.4f}, {t.ci_hi:+.4f}] -> {verdict}")


if __name__ == "__main__":
    main()
