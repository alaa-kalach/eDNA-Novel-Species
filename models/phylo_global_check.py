#!/usr/bin/env python3
"""
Global check of phylogenetic placement on the Fish Tree (models/phylo_placement.py), with NO
French Guiana data: does the French Guiana result hold on fish from around the world?

Held-out set (same as models/select_treeloss.py and the other global checks):
  global sequences from pretrain.load_global (no French Guiana species, no sequence identical to
  a French Guiana one); 10% of species hidden = "novel" (numpy rng(0), sorted species);
  one held-out sequence of every other species with >= 2 distinct sequences = "known";
  everything else = reference.

Methods (exactly the same rules as models/phylo_placement.py, applied once to this set):
  phylo              EPA-ng placement on the Fish Tree pruned to the reference species
  alignment nearest  same alignment, nearest reference sequence by identity, no tree
  raw                nearest reference species centre in raw k-mer space
  raw+taxonomy       raw + global-pretrained taxonomy model (models/pretrained_taxonomy*.npz,
                     seeds 0-4, averaged), same distance as placement_kmer_tree.py. Those models
                     were trained on the reference species only (the 10% novel species were held
                     out of training), so only NOVEL outcomes are compared for this method.

DECISION RULE, FIXED BEFORE RUNNING:
  The French Guiana result counts as confirmed if, on Teleo, phylo beats alignment nearest on
  FAMILY placement of novel species with a 95% bootstrap interval above 0 (bootstrap over novel
  species). Everything else (family_easy, genus_hard, tree_error, raw+taxonomy, long 12S,
  detection AUROC) is reported as secondary.

Outcomes: family (novel), family_easy (no same-genus species in the reference), genus_hard,
known_sp, tree_error / tree_error_easy (My), detection AUROC of the novelty scores.

Usage (repo root, tools as for phylo_placement.py):
    python models/phylo_global_check.py --threads=8                 # Teleo
    python models/phylo_global_check.py --marker=full --threads=8   # long 12S (slower)
Needs data/global_teleo.csv (or global_full12s.csv), data/global_taxonomy.csv,
data/global_tree_distances.npz, data/fish_tree_full.tre, models/pretrained_taxonomy*.npz (optional).
Output: outputs/results/phylo_global_check[_full]_per_seq.csv and _summary.csv
"""
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "models"))
import phylo_placement as pp  # noqa: E402  (tools, place_split, rules; reads --marker / --threads)
from models.kmers import kmer_profile  # noqa: E402
from models.pretrain import load_encoder, load_global, seed_suffix  # noqa: E402

DATA, OUT = ROOT / "data", ROOT / "outputs" / "results"
MARKER, MK = pp.MARKER, pp.MK


def held_out_set(g):
    """Same roles as select_treeloss.py."""
    rng = np.random.default_rng(0)
    species = np.array(sorted(g.species_name.unique()))
    novel = set(rng.choice(species, size=len(species) // 10, replace=False))
    role = np.where(g.species_name.isin(novel), "test_novel", "reference").astype(object)
    for sp, grp in g[~g.species_name.isin(novel)].groupby("species_name"):
        seqs = sorted(grp.teleo.unique())
        if len(seqs) >= 2:
            role[grp.index[grp.teleo == seqs[rng.integers(len(seqs))]]] = "test_known"
    return role


def centre_dist(V, ref_mask, ref_sp, test_mask):
    sps = np.unique(ref_sp)
    C = np.stack([V[ref_mask][ref_sp == s].mean(0) for s in sps])

    def dist(A, B):
        return np.sqrt(np.maximum((A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None] - 2 * A @ B.T, 0))
    cc = dist(C, C)
    return sps, dist(V[test_mask], C) / np.median(cc[np.triu_indices(len(C), 1)])


def main():
    opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    pp.THREADS = opts.get("threads", pp.THREADS)
    g = load_global(MARKER)
    g["teleo"] = g.teleo.str.lower()
    g["seq_id"] = [f"g{i}" for i in range(len(g))]
    g["role"] = held_out_set(g)
    tax = pd.read_csv(DATA / "global_taxonomy.csv").drop_duplicates("species_name").set_index("species_name")
    family = g.species_name.map(tax["Family"]).values
    genus = g.species_name.str.split().str[0].values
    ref, test = g.role.values == "reference", g.role.values != "reference"
    ref_genera = set(genus[ref])
    print(f"[{MARKER}] reference {ref.sum()} seqs / {g[ref].species_name.nunique()} species | "
          f"novel {(g.role == 'test_novel').sum()} seqs / {g[g.role == 'test_novel'].species_name.nunique()} species | "
          f"known {(g.role == 'test_known').sum()}", flush=True)

    # phylo + alignment nearest (one big "split")
    t0 = time.time()
    clean = g.set_index("seq_id")[["species_name"]]
    seqs = g.set_index("seq_id").teleo
    tree = pp.load_tree()
    with tempfile.TemporaryDirectory() as tmp:
        df = pp.place_split(g.seq_id.values[ref], g.seq_id.values[test], clean, seqs, tree, Path(tmp))
    print(f"placement done ({time.time() - t0:.0f}s)", flush=True)
    t = g[test].reset_index(drop=True)
    df = df.set_index("seq_id").loc[t.seq_id].reset_index()

    # raw k-mers (+ taxonomy model)
    X = np.array([kmer_profile(s) for s in g.teleo])
    sps, Rn = centre_dist(X, ref, g.species_name.values[ref], test)
    df["raw_species"] = sps[Rn.argmin(1)]
    tax_preds = []
    for seed in range(5):
        f = ROOT / "models" / f"pretrained_taxonomy{MK}{seed_suffix(seed)}.npz"
        if f.exists():
            Z = load_encoder(f).embed(X)
            _, Tn = centre_dist(Z, ref, g.species_name.values[ref], test)
            tax_preds.append(sps[(Rn + Tn).argmin(1)])
    print(f"raw+taxonomy seeds found: {len(tax_preds)}", flush=True)

    true = t.species_name.values
    nov = (t.role == "test_novel").values
    fam_true, gen_true = family[test], genus[test]
    easy = nov & np.array([x not in ref_genera for x in gen_true])
    hard = nov & ~easy
    fam_of = dict(zip(g.species_name, family))
    trow = {s: i for i, s in enumerate(np.load(DATA / "global_tree_distances.npz")["species"])}
    z = np.load(DATA / "global_tree_distances.npz")
    age = lambda p: z["D"][[trow[x] for x in p], [trow[x] for x in true]] * float(z["scale"]) / 2  # noqa: E731

    def per_seq(pred):
        fam = (np.array([fam_of.get(p) for p in pred], dtype=object) == fam_true) & pd.notna(fam_true)
        gen = np.array([p.split()[0] for p in pred]) == gen_true
        return dict(fam=fam, gen=gen, sp=np.asarray(pred) == true, age=age(pred))

    methods = {"phylo": [per_seq(df.phylo_species.values)],
               "alignment nearest": [per_seq(df.align_species.values)],
               "raw": [per_seq(df.raw_species.values)]}
    if tax_preds:
        methods["raw+taxonomy"] = [per_seq(p) for p in tax_preds]
    for m, runs in methods.items():
        for k in ["fam", "gen", "sp", "age"]:
            df[f"{m}|{k}"] = np.mean([r[k] for r in runs], axis=0)
    df["role"], df["easy"], df["species_name"] = t.role.values, easy, true
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / f"phylo_global_check{MK}_per_seq.csv", index=False)

    fam_ok = nov & pd.notna(fam_true)
    masks = dict(family=(fam_ok, "fam"), family_easy=(fam_ok & easy, "fam"), genus_hard=(hard, "gen"),
                 known_sp=(~nov, "sp"), tree_error=(nov, "age"), tree_error_easy=(easy, "age"))
    rows = []
    for m in methods:
        r = {"method": m}
        for o, (mask, k) in masks.items():
            r[o] = df.loc[mask, f"{m}|{k}"].mean() if not (m == "raw+taxonomy" and o == "known_sp") else np.nan
        rows.append(r)
    for m, col in [("phylo pendant", "pendant"), ("phylo tree distance", "tree_dist"),
                   ("alignment identity", "align_novelty")]:
        rows.append({"method": m, "auroc": roc_auc_score(nov, df[col].values)})
    summary = pd.DataFrame(rows).set_index("method")

    # bootstrap over novel species (paired)
    rng = np.random.default_rng(0)
    tests = []
    for b in [x for x in ["alignment nearest", "raw+taxonomy", "raw"] if x in methods]:
        for o, (mask, k) in masks.items():
            if o == "known_sp":
                continue
            d = (df.loc[mask, f"phylo|{k}"] - df.loc[mask, f"{b}|{k}"]).groupby(df.loc[mask, "species_name"]).mean()
            if len(d) < 2:
                continue
            boots = [rng.choice(d.values, len(d)).mean() for _ in range(2000)]
            lo, hi = np.percentile(boots, [2.5, 97.5])
            tests.append(dict(a="phylo", b=b, outcome=o, n_species=len(d), diff_species_mean=d.mean(),
                              ci_lo=lo, ci_hi=hi))
    tests = pd.DataFrame(tests)
    summary.to_csv(OUT / f"phylo_global_check{MK}_summary.csv")
    tests.to_csv(OUT / f"phylo_global_check{MK}_tests.csv", index=False)

    print(f"\n[{MARKER}] global held-out species (tree_error in My, lower = better)")
    print(summary.round(3).to_string())
    print("\nphylo minus other method, mean over novel species, 95% bootstrap CI over species "
          "(tree_error: negative = phylo better)")
    print(tests.round(4).to_string(index=False))
    if MARKER == "teleo":
        r = tests[(tests.b == "alignment nearest") & (tests.outcome == "family")].iloc[0]
        ok = r.ci_lo > 0
        print(f"\nDECISION (fixed in advance): phylo vs alignment nearest, family = {r.diff_species_mean:+.3f} "
              f"[{r.ci_lo:+.3f}, {r.ci_hi:+.3f}] -> {'CONFIRMED' if ok else 'NOT confirmed'}")
        for b in ["raw+taxonomy", "raw"]:
            x = tests[(tests.b == b) & (tests.outcome == "family")]
            if len(x):
                x = x.iloc[0]
                print(f"   (also report) phylo vs {b}, family = {x.diff_species_mean:+.3f} [{x.ci_lo:+.3f}, {x.ci_hi:+.3f}]")
    print(f"\nwrote outputs/results/phylo_global_check{MK}_per_seq.csv, _summary.csv, _tests.csv")


if __name__ == "__main__":
    main()
