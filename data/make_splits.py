#!/usr/bin/env python3
"""
Build 30 reproducible novel-species detection experiments (splits).

Input : data/guiana_teleo_clean.csv  (rows with in_tree == True)
Output: data/splits.csv              splits 1-10   seq_id, split_id, role, difficulty
        data/splits_extra.csv        splits 11-30  (same columns, same rules)
        data/excluded_shared_teleo.csv  rows dropped because their Teleo
                                        sequence is shared by 2+ species
The 30 splits are kept in two files so the existing results for splits 1-10 stay
valid; scripts run on the extra file with --splits=splits_extra.csv --tag=_extra.
Split k always uses seed BASE_SEED + k, so splits 1-10 are unchanged.

Rules
-----
* Drop every row whose Teleo sequence is carried by 2+ species (unresolvable).
* Per split (seed = BASE_SEED + split_id):
    1. Hide ~20% of species as novel  -> all their rows: role = test_novel.
    2. Known species, grouped by distinct Teleo sequence (all copies together):
         - >= 2 distinct sequences: one random distinct sequence -> test_known,
           the rest -> reference.
         - 1 distinct sequence: everything -> reference.
    3. Novel difficulty: 'hard' if a congeneric species is in the reference,
       otherwise 'easy'.

Usage: python data/make_splits.py [input_csv] [output_csv]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

N_SPLITS = 30          # total number of splits
N_MAIN = 10            # splits 1..N_MAIN -> splits.csv, the rest -> splits_extra.csv
NOVEL_FRAC = 0.20
BASE_SEED = 2026  # split k uses seed BASE_SEED + k

HERE = Path(__file__).resolve().parent
IN_CSV = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "guiana_teleo_clean.csv"
OUT_CSV = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "splits.csv"
EXTRA_CSV = OUT_CSV.with_name(OUT_CSV.stem + "_extra" + OUT_CSV.suffix)
EXCL_CSV = OUT_CSV.parent / "excluded_shared_teleo.csv"


def load():
    d = pd.read_csv(IN_CSV)
    d = d[d["in_tree"]].copy()
    d = d.rename(columns={"Specimen identification code": "seq_id"})
    d["teleo"] = d["teleo"].str.strip().str.lower()
    assert d["seq_id"].is_unique, "seq_id must be unique"
    print(f"in_tree rows: {len(d)}  species: {d.species_name.nunique()}")

    n_sp = d.groupby("teleo")["species_name"].nunique()
    shared = n_sp[n_sp > 1].index
    excl = d[d["teleo"].isin(shared)]
    d = d[~d["teleo"].isin(shared)].copy()
    excl[["seq_id", "species_name", "Genus", "teleo"]].sort_values(
        ["teleo", "species_name"]).to_csv(EXCL_CSV, index=False)

    lost = sorted(set(excl.species_name) - set(d.species_name))
    print(f"removed {len(shared)} shared Teleo sequences = {len(excl)} rows")
    print(f"species lost entirely ({len(lost)}): {', '.join(lost) or '-'}")
    print(f"working set: {len(d)} rows, {d.species_name.nunique()} species, "
          f"{d.teleo.nunique()} distinct Teleo sequences\n")
    # stable ordering so results don't depend on input row order
    return d.sort_values("seq_id").reset_index(drop=True)


def make_split(d, split_id):
    rng = np.random.default_rng(BASE_SEED + split_id)
    species = np.array(sorted(d["species_name"].unique()))
    n_novel = int(round(NOVEL_FRAC * len(species)))
    novel = set(rng.choice(species, size=n_novel, replace=False))

    role = pd.Series("reference", index=d.index)
    role[d["species_name"].isin(novel)] = "test_novel"

    known = d[~d["species_name"].isin(novel)]
    for sp, grp in known.groupby("species_name", sort=True):
        seqs = sorted(grp["teleo"].unique())
        if len(seqs) >= 2:
            held = seqs[rng.integers(len(seqs))]
            role[grp.index[grp["teleo"] == held]] = "test_known"

    ref_genera = set(d.loc[role == "reference", "Genus"])
    diff = pd.Series("", index=d.index)
    nov = role == "test_novel"
    diff[nov] = np.where(d.loc[nov, "Genus"].isin(ref_genera), "hard", "easy")

    return pd.DataFrame({"seq_id": d["seq_id"], "split_id": split_id,
                         "role": role, "difficulty": diff})


def check(d, s):
    m = s.join(d[["species_name", "Genus", "teleo"]])
    r = m["role"].value_counts()
    nov_sp = m[m.role == "test_novel"].groupby("species_name")["difficulty"].first()
    ref_seqs = set(m.loc[m.role == "reference", "teleo"])
    test_seqs = set(m.loc[m.role != "reference", "teleo"])
    leak = len(ref_seqs & test_seqs)
    # extra guards
    tk_sp = set(m.loc[m.role == "test_known", "species_name"])
    ref_sp = set(m.loc[m.role == "reference", "species_name"])
    assert tk_sp <= ref_sp, "a test_known species has no reference sequence"
    assert m.groupby("species_name")["role"].apply(
        lambda x: ("test_novel" in set(x)) == (set(x) == {"test_novel"})).all()
    assert leak == 0, "Teleo sequence leaked between reference and test"
    sid = int(s.split_id.iloc[0])
    print(f"split {sid:2d} | ref {r.get('reference',0):4d}  "
          f"test_known {r.get('test_known',0):3d} ({len(tk_sp):3d} sp)  "
          f"test_novel {r.get('test_novel',0):3d} ({len(nov_sp):2d} sp: "
          f"{(nov_sp=='hard').sum():2d} hard / {(nov_sp=='easy').sum():2d} easy) "
          f"| ref∩test Teleo seqs = {leak}")


def main():
    d = load()
    out = []
    for k in range(1, N_SPLITS + 1):
        s = make_split(d, k)
        check(d, s)
        out.append(s)
    out = pd.concat(out, ignore_index=True)
    main_part = out[out["split_id"] <= N_MAIN]
    extra_part = out[out["split_id"] > N_MAIN]
    main_part.to_csv(OUT_CSV, index=False)
    extra_part.to_csv(EXTRA_CSV, index=False)
    print(f"\nwrote {OUT_CSV} (splits 1-{N_MAIN}, {len(main_part)} rows), "
          f"{EXTRA_CSV} (splits {N_MAIN + 1}-{N_SPLITS}, {len(extra_part)} rows) "
          f"and {EXCL_CSV}")


if __name__ == "__main__":
    main()
