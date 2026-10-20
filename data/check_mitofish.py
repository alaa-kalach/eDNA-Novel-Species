"""
Check whether the global MitoFish 12S reference (Mitohelper, Zenodo 8428816,
CC BY 4.0) is usable for pretraining the encoder.

It:
  1. downloads mitofish.12S.Oct2023.tsv (~58 MB) into data/ (once),
  2. finds the sequence and species columns,
  3. cuts out the Teleo region (between the Teleo forward and reverse primers),
  4. reports how many species have a Teleo sequence, how many are in the fish
     tree, and how many are NOT French Guiana species (usable for pretraining),
  5. saves data/global_teleo.csv  (species_name, teleo, in_tree, in_guiana).

Usage (from the repo root, with the .venv active):
    python -m pip install regex
    python data/check_mitofish.py
"""
import urllib.request
from pathlib import Path

import dendropy
import pandas as pd
import regex

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
URL = "https://zenodo.org/records/8428816/files/mitofish.12S.Oct2023.tsv?download=1"
TSV = DATA / "mitofish.12S.Oct2023.tsv"

TELEO_F = "acaccgcccgtcactct"            # Teleo forward primer
TELEO_R_RC = "catggtaagtgtaccggaag"      # reverse complement of TeleoR (CTTCCGGTACACTTACCATG)
MAX_ERR = 3                              # mismatches allowed in each primer
PAT = regex.compile(f"({TELEO_F}){{s<={MAX_ERR}}}(.{{20,120}}?)({TELEO_R_RC}){{s<={MAX_ERR}}}")


def download():
    if TSV.exists():
        print(f"already downloaded: {TSV}")
        return
    print("downloading (~58 MB) ...")
    urllib.request.urlretrieve(URL, TSV)
    print(f"saved {TSV}")


def find_columns(df):
    sample = df.head(200).astype(str)
    acgt = {c: sample[c].str.lower().str.count("[acgt]").mean() for c in df.columns}
    seq_col = max(acgt, key=acgt.get)
    named = [c for c in df.columns if "species" in c.lower()]
    sp_col = named[0] if named else None
    return seq_col, sp_col


def extract_teleo(seq):
    m = PAT.search(seq)
    return m.group(2) if m else None


def main():
    download()
    df = pd.read_csv(TSV, sep="\t", low_memory=False)
    print("\ncolumns:", list(df.columns))
    print("first row:\n", df.iloc[0].to_string()[:1500], "\n")

    seq_col, sp_col = find_columns(df)
    print(f"sequence column -> {seq_col!r}   species column -> {sp_col!r}")
    if sp_col is None:
        print("!! could not find a species column automatically; send me the column list above")
        return

    df = df[[sp_col, seq_col]].dropna()
    df.columns = ["species_name", "seq"]
    df["species_name"] = df["species_name"].astype(str).str.replace("_", " ").str.strip()
    df["seq"] = df["seq"].astype(str).str.lower().str.replace(r"[^acgtn]", "", regex=True)
    print(f"records with species + sequence: {len(df)} | species: {df.species_name.nunique()}")
    print("sequence length (median):", int(df.seq.str.len().median()))

    print("\nextracting Teleo region (can take several minutes) ...")
    df["teleo"] = df["seq"].map(extract_teleo)
    t = df.dropna(subset=["teleo"])
    t = t[t.species_name.str.split().str.len() == 2]               # binomial names only
    t = t[~t.species_name.str.contains(r"\bsp\.|cf\.|aff\.", regex=True)]
    print(f"records with a Teleo region: {len(t)} | species: {t.species_name.nunique()}")
    print("Teleo length (median):", int(t.teleo.str.len().median()))

    tree = dendropy.Tree.get(path=str(DATA / "fish_tree_full.tre"), schema="newick",
                             preserve_underscores=True)
    tree_sp = {l.taxon.label.replace("_", " ").strip("'") for l in tree.leaf_node_iter()}
    guiana = set(pd.read_csv(DATA / "guiana_teleo_clean.csv")["species_name"])
    t = t.drop_duplicates(["species_name", "teleo"]).copy()
    t["in_tree"] = t.species_name.isin(tree_sp)
    t["in_guiana"] = t.species_name.isin(guiana)

    usable = t[t.in_tree & ~t.in_guiana]
    print("\n================ SUMMARY ================")
    print(f"distinct (species, Teleo) pairs        : {len(t)}")
    print(f"species with Teleo                     : {t.species_name.nunique()}")
    print(f"  ... in the fish tree                 : {t[t.in_tree].species_name.nunique()}")
    print(f"  ... also French Guiana species       : {t[t.in_guiana].species_name.nunique()}")
    print(f"USABLE for pretraining (in tree, not FG): {usable.species_name.nunique()} species, "
          f"{len(usable)} distinct sequences")
    t[["species_name", "teleo", "in_tree", "in_guiana"]].to_csv(DATA / "global_teleo.csv", index=False)
    print(f"saved {DATA / 'global_teleo.csv'}")


if __name__ == "__main__":
    main()
