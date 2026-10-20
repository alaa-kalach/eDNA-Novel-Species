"""
Build data/global_taxonomy.csv: Genus, Family, Order for every species in the
global tree-distance matrix (data/global_tree_distances.npz). Needed by B6
(taxonomy target) so it gets the same global pretraining as the main method.

Sources
    French Guiana species -> data/guiana_teleo_clean.csv
    all other species     -> data/mitofish.12S.Oct2023.tsv (downloaded by check_mitofish.py)
    Genus always comes from the species name (first word) if nothing else is found.

Usage (from the repo root):  python data/add_global_taxonomy.py
Output: data/global_taxonomy.csv  (species_name, Genus, Family, Order)
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
TSV = DATA / "mitofish.12S.Oct2023.tsv"


def find(cols, name):
    hits = [c for c in cols if c.strip().lower() == name]
    return hits[0] if hits else None


def main():
    species = np.load(DATA / "global_tree_distances.npz")["species"]
    tax = pd.DataFrame({"species_name": species})
    tax["Genus"] = tax.species_name.str.split().str[0]

    if not TSV.exists():
        raise FileNotFoundError(f"{TSV} not found - run data/check_mitofish.py first")
    m = pd.read_csv(TSV, sep="\t", low_memory=False)
    cols = list(m.columns)
    sp_col = next((c for c in cols if "species" in c.lower()), None)
    fam_col, ord_col = find(cols, "family"), find(cols, "order")
    if sp_col is None or fam_col is None or ord_col is None:
        raise SystemExit(f"could not find species/family/order columns in {cols} - send me this list")
    m = m[[sp_col, fam_col, ord_col]].dropna()
    m.columns = ["species_name", "Family", "Order"]
    m["species_name"] = m.species_name.astype(str).str.replace("_", " ").str.strip()
    m = m.drop_duplicates("species_name").set_index("species_name")

    fg = pd.read_csv(DATA / "guiana_teleo_clean.csv").drop_duplicates("species_name")
    fg = fg.set_index("species_name")[["Family", "Order"]]
    src = pd.concat([fg, m[~m.index.isin(fg.index)]])      # French Guiana labels win

    tax = tax.join(src, on="species_name")
    print(f"species: {len(tax)} | with family: {tax.Family.notna().sum()} "
          f"| with order: {tax.Order.notna().sum()}")
    tax.to_csv(DATA / "global_taxonomy.csv", index=False)
    print(f"saved {DATA / 'global_taxonomy.csv'}")


if __name__ == "__main__":
    main()
