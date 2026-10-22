"""
Build the LONG-marker version of the data: the ~600 bp of 12S that end with the Teleo
region, for the French Guiana library and for the global MitoFish data.

Question it serves: Teleo (~64 bp) carries little tree signal. Does the tree beat
taxonomy when the marker is long enough to carry that signal?

Same species, same splits, same tree, same taxonomy as the Teleo runs -
only the DNA region changes.

Region: from up to UP bases before the Teleo forward primer, through the end of the
Teleo region (this is what the French Guiana library contains; UP = the median
upstream length found in the French Guiana sequences).

Inputs (data/):
    Data_12S Ref library_Fish Guiana.csv  original French Guiana library (12S sequence column)
    mitofish.12S.Oct2023.tsv              MitoFish (python data/check_mitofish.py downloads it)
    guiana_teleo_clean.csv, global_teleo.csv
Outputs (data/):
    guiana_full12s.csv          seq_id, seq          (French Guiana, long region)
    kmer_features_full12S.npz   seq_id, X            (k = 3..6 profiles of those)
    global_full12s.csv          species_name, seq, in_tree, in_guiana
                                (only species used for the Teleo pretraining)
Usage (from the repo root):  python -m pip install regex && python data/make_full12s.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import regex
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
sys.path.insert(0, str(ROOT))
from models.kmers import kmer_profile  # noqa: E402

FG_RAW = DATA / "Data_12S Ref library_Fish Guiana.csv"
TSV = DATA / "mitofish.12S.Oct2023.tsv"
TELEO_F = "acaccgcccgtcactct"
TELEO_R_RC = "catggtaagtgtaccggaag"
F_PAT = regex.compile(f"(?b)({TELEO_F}){{s<=3}}")
FULL_PAT = regex.compile(f"(?b)({TELEO_F}){{s<=3}}(.{{20,120}}?)({TELEO_R_RC}){{s<=3}}")


def pdist(A, B=None):
    """Euclidean distances without building a huge 3-D array."""
    B = A if B is None else B
    return np.sqrt(np.maximum((A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None] - 2 * A @ B.T, 0))


def clean_seq(s):
    """lower case, drop gaps/spaces, keep ambiguity letters as 'n' (so positions don't shift)"""
    s = regex.sub(r"[^a-z]", "", str(s).lower())
    return regex.sub(r"[^acgt]", "n", s)


def main():
    # ---------------- French Guiana: long region = everything up to the end of Teleo
    if not FG_RAW.exists():
        raise SystemExit(f"put the original library in {FG_RAW}")
    raw = pd.read_csv(FG_RAW, sep=None, engine="python")
    raw["seq_id"] = raw["Specimen identification code"].astype(str)
    raw = raw.dropna(subset=["12S sequence"]).drop_duplicates("seq_id").set_index("seq_id")
    clean = pd.read_csv(DATA / "guiana_teleo_clean.csv")
    ids = clean.loc[clean.in_tree, "Specimen identification code"].astype(str)
    fg, up_len = {}, []
    for sid in ids:
        s = clean_seq(raw.at[sid, "12S sequence"])
        m = F_PAT.search(s)
        fg[sid] = s                          # the library sequences end with the Teleo region
        if m is not None:
            up_len.append(m.start())
    UP = int(np.median(up_len))
    print(f"French Guiana: {len(fg)} / {len(ids)} sequences ({len(up_len)} with the Teleo primer found) "
          f"| bases before the Teleo primer: "
          f"median {UP}, range {min(up_len)}-{max(up_len)}")
    fg_df = pd.DataFrame({"seq_id": list(fg), "seq": list(fg.values())})
    fg_df.to_csv(DATA / "guiana_full12s.csv", index=False)
    X = np.array([kmer_profile(s) for s in fg_df.seq])
    np.savez(DATA / "kmer_features_full12S.npz", seq_id=fg_df.seq_id.values, X=X)
    print(f"saved guiana_full12s.csv and kmer_features_full12S.npz {X.shape}")

    # ---------------- global: same region, same species as the Teleo pretraining
    gt = pd.read_csv(DATA / "global_teleo.csv")
    keep_sp = set(gt.loc[gt.in_tree & ~gt.in_guiana, "species_name"])
    m = pd.read_csv(TSV, sep="\t", low_memory=False)
    sample = m.head(200).astype(str)
    seq_col = max(m.columns, key=lambda c: sample[c].str.lower().str.count("[acgt]").mean())
    sp_col = [c for c in m.columns if "species" in c.lower()][0]
    m = m[[sp_col, seq_col]].dropna()
    m.columns = ["species_name", "seq"]
    m["species_name"] = m.species_name.astype(str).str.replace("_", " ").str.strip()
    fg_species = set(clean.loc[clean.in_tree, "species_name"])
    m = m[m.species_name.isin(keep_sp | fg_species)]
    print(f"\nMitoFish records of the {len(keep_sp)} Teleo pretraining species "
          f"(+ French Guiana species, only for the check below): {len(m)}")

    out = []
    for sp, s in zip(m.species_name, m.seq.map(clean_seq)):
        hit = FULL_PAT.search(s)
        if hit is None:
            continue
        out.append((sp, s[max(0, hit.start() - UP):hit.start(3)]))   # ends where FG ends
    allg = pd.DataFrame(out, columns=["species_name", "seq"]).drop_duplicates()
    allg = allg[allg.seq.str.len() >= 300]
    check_same_species(allg[allg.species_name.isin(fg_species)], fg_df, clean)
    g = allg[allg.species_name.isin(keep_sp)].copy()
    g = g[g.seq.str.len() >= 300]                      # drop records that are mostly Teleo only
    g["in_tree"], g["in_guiana"] = True, False
    g.to_csv(DATA / "global_full12s.csv", index=False)
    print(f"global long region: {len(g)} sequences, {g.species_name.nunique()} species "
          f"(Teleo set: {len(keep_sp)} species) | length median {int(g.seq.str.len().median())}")
    print("saved global_full12s.csv")

    # ---------------- sanity check: is the tree signal stronger in the long region?
    tz = np.load(DATA / "global_tree_distances.npz")
    row = {s: i for i, s in enumerate(tz["species"])}
    fgc = clean[clean.in_tree].assign(seq_id=lambda d: d["Specimen identification code"].astype(str))
    fgc = fgc.drop_duplicates("seq_id").set_index("seq_id")
    for name, seqs in [("Teleo", fgc.teleo.str.lower()), ("long 12S", fg_df.set_index("seq_id").seq)]:
        P = pd.DataFrame([kmer_profile(x) for x in seqs], index=seqs.index)
        C = P.groupby(fgc.loc[P.index, "species_name"].values).mean()
        D = pdist(C.values)
        r = [row[x] for x in C.index]
        iu = np.triu_indices(len(C), 1)
        print(f"tree signal, French Guiana species, {name:8s}: {spearmanr(D[iu], tz['D'][np.ix_(r, r)][iu])[0]:.2f}")
    for name, d in [("Teleo", gt[gt.species_name.isin(keep_sp)].rename(columns={"teleo": "seq"})),
                    ("long 12S", g)]:
        one = d.drop_duplicates("species_name")
        one = one[one.species_name.isin(set(g.species_name))]      # same species for both
        pick = one.sample(n=min(400, len(one)), random_state=0)
        P = np.array([kmer_profile(s) for s in pick.seq])
        D = pdist(P)
        r = pick.species_name.map(row).values
        iu = np.triu_indices(len(pick), 1)
        rho = spearmanr(D[iu], tz["D"][np.ix_(r, r)][iu])[0]
        print(f"tree signal, global species (400 random), {name:8s}: {rho:.2f}")


def check_same_species(gfg, fg_df, clean):
    """MitoFish also has some French Guiana species. If the global cut is the same region
    as the French Guiana library, their sequences should be almost identical."""
    sp_of = clean.assign(seq_id=clean["Specimen identification code"].astype(str)) \
                 .drop_duplicates("seq_id").set_index("seq_id")["species_name"]
    fgs = fg_df.assign(species_name=fg_df.seq_id.map(sp_of)).dropna()
    common = sorted(set(gfg.species_name) & set(fgs.species_name))
    if not common:
        print("check: no French Guiana species in MitoFish long region")
        return
    A = np.array([kmer_profile(gfg[gfg.species_name == s].seq.iloc[0]) for s in common])
    B = np.array([kmer_profile(fgs[fgs.species_name == s].seq.iloc[0]) for s in common])
    D = pdist(A, B)
    same, other = np.diag(D), D[~np.eye(len(common), dtype=bool)]
    lens = gfg.seq.str.len()
    print(f"check ({len(common)} species in both): k-mer distance MitoFish vs French Guiana, "
          f"same species median {np.median(same):.3f} | different species median {np.median(other):.3f} "
          f"| same species is the closest for {(D.argmin(1) == np.arange(len(common))).mean():.0%}")
    print(f"       global long-region length: 5% {int(lens.quantile(.05))}, median {int(lens.median())}, "
          f"95% {int(lens.quantile(.95))} | French Guiana median {int(fgs.seq.str.len().median())}")


if __name__ == "__main__":
    main()
