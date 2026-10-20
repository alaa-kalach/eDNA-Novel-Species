"""
Shared helpers for the baselines (B1 forced assignment, B2 VSEARCH, B4 raw k-mers).

Conventions (shared team format, agreed with Person A):
  results_<method>[<tag>].csv : seq_id, split_id, novelty_score,
                                nearest_species, predicted_species
      - one row per TEST sequence (test_known + test_novel)
      - higher novelty_score = more likely novel
      - nearest_species  : ALWAYS filled - closest known (reference) species, used to
                           score genus/family placement of novel sequences
      - predicted_species: NOVEL_LABEL when called novel, otherwise the species name
                           (= nearest_species)
  clusters_<method>[<tag>].csv: seq_id, split_id, cluster_id  (predicted-novel rows only)

Splits (30 in total, same rule and seeds, data/make_splits.py):
  data/splits.csv        split 1-10   -> default, no tag
  data/splits_extra.csv  split 11-30  -> run with  --splits=splits_extra.csv --tag=_extra
"""
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
NOVEL_LABEL = "novel"
PSEUDO_FRAC = 0.20     # share of reference species hidden in each pseudo-novel split
PSEUDO_REPEATS = 5     # pseudo-novel splits per real split, pooled for threshold tuning


# ----------------------------------------------------------------------------- data
def parse_opts(argv=None):
    """--splits=<file in data/> (default splits.csv) and --tag=<suffix> (default '').
    Same options as Person A's scripts, e.g. --splits=splits_extra.csv --tag=_extra"""
    argv = sys.argv[1:] if argv is None else argv
    opts = dict(a[2:].split("=", 1) for a in argv if a.startswith("--") and "=" in a)
    return opts.get("splits", "splits.csv"), opts.get("tag", "")


def load_data(splits_file="splits.csv"):
    """Return (seqs, splits).
    seqs  : DataFrame indexed by seq_id with species_name, Genus, teleo (upper case)
    splits: data/<splits_file> with seq_id read as string
    """
    d = pd.read_csv(DATA / "guiana_teleo_clean.csv", dtype={"Specimen identification code": str})
    d = d[d["in_tree"]].rename(columns={"Specimen identification code": "seq_id"})
    d["teleo"] = d["teleo"].str.strip().str.upper()
    splits = pd.read_csv(DATA / splits_file, dtype={"seq_id": str, "difficulty": str})
    seqs = d.set_index("seq_id").loc[sorted(splits["seq_id"].unique()),
                                     ["species_name", "Genus", "teleo"]]
    return seqs, splits


def split_frames(seqs, splits, split_id):
    """Reference rows and test rows (test_known + test_novel) of one split."""
    s = splits[splits["split_id"] == split_id].set_index("seq_id")
    m = seqs.join(s[["role", "difficulty"]], how="inner")
    return m[m["role"] == "reference"].copy(), m[m["role"] != "reference"].copy()


def pseudo_split(ref, rng, frac=PSEUDO_FRAC):
    """Pseudo-novel split built ONLY from reference rows (same rules as make_splits.py).
    Returns (pseudo_ref, pseudo_query) where pseudo_query has a boolean column is_novel."""
    species = np.array(sorted(ref["species_name"].unique()))
    novel = set(rng.choice(species, size=int(round(frac * len(species))), replace=False))
    is_nov = ref["species_name"].isin(novel)
    query_idx = list(ref.index[is_nov])
    for sp, grp in ref[~is_nov].groupby("species_name", sort=True):
        u = sorted(grp["teleo"].unique())
        if len(u) >= 2:
            held = u[rng.integers(len(u))]
            query_idx += list(grp.index[grp["teleo"] == held])
    q = ref.loc[query_idx].copy()
    q["is_novel"] = q["species_name"].isin(novel)
    return ref.drop(index=query_idx), q


def pseudo_rng(split_id, r):
    return np.random.default_rng(100_000 + 100 * split_id + r)


def tune_threshold(scores, is_novel):
    """Threshold t maximising F1 of the novel class for the rule `novel if score >= t`.
    Returns (t, best_f1). t is placed halfway between the chosen score and the next
    lower one, so it does not sit exactly on a training value."""
    scores = np.asarray(scores, float)
    y = np.asarray(is_novel, bool)
    order = np.argsort(-scores, kind="stable")
    s, yy = scores[order], y[order]
    tp = np.cumsum(yy)
    fp = np.cumsum(~yy)
    last = np.r_[s[1:] != s[:-1], True]          # only cut between distinct values
    tp, fp, s_cut = tp[last], fp[last], s[last]
    f1 = 2 * tp / (tp + fp + y.sum())
    i = int(np.argmax(f1))
    lower = s_cut[i + 1] if i + 1 < len(s_cut) else s_cut[i] - 1e-6
    return float((s_cut[i] + lower) / 2), float(f1[i])


# -------------------------------------------------------------------------- vsearch
def find_vsearch():
    """VSEARCH binary: $VSEARCH, then PATH, then <repo>/tools/vsearch[.exe]."""
    cands = [os.environ.get("VSEARCH"), shutil.which("vsearch"),
             ROOT / "tools" / "vsearch.exe", ROOT / "tools" / "vsearch"]
    for c in cands:
        if c and Path(c).is_file():
            return str(c)
    raise FileNotFoundError(
        "VSEARCH not found. Put vsearch(.exe) in <repo>/tools/, add it to PATH, "
        "or set the VSEARCH environment variable to its full path.")


def write_fasta(df, path):
    with open(path, "w", newline="\n") as fh:
        for sid, seq in zip(df.index, df["teleo"]):
            fh.write(f">{sid}\n{seq}\n")


def _run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"VSEARCH failed ({p.returncode}):\n{p.stderr[-2000:]}")


def vsearch_best_hits(query, ref, vsearch, threads=0):
    """Exhaustive global alignment of every query against every reference sequence.
    Returns DataFrame indexed like `query` with best_id (0-1) and best_species.
    Ties at the best identity -> species with most tied hits, then alphabetical."""
    with tempfile.TemporaryDirectory() as tmp:
        qf, rf, of = (os.path.join(tmp, x) for x in ("q.fa", "r.fa", "hits.tsv"))
        write_fasta(query, qf)
        write_fasta(ref, rf)
        _run([vsearch, "--usearch_global", qf, "--db", rf, "--id", "0.0",
              "--maxaccepts", "0", "--maxrejects", "0", "--top_hits_only",
              # 0 = no shared-word prefilter, so distant queries still get aligned
              "--minwordmatches", "0",
              "--strand", "plus", "--qmask", "none", "--dbmask", "none",
              "--userout", of, "--userfields", "query+target+id",
              "--output_no_hits", "--threads", str(threads), "--quiet"])
        hits = pd.read_csv(of, sep="\t", header=None, names=["q", "t", "id"],
                           dtype={"q": str, "t": str})
    sp_of = ref["species_name"]
    hits = hits[hits["t"] != "*"]
    hits["sp"] = sp_of.reindex(hits["t"]).values
    best = {}
    for q, g in hits.groupby("q", sort=False):
        counts = g["sp"].value_counts()
        top = sorted(counts.index[counts == counts.max()])[0]
        best[q] = (g["id"].max() / 100.0, top)
    out = pd.DataFrame(index=query.index)
    out["best_id"] = [best.get(q, (0.0, None))[0] for q in query.index]
    out["best_species"] = [best.get(q, (0.0, None))[1] for q in query.index]
    return out


def vsearch_cluster(seqs_df, identity, vsearch, threads=0):
    """Greedy clustering (cluster_fast) at `identity` (0-1). Returns Series seq_id -> cluster."""
    if len(seqs_df) == 0:
        return pd.Series(dtype=int)
    with tempfile.TemporaryDirectory() as tmp:
        f, uc = os.path.join(tmp, "n.fa"), os.path.join(tmp, "n.uc")
        write_fasta(seqs_df, f)
        _run([vsearch, "--cluster_fast", f, "--id", f"{identity:.6f}",
              "--strand", "plus", "--qmask", "none", "--uc", uc,
              "--threads", str(threads), "--quiet"])
        uc_df = pd.read_csv(uc, sep="\t", header=None, dtype={8: str})
    uc_df = uc_df[uc_df[0].isin(["S", "H"])]
    return pd.Series(uc_df[1].astype(int).values, index=uc_df[8].values)


# ------------------------------------------------------------------------- reporting
def results_frame(test, split_id, novelty, nearest, is_novel):
    """Shared results format. nearest = closest known species (always filled);
    is_novel = boolean call -> predicted_species is NOVEL_LABEL or the nearest species."""
    nearest = np.asarray(nearest, dtype=object)
    if pd.isna(nearest).any():
        raise ValueError("nearest_species must be filled for every test sequence")
    is_novel = np.broadcast_to(np.asarray(is_novel, bool), nearest.shape)
    return pd.DataFrame({"seq_id": test.index, "split_id": split_id,
                         "novelty_score": np.round(np.broadcast_to(
                             np.asarray(novelty, float), nearest.shape), 6),
                         "nearest_species": nearest,
                         "predicted_species": np.where(is_novel, NOVEL_LABEL, nearest)})


def quick_summary(res, test):
    """One-line sanity summary (not the official evaluation)."""
    t = test.loc[res["seq_id"]]
    pred = res["predicted_species"].values
    nov = (t["role"] == "test_novel").values
    called = pred == NOVEL_LABEL
    tp = (called & nov).sum()
    f1 = 2 * tp / (called.sum() + nov.sum()) if (called.sum() + nov.sum()) else 0.0
    known_acc = (pred[~nov] == t["species_name"].values[~nov]).mean()
    misassigned = (~called & nov).sum() / max(nov.sum(), 1)
    return f"novel-F1 {f1:.3f} | novel->known {misassigned:.1%} | known acc {known_acc:.3f}"
