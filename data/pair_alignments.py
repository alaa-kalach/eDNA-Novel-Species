#!/usr/bin/env python3
"""
Where do two sequences differ? For every in-tree French Guiana sequence and its K closest
sequences, keep VSEARCH's own pairwise alignment (same settings as B2) and record, for each
difference, the shared POSITION it falls at, so differences can later be weighted by position
(models/mismatch_weights.py) without changing the alignment.

Two VSEARCH runs, both global alignment, plus strand, no masking:
  1. coordinates: every sequence aligned to one centre sequence (the medoid = highest mean
     identity in data/pairwise_identity_<marker>.npz; label-free). This only maps each base of
     each sequence to a shared position j = 0..L-1 (inserted bases -> the position before them,
     overhanging ends -> the first / last aligned position).
  2. differences: every sequence against all others, keeping its K best hits (--maxhits).
     For each pair, walk VSEARCH's alignment (terminal gaps excluded, as in VSEARCH's identity):
     every alignment column counts in the length; a column is a difference when the two bases
     are not compatible (IUPAC codes compatible = match, as VSEARCH counts them) or one is a gap.
     A difference is placed at the shared position of the query base (or of the target base
     when the query has the gap).
  => with every position weight = 1, distance = differences / alignment length = 1 - VSEARCH
     identity for the same pair (checked and printed at the end).

Usage (repo root):  python data/pair_alignments.py               # Teleo     (~1 min)
                    python data/pair_alignments.py --marker=full  # long 12S  (a few min)
Needs VSEARCH and data/pairwise_identity_<marker>.npz (python data/pairwise_identity.py).
Output: data/pair_alignments_<teleo|full12s>.npz
        seq_id (n,), L, centre_id, centre_seq, cov (n, L) bool (positions each sequence spans),
        pair_q, pair_t (indices into seq_id), aln_len, vsearch_id (0-1),
        diff_ptr, diff_pos  (CSR: positions of the differences of pair p =
                             diff_pos[diff_ptr[p]:diff_ptr[p+1]], one entry per difference)
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
sys.path.insert(0, str(ROOT / "baselines"))
sys.path.insert(0, str(ROOT / "data"))
from common import find_vsearch  # noqa: E402
from pairwise_identity import load_sequences  # noqa: E402

K = 100                     # best hits kept per sequence
IUPAC = {"A": "A", "C": "C", "G": "G", "T": "T", "U": "T", "R": "AG", "Y": "CT", "S": "CG",
         "W": "AT", "K": "GT", "M": "AC", "B": "CGT", "D": "AGT", "H": "ACT", "V": "ACG", "N": "ACGT"}
FIELDS = ["query", "target", "id", "qrow", "trow", "qilo", "qihi", "tilo", "tihi"]


def compatible(a, b):
    return bool(set(IUPAC.get(a, a)) & set(IUPAC.get(b, b)))


def vsearch(vs, query_fa, db_fa, out, extra):
    p = subprocess.run([vs, "--usearch_global", query_fa, "--db", db_fa, "--id", "0.0",
                        "--maxaccepts", "0", "--maxrejects", "0", "--minwordmatches", "0",
                        "--minseqlength", "1", "--strand", "plus", "--qmask", "none", "--dbmask", "none",
                        "--userout", out, "--userfields", "+".join(FIELDS),
                        "--threads", str(os.cpu_count() or 1), "--quiet"] + extra,
                       capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr[-2000:])
    return pd.read_csv(out, sep="\t", header=None, names=FIELDS, keep_default_na=False,
                       dtype={"query": str, "target": str, "qrow": str, "trow": str})


def centre_map(seqlen, qrow, trow, qilo, qihi, tilo, tihi, L):
    """shared position of every base of one sequence + the positions it spans."""
    pos = np.empty(seqlen, dtype=np.int32)
    pos[:qilo - 1] = tilo - 1                       # left overhang
    pos[qihi:] = tihi - 1                           # right overhang
    c, qi = tilo - 2, qilo - 1
    for qc, tc in zip(qrow, trow):
        if tc != "-":
            c += 1
        if qc != "-":
            pos[qi] = max(c, tilo - 1)
            qi += 1
    assert qi == qihi and c == tihi - 1, "centre alignment does not match its coordinates"
    cov = np.zeros(L, dtype=bool)
    cov[tilo - 1:tihi] = True
    return pos, cov


def main():
    opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    marker = {"teleo": "teleo", "full": "full12s", "full12s": "full12s"}[opts.get("marker", "teleo")]
    seq = load_sequences(marker)
    n = len(seq)
    z = np.load(DATA / f"pairwise_identity_{marker}.npz", allow_pickle=True)
    assert list(z["seq_id"]) == list(seq.index)
    c = int(np.argmax(z["ID"].mean(1)))
    centre_id, centre = seq.index[c], seq.iloc[c]
    L = len(centre)
    vs = find_vsearch()

    with tempfile.TemporaryDirectory() as tmp:
        allfa, cfa = os.path.join(tmp, "all.fa"), os.path.join(tmp, "c.fa")
        with open(allfa, "w", newline="\n") as fh:
            for i, s in enumerate(seq.values):
                fh.write(f">s{i}\n{s}\n")
        with open(cfa, "w", newline="\n") as fh:
            fh.write(f">centre\n{centre}\n")
        cen = vsearch(vs, allfa, cfa, os.path.join(tmp, "c.tsv"), [])
        print(f"aligned {len(cen)} sequences to the centre ({centre_id}, {L} positions)", flush=True)
        hits = vsearch(vs, allfa, allfa, os.path.join(tmp, "h.tsv"), ["--maxhits", str(K + 1)])
        print(f"{len(hits)} best-hit alignments", flush=True)

    if len(cen) != n:
        raise SystemExit(f"{n - len(cen)} sequences did not align to the centre")
    posmap, cov = [None] * n, np.zeros((n, L), dtype=bool)
    for r in cen.itertuples():
        i = int(r.query[1:])
        posmap[i], cov[i] = centre_map(len(seq.iloc[i]), r.qrow, r.trow, int(r.qilo), int(r.qihi),
                                       int(r.tilo), int(r.tihi), L)

    hits["q"] = hits["query"].str[1:].astype(int)
    hits["t"] = hits["target"].str[1:].astype(int)
    hits = hits[hits.q != hits.t].drop_duplicates(["q", "t"])
    pq, pt, alen, vid, ptr, dpos = [], [], [], [], [0], []
    agree = 0
    for r in hits.itertuples():
        qmap, tmap = posmap[r.q], posmap[r.t]
        qi, ti, nd = int(r.qilo) - 1, int(r.tilo) - 1, 0
        for qc, tc in zip(r.qrow, r.trow):
            if qc == "-":
                dpos.append(tmap[ti]); ti += 1; nd += 1
            elif tc == "-":
                dpos.append(qmap[qi]); qi += 1; nd += 1
            else:
                if not compatible(qc, tc):
                    dpos.append(qmap[qi]); nd += 1
                qi += 1; ti += 1
        ln = len(r.qrow)
        pq.append(r.q); pt.append(r.t); alen.append(ln); vid.append(float(r.id) / 100)
        ptr.append(len(dpos))
        agree += abs((1 - nd / ln) - float(r.id) / 100) < 0.0006      # VSEARCH prints 1 decimal
    path = DATA / f"pair_alignments_{marker}.npz"
    np.savez_compressed(path, seq_id=np.array(list(seq.index)), L=L, centre_id=centre_id, centre_seq=centre,
                        cov=cov, pair_q=np.array(pq, np.int32), pair_t=np.array(pt, np.int32),
                        aln_len=np.array(alen, np.int32), vsearch_id=np.array(vid),
                        diff_ptr=np.array(ptr, np.int64), diff_pos=np.array(dpos, np.int32))
    print(f"wrote {path}: {len(pq)} pairs ({K} best hits per sequence), {len(dpos)} differences")
    print(f"CHECK: unweighted distance = 1 - VSEARCH identity for {agree / len(pq):.2%} of pairs "
          f"(should be ~100%)")


if __name__ == "__main__":
    main()
