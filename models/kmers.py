"""
k-mer frequency profiles for k = 3..6 (as in the proposal).

Each sequence becomes one long vector: the frequencies of all 3-mers, then all
4-mers, 5-mers and 6-mers (64 + 256 + 1024 + 4096 = 5440 numbers). Each block
sums to 1. Words containing N or other ambiguity letters are skipped.

Usage (from the repo root):  python models/kmers.py
Output: data/kmer_features_k3to6.npz   arrays seq_id (n,), X (n, 5440), ks
"""
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
KS = (3, 4, 5, 6)


def kmer_profile(seq, ks=KS):
    parts = []
    for k in ks:
        index = {"".join(p): i for i, p in enumerate(itertools.product("acgt", repeat=k))}
        v = np.zeros(len(index))
        for i in range(len(seq) - k + 1):
            j = index.get(seq[i:i + k])
            if j is not None:
                v[j] += 1
        parts.append(v / max(v.sum(), 1))
    return np.concatenate(parts)


def main():
    df = pd.read_csv(ROOT / "data" / "guiana_teleo_clean.csv")
    df = df[df["in_tree"]].reset_index(drop=True)
    X = np.array([kmer_profile(s.strip().lower()) for s in df["teleo"]])
    out = ROOT / "data" / "kmer_features_k3to6.npz"
    np.savez(out, seq_id=df["Specimen identification code"].astype(str).values,
             X=X, ks=np.array(KS))
    print(f"features {X.shape} -> {out}")


if __name__ == "__main__":
    main()
