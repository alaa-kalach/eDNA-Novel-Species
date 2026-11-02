#!/usr/bin/env python3
"""
Group the sequences called "novel" into putative new species (HDBSCAN) and guess
the genus and family of each group.

Steps, for each split:
  1. Which test sequences are novel?
       --novel-from=<results csv>  sequences whose predicted_species == "novel"
                                   (e.g. Alaa's outputs/results_kmer.csv or results_vsearch.csv)
       --novel-from=true           the true novel sequences (only to study clustering
                                   on its own, NOT for the final pipeline)
  2. Cluster them with HDBSCAN(min_cluster_size=2, min_samples=1) in the chosen space.
     Points HDBSCAN leaves alone get cluster_id = -1 (noise = a group of one).
  3. Name each group: the group's centre -> nearest reference species centre
     -> that species' genus and family (noise points are named one by one).

Spaces (--emb):
    kmer                    raw k-mer profiles (data/kmer_features_k3to6.npz) -> B4
    ft_<target>[_s<N>]      fine-tuned encoder embeddings saved by models/finetune_eval.py
                            in outputs/embeddings/ (e.g. ft_tree2g_w10m0.4, ft_labels, ft_taxonomy)
    --with-raw              use the SAME combined distance as the final placement method:
                            D = D_raw / median_raw + D_model / median_model (w = 1), medians =
                            median distance between reference species centres in each space.
                            Grouping: HDBSCAN on this distance between the novel sequences.
                            Naming: the group's mean combined distance to each reference species
                            centre -> nearest species. Output gets "_withraw" in its name.

Usage (from the repo root):
    python clustering/cluster.py --emb=ft_tree2g_w10m0.4 --novel-from=outputs/results_kmer.csv
    python clustering/cluster.py --emb=kmer --novel-from=outputs/results_kmer.csv
    python clustering/cluster.py --emb=ft_labels --novel-from=true --splits=splits_extra.csv --tag=_extra
    python clustering/cluster.py --emb=ft_treecap90 --with-raw --novel-from=outputs/results_kmer.csv
Output: outputs/results/clusters_<emb>[_withraw][_truenovel][<tag>].csv
        seq_id, split_id, cluster_id (-1 = noise), guess_genus, guess_family
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN

ROOT = Path(__file__).resolve().parents[1]
DATA, OUT, EMB = ROOT / "data", ROOT / "outputs" / "results", ROOT / "outputs" / "embeddings"
NOVEL_LABEL = "novel"
HDBSCAN_ARGS = dict(min_cluster_size=2, min_samples=1)


def load_meta():
    clean = pd.read_csv(DATA / "guiana_teleo_clean.csv")
    clean["seq_id"] = clean["Specimen identification code"].astype(str)
    clean = clean[clean.in_tree].drop_duplicates("seq_id").set_index("seq_id")
    k = np.load(DATA / "kmer_features_k3to6.npz", allow_pickle=True)
    raw = pd.DataFrame(k["X"], index=k["seq_id"].astype(str))
    return clean, raw


def load_space(emb, sid, raw):
    """DataFrame (seq_id -> vector) for one split."""
    if emb == "kmer":
        return raw
    z = np.load(EMB / f"embeddings_{emb}_split{sid}.npz", allow_pickle=True)
    return pd.DataFrame(z["emb"], index=z["seq_id"].astype(str))


def dist(A, B):
    return np.sqrt(np.maximum((A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None] - 2 * A @ B.T, 0))


def species_centres(V, ref_ids, ref_species):
    sps = np.unique(ref_species)
    return sps, np.array([V.loc[ref_ids[ref_species == s]].values.mean(0) for s in sps])


def median_centre_dist(C):
    return np.median(dist(C, C)[np.triu_indices(len(C), 1)])


def combined(raw, V, ref, rsp, nov):
    """raw + model distances (each / its median centre distance) for the novel sequences:
    (species, novel x novel distance, novel x species-centre distance)."""
    sps, Cr = species_centres(raw, ref, rsp)
    _, Cm = species_centres(V, ref, rsp)
    mr, mm = median_centre_dist(Cr), median_centre_dist(Cm)
    Xr, Xm = raw.loc[nov].values, V.loc[nov].values
    DD = dist(Xr, Xr) / mr + dist(Xm, Xm) / mm
    DD = (DD + DD.T) / 2                       # remove tiny rounding differences
    np.fill_diagonal(DD, 0)
    DC = dist(Xr, Cr) / mr + dist(Xm, Cm) / mm
    return sps, DD, DC


def main():
    opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    with_raw = "--with-raw" in sys.argv[1:]
    emb = opts.get("emb", "kmer")
    novel_from = opts.get("novel-from", "true")
    splits_file, tag = opts.get("splits", "splits.csv"), opts.get("tag", "")

    clean, raw = load_meta()
    info = clean.drop_duplicates("species_name").set_index("species_name")
    splits = pd.read_csv(DATA / splits_file)
    splits["seq_id"] = splits.seq_id.astype(str)
    if novel_from != "true":
        called = pd.read_csv(ROOT / novel_from, dtype={"seq_id": str})
        called = called[called.predicted_species == NOVEL_LABEL]

    rows = []
    for sid in sorted(splits.split_id.unique()):
        s = splits[splits.split_id == sid]
        ref = s[s.role == "reference"].seq_id.values
        if novel_from == "true":
            nov = s[s.role == "test_novel"].seq_id.values
        else:
            nov = called[called.split_id == sid].seq_id.values
            if len(nov) == 0 and sid not in set(called.split_id):
                raise SystemExit(f"split {sid} is missing from {novel_from}")
        if len(nov) == 0:
            continue
        V = load_space(emb, sid, raw)
        rsp = clean.loc[ref, "species_name"].values

        if with_raw:
            sps, DD, DC = combined(raw, V, ref, rsp, nov)
            # 1. cluster on the combined distance
            lab = (HDBSCAN(metric="precomputed", **HDBSCAN_ARGS).fit_predict(DD.astype(np.float64))
                   if len(nov) >= 2 else np.array([-1]))
        else:
            # 1. cluster
            lab = HDBSCAN(**HDBSCAN_ARGS).fit_predict(V.loc[nov].values) if len(nov) >= 2 else np.array([-1])
            sps, C = species_centres(V, ref, rsp)

        # 2. name each group (noise points one by one)
        guess = np.empty(len(nov), dtype=object)
        groups = [np.where(lab == c)[0] for c in np.unique(lab[lab >= 0])] + [[i] for i in np.where(lab < 0)[0]]
        for idx in groups:
            if with_raw:
                d = DC[idx].mean(0)
            else:
                d = dist(V.loc[nov[idx]].values.mean(0, keepdims=True), C)[0]
            guess[idx] = sps[np.argmin(d)]
        rows.append(pd.DataFrame({"seq_id": nov, "split_id": sid, "cluster_id": lab,
                                  "guess_genus": info.loc[guess, "Genus"].values,
                                  "guess_family": info.loc[guess, "Family"].values}))
        print(f"split {sid:2d} | {len(nov)} novel seqs -> {lab.max() + 1} groups + {(lab < 0).sum()} noise", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    extra = ("_withraw" if with_raw else "") + ("_truenovel" if novel_from == "true" else "")
    out = OUT / f"clusters_{emb}{extra}{tag}.csv"
    pd.concat(rows).to_csv(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
