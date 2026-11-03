#!/usr/bin/env python3
"""
Species delimitation on a gene tree (PTP, the model behind mPTP): DETECTION and GROUPING in one step.

Per split, one gene tree is built from the DNA of the reference AND test sequences together.
PTP then splits that tree into species: branches between species are long (speciation),
branches inside a species are short (coalescent). Each delimited species is a group.
  - DETECTION: a test sequence whose delimited species contains no reference sequence is "new".
  - GROUPING:  test sequences in the same delimited species form one group.

RULES, FIXED BEFORE RUNNING
  1. Tips = distinct Teleo sequences of the split (reference + test); identical sequences share a tip.
  2. Alignment: MAFFT --auto. Gene tree: FastTree -nt -gtr (default settings). Rooted at the midpoint.
  3. Delimitation: single-rate PTP (Zhang et al. 2013), solved EXACTLY by dynamic programming as in
     mPTP (Kapli et al. 2017, --single). Branches shorter than 0.0001 are ignored (mPTP's default
     minimum branch length). Maximum-likelihood delimitation is used.
  4. Detection call: new <=> no reference sequence in its delimited species. Otherwise the predicted
     species = the reference tip nearest on the gene tree inside its delimited species.
     Continuous score (secondary, for AUROC only) = gene-tree distance to the nearest reference tip.
  5. Grouping, two outputs:
       ptp      the sequences called new by B4 (same input as every other clustering file),
                grouped by their PTP species   -> fair comparison of the grouping step
       ptp_own  the sequences PTP itself calls new, grouped by PTP species (end-to-end, reported only)
     Naming: each member's nearest reference tip on the gene tree gives a species; the group takes the
     most common one (ties -> alphabetically first).
  Main comparisons: detection F1 and "new species named as known" vs VSEARCH (B2) and B4;
  grouping ARI / B-cubed F1 of `ptp` vs kmer, gap_align (clustering/score_clusters.py).

Teleo only. Tools: mafft and FastTree (on the PATH, or MAFFT= / FASTTREE= paths).
Usage (repo root):
    python clustering/ptp_delimit.py
    python clustering/score_clusters.py kmer gap_align ptp phylo_tree
Output: outputs/results/results_ptp.csv          (detection, one row per test sequence, all 30 splits)
        outputs/results/ptp_detection_per_split.csv, ptp_detection_tests.csv
        outputs/results/clusters_ptp[_extra].csv, clusters_ptp_own[_extra].csv
"""
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import dendropy
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.setrecursionlimit(100000)
DATA, OUT = ROOT / "data", ROOT / "outputs" / "results"
MINBR = 1e-4


def tool(names, env):
    for n in [os.environ.get(env)] + names:
        if n and shutil.which(n):
            return shutil.which(n)
    sys.exit(f"cannot find {names[0]}: install it or set {env}=/path/to/it")


MAFFT, FASTTREE = tool(["mafft"], "MAFFT"), tool(["FastTree", "fasttree"], "FASTTREE")


def run(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode != 0:
        sys.exit(f"command failed: {' '.join(map(str, cmd))}\n{r.stderr[-2000:]}")
    return r.stdout


# ------------------------------------------------------------------ PTP (exact, single rate)
def ptp(tree):
    """Maximum-likelihood single-rate PTP delimitation. Returns {leaf taxon label: species id}."""
    nodes = list(tree.postorder_node_iter())
    elen = {nd: (nd.edge.length or 0.0) for nd in nodes}
    counted = {nd: (nd.parent_node is not None and elen[nd] >= MINBR) for nd in nodes}
    E = sum(counted.values())
    L = sum(elen[nd] for nd in nodes if counted[nd])
    tab = {}                      # node -> (Smin, Smax, back_min, back_max); arrays indexed by k
    for nd in nodes:
        kids = nd.child_nodes()
        if not kids:
            tab[nd] = (np.zeros(1), np.zeros(1), None, None)
            continue
        # speciation option: every child edge is a speciation edge
        cur_min, cur_max = np.zeros(1), np.zeros(1)
        splits_min, splits_max = [], []
        for c in kids:
            cmin, cmax = tab[c][0], tab[c][1]
            if counted[c]:
                cmin = np.concatenate([[np.inf], cmin + elen[c]])
                cmax = np.concatenate([[-np.inf], cmax + elen[c]])
            n = len(cur_min) + len(cmin) - 1
            nmin, nmax = np.full(n, np.inf), np.full(n, -np.inf)
            amin, amax = np.zeros(n, int), np.zeros(n, int)
            for i in range(len(cur_min)):
                v = cur_min[i] + cmin
                better = v < nmin[i:i + len(cmin)]
                nmin[i:i + len(cmin)][better] = v[better]; amin[i:i + len(cmin)][better] = i
                v = cur_max[i] + cmax
                better = v > nmax[i:i + len(cmax)]
                nmax[i:i + len(cmax)][better] = v[better]; amax[i:i + len(cmax)][better] = i
            splits_min.append(amin); splits_max.append(amax)
            cur_min, cur_max = nmin, nmax
        # coalescent option (whole subtree is one species): k = 0, S = 0
        # (k = 0 means no speciation edge below this node, i.e. the whole subtree is one species)
        smin, smax = cur_min.copy(), cur_max.copy()
        coal_min = np.zeros(len(smin), bool); coal_max = np.zeros(len(smax), bool)
        coal_min[0] = coal_max[0] = True
        smin[0] = smax[0] = 0.0
        tab[nd] = (smin, smax, (coal_min, splits_min), (coal_max, splits_max))

    root = tree.seed_node

    def term(n, S):
        if n == 0:
            return 0.0
        return n * np.log(n / S) - n if S > 0 else -np.inf

    best = (-np.inf, 0, "min")
    Smin, Smax = tab[root][0], tab[root][1]
    for k in range(len(Smin)):
        for S, kind in [(Smin[k], "min"), (Smax[k], "max")]:
            if np.isfinite(S):
                ll = term(k, S) + term(E - k, L - S)
                if ll > best[0] + 1e-12:
                    best = (ll, k, kind)

    # backtrack: which nodes are coalescent roots (species)
    species, sid = {}, [0]

    def assign(nd, k, kind):
        kids = nd.child_nodes()
        if not kids:
            species[nd.taxon.label] = sid[0]; sid[0] += 1
            return
        coal, splits = tab[nd][2] if kind == "min" else tab[nd][3]
        if k == 0 and coal[0]:
            for lf in nd.leaf_iter():
                species[lf.taxon.label] = sid[0]
            sid[0] += 1
            return
        # undo the child-by-child convolution, last child first
        ks = []
        for c, sp in zip(reversed(kids), reversed(splits)):
            i = sp[k]
            ks.append((c, k - i))
            k = i
        for c, kc in reversed(ks):
            if counted[c]:
                kc -= 1
            assign(c, kc, kind)

    assign(root, best[1], best[2])
    return species


# ------------------------------------------------------------------ one split
def delimit_split(ref_ids, test_ids, clean, seqs, work):
    ids = np.concatenate([ref_ids, test_ids])
    uniq = sorted(set(seqs.loc[ids]))
    tip = {s: f"t{i}" for i, s in enumerate(uniq)}
    with open(work / "u.fa", "w") as fh:
        for s, t in tip.items():
            fh.write(f">{t}\n{s}\n")
    aln = run([MAFFT, "--auto", "--quiet", str(work / "u.fa")])
    (work / "aln.fa").write_text(aln)
    nwk = run([FASTTREE, "-nt", "-gtr", "-quiet", "-nopr", str(work / "aln.fa")])
    tree = dendropy.Tree.get(data=nwk, schema="newick", preserve_underscores=True)
    tree.reroot_at_midpoint(update_bipartitions=False)
    tree.resolve_polytomies()
    sp_of_tip = ptp(tree)
    pdm = tree.phylogenetic_distance_matrix()
    taxa = {t.label: t for t in tree.taxon_namespace}

    ref_sp = clean.loc[ref_ids, "species_name"].values
    ref_tips = {}
    for sid_, s in zip(ref_ids, ref_sp):
        ref_tips.setdefault(tip[seqs[sid_]], []).append(s)
    ref_tip_sp = {t: pd.Series(v).value_counts().sort_index().idxmax() for t, v in ref_tips.items()}
    ref_tip_list = list(ref_tip_sp)

    rows = []
    cache = {}
    for sid_ in test_ids:
        t = tip[seqs[sid_]]
        if t not in cache:
            d = np.array([pdm.distance(taxa[t], taxa[r]) for r in ref_tip_list])
            same = np.array([sp_of_tip[r] == sp_of_tip[t] for r in ref_tip_list])
            nearest = ref_tip_sp[ref_tip_list[int(d.argmin())]]
            if same.any():
                inside = np.where(same)[0]
                pred = ref_tip_sp[ref_tip_list[inside[int(d[inside].argmin())]]]
            else:
                pred = "novel"
            cache[t] = (pred, nearest, float(d.min()), sp_of_tip[t])
        pred, nearest, dmin, spid = cache[t]
        rows.append(dict(seq_id=sid_, predicted_species=pred, nearest_species=nearest,
                         novelty_score=dmin, ptp_species=spid))
    return pd.DataFrame(rows)


def name_groups(lab, nearest):
    guess = np.empty(len(lab), dtype=object)
    for c in np.unique(lab):
        m = lab == c
        vc = pd.Series(nearest[m]).value_counts()
        guess[m] = sorted(vc.index[vc == vc.max()])[0]
    return guess


# ------------------------------------------------------------------ detection scores
def detection_scores(df, splits, clean):
    out = []
    for k, g in df.groupby("split_id"):
        r = g.merge(splits[splits.split_id == k][["seq_id", "role"]], on="seq_id")
        r = r[r.role != "reference"]
        y = (r.role == "test_novel").values
        called = (r.predicted_species == "novel").values
        tp, fp, fn = (called & y).sum(), (called & ~y).sum(), (~called & y).sum()
        true_sp = clean.loc[r.seq_id, "species_name"].values
        row = dict(split=k, f1=2 * tp / max(2 * tp + fp + fn, 1),
                   novel_named_known=(~called[y]).mean(),
                   known_correct=((~called) & (r.predicted_species.values == true_sp))[~y].mean(),
                   known_called_new=called[~y].mean())
        if "novelty_score" in r and r.novelty_score.notna().all():
            row["auroc"] = roc_auc_score(y, r.novelty_score.values)
        out.append(row)
    return pd.DataFrame(out)


def main():
    clean = pd.read_csv(DATA / "guiana_teleo_clean.csv")
    clean["seq_id"] = clean["Specimen identification code"].astype(str)
    clean = clean[clean.in_tree].drop_duplicates("seq_id").set_index("seq_id")
    seqs = clean.teleo.str.strip().str.lower()
    info = clean.drop_duplicates("species_name").set_index("species_name")
    OUT.mkdir(parents=True, exist_ok=True)
    all_det = []
    for called_file, split_file, tag in [("results_kmer.csv", "splits.csv", ""),
                                         ("results_kmer_extra.csv", "splits_extra.csv", "_extra")]:
        b4 = pd.read_csv(ROOT / "outputs" / called_file, dtype={"seq_id": str})
        b4_new = b4[b4.predicted_species == "novel"]
        splits = pd.read_csv(DATA / split_file)
        splits["seq_id"] = splits.seq_id.astype(str)
        groups = {"ptp": [], "ptp_own": []}
        for k in sorted(splits.split_id.unique()):
            t0 = time.time()
            s = splits[splits.split_id == k]
            ref, test = s[s.role == "reference"].seq_id.values, s[s.role != "reference"].seq_id.values
            with tempfile.TemporaryDirectory() as tmp:
                det = delimit_split(ref, test, clean, seqs, Path(tmp))
            det.insert(1, "split_id", k)
            all_det.append(det)
            d = det.set_index("seq_id")
            for name, nov in [("ptp", b4_new[b4_new.split_id == k].seq_id.values),
                              ("ptp_own", d.index[d.predicted_species == "novel"].values)]:
                if len(nov) == 0:
                    continue
                lab = pd.factorize(d.loc[nov, "ptp_species"].values)[0]
                guess = name_groups(lab, d.loc[nov, "nearest_species"].values)
                groups[name].append(pd.DataFrame({"seq_id": nov, "split_id": k, "cluster_id": lab,
                                                  "guess_genus": info.loc[guess, "Genus"].values,
                                                  "guess_family": info.loc[guess, "Family"].values}))
            print(f"split {k:2d} | PTP species {d.ptp_species.nunique()} | called new {(d.predicted_species == 'novel').sum()} "
                  f"of {len(d)} test seqs ({time.time() - t0:.0f}s)", flush=True)
        for name, g in groups.items():
            pd.concat(g).to_csv(OUT / f"clusters_{name}{tag}.csv", index=False)
    det = pd.concat(all_det)
    det.to_csv(OUT / "results_ptp.csv", index=False)

    # detection: PTP vs VSEARCH (B2) and B4, same splits
    splits = pd.concat([pd.read_csv(DATA / "splits.csv"), pd.read_csv(DATA / "splits_extra.csv")])
    splits["seq_id"] = splits.seq_id.astype(str)
    per = {"PTP": detection_scores(det, splits, clean)}
    for name, f in [("VSEARCH", "results_vsearch"), ("B4 raw k-mers", "results_kmer")]:
        files = [ROOT / "outputs" / f"{f}.csv", ROOT / "outputs" / f"{f}_extra.csv"]
        if all(x.exists() for x in files):
            per[name] = detection_scores(pd.concat([pd.read_csv(x, dtype={"seq_id": str}) for x in files]),
                                         splits, clean)
    tab = pd.concat([v.assign(method=m) for m, v in per.items()])
    tab.to_csv(OUT / "ptp_detection_per_split.csv", index=False)
    print("\nDETECTION (mean over splits; auroc for PTP = gene-tree distance to nearest reference, secondary)")
    print(tab.groupby("method").mean(numeric_only=True).drop(columns="split").round(3).to_string())
    rows = []
    for b in [m for m in per if m != "PTP"]:
        A, B = per["PTP"].set_index("split"), per[b].set_index("split")
        for c in ["f1", "novel_named_known", "known_correct", "known_called_new", "auroc"]:
            if c in A and c in B:
                d = (A[c] - B[c]).dropna()
                p = wilcoxon(d).pvalue if (d != 0).any() else 1.0
                rows.append(dict(a="PTP", b=b, outcome=c, mean_a=A[c].mean(), mean_b=B[c].mean(),
                                 diff=d.mean(), n=len(d), wilcoxon_p=p))
    tests = pd.DataFrame(rows)
    tests.to_csv(OUT / "ptp_detection_tests.csv", index=False)
    print("\nPTP vs baselines (novel_named_known and known_called_new: lower = better)")
    print(tests.round(4).to_string(index=False))
    print("\nwrote results_ptp.csv, clusters_ptp[_extra].csv, clusters_ptp_own[_extra].csv, ptp_detection_*.csv")


if __name__ == "__main__":
    main()
