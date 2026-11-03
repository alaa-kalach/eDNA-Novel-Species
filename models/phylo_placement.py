#!/usr/bin/env python3
"""
Phylogenetic placement of test sequences directly ON the Fish Tree of Life (EPA-ng).

Idea: so far the tree was only used indirectly (an embedding trained on tree distances).
Here the tree itself is the reference: each test sequence is grafted onto the Fish Tree
of the reference species, using a model of DNA evolution.
  - WHERE it attaches  -> its nearest reference species -> guessed genus / family (placement)
  - HOW LONG its own new branch is (pendant length) -> how different it is from everything
    on the tree (novelty score for detection)

RULES, FIXED BEFORE ANY RUN (do not change after seeing results)
  Per split, reference data only:
  1. One representative sequence per reference species = its most common reference sequence
     (ties -> alphabetically first).
  2. Align the representatives with MAFFT (--auto); add every distinct test sequence to that
     alignment with `mafft --addfragments --keeplength` (the reference columns stay fixed).
  3. Topology = the Fish Tree pruned to the reference species (shape kept as it is).
     Branch lengths are re-estimated from the 12S alignment with RAxML-NG --evaluate,
     model GTR+G, seed 1.
  4. EPA-ng places each test sequence (default settings). The placement with the highest
     likelihood weight ratio (LWR) is used.
  5. Guessed species = the reference leaf closest (path length on the re-fitted tree) to the
     attachment point.
  6. Novelty scores: MAIN = pendant length (length of the new branch to the test sequence);
     secondary = pendant length + path from the attachment point to the nearest leaf.
  Controls computed on the SAME alignment: "alignment nearest" = the reference sequence with
  the highest identity (aligned positions where both have a base; ties -> first), its species
  = guess, 1 - identity = novelty score (a VSEARCH-like control).

Outcomes (same definitions as models/placement_kmer_tree.py):
  placement, truly novel sequences: family, family_easy, genus_hard, tree_error(_easy) (My)
  known_sp: species correct for test_known sequences
  detection: AUROC of each novelty score, test_novel vs test_known
Comparisons (paired Wilcoxon over the 30 splits, Holm within the primary family):
  primary   : phylo vs raw+taxonomy (placement; from placement_kmer_tree[_full]_per_split.csv,
              5-seed average), phylo vs alignment nearest (placement: same alignment, with vs
              without the tree = the tree's own effect), and phylo pendant vs alignment
              identity (detection AUROC)
  secondary : phylo vs raw, phylo vs raw+treeclade, phylo vs B4 (placement),
              phylo tree distance / pendant vs VSEARCH (Teleo only, outputs/results_vsearch*.csv)

Tools: mafft, raxml-ng, epa-ng on the PATH, or set MAFFT=, RAXML=, EPANG= to their paths.
Python: pip install dendropy
Usage (repo root):
    python models/phylo_placement.py                  # Teleo, all 30 splits
    python models/phylo_placement.py --marker=full    # long 12S
    python models/phylo_placement.py --splits=1,2     # quick check on two splits
    add --threads=8 to use more CPU cores (default 4)
Output: outputs/results/phylo_placement[_full]_per_seq.csv   (one row per test sequence)
        outputs/results/phylo_placement[_full]_per_split.csv
        outputs/results/phylo_placement[_full]_tests.csv
"""
import json
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
sys.path.insert(0, str(ROOT / "models"))
import placement_kmer_tree as pk  # noqa: E402  (outcomes, split_age, LOWER_BETTER; reads --marker)

DATA, OUT = ROOT / "data", ROOT / "outputs" / "results"
MARKER, MK = pk.MARKER, pk.MK
OUTCOMES = ["family", "family_easy", "genus_hard", "known_sp", "tree_error", "tree_error_easy"]


def tool(name, env):
    path = os.environ.get(env) or shutil.which(name)
    if not path:
        sys.exit(f"cannot find {name}: install it (see the header) or set {env}=/path/to/{name}")
    return path


MAFFT, RAXML, EPANG = tool("mafft", "MAFFT"), tool("raxml-ng", "RAXML"), tool("epa-ng", "EPANG")
THREADS = "4"                                  # --threads=<n> to change (speed only, same results)


def run(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode != 0:
        sys.exit(f"command failed: {' '.join(map(str, cmd))}\n{r.stderr[-2000:]}")
    return r.stdout


# ------------------------------------------------------------------ data
def load():
    clean = pd.read_csv(DATA / "guiana_teleo_clean.csv")
    clean["seq_id"] = clean["Specimen identification code"].astype(str)
    clean = clean[clean.in_tree].drop_duplicates("seq_id").set_index("seq_id")
    if MARKER == "full":
        f = pd.read_csv(DATA / "guiana_full12s.csv", dtype={"seq_id": str}).set_index("seq_id")
        seqs = f.seq.str.lower()
    else:
        seqs = clean.teleo.str.strip().str.lower()
    seqs = seqs[seqs.index.isin(clean.index)]
    splits = pd.concat([pd.read_csv(DATA / "splits.csv"), pd.read_csv(DATA / "splits_extra.csv")])
    splits["seq_id"] = splits.seq_id.astype(str)
    return clean, seqs, splits, load_tree()


def load_tree():
    return dendropy.Tree.get(path=str(DATA / "fish_tree_full.tre"), schema="newick", preserve_underscores=True)


def write_fasta(path, names, seqs):
    with open(path, "w") as fh:
        for n, s in zip(names, seqs):
            fh.write(f">{n}\n{s}\n")


def read_fasta(path):
    out, name = {}, None
    for line in open(path):
        line = line.strip()
        if line.startswith(">"):
            name = line[1:].split()[0]
            out[name] = []
        elif name:
            out[name].append(line)
    return {k: "".join(v).lower() for k, v in out.items()}


# ------------------------------------------------------------------ jplace tree
class Node:
    __slots__ = ("name", "length", "edge", "parent", "kids")

    def __init__(self):
        self.name, self.length, self.edge, self.parent, self.kids = None, 0.0, None, None, []


def parse_jplace_tree(s):
    """Newick with {edge_num} after each branch length (jplace format)."""
    s = s.strip().rstrip(";")
    root, cur, i, n = Node(), None, 0, len(s)
    cur = root
    stack = []
    while i < n:
        c = s[i]
        if c == "(":
            child = Node(); child.parent = cur; cur.kids.append(child)
            stack.append(cur); cur = child; i += 1
        elif c == ",":
            parent = stack[-1]
            child = Node(); child.parent = parent; parent.kids.append(child)
            cur = child; i += 1
        elif c == ")":
            cur = stack.pop(); i += 1
        elif c == ":":
            j = i + 1
            while j < n and s[j] not in ",(){}[]":
                j += 1
            cur.length = float(s[i + 1:j]); i = j
        elif c == "{":
            j = s.index("}", i)
            cur.edge = int(s[i + 1:j]); i = j + 1
        elif c == "[":
            i = s.index("]", i) + 1
        else:
            j = i
            while j < n and s[j] not in ",():{}[]":
                j += 1
            cur.name = s[i:j].strip().strip("'"); i = j
    return root


def nearest_leaf_tables(root):
    """down[v] = (dist, leaf) nearest leaf inside v's subtree; out[v] = nearest leaf outside it,
    measured from v (through its parent)."""
    order, stack = [], [root]
    while stack:
        v = stack.pop(); order.append(v); stack.extend(v.kids)
    down = {}
    for v in reversed(order):
        down[v] = (0.0, v.name) if not v.kids else min((k.length + down[k][0], down[k][1]) for k in v.kids)
    out = {root: (np.inf, None)}
    for v in order:
        for k in v.kids:
            opts = [out[v]] + [(s.length + down[s][0], down[s][1]) for s in v.kids if s is not k]
            best = min(opts, key=lambda x: x[0])
            out[k] = (k.length + best[0], best[1])
    return down, out


# ------------------------------------------------------------------ one split
def place_split(ref_ids, test_ids, clean, seqs, tree, work, positions=False):
    """positions=True also returns where each test sequence attaches: {seq_id: (node, distal, pendant,
    query name)} and the root of the re-fitted tree (used by clustering/tree_grouping.py)."""
    rsp = clean.loc[ref_ids, "species_name"].values
    rseq = seqs.loc[ref_ids].values
    reps = {}
    for sp in sorted(set(rsp)):
        vc = pd.Series(rseq[rsp == sp]).value_counts()
        reps[sp] = sorted(vc.index[vc == vc.max()])[0]
    labels = [sp.replace(" ", "_") for sp in reps]
    write_fasta(work / "ref.fa", labels, reps.values())
    aln = run([MAFFT, "--auto", "--quiet", "--thread", THREADS, str(work / "ref.fa")])
    (work / "ref_aln.fa").write_text(aln)

    tseq = seqs.loc[test_ids]
    uniq = sorted(set(tseq.values))
    qname = {s: f"q{i}" for i, s in enumerate(uniq)}
    write_fasta(work / "q.fa", qname.values(), qname.keys())
    full = run([MAFFT, "--addfragments", str(work / "q.fa"), "--keeplength", "--quiet", "--thread", THREADS,
                str(work / "ref_aln.fa")])
    (work / "all_aln.fa").write_text(full)
    A = read_fasta(work / "all_aln.fa")
    write_fasta(work / "ref_only.fa", labels, [A[l] for l in labels])
    write_fasta(work / "q_only.fa", qname.values(), [A[q] for q in qname.values()])

    sub = tree.extract_tree_with_taxa_labels(labels=labels)
    sub.write(path=str(work / "topo.nwk"), schema="newick", suppress_rooting=True, unquoted_underscores=True,
              suppress_internal_node_labels=True)
    run([RAXML, "--evaluate", "--msa", str(work / "ref_only.fa"), "--tree", str(work / "topo.nwk"),
         "--model", "GTR+G", "--prefix", str(work / "rx"), "--seed", "1", "--threads", "1", "--redo",
         "--log", "ERROR"])
    run([EPANG, "--tree", str(work / "rx.raxml.bestTree"), "--ref-msa", str(work / "ref_only.fa"),
         "--query", str(work / "q_only.fa"), "--model", str(work / "rx.raxml.bestModel"),
         "--outdir", str(work), "--redo", "--threads", THREADS])
    jp = json.load(open(work / "epa_result.jplace"))
    root = parse_jplace_tree(jp["tree"])
    down, out = nearest_leaf_tables(root)
    by_edge, stack = {}, [root]
    while stack:
        v = stack.pop(); stack.extend(v.kids)
        if v.edge is not None:
            by_edge[v.edge] = v
    f = jp["fields"]
    iE, iW, iD, iP = f.index("edge_num"), f.index("like_weight_ratio"), f.index("distal_length"), \
        f.index("pendant_length")
    res, where = {}, {}
    for pl in jp["placements"]:
        best = max(pl["p"], key=lambda x: x[iW])
        v, a, pend = by_edge[best[iE]], best[iD], best[iP]          # a = distance attachment -> child node
        cand = [(a + down[v][0], down[v][1]), (max(v.length - a, 0) + out[v][0] - v.length, out[v][1])]
        d, leaf = min(cand, key=lambda x: x[0])
        names = [x[0] for x in pl["nm"]] if "nm" in pl else pl["n"]
        for q in names:
            res[q] = dict(phylo_species=leaf.replace("_", " "), pendant=pend, tree_dist=pend + d,
                          lwr=best[iW])
            where[q] = (v, a, pend)

    # alignment-nearest control (same alignment, reference representatives + all reference seqs)
    ref_full = {}
    for sid, sp in zip(ref_ids, rsp):
        ref_full.setdefault(seqs[sid], sp)
    rs = list(reps.items())
    R = np.array([list(A[sp.replace(" ", "_")]) for sp, _ in rs])
    Rb = R != "-"
    rows = []
    for sid in test_ids:
        s = tseq[sid]
        r = res[qname[s]]
        q = np.array(list(A[qname[s]]))
        both = Rb & (q != "-")[None, :]
        ident = ((R == q[None, :]) & both).sum(1) / np.maximum(both.sum(1), 1)
        j = int(np.argmax(ident))
        rows.append(dict(seq_id=sid, **r, align_species=rs[j][0], align_novelty=1 - ident[j]))
    if positions:
        pos = {sid: (*where[qname[tseq[sid]]], qname[tseq[sid]]) for sid in test_ids}
        return pd.DataFrame(rows), pos, root
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ scoring
def score_split(df, test, clean, info):
    t = test.set_index("seq_id").loc[df.seq_id].reset_index()
    y = (t.role == "test_novel").values
    out = {}
    for m, col in [("phylo", "phylo_species"), ("alignment nearest", "align_species")]:
        out[m] = pk.outcomes(df[col].values, t, clean, info)
    for m, col in [("phylo pendant", "pendant"), ("phylo tree distance", "tree_dist"),
                   ("alignment identity", "align_novelty")]:
        out.setdefault(m, {})["auroc"] = roc_auc_score(y, df[col].values)
    return out


def holm(p):
    p = np.asarray(p, float)
    o, adj, run_ = np.argsort(p), np.empty(len(p)), 0.0
    for r, i in enumerate(o):
        run_ = max(run_, (len(p) - r) * p[i]); adj[i] = min(run_, 1.0)
    return adj


def compare(per, a, b, outs, family, rng):
    A, B = per[per.method == a].set_index("split"), per[per.method == b].set_index("split")
    rows = []
    for c in outs:
        d = (A[c] - B[c]).dropna()
        if len(d) < 2:
            continue
        boots = [rng.choice(d.values, len(d)).mean() for _ in range(5000)]
        p = wilcoxon(d.values).pvalue if (d != 0).any() else 1.0
        aw, bw = ((d < 0), (d > 0)) if c in pk.LOWER_BETTER else ((d > 0), (d < 0))
        rows.append(dict(family=family, a=a, b=b, outcome=c, mean_a=A[c].mean(), mean_b=B[c].mean(),
                         diff=d.mean(), ci_lo=np.percentile(boots, 2.5), ci_hi=np.percentile(boots, 97.5),
                         a_better=int(aw.sum()), b_better=int(bw.sum()), n=len(d), wilcoxon_p=p))
    return rows


def main():
    global THREADS
    opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    THREADS = opts.get("threads", THREADS)
    clean, seqs, splits, tree = load()
    info = clean.drop_duplicates("species_name").set_index("species_name")
    sids = sorted(splits.split_id.unique())
    if "splits" in opts:
        sids = [int(x) for x in opts["splits"].split(",")]
    OUT.mkdir(parents=True, exist_ok=True)
    per_seq, rows = [], []
    for k in sids:
        t0 = time.time()
        s = splits[splits.split_id == k]
        s = s[s.seq_id.isin(seqs.index)]
        ref, test = s[s.role == "reference"], s[s.role != "reference"]
        with tempfile.TemporaryDirectory() as tmp:
            df = place_split(ref.seq_id.values, test.seq_id.values, clean, seqs, tree, Path(tmp))
        df.insert(1, "split_id", k)
        per_seq.append(df)
        for m, o in score_split(df, test, clean, info).items():
            rows.append(dict(split=k, method=m, **o))
        print(f"[{MARKER}] split {k:2d} done ({time.time() - t0:.0f}s)", flush=True)
    pd.concat(per_seq).to_csv(OUT / f"phylo_placement{MK}_per_seq.csv", index=False)
    per = pd.DataFrame(rows)

    # comparison methods from earlier runs (same splits)
    pkf = ROOT / "outputs" / f"placement_kmer_tree{MK}_per_split.csv"
    if pkf.exists():
        old = pd.read_csv(pkf)
        keep = [("raw", 0.0), ("raw+taxonomy", 1.0), ("raw+treeclade", 1.0), ("B4 nearest sequence", 0.0)]
        old = pd.concat([old[(old.method == m) & (old.w == w)] for m, w in keep])
        per = pd.concat([per, old[old.split.isin(sids)].drop(columns="w")], ignore_index=True)
    else:
        print(f"note: {pkf.name} not found - no raw / raw+taxonomy comparison")
    if MARKER == "teleo":
        vf = [ROOT / "outputs" / "results_vsearch.csv", ROOT / "outputs" / "results_vsearch_extra.csv"]
        if all(f.exists() for f in vf):
            v = pd.concat([pd.read_csv(f, dtype={"seq_id": str}) for f in vf])
            for k in sids:
                x = v[v.split_id == k].merge(splits[splits.split_id == k][["seq_id", "role"]], on="seq_id")
                x = x[x.role != "reference"]
                per = pd.concat([per, pd.DataFrame([dict(split=k, method="VSEARCH",
                                 auroc=roc_auc_score(x.role == "test_novel", x.novelty_score))])], ignore_index=True)
    per.to_csv(OUT / f"phylo_placement{MK}_per_split.csv", index=False)

    rng = np.random.default_rng(0)
    have = set(per.method)
    prim = []
    if "raw+taxonomy" in have:
        prim += compare(per, "phylo", "raw+taxonomy", OUTCOMES, "primary", rng)
    prim += compare(per, "phylo", "alignment nearest", OUTCOMES, "primary", rng)   # the tree's own effect
    prim += compare(per, "phylo pendant", "alignment identity", ["auroc"], "primary", rng)
    sec = []
    for b in ["raw", "raw+treeclade", "B4 nearest sequence"]:
        if b in have:
            sec += compare(per, "phylo", b, OUTCOMES, "secondary", rng)
    for b in ["VSEARCH", "alignment identity"]:
        if b in have:
            sec += compare(per, "phylo tree distance", b, ["auroc"], "secondary", rng)
    if "VSEARCH" in have:
        sec += compare(per, "phylo pendant", "VSEARCH", ["auroc"], "secondary", rng)
    tests = pd.DataFrame(prim + sec, columns=["family", "a", "b", "outcome", "mean_a", "mean_b", "diff", "ci_lo",
                                              "ci_hi", "a_better", "b_better", "n", "wilcoxon_p"])
    tests["holm_p"] = np.nan
    for fam in ["primary", "secondary"]:
        m = tests.family == fam
        if m.any():
            tests.loc[m, "holm_p"] = holm(tests.loc[m, "wilcoxon_p"].values)
    tests.to_csv(OUT / f"phylo_placement{MK}_tests.csv", index=False)

    cols = OUTCOMES + ["auroc"]
    order = [m for m in ["phylo", "alignment nearest", "raw", "B4 nearest sequence", "raw+taxonomy", "raw+treeclade",
                         "phylo pendant", "phylo tree distance", "alignment identity", "VSEARCH"] if m in have]
    print(f"\n[{MARKER}] {len(sids)} splits | mean over splits (tree_error in My, lower = better)")
    print(per.groupby("method")[[c for c in cols if c in per]].mean().loc[order].round(3).to_string())
    for fam in ["primary", "secondary"]:
        t = tests[tests.family == fam]
        print(f"\n{fam} comparisons (Holm within this family)")
        print(t[["a", "b", "outcome", "mean_a", "mean_b", "diff", "ci_lo", "ci_hi", "a_better", "b_better",
                 "wilcoxon_p", "holm_p"]].round(4).to_string(index=False))
    print(f"\nwrote outputs/results/phylo_placement{MK}_per_seq.csv, _per_split.csv, _tests.csv")


if __name__ == "__main__":
    main()
