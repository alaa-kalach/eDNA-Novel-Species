#!/usr/bin/env python3
"""
Turn the raw Illumina reads of Cilleros et al. (2019) into a table of Teleo DNA sequences per
sample, so our pipeline can run on REAL water samples.

Data: Dryad doi:10.5061/dryad.dc25730, one folder per run (Run1, Run2, ...) with
  *_R1.fastq.tar.gz, *_R2.fastq.tar.gz and a tag file (Fish_Guiana_Tag*.dat, OBITools ngsfilter
  format: experiment, PCR name, 8-letter tag, forward primer, reverse primer).
Read layout (checked on Run1): [0-4 spacer letters][tag][Teleo forward primer][insert]
                                [reverse complement of reverse primer][reverse complement of tag][spacer]
The same tag sits on both ends (one tag per PCR replicate).

RULES, FIXED BEFORE RUNNING (standard eDNA practice; written before looking at any result)
  1. Merge read pairs: vsearch --fastq_mergepairs (staggered merges allowed, overlap >= 20,
     <= 10 mismatches), then keep merged reads with expected errors <= 1 (vsearch --fastq_maxee 1).
  2. Find the primers in either orientation: the 12 last letters of each primer must match exactly,
     the whole primer with <= 2 mismatches. Tag = the 8 letters before the forward primer; the
     8 letters after the reverse primer must be its reverse complement; both must equal a tag
     of the tag file exactly (no mismatch), else the read is dropped.
  3. Insert = letters between the primers; keep 20-100 letters, A/C/G/T only.
  4. Denoise all PCRs of the run together: vsearch --cluster_unoise (minsize 8, alpha 2), then
     remove chimeras (vsearch --uchime3_denovo). Each sequence variant kept = one "ZOTU".
     Every insert is then mapped to its ZOTU with vsearch --usearch_global --id 0.97.
  5. Per PCR: ZOTU counts. Control PCRs (C_PCR, CPCR, C-PCR, Cext...: regex ^C(?:_|-|PCR|ext)) give a per-ZOTU
     contamination level = the largest count in any control; counts at or below it are set to 0.
  6. A ZOTU is kept in a SAMPLE (PCR name without its _<replicate> suffix) if it has reads in at
     least 2 of the sample's PCR replicates.
Usage (repo root):
    python real_edna/process_reads.py data/real_edna/raw/Run1
    python real_edna/process_reads.py data/real_edna/raw/Run1 --max-reads=200000   # quick test
    python real_edna/process_reads.py data/real_edna/raw/Run2 --refilter   # redo rules 5-6 only
Needs vsearch on the PATH (micromamba install -p ~/phylo-tools -c bioconda vsearch).
Output (data/real_edna/processed/<run>/):
    zotus.fasta            ZOTU sequences (>zotu_<n>;size=<reads>)
    zotu_pcr_counts.csv    zotu, pcr, count   (after the control filter)
    zotu_samples.csv       zotu, sample, n_pcr, reads   (kept by rule 6)
    summary.txt            reads kept at every step
"""
import collections
import gzip
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

import pandas as pd

COMP = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def rc(s):
    return s.translate(COMP)[::-1]


def vsearch():
    v = os.environ.get("VSEARCH") or shutil.which("vsearch")
    if not v:
        sys.exit("vsearch not found: micromamba install -y -p ~/phylo-tools -c conda-forge -c bioconda vsearch")
    return v


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"failed: {' '.join(map(str, cmd))}\n{r.stderr[-2000:]}")
    return r


def extract(tgz, out, max_reads):
    """Stream the single fastq inside a .fastq.tar.gz to `out` (optionally the first reads only)."""
    with tarfile.open(tgz, "r:gz") as t:
        m = [x for x in t.getmembers() if x.isfile()][0]
        src = t.extractfile(m)
        with open(out, "wb") as fh:
            if max_reads:
                for i, line in enumerate(src):
                    if i >= 4 * max_reads:
                        break
                    fh.write(line)
            else:
                shutil.copyfileobj(src, fh, 1 << 24)


def hamming_ok(a, b, k=2):
    return len(a) == len(b) and sum(x != y for x, y in zip(a, b)) <= k


def demux(fasta, tags, fwd, rev):
    """Yields (pcr, insert) for every read that passes rule 2-3; returns a Counter of drops."""
    rrev = rc(rev)
    f_core, r_core = fwd[-12:], rrev[:12]
    drop = collections.Counter()

    def parse(s):
        i = s.find(f_core)
        if i < 0:
            return None
        fs = i - (len(fwd) - 12)
        if fs < 8 or not hamming_ok(s[fs:i + 12], fwd):
            return None
        j = s.find(r_core, i + 12)
        if j < 0 or not hamming_ok(s[j:j + len(rrev)], rrev):
            return None
        t5, t3 = s[fs - 8:fs], s[j + len(rrev):j + len(rrev) + 8]
        if len(t3) < 8 or rc(t3) != t5:
            return "tagmismatch"
        if t5 not in tags:
            return "unknowntag"
        return tags[t5], s[i + 12:j]

    name, seq = None, []
    out = []

    def flush():
        if name is None:
            return
        s = "".join(seq).upper()
        r = parse(s)
        if r is None:
            r = parse(rc(s))
        if r is None:
            drop["no_primers"] += 1
        elif isinstance(r, str):
            drop[r] += 1
        else:
            pcr, ins = r
            if not (20 <= len(ins) <= 100) or set(ins) - set("ACGT"):
                drop["length_or_N"] += 1
            else:
                out.append((pcr, ins))

    with open(fasta) as fh:
        for line in fh:
            if line.startswith(">"):
                flush()
                name, seq = line, []
            else:
                seq.append(line.strip())
        flush()
    return out, drop


CONTROL = r"^C(?:_|-|PCR|ext)"     # control PCR names: C_PCR, CPCR, C-PCR, Cext_1102, Cext-28-10


def control_filter(pc):
    """Rules 5-6 on a (zotu, pcr, count) table. Returns (pcr table, sample table, log line)."""
    ctrl = pc.pcr.str.contains(CONTROL, regex=True)
    cnames = sorted(set(pc.pcr[ctrl].str.replace(r"_\d+$", "", regex=True)))
    cont = pc[ctrl].groupby("zotu")["count"].max()
    pc = pc[~ctrl].copy()
    pc["count"] = pc["count"].where(pc["count"] > pc.zotu.map(cont).fillna(0), 0)
    pc = pc[pc["count"] > 0]
    pc["sample"] = pc.pcr.str.replace(r"_\d+$", "", regex=True)
    smp = pc.groupby(["zotu", "sample"]).agg(n_pcr=("pcr", "nunique"), reads=("count", "sum")).reset_index()
    smp = smp[smp.n_pcr >= 2]
    msg = (f"after controls {cnames} and >= 2 PCR replicates: "
           f"{smp.zotu.nunique()} ZOTUs in {smp['sample'].nunique()} samples, {smp.reads.sum()} reads")
    return pc, smp, msg


def refilter(out):
    """--refilter: redo rules 5-6 from a saved zotu_pcr_counts.csv (no need to reprocess the reads)."""
    pc = pd.read_csv(out / "zotu_pcr_counts.csv").drop(columns=["sample"], errors="ignore")
    pc, smp, msg = control_filter(pc)
    keep = set(smp.zotu)
    lines, name = [], None
    for line in open(out / "zotus.fasta"):
        if line.startswith(">"):
            name = line[1:].split(";")[0]
        if name in keep:
            lines.append(line)
    (out / "zotus.fasta").write_text("".join(lines))
    pc.to_csv(out / "zotu_pcr_counts.csv", index=False)
    smp.to_csv(out / "zotu_samples.csv", index=False)
    with open(out / "summary.txt", "a") as fh:
        fh.write("refilter (control names fixed): " + msg + "\n")
    print(msg)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    if not args:
        sys.exit("usage: python real_edna/process_reads.py data/real_edna/raw/Run1 [--max-reads=N]")
    rundir = Path(args[0])
    if "--refilter" in sys.argv:
        return refilter(Path(__file__).resolve().parents[1] / "data" / "real_edna" / "processed" / rundir.name)
    max_reads = int(opts.get("max-reads", 0))
    root = Path(__file__).resolve().parents[1]
    out = root / "data" / "real_edna" / "processed" / (rundir.name + ("_test" if max_reads else ""))
    out.mkdir(parents=True, exist_ok=True)
    V = vsearch()
    r1 = sorted(rundir.glob("*_R1.fastq.tar.gz"))[0]
    r2 = sorted(rundir.glob("*_R2.fastq.tar.gz"))[0]
    tagfile = sorted(rundir.glob("*.dat"))[0]
    tg = pd.read_csv(tagfile, sep="\t", header=None, names=["exp", "pcr", "tag", "fwd", "rev", "x"])
    tags = dict(zip(tg.tag.str.upper(), tg.pcr))
    fwd, rev = tg.fwd.iloc[0].upper(), tg.rev.iloc[0].upper()
    log = []

    with tempfile.TemporaryDirectory(dir=out) as tmp:
        tmp = Path(tmp)
        print("extracting reads ...", flush=True)
        extract(r1, tmp / "R1.fq", max_reads)
        extract(r2, tmp / "R2.fq", max_reads)
        print("merging pairs ...", flush=True)
        r = run([V, "--fastq_mergepairs", tmp / "R1.fq", "--reverse", tmp / "R2.fq",
                 "--fastq_allowmergestagger", "--fastq_minovlen", "20", "--fastq_maxdiffs", "10",
                 "--fastqout", tmp / "merged.fq", "--threads", "0"])
        log.append("merge:\n" + "\n".join(l for l in r.stderr.splitlines() if "airs" in l or "erged" in l))
        run([V, "--fastq_filter", tmp / "merged.fq", "--fastq_maxee", "1", "--fastaout", tmp / "merged.fa",
             "--fasta_width", "0"])
        (tmp / "R1.fq").unlink(); (tmp / "R2.fq").unlink()
        print("demultiplexing ...", flush=True)
        reads, drop = demux(tmp / "merged.fa", tags, fwd, rev)
        log.append(f"reads after quality filter and primers/tags: {len(reads)}; dropped: {dict(drop)}")
        cnt = collections.Counter(reads)                      # (pcr, insert) -> reads
        tot = collections.Counter()
        for (p, s), c in cnt.items():
            tot[s] += c
        uniq = sorted(tot, key=lambda s: -tot[s])
        with open(tmp / "uniq.fa", "w") as fh:
            for i, s in enumerate(uniq):
                fh.write(f">u{i};size={tot[s]}\n{s}\n")
        print(f"denoising {len(uniq)} distinct inserts ...", flush=True)
        run([V, "--cluster_unoise", tmp / "uniq.fa", "--minsize", "8", "--unoise_alpha", "2",
             "--centroids", tmp / "denoised.fa", "--sizein", "--sizeout", "--fasta_width", "0"])
        run([V, "--uchime3_denovo", tmp / "denoised.fa", "--nonchimeras", tmp / "zotus_raw.fa",
             "--sizein", "--sizeout", "--fasta_width", "0"])
        z = {}
        name = None
        for line in open(tmp / "zotus_raw.fa"):
            if line.startswith(">"):
                name = line[1:].split(";")[0]
            else:
                z[name] = line.strip()
        zid = {k: f"zotu_{i + 1}" for i, k in enumerate(z)}
        with open(tmp / "zotus.fa", "w") as fh:
            for k, s in z.items():
                fh.write(f">{zid[k]}\n{s}\n")
        run([V, "--usearch_global", tmp / "uniq.fa", "--db", tmp / "zotus.fa", "--id", "0.97",
             "--userout", tmp / "map.tsv", "--userfields", "query+target", "--strand", "plus",
             "--maxaccepts", "1", "--threads", "0"])
        mp = pd.read_csv(tmp / "map.tsv", sep="\t", header=None, names=["q", "t"])
        mp["q"] = mp.q.str.split(";").str[0]
        u2z = dict(zip(mp.q, mp.t))
        useq = {f"u{i}": s for i, s in enumerate(uniq)}
        s2z = {useq[u]: t for u, t in u2z.items()}

    rows = collections.Counter()
    for (p, s), c in cnt.items():
        if s in s2z:
            rows[(s2z[s], p)] += c
    pc = pd.DataFrame([(zz, p, c) for (zz, p), c in rows.items()], columns=["zotu", "pcr", "count"])
    log.append(f"reads assigned to ZOTUs: {pc['count'].sum()} in {pc.zotu.nunique()} ZOTUs")
    pc, smp, msg = control_filter(pc)
    log.append(msg)

    with open(out / "zotus.fasta", "w") as fh:
        size = pc.groupby("zotu")["count"].sum()
        for k, s in z.items():
            if zid[k] in set(smp.zotu):
                fh.write(f">{zid[k]};size={int(size.get(zid[k], 0))}\n{s}\n")
    pc.to_csv(out / "zotu_pcr_counts.csv", index=False)
    smp.to_csv(out / "zotu_samples.csv", index=False)
    (out / "summary.txt").write_text("\n".join(log) + "\n")
    print("\n".join(log))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
