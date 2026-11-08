# Where does phylogeny help? Unseen fish species from 12S eDNA

This repository has the code, data splits and results for our CMPS 355 project
(Leen Itani and Alaa Kalach, AUB, Fall 2026). We hide species from the French Guiana
12S reference library and test how well different methods can (1) tell that a sequence
comes from a species that isn't in the library, (2) place it in the right genus and family,
and (3) group new sequences into species, with and without the Fish Tree of Life. At the
end we run the same pipeline on real river-water eDNA from Cilleros et al. (2019).

If you only want to check the numbers in the paper, skip to "Just the paper" below:
everything is already computed and one script rebuilds all tables and figures in a few
seconds. The rest of this file explains how to redo the experiments from scratch, in the
order we ran them.

## Setup

We used Python 3.9 on an Apple silicon laptop (10 cores, 16 GB RAM).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

You also need a few command-line tools. We installed them with micromamba into
`~/phylo-tools`, but any install works as long as they are on your PATH:

```bash
micromamba create -y -p ~/phylo-tools -c conda-forge -c bioconda \
    vsearch=2.32.0 mafft=7.526 raxml-ng=2.0.4 epa-ng=0.3.8 fasttree=2.2.0
export PATH="$PATH:$HOME/phylo-tools/bin"
```

Add the tools to the *end* of the PATH (as above). If you put them first, the conda
Python inside `~/phylo-tools` hides the `.venv` one and you get "No module named pandas".
Run every command below from the repository root.

## Just the paper

```bash
python paper/evaluate.py         # Tables 1-3, Figures 2-6 and S1, paper/tables/key_numbers.csv
python paper/fig1_schematic.py   # Figure 1
```

`paper/tables/key_numbers.csv` lists every number we quote in the manuscript together with
the result file it comes from, so you can trace any value back to the per-split results.

## Redoing everything

Most scripts loop over the 30 splits. Splits 1-10 live in `data/splits.csv` and 11-30 in
`data/splits_extra.csv`; for the second file add `--splits=splits_extra.csv --tag=_extra`.
The shell scripts in the root folder already do both.

**1. Data.** The French Guiana library is cleaned in `notebooks/eDNA_teleo_dataset.ipynb`
(output: `data/guiana_teleo_clean.csv`). Then:

```bash
python data/check_mitofish.py        # downloads MitoFish (Oct 2023) -> data/global_teleo.csv
python data/add_global_taxonomy.py   # genus/family/order for global species
python data/make_full12s.py          # the ~600 bp long 12S version
python data/make_splits.py           # 30 splits, 20% of species hidden (split k uses seed 2026 + k)
python models/kmers.py               # k-mer profiles, k = 3-6
python models/tree_distances.py      # distances on the Fish Tree
python data/pairwise_identity.py && python data/pairwise_identity.py --marker=full
python data/pair_alignments.py   && python data/pair_alignments.py --marker=full
```

**2. Baselines.**

```bash
for b in b1_forced b2_vsearch b3_naive_bayes b4_kmer; do
  python baselines/$b.py
  python baselines/$b.py --splits=splits_extra.csv --tag=_extra
done
python baselines/b5_labels.py   --seeds=0,1,2,3,4
python baselines/b6_taxonomy.py --seeds=0,1,2,3,4
```

**3. Tree-trained encoder.** `select_treeloss.py` compares the tree targets on held-out
global species; the paper uses the 90-million-year cap.

```bash
python models/select_treeloss.py
bash run_treefix.sh treecap90
bash run_treefix.sh treecap90 full
bash run_full12s.sh
```

**4. Placement.**

```bash
python models/placement_kmer_tree.py && python models/placement_kmer_tree.py --marker=full
python models/placement_vsearch_check.py --threads=8
python models/global_library.py --threads=8 && python models/global_library.py --marker=full --threads=8
bash run_phylo_placement.sh              # EPA-ng on the Fish Tree (the slowest step)
python models/phylo_global_check.py --threads=8
python models/tiebreak_global_check.py --threads=8
python models/when_tree_helps.py
```

**5. Detection and the conformal layer.**

```bash
python models/conformal_detect.py && python models/conformal_detect.py --marker=full
python models/end_to_end.py       && python models/end_to_end.py --marker=full
python models/conformal_any.py           # 6 tools x 4 rules on all 30 splits
python models/conformal_tradeoff.py      # summary table, no re-run needed
```

**6. Grouping.**

```bash
bash run_clustering.sh
python clustering/gap_grouping.py        # calibrated barcode gap
python clustering/fixed_grouping.py      # fixed 95-100% thresholds
python clustering/ptp_delimit.py         # PTP on a FastTree gene tree
python clustering/tree_grouping.py --threads=8
python clustering/score_clusters.py kmer gap_align ptp phylo_tree
python clustering/score_clusters.py --out=_fixed gap_align_vs fixed97_single_vs \
    fixed98_single_vs fixed99_single_vs fixed100_single_vs
```

**7. Real water samples.** The raw reads are about 10 GB, so they are not in the repo.
Download them from Dryad (doi:10.5061/dryad.dc25730): put each run's two `.fastq.tar.gz`
files and its tag file in `data/real_edna/raw/Run1` ... `Run4`. The small study tables
(`site_data.txt`, `traditional_result.txt`, `assignment_result.txt`,
`Reference_sequence.fasta`) are already in `data/real_edna/`.

```bash
for r in 1 2 3 4; do python real_edna/process_reads.py data/real_edna/raw/Run$r; done
python real_edna/run_real.py
```

Processing takes 10-30 minutes per run; `run_real.py` takes about a minute.

**8. Paper.** `python analysis/confirm_positive.py` checks the four pre-registered claims,
then run the two commands under "Just the paper".

## Things worth knowing

- All random choices are seeded: splits use 2026 + split number, the pseudo-splits used to
  pick cut-offs use 100000 + 100 x split + r (r = 0-4), encoders use seeds 0-4, RAxML-NG
  uses seed 1, and bootstraps (5,000 resamples) and permutation tests (10,000) use seed 0.
- Wilcoxon p-values can come out slightly different with another SciPy version (exact vs
  approximate method). Means and confidence intervals don't change.
- The hidden species of a split are never used to choose a cut-off or a model; all choices
  are made on the reference alone.
