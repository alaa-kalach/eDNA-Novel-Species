#!/bin/bash
# Phylogenetic placement on the Fish Tree (models/phylo_placement.py), both markers, all 30 splits.
#
# One-time setup on a Mac (Homebrew):
#   brew install mafft micromamba
#   micromamba create -y -p ~/phylo-tools -c conda-forge -c bioconda raxml-ng epa-ng
#   pip install dendropy            # inside the project's .venv
#
# Usage (repo root, venv active):  bash run_phylo_placement.sh
set -e
export PATH="$HOME/phylo-tools/bin:$PATH"
for t in mafft raxml-ng epa-ng; do
  command -v $t >/dev/null || { echo "missing $t - see the setup lines at the top of this file"; exit 1; }
done
python -W ignore models/phylo_placement.py --threads=8
python -W ignore models/phylo_placement.py --marker=full --threads=8
