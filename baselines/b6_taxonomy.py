#!/usr/bin/env python3
"""
B6 - taxonomy encoder.

Same encoder, global pretraining and fine-tuning as the main method, but the
target is taxonomic distance instead of tree distance:
    same species 0 | same genus 0.25 | same family 0.5 | same order 0.75 | other 1
Comparing it with the main method shows whether the real tree beats simple
taxonomic ranks.

Needs data/global_taxonomy.csv once:  python data/add_global_taxonomy.py

Usage (from the repo root):
    python baselines/b6_taxonomy.py                     # seed 0
    python baselines/b6_taxonomy.py --seeds=0,1,2,3,4   # several training seeds
Output: models/pretrained_taxonomy[_s<N>].npz
        outputs/results/results_ft_taxonomy[_s<N>][_extra].csv  (main result)
        outputs/results/results_pre_taxonomy...csv               (pretrained only)
"""
from _encoder_baseline import main

if __name__ == "__main__":
    main("taxonomy")
