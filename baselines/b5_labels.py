#!/usr/bin/env python3
"""
B5 - labels-only encoder.

Same encoder, global pretraining and fine-tuning as the main method, but the
target only says "same species" (0) or "different species" (1). No tree.
Comparing it with the main method shows what the tree itself adds.

Usage (from the repo root):
    python baselines/b5_labels.py                     # seed 0
    python baselines/b5_labels.py --seeds=0,1,2,3,4   # several training seeds
Output: models/pretrained_labels[_s<N>].npz
        outputs/results/results_ft_labels[_s<N>][_extra].csv  (main result)
        outputs/results/results_pre_labels...csv               (pretrained only, no fine-tuning)
"""
from _encoder_baseline import main

if __name__ == "__main__":
    main("labels")
