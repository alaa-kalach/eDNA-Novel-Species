# Conformal layer: what each setting costs

False alarm = known fish wrongly called new (mean over 30 splits, 95% CI). Caught = share of hidden (new) species called new; close sisters = new species within 3 Teleo letters of a known species. 'vs F1' = change in catch rate compared with the F1-tuned cut-off of the same tool (percentage points).

| Tool | Rule | False alarm (95% CI) | Realised / target | New caught | Close sisters caught | New caught vs F1 |
|---|---|---|---|---|---|---|
| VSEARCH | standard | 14.7% (13.5%-15.9%) | - | 97.2% | 73.1% | +5.7 |
| VSEARCH | f1 | 8.4% (6.6%-10.2%) | - | 91.5% | 32.8% | - |
| VSEARCH | conf05 | 1.5% (1.0%-2.0%) | 0.30 | 59.8% | 0.0% | -31.7 |
| VSEARCH | conf10 | 3.2% (2.6%-3.8%) | 0.32 | 76.1% | 0.5% | -15.3 |
| SINTAX | standard | 5.6% (4.5%-6.6%) | - | 66.7% | 19.7% | -17.8 |
| SINTAX | f1 | 12.8% (11.7%-13.8%) | - | 84.5% | 44.4% | - |
| SINTAX | conf05 | 2.3% (1.4%-3.3%) | 0.46 | 44.4% | 6.6% | -40.0 |
| SINTAX | conf10 | 7.5% (6.3%-8.7%) | 0.75 | 72.7% | 24.3% | -11.8 |
| RDP | standard | 21.9% (20.5%-23.3%) | - | 95.3% | 72.9% | +6.0 |
| RDP | f1 | 14.8% (12.5%-17.1%) | - | 89.4% | 43.5% | - |
| RDP | conf05 | 1.5% (0.9%-2.1%) | 0.30 | 57.9% | 1.2% | -31.4 |
| RDP | conf10 | 3.8% (2.9%-4.6%) | 0.38 | 77.5% | 12.6% | -11.8 |
| Raw k-mers | f1 | 19.0% (15.9%-22.0%) | - | 90.6% | 58.6% | - |
| Raw k-mers | conf05 | 0.8% (0.3%-1.3%) | 0.15 | 42.2% | 5.9% | -48.4 |
| Raw k-mers | conf10 | 4.1% (3.3%-4.9%) | 0.41 | 72.2% | 19.0% | -18.4 |
| Tree placement | f1 | 16.6% (13.9%-19.3%) | - | 90.5% | 36.0% | - |
| Tree placement | conf05 | 2.6% (1.4%-3.9%) | 0.52 | 37.9% | 0.6% | -52.6 |
| Tree placement | conf10 | 5.1% (3.7%-6.5%) | 0.51 | 59.1% | 1.3% | -31.4 |
| Tree-trained k-mers | f1 | 18.7% (15.8%-21.6%) | - | 91.6% | 52.2% | - |
| Tree-trained k-mers | conf05 | 0.4% (0.2%-0.6%) | 0.08 | 7.2% | 0.5% | -84.4 |
| Tree-trained k-mers | conf10 | 1.9% (1.1%-2.6%) | 0.19 | 40.4% | 5.4% | -51.3 |

How cautious the conformal setting is in practice: at 5% the realised false alarm is 0.4%-2.6%, at 10% it is 1.9%-7.5%.
A very low false alarm with a very low catch rate is not a good result: the tool barely flags anything.

## Equal-error comparison (each tool vs VSEARCH at the same conformal setting)

Pre-registered secondary tests (Wilcoxon over 30 splits, Holm within each family).

| Setting | Tool | Outcome | Tool | VSEARCH | Difference | Holm p |
|---|---|---|---|---|---|---|
| conf05 | SINTAX | novel_recall | 44.4% | 59.8% | -15.3 pts | 8.8e-05 |
| conf05 | SINTAX | close_recall | 6.6% | 0.0% | +6.6 pts | 0.017 |
| conf10 | SINTAX | novel_recall | 72.7% | 76.1% | -3.4 pts | 0.1 |
| conf10 | SINTAX | close_recall | 24.3% | 0.5% | +23.9 pts | 0.00015 |
| conf05 | RDP | novel_recall | 57.9% | 59.8% | -1.9 pts | 0.51 |
| conf05 | RDP | close_recall | 1.2% | 0.0% | +1.2 pts | 0.17 |
| conf10 | RDP | novel_recall | 77.5% | 76.1% | +1.4 pts | 0.27 |
| conf10 | RDP | close_recall | 12.6% | 0.5% | +12.2 pts | 0.0012 |
| conf05 | Raw k-mers | novel_recall | 42.2% | 59.8% | -17.6 pts | 0.00023 |
| conf05 | Raw k-mers | close_recall | 5.9% | 0.0% | +5.9 pts | 0.0059 |
| conf10 | Raw k-mers | novel_recall | 72.2% | 76.1% | -4.0 pts | 0.068 |
| conf10 | Raw k-mers | close_recall | 19.0% | 0.5% | +18.6 pts | 0.00062 |
| conf05 | Tree-trained k-mers | novel_recall | 7.2% | 59.8% | -52.6 pts | 1.9e-08 |
| conf05 | Tree-trained k-mers | close_recall | 0.5% | 0.0% | +0.5 pts | 0.36 |
| conf10 | Tree-trained k-mers | novel_recall | 40.4% | 76.1% | -35.8 pts | 1.9e-08 |
| conf10 | Tree-trained k-mers | close_recall | 5.4% | 0.5% | +5.0 pts | 0.044 |
| conf05 | Tree placement | novel_recall | 37.9% | 59.8% | -21.9 pts | 0.00026 |
| conf05 | Tree placement | close_recall | 0.6% | 0.0% | +0.6 pts | 0.17 |
| conf10 | Tree placement | novel_recall | 59.1% | 76.1% | -17.0 pts | 5e-08 |
| conf10 | Tree placement | close_recall | 1.3% | 0.5% | +0.8 pts | 0.27 |
