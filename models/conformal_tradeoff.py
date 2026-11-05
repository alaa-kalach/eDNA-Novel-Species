#!/usr/bin/env python3
"""
Fix 2 of the review: report the conformal layer as a TRADE-OFF, not as "fewer false alarms than F1"
(that test is close to automatic, because conformal simply uses a stricter cut-off).

Reads the files written by models/conformal_any.py (no rerun needed):
    outputs/results/conformal_any_per_split.csv, conformal_any_tests.csv
Writes:
    outputs/results/conformal_tradeoff_table.csv   one row per tool x rule:
        target error, realised false alarm (mean, 95% CI over the 30 splits), how cautious it is
        (realised / target), catch rate of new species, catch rate of close sisters (<= 3 letters
        from a known species), and the cost vs the F1-tuned cut-off of the same tool
    outputs/results/conformal_tradeoff.md          the same as a readable table + the equal-error
        comparison between tools (Wilcoxon + Holm, already pre-registered as secondary tests)
Usage (repo root):  python models/conformal_tradeoff.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "outputs" / "results"
TOOLS = ["vsearch", "sintax", "rdp", "kmer", "epa", "treekmer"]
NAMES = {"vsearch": "VSEARCH", "sintax": "SINTAX", "rdp": "RDP", "kmer": "Raw k-mers",
         "epa": "Tree placement", "treekmer": "Tree-trained k-mers"}
RULES = ["standard", "f1", "conf05", "conf10"]
TARGET = {"conf05": 0.05, "conf10": 0.10}


def ci(x):
    x = np.asarray(x, float)
    m = x.mean()
    h = 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan
    return m, m - h, m + h


def main():
    ps = pd.read_csv(RES / "conformal_any_per_split.csv")
    rows = []
    for t in TOOLS:
        f1 = ps[(ps.tool == t) & (ps.rule == "f1")].set_index("split")
        for r in RULES:
            g = ps[(ps.tool == t) & (ps.rule == r)].set_index("split")
            if g.empty:
                continue
            fp, lo, hi = ci(g.known_fp)
            d_rec = (g.novel_recall - f1.novel_recall.reindex(g.index)).mean()
            d_close = (g.close_recall - f1.close_recall.reindex(g.index)).mean()
            rows.append(dict(tool=t, rule=r, target=TARGET.get(r, np.nan), false_alarm=fp, fa_lo=lo, fa_hi=hi,
                             cautious=fp / TARGET[r] if r in TARGET else np.nan,
                             splits_over_target=int((g.known_fp > TARGET[r]).sum()) if r in TARGET else np.nan,
                             new_caught=g.novel_recall.mean(), close_caught=g.close_recall.mean(),
                             new_caught_vs_f1=d_rec if r != "f1" else 0.0,
                             close_caught_vs_f1=d_close if r != "f1" else 0.0, n_splits=len(g)))
    tab = pd.DataFrame(rows)
    tab.to_csv(RES / "conformal_tradeoff_table.csv", index=False)

    pct = lambda v: "-" if pd.isna(v) else f"{100 * v:.1f}%"
    md = ["# Conformal layer: what each setting costs", "",
          "False alarm = known fish wrongly called new (mean over 30 splits, 95% CI). "
          "Caught = share of hidden (new) species called new; close sisters = new species within 3 Teleo letters "
          "of a known species. 'vs F1' = change in catch rate compared with the F1-tuned cut-off of the same tool "
          "(percentage points).", "",
          "| Tool | Rule | False alarm (95% CI) | Realised / target | New caught | Close sisters caught | New caught vs F1 |",
          "|---|---|---|---|---|---|---|"]
    for r_ in tab.itertuples():
        md.append(f"| {NAMES[r_.tool]} | {r_.rule} | {pct(r_.false_alarm)} ({pct(r_.fa_lo)}-{pct(r_.fa_hi)}) | "
                  f"{'-' if pd.isna(r_.cautious) else f'{r_.cautious:.2f}'} | {pct(r_.new_caught)} | "
                  f"{pct(r_.close_caught)} | {'-' if r_.rule == 'f1' else f'{100 * r_.new_caught_vs_f1:+.1f}'} |")
    c = tab[tab.rule.isin(TARGET)]
    md += ["", f"How cautious the conformal setting is in practice: at 5% the realised false alarm is "
           f"{pct(c[c.rule == 'conf05'].false_alarm.min())}-{pct(c[c.rule == 'conf05'].false_alarm.max())}, "
           f"at 10% it is {pct(c[c.rule == 'conf10'].false_alarm.min())}-{pct(c[c.rule == 'conf10'].false_alarm.max())}.",
           "A very low false alarm with a very low catch rate is not a good result: the tool barely flags anything.", ""]
    te = pd.read_csv(RES / "conformal_any_tests.csv")
    te = te[te.family.str.contains("recall vs vsearch")]
    md += ["## Equal-error comparison (each tool vs VSEARCH at the same conformal setting)", "",
           "Pre-registered secondary tests (Wilcoxon over 30 splits, Holm within each family).", "",
           "| Setting | Tool | Outcome | Tool | VSEARCH | Difference | Holm p |", "|---|---|---|---|---|---|---|"]
    for r_ in te.itertuples():
        tool, rule = r_.a.split(":")
        md.append(f"| {rule} | {NAMES[tool]} | {r_.outcome} | {pct(r_.mean_a)} | {pct(r_.mean_b)} | "
                  f"{100 * r_.diff:+.1f} pts | {r_.holm_p:.2g} |")
    (RES / "conformal_tradeoff.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\nwrote {RES}/conformal_tradeoff_table.csv and conformal_tradeoff.md")


if __name__ == "__main__":
    main()
