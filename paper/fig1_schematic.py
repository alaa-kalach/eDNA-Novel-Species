"""Figure 1: study design and the open-set pipeline (a drawing, no data).

a  Evaluation design: the French Guiana library and the Fish Tree feed 30 splits that hide ~20% of
   species as "new"; the global MitoFish library (no French Guiana species) is used for pretraining
   and as the global reference.
b  The pipeline: detect -> place -> group, with what each step uses and what the user gets.
c  Where it is tested.

Same fonts, colours and sizes as paper/evaluate.py (Figures 2-6). Edit the TEXT dict to change labels.
Usage (repo root):  python paper/fig1_schematic.py
Output: paper/figures/fig1_schematic.png
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "paper" / "figures"

GREY, DARK, LIGHT, BLUE, LBLUE, PALE = "#a6a6a6", "#555555", "#d4d4d4", "#2f6db5", "#9dbbe0", "#f2f2f2"
PALEBLUE = "#e8f0fa"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 6.3, "savefig.dpi": 300, "pdf.fonttype": 42})

TEXT = {
    "lib": "French Guiana 12S library\n1,200 sequences, 263 species\nTeleo (64 bp) and long 12S (~600 bp)",
    "tree": "Fish Tree of Life\n(ray-finned fish, time-calibrated)",
    "glob": "Global MitoFish 12S\n(no French Guiana species)\nused for pretraining and\nas the global placement library",
    "splits": "30 splits\nhide ~20% of species\nas \u201cnew\u201d",
    "ref": "Reference\n~725 seq",
    "known": "Known test\n~230 seq",
    "new": "New test\n~245 seq",
    "query": "eDNA\nsequence",
    "detect": "1  Detect\nVSEARCH identity to the library;\nF1-tuned or conformal cut-off",
    "known_out": "Known species\n(named)",
    "place": "2  Place\nnearest match in the local\nand global library\n\u2192 genus and family",
    "group": "3  Group\nbarcode gap calibrated\non the reference library",
    "new_out": "New-species\ngroups with\ngenus, family",
    "tree_tests": "Tree alternatives tested: per-group cut-offs and pendant\n"
                  "length (detect); tree-trained encoder and EPA-ng\n(place); PTP and tree grouping (group)",
    "tests": "Tested on: 30 hidden-species splits\nglobal held-out fish (no French Guiana data)\n"
             "real water samples, 39 sites (Cilleros et al. 2019)",
}


def box(ax, x, y, w, h, text, fc=PALE, ec=GREY, tc=DARK, lw=0.7, bold_first=False, size=6.3):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.2", fc=fc, ec=ec, lw=lw))
    if bold_first:
        first, rest = text.split("\n", 1)
        ax.text(x + w / 2, y + h - 2.6, first, ha="center", va="center", color=tc, fontsize=size + 1,
                fontweight="bold")
        ax.text(x + w / 2, y + (h - 4.5) / 2, rest, ha="center", va="center", color=tc, fontsize=size - 0.3,
                linespacing=1.3)
    else:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=tc, fontsize=size, linespacing=1.3)


def arrow(ax, p, q, color=DARK, lw=0.9, style="-|>", rad=0.0, label=None, lpos=0.5, loff=(0, 1.4)):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=8, color=color, lw=lw,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=1, shrinkB=1))
    if label:
        x = p[0] + (q[0] - p[0]) * lpos + loff[0]
        y = p[1] + (q[1] - p[1]) * lpos + loff[1]
        ax.text(x, y, label, ha="center", va="bottom", fontsize=6.3, color=color, style="italic")


def main():
    fig, ax = plt.subplots(figsize=(7.2, 3.7))
    ax.set_xlim(0, 144)
    ax.set_ylim(0, 74)
    ax.axis("off")

    # ---------------------------------------------------------------- a: evaluation design
    ax.text(0, 73, "a", fontsize=10, fontweight="bold", va="top")
    ax.text(4, 73, "Evaluation design", fontsize=8.5, va="top", color="black")
    box(ax, 1, 52, 44, 13, TEXT["lib"])
    box(ax, 1, 40, 44, 9, TEXT["tree"])
    box(ax, 1, 21, 44, 15, TEXT["glob"])
    box(ax, 51, 46, 26, 14, TEXT["splits"])
    arrow(ax, (45.5, 58.5), (50.5, 55))
    arrow(ax, (45.5, 44.5), (50.5, 50))
    for i, k in enumerate(["ref", "known", "new"]):
        x = 48.5 + i * 11.5
        fc, ec, tc = (PALEBLUE, BLUE, BLUE) if k == "new" else (PALE, GREY, DARK)
        box(ax, x, 26, 10.5, 10, TEXT[k], fc=fc, ec=ec, tc=tc, size=5.4)
        arrow(ax, (64, 45.5), (x + 5.25, 36.5), color=GREY, lw=0.6)

    # ---------------------------------------------------------------- b: pipeline
    ax.text(86, 73, "b", fontsize=10, fontweight="bold", va="top")
    ax.text(90, 73, "Open-set pipeline", fontsize=8.5, va="top", color="black")
    box(ax, 86, 54, 12, 10, TEXT["query"], fc="white", ec=DARK)
    box(ax, 103, 52, 40, 14, TEXT["detect"], fc=PALEBLUE, ec=BLUE, lw=1.0, bold_first=True)
    arrow(ax, (98.5, 59), (102.5, 59))
    box(ax, 86, 34, 17, 10, TEXT["known_out"], fc="white", ec=GREY)
    arrow(ax, (113, 51.5), (97, 44.5), label="known", lpos=0.5, loff=(-5.5, 0.2))
    box(ax, 108, 29, 35, 17, TEXT["place"], fc=PALEBLUE, ec=BLUE, lw=1.0, bold_first=True)
    arrow(ax, (125.5, 51.5), (125.5, 46.5), color=BLUE, label="new", lpos=0.5, loff=(4, -0.9))
    box(ax, 108, 9, 35, 15, TEXT["group"], fc=PALEBLUE, ec=BLUE, lw=1.0, bold_first=True)
    arrow(ax, (125.5, 28.5), (125.5, 24.5), color=BLUE)
    box(ax, 86, 10, 17, 13, TEXT["new_out"], fc="white", ec=BLUE, tc=BLUE)
    arrow(ax, (107.5, 16.5), (103.5, 16.5), color=BLUE)
    ax.text(86, 2.5, TEXT["tree_tests"], fontsize=5.6, color=DARK, va="center", linespacing=1.3, style="italic")

    # ---------------------------------------------------------------- c: where it is tested
    ax.text(0, 17, "c", fontsize=10, fontweight="bold", va="top")
    box(ax, 4, 2, 74, 13, TEXT["tests"], fc="white", ec=GREY)

    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "fig1_schematic.png", bbox_inches="tight")
    print("wrote", FIG / "fig1_schematic.png")


if __name__ == "__main__":
    main()
