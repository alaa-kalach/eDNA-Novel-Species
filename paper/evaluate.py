"""Paper evaluation: Tables 1-3 and Figures 2-6, built from the per-split results already on disk.

No experiment is re-run. Every number comes from the per-split / per-sequence files written by the
experiment scripts, and every mean, 95% CI and paired test is recomputed here, the same way as the
rest of the repo:
    one split = one observation (30 splits); bootstrap 95% CI of the mean (5000 resamples, seed 0);
    paired comparisons = Wilcoxon signed-rank on the per-split differences; Holm within a table.
(The bootstrap draws differ from analysis/confirm_positive.py, so a CI can move in the 3rd decimal.)

Inputs  outputs/results/*_per_split.csv, outputs/placement_kmer_tree_per_split.csv,
        outputs/results/clusters_*.csv, data/*.csv, data/real_edna/results/*
Outputs paper/tables/table1_dataset.csv       dataset and split sizes
        paper/tables/table2_detection.csv     detection tools x cut-off rules
        paper/tables/table3_teleo_vs_long12s.csv
        paper/tables/key_numbers.csv               every number the manuscript quotes, with its source
        paper/figures/fig2_detection.png     ... fig6_real_edna.png
        paper/figures/figS1_tree_helps_by_age.png  supplement: Fig. 4 grouped by tree age
Figure 1 (pipeline schematic) is a drawing, not data, and is made separately.

Usage (repo root):  python paper/evaluate.py
Needs: numpy, pandas, scipy, matplotlib.
"""
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import beta, wilcoxon

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "outputs" / "results"
OUT = ROOT / "outputs"
DATA = ROOT / "data"
REAL = DATA / "real_edna" / "results"
FIG = ROOT / "paper" / "figures"
TAB = ROOT / "paper" / "tables"
NBOOT = 5000
rng = np.random.default_rng(0)

# ---------------------------------------------------------------- style: grey first, one accent
GREY, DARK, LIGHT, BLUE, LBLUE = "#a6a6a6", "#555555", "#d4d4d4", "#2f6db5", "#9dbbe0"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "axes.spines.top": False,
    "axes.spines.right": False, "axes.edgecolor": DARK, "axes.linewidth": 0.6, "xtick.color": DARK,
    "ytick.color": DARK, "savefig.dpi": 300, "pdf.fonttype": 42,
})

KEY = []  # (section, quantity, value, ci_lo, ci_hi, source)


def key(section, quantity, value, lo=np.nan, hi=np.nan, source=""):
    KEY.append(dict(section=section, quantity=quantity, value=value, ci_lo=lo, ci_hi=hi, source=source))


# ---------------------------------------------------------------- statistics
def ci_mean(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    b = rng.choice(x, (NBOOT, len(x))).mean(axis=1)
    lo, hi = np.percentile(b, [2.5, 97.5])
    return x.mean(), lo, hi


def paired(a, b):
    """a, b: Series indexed by split. diff = a - b."""
    d = (a - b).dropna()
    m, lo, hi = ci_mean(d.values)
    p = wilcoxon(d.values).pvalue if (d != 0).any() else 1.0
    return dict(mean_a=a.loc[d.index].mean(), mean_b=b.loc[d.index].mean(), diff=m, ci_lo=lo, ci_hi=hi,
                n=len(d), p=p)


def holm(p):
    p = np.asarray(p, float)
    order, m = np.argsort(p), len(p)
    adj, run = np.empty(m), 0.0
    for r, i in enumerate(order):
        run = max(run, (m - r) * p[i])
        adj[i] = min(run, 1.0)
    return adj


def clopper(k, n):
    lo = beta.ppf(0.025, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(0.975, k + 1, n - k) if k < n else 1.0
    return k / n, lo, hi


def by_split(df, col, **sel):
    for k, v in sel.items():
        df = df[df[k] == v]
    s = df.set_index("split")[col]
    assert s.index.is_unique and len(s) == 30, f"{sel} {col}: {len(s)} splits"
    return s


def fmt_pct(m, lo, hi, d=1):
    return f"{100 * m:.{d}f} [{100 * lo:.{d}f}, {100 * hi:.{d}f}]"


def save_table(df, name, note=""):
    TAB.mkdir(parents=True, exist_ok=True)
    df.to_csv(TAB / f"{name}.csv", index=False)   # note = the table's caption text, kept here for the manuscript


def save_fig(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def panel(ax, letter):
    ax.text(-0.02, 1.04, letter, transform=ax.transAxes, fontsize=10, fontweight="bold", ha="right", va="bottom")


# ================================================================ Table 1: dataset
def table1():
    raw = pd.read_csv(DATA / "Data_12S Ref library_Fish Guiana.csv")
    clean = pd.read_csv(DATA / "guiana_teleo_clean.csv")
    excl = pd.read_csv(DATA / "excluded_shared_teleo.csv")
    intree = clean[clean.in_tree]
    used = intree[~intree["Specimen identification code"].astype(str).isin(excl.seq_id.astype(str))]
    distinct = used.assign(t=used.teleo.str.upper()).groupby("species_name").t.nunique()
    sp = pd.concat([pd.read_csv(DATA / "splits.csv"), pd.read_csv(DATA / "splits_extra.csv")])
    sp["seq_id"] = sp.seq_id.astype(str)
    spec = clean.assign(seq_id=clean["Specimen identification code"].astype(str)).set_index("seq_id").species_name
    sp["species"] = sp.seq_id.map(spec)
    counts = sp.groupby(["split_id", "role"]).size().unstack()
    nov = sp[sp.role == "test_novel"]
    nov_sp = nov.groupby("split_id").species.nunique()
    hard = nov.drop_duplicates(["split_id", "species"]).groupby("split_id").difficulty.apply(lambda s: (s == "hard").mean())
    rows = [
        ("Sequences in the published library", len(raw)),
        ("After removing open names (sp., aff., cf.)", f"{len(clean)} sequences, {clean.species_name.nunique()} species"),
        ("Species in the Fish Tree of Life", f"{intree.species_name.nunique()} ({intree.species_name.nunique() / clean.species_name.nunique():.0%})"),
        ("Sequences of in-tree species", len(intree)),
        ("Removed: Teleo sequence shared by 2+ species", f"{len(excl)} sequences ({excl.teleo.str.upper().nunique()} distinct)"),
        ("Used in the splits", f"{len(used)} sequences, {used.species_name.nunique()} species"),
        ("Species with >= 2 distinct Teleo sequences", int((distinct >= 2).sum())),
        ("Test splits", sp.split_id.nunique()),
        ("Hidden (new) species per split, mean", f"{nov_sp.mean():.1f} ({nov_sp.mean() / used.species_name.nunique():.0%} of species)"),
        ("  of which 'hard' (same-genus species in the reference)", f"{hard.mean():.0%}"),
        ("Reference sequences per split, mean", f"{counts.reference.mean():.0f}"),
        ("Known test sequences per split, mean", f"{counts.test_known.mean():.0f}"),
        ("New test sequences per split, mean", f"{counts.test_novel.mean():.0f}"),
    ]
    t = pd.DataFrame(rows, columns=["Quantity", "Value"])
    save_table(t, "table1_dataset", "French Guiana 12S reference library, Teleo region. Splits: data/make_splits.py.")
    for q, v in rows:
        key("Table 1", q, v, source="data/*.csv")


# ================================================================ Table 2: detection
TOOLS = {"vsearch": "VSEARCH identity", "sintax": "SINTAX", "rdp": "RDP naive Bayes", "kmer": "Raw k-mers (B4)",
         "epa": "Tree placement (EPA-ng)", "treekmer": "Tree-trained k-mers"}
RULES = {"standard": "Standard cut-off", "f1": "F1-tuned cut-off", "conf10": "Conformal, alpha = 0.10",
         "conf05": "Conformal, alpha = 0.05"}


def table2():
    ca = pd.read_csv(RES / "conformal_any_per_split.csv")
    e2e = pd.read_csv(RES / "end_to_end_per_split.csv")
    rows = []
    a = e2e[e2e.pipeline == "A forced"]
    rows.append(["Forced assignment (B1)", "none (closed set)", fmt_pct(0, 0, 0), fmt_pct(0, 0, 0), fmt_pct(0, 0, 0), "0.500"])
    for tool, tname in TOOLS.items():
        for rule, rname in RULES.items():
            g = ca[(ca.tool == tool) & (ca.rule == rule)]
            if g.empty:
                continue
            assert len(g) == 30
            fp, nr, cr = ci_mean(g.known_fp), ci_mean(g.novel_recall), ci_mean(g.close_recall)
            rows.append([tname, rname, fmt_pct(*fp), fmt_pct(*nr), fmt_pct(*cr), f"{g.auroc.mean():.3f}"])
            if tool == "vsearch":
                key("Results 1-2", f"VSEARCH {rule}: known called new", *fp, source="conformal_any_per_split")
                key("Results 1-2", f"VSEARCH {rule}: new caught", *nr, source="conformal_any_per_split")
                key("Results 1-2", f"VSEARCH {rule}: close sisters caught", *cr, source="conformal_any_per_split")
        key("Results 1-3", f"{tname}: AUROC", *ci_mean(ca[(ca.tool == tool) & (ca.rule == 'f1')].auroc),
            source="conformal_any_per_split")
    t = pd.DataFrame(rows, columns=["Tool", "Cut-off rule", "Known fish called new, % [95% CI]",
                                    "New fish caught, %", "Close sisters caught, %", "AUROC"])
    save_table(t, "table2_detection",
               "Teleo, mean of 30 splits, bootstrap 95% CI. Close sisters = new species within 3 Teleo differences "
               "of a known species. Standard cut-off: VSEARCH identity < 97%; SINTAX / RDP confidence < 0.8. "
               "Conformal: the target is the share of known fish called new.")
    # forced vs open set (claim 4)
    b = e2e[e2e.pipeline == "B current (B2)"]
    for name, g in [("forced", a), ("VSEARCH F1", b)]:
        key("Results 1", f"{name}: new fish given a known name", *ci_mean(g.novel_named_known), source="end_to_end_per_split")
        key("Results 1", f"{name}: balanced accuracy", *ci_mean(g.balanced_correct), source="end_to_end_per_split")
        key("Results 1", f"{name}: known fish named correctly", *ci_mean(g.known_correct), source="end_to_end_per_split")


# ================================================================ Table 3: Teleo vs long 12S
def table3():
    rows = []
    gt, gf = pd.read_csv(RES / "group_cutoffs_per_split.csv"), pd.read_csv(RES / "group_cutoffs_full_per_split.csv")
    for o, lab in [("auroc", "Detection AUROC"), ("f1", "Detection F1"), ("novel_recall", "New fish caught"),
                   ("close_recall", "Close sisters caught"), ("known_fp", "Known fish called new")]:
        rows.append((lab, "VSEARCH, F1 cut-off", paired(by_split(gf, o, method="global"), by_split(gt, o, method="global"))))
    pt, pf = pd.read_csv(RES / "phylo_placement_per_split.csv"), pd.read_csv(RES / "phylo_placement_full_per_split.csv")
    for meth, mlab in [("alignment nearest", "nearest sequence (alignment)"), ("phylo", "tree placement (EPA-ng)")]:
        for o, lab in [("family", "Correct family, new fish"), ("family_easy", "Correct family, no same-genus relative"),
                       ("genus_hard", "Correct genus, same-genus relative present"),
                       ("tree_error", "Placement error on the tree, My")]:
            rows.append((lab, mlab, paired(by_split(pf, o, method=meth), by_split(pt, o, method=meth))))
    lt, lf = pd.read_csv(RES / "global_library_per_split.csv"), pd.read_csv(RES / "global_library_full_per_split.csv")
    rows.append(("Correct family, new fish", "VSEARCH + global library",
                 paired(by_split(lf, "family", method="local+global"), by_split(lt, "family", method="local+global"))))
    et, ef = pd.read_csv(RES / "end_to_end_per_split.csv"), pd.read_csv(RES / "end_to_end_full_per_split.csv")
    rows.append(("Grouping ARI, end to end", "pipeline B (VSEARCH F1)",
                 paired(by_split(ef, "ARI", pipeline="B current (B2)"), by_split(et, "ARI", pipeline="B current (B2)"))))
    ph = holm([r[2]["p"] for r in rows])
    out = []
    for (lab, meth, r), hp in zip(rows, ph):
        pct = "My" not in lab and "AUROC" not in lab and "F1" not in lab and "ARI" not in lab
        f = (lambda v: f"{100 * v:.1f}") if pct else (lambda v: f"{v:.3f}" if abs(v) < 2 else f"{v:.1f}")
        out.append([lab + (" (%)" if pct else ""), meth, f(r["mean_a"]), f(r["mean_b"]),
                    f"{f(r['diff'])} [{f(r['ci_lo'])}, {f(r['ci_hi'])}]", f"{hp:.2g}"])
        key("Results 9", f"{lab} | {meth}: long 12S", r["mean_a"], source="Table 3")
        key("Results 9", f"{lab} | {meth}: Teleo", r["mean_b"], source="Table 3")
    t = pd.DataFrame(out, columns=["Outcome", "Method", "Long 12S", "Teleo", "Difference [95% CI]", "Holm p"])
    save_table(t, "table3_teleo_vs_long12s",
               "Same species, splits and test sequences for both markers; long 12S = about 600 bp ending at Teleo. "
               "Wilcoxon signed-rank over 30 splits, Holm over the rows of this table.")


# ================================================================ Figure 2: detection
def fig2():
    e2e = pd.read_csv(RES / "end_to_end_per_split.csv")
    ca = pd.read_csv(RES / "conformal_any_per_split.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw=dict(width_ratios=[1, 1.15], wspace=0.35))
    # (a) misnaming vs false alarms
    pipes = [("A forced", "Forced"), ("B current (B2)", "VSEARCH\nF1 cut-off"),
             ("C proposed a=0.10", "Conformal\nα = 0.10"), ("C proposed a=0.05", "Conformal\nα = 0.05")]
    x = np.arange(len(pipes))
    for j, (col, lab, colr) in enumerate([("novel_named_known", "New fish misnamed", BLUE),
                                          ("known_called_novel", "Known fish called new", GREY)]):
        for i, (p, _) in enumerate(pipes):
            m, lo, hi = ci_mean(e2e[e2e.pipeline == p][col])
            xx = i + (j - 0.5) * 0.36
            a.bar(xx, 100 * m, 0.34, color=colr, label=lab if i == 0 else None)
            a.errorbar(xx, 100 * m, [[100 * (m - lo)], [100 * (hi - m)]], color=DARK, lw=0.7, capsize=1.5)
            a.text(xx, 100 * hi + 2, f"{100 * m:.0f}", ha="center", fontsize=6.5, color=DARK)
    a.set_xticks(x, [p[1] for p in pipes], fontsize=7)
    a.set_ylabel("Share of test sequences (%)")
    a.set_ylim(0, 112)
    a.legend(frameon=False, loc="upper right")
    a.set_title("Open-set detection stops the misnaming", loc="left")
    panel(a, "a")
    # (b) trade-off: false alarm vs new caught
    order = ["f1", "conf10", "conf05"]
    marks = {"f1": "o", "conf10": "s", "conf05": "^"}
    ends = []
    for tool, tname in TOOLS.items():
        pts = []
        for r in order:
            g = ca[(ca.tool == tool) & (ca.rule == r)]
            pts.append((100 * g.known_fp.mean(), 100 * g.novel_recall.mean()))
        pts = np.array(pts)
        c = BLUE if tool == "vsearch" else GREY
        b.plot(pts[:, 0], pts[:, 1], color=c, lw=1.0 if tool == "vsearch" else 0.7, zorder=2 if tool == "vsearch" else 1)
        for (px, py), r in zip(pts, order):
            b.scatter(px, py, marker=marks[r], s=18, color=c, zorder=3, edgecolor="white", linewidth=0.4)
        ends.append((pts[0, 1], pts[0, 0], tname, tool))
    # labels in a column right of the data, spread apart, with thin leader lines to each tool's F1 point
    ends.sort(reverse=True)
    ly, prev = [], None
    for y, *_ in ends:
        y = min(y, prev - 5.5) if prev is not None else y + 4
        ly.append(y)
        prev = y
    for (y, x, tname, tool), yl in zip(ends, ly):
        col = BLUE if tool == "vsearch" else DARK
        b.plot([x + 0.3, 21.6], [y, yl], color=LIGHT, lw=0.5, zorder=0, clip_on=False)
        b.text(21.8, yl, tname, fontsize=6.5, va="center", color=col)
    for t in (5, 10):
        b.axvline(t, color=LIGHT, lw=0.8, ls="--", zorder=0)
        b.text(t + 0.3, 99, f"{t}% target", fontsize=6.5, color=DARK, rotation=90, va="top")
    for r, lab in [("f1", "F1-tuned"), ("conf10", "conformal 0.10"), ("conf05", "conformal 0.05")]:
        b.scatter([], [], marker=marks[r], color=DARK, s=18, label=lab)
    b.legend(frameon=False, loc="lower right", title="Cut-off", title_fontsize=7)
    b.set_xlabel("Known fish called new (%)")
    b.set_ylabel("New fish caught (%)")
    b.set_xlim(0, 21)
    b.spines["bottom"].set_bounds(0, 21)
    b.set_ylim(0, 100)
    b.set_title("Fewer false alarms cost missed new species", loc="left")
    panel(b, "b")
    save_fig(fig, "fig2_detection")


# ================================================================ Figure 3: placement
def fig3():
    pk = pd.read_csv(OUT / "placement_kmer_tree_per_split.csv")
    ph = pd.read_csv(RES / "phylo_placement_per_split.csv")
    vs = pd.read_csv(RES / "placement_vsearch_check_per_split.csv")
    gl = pd.read_csv(RES / "global_library_per_split.csv")
    ladder = [
        ("Raw k-mers", by_split(pk, "family", method="raw"), False),
        ("Raw k-mers + labels model", by_split(pk, "family", method="raw+labels", w=1.0), False),
        ("Raw k-mers + taxonomy model", by_split(pk, "family", method="raw+taxonomy", w=1.0), False),
        ("Raw k-mers + tree model", by_split(pk, "family", method="raw+treecap90", w=1.0), True),
        ("Tree placement (EPA-ng)", by_split(ph, "family", method="phylo"), True),
        ("VSEARCH nearest match", by_split(vs, "family", method="VSEARCH all"), False),
        ("VSEARCH + global library", by_split(gl, "family", method="local+global"), False),
    ]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw=dict(width_ratios=[1.35, 1], wspace=0.75))
    for i, (lab, s, tree) in enumerate(ladder):
        m, lo, hi = ci_mean(s)
        y = len(ladder) - 1 - i
        a.barh(y, 100 * m, 0.62, color=BLUE if tree else GREY)
        a.errorbar(100 * m, y, xerr=[[100 * (m - lo)], [100 * (hi - m)]], color=DARK, lw=0.7, capsize=1.5)
        a.text(100 * hi + 1.2, y, f"{100 * m:.1f}", va="center", fontsize=7, color=DARK)
        key("Results 4", f"correct family: {lab}", m, lo, hi, source="placement per-split files")
    a.set_yticks(range(len(ladder)), [l[0] for l in ladder][::-1])
    a.set_xlim(0, 100)
    a.set_xlabel("New fish placed in the correct family (%)")
    a.set_title("French Guiana: the tree only ties VSEARCH", loc="left")
    for tl, (_, _, tree) in zip(a.get_yticklabels(), ladder[::-1]):
        tl.set_color(BLUE if tree else DARK)
    panel(a, "a")
    # paired tests behind the ladder
    for (la, sa), (lb, sb) in [((ladder[3][0], ladder[3][1]), (ladder[1][0], ladder[1][1])),
                               ((ladder[3][0], ladder[3][1]), (ladder[0][0], ladder[0][1])),
                               ((ladder[4][0], ladder[4][1]), (ladder[5][0], ladder[5][1])),
                               ((ladder[6][0], ladder[6][1]), (ladder[5][0], ladder[5][1]))]:
        r = paired(sa, sb)
        key("Results 4/7", f"{la} minus {lb} (family)", r["diff"], r["ci_lo"], r["ci_hi"], source=f"Wilcoxon p = {r['p']:.2g}")
    v1 = by_split(vs, "family", method="VSEARCH 1 per species")
    r = paired(by_split(vs, "family", method="tree placement"), v1)
    key("Results 4", "tree placement minus VSEARCH 1 per species (family)", r["diff"], r["ci_lo"], r["ci_hi"],
        source=f"Wilcoxon p = {r['p']:.2g}")
    for o in ["family_easy", "genus_easy", "share_global"]:
        key("Results 7", f"VSEARCH + global library: {o}", *ci_mean(by_split(gl, o, method="local+global")),
            source="global_library_per_split")
        if o != "share_global":
            key("Results 7", f"VSEARCH local only: {o}", *ci_mean(by_split(gl, o, method="local")), source="global_library_per_split")
    # (b) French Guiana vs global held-out fish
    gs = pd.read_csv(RES / "phylo_global_check_summary.csv").set_index("method")
    meths = [("phylo", "Tree placement"), ("alignment nearest", "Nearest sequence\n(same alignment)"),
             ("raw", "Raw k-mers"), ("raw+taxonomy", "Raw k-mers +\ntaxonomy model")]
    yy = np.arange(len(meths))[::-1]
    for y, (m_, lab) in zip(yy, meths):
        fg = by_split(ph, "family", method=m_).mean()
        gv = gs.loc[m_, "family"]
        b.plot([100 * fg, 100 * gv], [y, y], color=LIGHT, lw=1.2, zorder=1)
        b.scatter(100 * fg, y, color=GREY, s=26, zorder=2, label="French Guiana (30 splits)" if y == yy[0] else None)
        b.scatter(100 * gv, y, color=DARK, s=26, zorder=3, label="Global held-out fish" if y == yy[0] else None)
        key("Results 5", f"global check, family: {m_}", gv, source="phylo_global_check_summary")
    b.set_yticks(yy, [m[1] for m in meths])
    b.set_xlim(40, 90)
    b.set_xlabel("New fish placed in the correct family (%)")
    b.set_title("Global fish: the tree falls behind", loc="left")
    b.set_ylim(-0.6, len(meths) - 0.4)
    b.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.4, -0.27), ncol=2, fontsize=6.5)
    panel(b, "b")
    tb =pd.read_csv(RES / "tiebreak_global_summary.csv")
    r = tb[(tb.subset.str.startswith("ambiguous")) & (tb.outcome == "family")].iloc[0]
    key("Results 5", "global tie-break: tree minus VSEARCH (family, cross-genus ties)", r.diff_species_mean, r.ci_lo, r.ci_hi,
        source="tiebreak_global_summary")
    save_fig(fig, "fig3_placement")


# ================================================================ Figure 4: when the tree helps
def fig4():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), sharey=True, gridspec_kw=dict(wspace=0.12))
    bins = ["Q1 (closest)", "Q2", "Q3", "Q4 (most isolated)"]
    for ax, (f, title, letter) in zip(axes, [("when_tree_helps_per_split.csv", "Teleo (64 bp)", "a"),
                                             ("when_tree_helps_full_per_split.csv", "Long 12S (about 600 bp)", "b")]):
        d = pd.read_csv(RES / f)
        d = d[d.isolation == "seq"]
        for (ma, mb, lab, c, off) in [("raw+treecap90", "raw", "Tree model vs raw k-mers", BLUE, -0.06),
                                      ("phylo", "alignment nearest", "Tree placement vs nearest sequence", GREY, 0.06)]:
            ms, los, his = [], [], []
            for bn in bins:
                A = d[(d.bin == bn) & (d.method == ma)].set_index("split").family
                B = d[(d.bin == bn) & (d.method == mb)].set_index("split").family
                m, lo, hi = ci_mean((A - B).dropna())
                ms.append(100 * m), los.append(100 * lo), his.append(100 * hi)
                key("Results 6", f"{title}: {lab}, {bn}", m, lo, hi, source=f)
            x = np.arange(len(bins)) + off
            ax.plot(x, ms, color=c, lw=1.2, marker="o", ms=3.5, label=lab)
            ax.vlines(x, los, his, color=c, lw=0.8)
        ax.axhline(0, color=DARK, lw=0.6)
        ax.set_xticks(range(len(bins)), ["Closest", "Q2", "Q3", "Most\nisolated"])
        ax.set_title(title, loc="left")
        panel(ax, letter)
    axes[0].set_ylabel("Gain in correct family\n(percentage points)")
    axes[0].legend(frameon=False, loc="upper left")
    fig.supxlabel("Distance of the new species to its closest library sequence (quartile)", fontsize=8, y=-0.06)
    fig.suptitle("The tree model's gain over raw k-mers grows as new species get more isolated",
                 x=0.07, ha="left", y=1.04, fontsize=9, fontweight="bold")
    for f, lab in [("when_tree_helps_per_split.csv", "Teleo"), ("when_tree_helps_full_per_split.csv", "long 12S")]:
        d = pd.read_csv(RES / f)
        d = d[d.isolation == "seq"]
        g = lambda bn, m: d[(d.bin == bn) & (d.method == m)].set_index("split").family  # noqa: E731
        trend = (g(bins[3], "raw+treecap90") - g(bins[3], "raw")) - (g(bins[0], "raw+treecap90") - g(bins[0], "raw"))
        m, lo, hi = ci_mean(trend.dropna())
        p = wilcoxon(trend.dropna()).pvalue
        key("Results 6", f"{lab}: isolation trend (Q4 - Q1 gain), tree model vs raw", m, lo, hi, source=f"Wilcoxon p = {p:.2g}")
    save_fig(fig, "fig4_when_tree_helps")


# ================================================================ Figure 5: grouping
def load_clusters(name):
    fs = [RES / f"clusters_{name}.csv", RES / f"clusters_{name}_extra.csv"]
    return pd.concat([pd.read_csv(f, dtype={"seq_id": str}) for f in fs if f.exists()])


def count_error(name, species):
    df = load_clusters(name)
    out = {}
    for k, g in df.groupby("split_id"):
        lab = g.cluster_id.values
        n_groups = len(np.unique(lab[lab >= 0])) + int((lab < 0).sum())
        out[k] = abs(n_groups - species.loc[g.seq_id].nunique())
    return pd.Series(out)


def fig5():
    clean = pd.read_csv(DATA / "guiana_teleo_clean.csv")
    clean["seq_id"] = clean["Specimen identification code"].astype(str)
    species = clean.drop_duplicates("seq_id").set_index("seq_id").species_name
    fx = pd.read_csv(RES / "cluster_scores_fixed_per_split.csv")
    cs = pd.read_csv(RES / "cluster_scores_per_split.csv")
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(7.2, 2.5), gridspec_kw=dict(wspace=0.75))
    meths = [("gap_align_vs", "Calibrated gap"), ("fixed97_single_vs", "Fixed 97%"), ("fixed98_single_vs", "Fixed 98%"),
             ("fixed99_single_vs", "Fixed 99%"), ("fixed100_single_vs", "Exact (100%)")]
    gap_ari, gap_err = by_split(fx, "ARI", method="gap_align_vs"), count_error("gap_align_vs", species)
    for i, (m_, lab) in enumerate(meths):
        y = len(meths) - 1 - i
        col = BLUE if i == 0 else GREY
        s = by_split(fx, "ARI", method=m_)
        m, lo, hi = ci_mean(s)
        a.barh(y, m, 0.62, color=col)
        a.errorbar(m, y, xerr=[[m - lo], [hi - m]], color=DARK, lw=0.7, capsize=1.5)
        a.text(hi + 0.01, y, f"{m:.2f}", va="center", fontsize=7, color=DARK)
        e = count_error(m_, species)
        me, loe, hie = ci_mean(e)
        b.barh(y, me, 0.62, color=col)
        b.errorbar(me, y, xerr=[[me - loe], [hie - me]], color=DARK, lw=0.7, capsize=1.5)
        b.text(hie + 0.8, y, f"{me:.1f}", va="center", fontsize=7, color=DARK)
        key("Results 8", f"{lab}: ARI (VSEARCH-called)", m, lo, hi, source="cluster_scores_fixed_per_split")
        key("Results 8", f"{lab}: species-count error", me, loe, hie, source="clusters_*_vs.csv")
        if i:
            r = paired(gap_ari, s)
            key("Results 8", f"gap minus {lab}: ARI", r["diff"], r["ci_lo"], r["ci_hi"], source=f"Wilcoxon p = {r['p']:.2g}")
            r = paired(gap_err, e)
            key("Results 8", f"gap minus {lab}: count error", r["diff"], r["ci_lo"], r["ci_hi"], source=f"Wilcoxon p = {r['p']:.2g}")
    for ax in (a, b):
        ax.set_yticks(range(len(meths)), [m[1] for m in meths][::-1])
    a.set_xlim(0, 1.08)
    a.set_xlabel("ARI vs true species")
    a.set_title("Grouping accuracy", loc="left")
    b.set_yticklabels([])
    b.set_xlim(0, 45)
    b.set_xlabel("|groups - true new species|")
    b.set_title("Species-count error", loc="left")
    panel(a, "a")
    panel(b, "b")
    others = [("gap_align", "Calibrated gap"), ("ptp", "PTP"), ("kmer", "HDBSCAN"),
              ("phylo_tree", "Tree grouping")]
    for i, (m_, lab) in enumerate(others):
        y = len(others) - 1 - i
        m, lo, hi = ci_mean(by_split(cs, "ARI", method=m_))
        c.barh(y, m, 0.62, color=BLUE if i == 0 else GREY)
        c.errorbar(m, y, xerr=[[m - lo], [hi - m]], color=DARK, lw=0.7, capsize=1.5)
        c.text(hi + 0.01, y, f"{m:.2f}", va="center", fontsize=7, color=DARK)
        key("Results 8", f"{lab}: ARI (B4-called)", m, lo, hi, source="cluster_scores_per_split")
    c.set_yticks(range(len(others)), [o[1] for o in others][::-1])
    c.set_xlim(0, 1.08)
    c.set_xlabel("ARI vs true species")
    c.set_title("Other grouping methods", loc="left")
    panel(c, "c")
    fig.text(0.07, -0.08, "a, b: sequences called new by VSEARCH; c: sequences called new by raw k-mers (B4). "
             "Single linkage for fixed cut-offs. Mean of 30 splits, 95% CI.", fontsize=6.5, color=DARK)
    save_fig(fig, "fig5_grouping")


# ================================================================ Figure 6: real eDNA
REAL_RULES = [("vsearch97", "VSEARCH 97%"), ("b2_f1", "VSEARCH F1"), ("conf10", "Conformal 0.10"), ("conf05", "Conformal 0.05")]


def fig6():
    z = pd.read_csv(REAL / "zotu_calls.csv")
    perm = pd.read_csv(REAL / "net_check_permutation.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.8), gridspec_kw=dict(width_ratios=[1, 1.5], wspace=0.3))
    known, new = z[z.proxy == "known"], z[z.proxy == "new"]
    x = np.arange(len(REAL_RULES))
    for j, (sub, lab, colr) in enumerate([(new, "New fish caught", BLUE), (known, "Known fish called new", GREY)]):
        for i, (r, _) in enumerate(REAL_RULES):
            k = int(sub[r].astype(bool).sum())
            m, lo, hi = clopper(k, len(sub))
            xx = i + (j - 0.5) * 0.36
            a.bar(xx, 100 * m, 0.34, color=colr, label=lab if i == 0 else None)
            a.errorbar(xx, 100 * m, [[100 * (m - lo)], [100 * (hi - m)]], color=DARK, lw=0.7, capsize=1.5)
            a.text(xx, 100 * hi + 2, f"{k}/{len(sub)}", ha="center", fontsize=6, color=DARK)
            key("Results 11", f"{r}: {lab} (proxy labels)", m, lo, hi, source="zotu_calls (Clopper-Pearson)")
    a.set_xticks(x, ["97%", "F1", "α.10", "α.05"], fontsize=7)
    a.set_ylim(0, 118)
    a.set_ylabel("Share of proxy-labelled variants (%)")
    a.set_title(f"Proxy labels: {len(new)} new, {len(known)} known", loc="left")
    a.legend(frameon=False, loc="upper right", fontsize=6.5)
    panel(a, "a")
    # (b) net-fishing check, with and without the post-hoc fish filter
    groups = [("site_check.csv", "none", "All variants"), ("site_check_filtered.csv", "fish80", "Fish filter (post hoc)")]
    w = 0.18
    for gi, (f, filt, glab) in enumerate(groups):
        s = pd.read_csv(REAL / f).groupby("rule")[["zotus", "called_new", "new_fit_genus", "known_fit_genus"]].sum()
        for i, (r, rlab) in enumerate(REAL_RULES):
            nw = s.loc[r, "new_fit_genus"] / s.loc[r, "called_new"]
            kn = s.loc[r, "known_fit_genus"] / (s.loc[r, "zotus"] - s.loc[r, "called_new"])
            base = gi * (len(REAL_RULES) + 1) + i
            b.bar(base - w / 2 - 0.01, 100 * nw, w, color=BLUE, label="Called new" if (gi, i) == (0, 0) else None)
            b.bar(base + w / 2 + 0.01, 100 * kn, w, color=GREY, label="Called known" if (gi, i) == (0, 0) else None)
            p = perm[(perm["filter"] == filt) & (perm.rule == r) & (perm.level == "genus") & (perm.shuffle == "site")].p
            ptxt = "p<0.001" if len(p) and p.iloc[0] < 0.001 else (f"p={p.iloc[0]:.2f}" if len(p) else "")
            b.text(base, 100 * max(nw, kn) + 0.8, ptxt, ha="center", fontsize=5.5, color=DARK, rotation=90, va="bottom")
            key("Results 11", f"net check genus fit, {filt}, {r}: new", nw, source=f)
            key("Results 11", f"net check genus fit, {filt}, {r}: known", kn, source=f"{f}; site-shuffle p = {p.iloc[0] if len(p) else np.nan:.2g}")
    ticks = [gi * (len(REAL_RULES) + 1) + i for gi in range(2) for i in range(len(REAL_RULES))]
    short = {"vsearch97": "97%", "b2_f1": "F1", "conf10": "α.10", "conf05": "α.05"}
    b.set_xticks(ticks, [short[r[0]] for _ in groups for r in REAL_RULES], fontsize=6.5)
    for gi, (_, _, glab) in enumerate(groups):
        b.text(gi * (len(REAL_RULES) + 1) + 1.5, -4.2, glab, ha="center", fontsize=7, color=DARK, clip_on=False)
    b.text(0.0, -0.2, "97% = VSEARCH 97%, F1 = VSEARCH F1 cut-off, α = conformal",
           transform=b.transAxes, fontsize=6, color=DARK)
    b.set_ylim(0, 30)
    b.set_ylabel("Occurrences matching a net-caught\ngenus missing from the library (%)")
    b.set_title("Net-fishing check: new calls fit the missing fish", loc="left")
    b.legend(frameon=False, loc="upper right", fontsize=6.5)
    panel(b, "b")
    key("Results 11", "sequence variants", len(z), source="zotu_calls")
    save_fig(fig, "fig6_real_edna")


# ================================================================ Figure S1: tree help by age of the closest relative
def figS1():
    bins = ["< 30 My", "30-60 My", ">= 60 My"]
    meths = [("alignment nearest", "Nearest sequence (no tree)", GREY, "o"),
             ("phylo", "Tree placement (EPA-ng)", BLUE, "s"),
             ("raw+treecap90", "Raw k-mers + tree model", LBLUE, "^")]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), sharey=True, gridspec_kw=dict(wspace=0.1))
    for ax, (f, title, letter) in zip(axes, [("when_tree_helps_per_split.csv", "Teleo (64 bp)", "a"),
                                             ("when_tree_helps_full_per_split.csv", "Long 12S (about 600 bp)", "b")]):
        d = pd.read_csv(RES / f)
        d = d[d.isolation == "tree"]
        for j, (m_, lab, c, mk) in enumerate(meths):
            ms, los, his = [], [], []
            for bn in bins:
                m, lo, hi = ci_mean(d[(d.bin == bn) & (d.method == m_)].family)
                ms.append(100 * m), los.append(100 * lo), his.append(100 * hi)
                key("Supplement S1", f"{title}: {lab}, {bn}", m, lo, hi, source=f)
            x = np.arange(len(bins)) + (j - 1) * 0.08
            ax.plot(x, ms, color=c, lw=1.2, marker=mk, ms=4, label=lab)
            ax.vlines(x, los, his, color=c, lw=0.8)
        ns = [int(d[(d.bin == bn) & (d.method == "phylo")].n.sum()) for bn in bins]
        ax.set_xticks(range(len(bins)), [f"{b}\n(n = {n:,})" for b, n in zip(bins, ns)])
        ax.set_ylim(0, 100)
        ax.set_title(title, loc="left")
        panel(ax, letter)
    axes[0].set_ylabel("New fish placed in the\ncorrect family (%)")
    axes[0].legend(frameon=False, loc="lower left")
    fig.supxlabel("Age of the split to the closest species in the library (million years)", fontsize=8, y=-0.1)
    fig.suptitle("Grouped by tree age, the tree's advantage is small and shrinks for the most isolated species",
                 x=0.07, ha="left", y=1.04, fontsize=9, fontweight="bold")
    fig.text(0.07, -0.2, "Mean of 30 splits, 95% CI; n = new test sequences summed over splits. Model methods averaged "
             "over 5 seeds. Compare Fig. 4, which groups by sequence distance.", fontsize=6.5, color=DARK)
    save_fig(fig, "figS1_tree_helps_by_age")


def main():
    table1()
    table2()
    table3()
    fig2()
    fig3()
    fig4()
    fig5()
    fig6()
    figS1()
    k = pd.DataFrame(KEY)
    k.to_csv(TAB / "key_numbers.csv", index=False)
    print(f"wrote {len(k)} key numbers, 3 tables and 6 figures to {TAB.parent}")


if __name__ == "__main__":
    main()
