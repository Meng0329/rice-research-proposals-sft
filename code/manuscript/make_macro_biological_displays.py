#!/usr/bin/env python3
"""Create dense, publication-ready biological atlas displays for the rice SFT corpus."""

from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch, Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PROFILE = ROOT / "source_data/biological_question_profile"
FIG_DIR = ROOT / "figures"
ROWS_PATH = PROFILE / "record_candidate_labels.csv"

OBJECTIVES = [
    "Yield and plant architecture", "Grain quality and nutrition", "Abiotic stress adaptation",
    "Biotic stress resistance", "Flowering and development", "Nutrient use and fertilizer response",
    "Hybrid breeding and fertility", "Germplasm and diversity", "Physiology and plant processes",
    "Production environment and management",
]
OBJECTIVE_SHORT = {
    "Yield and plant architecture": "Yield and architecture", "Grain quality and nutrition": "Grain quality",
    "Abiotic stress adaptation": "Abiotic stress", "Biotic stress resistance": "Biotic stress",
    "Flowering and development": "Flowering and development", "Nutrient use and fertilizer response": "Nutrient use",
    "Hybrid breeding and fertility": "Hybrid and fertility", "Germplasm and diversity": "Germplasm and diversity",
    "Physiology and plant processes": "Physiology", "Production environment and management": "Production system",
}
TRAITS = [
    "Yield", "Plant or panicle architecture", "Grain quality or composition", "Grain shape or weight",
    "Drought or water response", "Disease or pest resistance", "Salinity or ion response",
    "Flowering or maturity", "Root or seedling traits", "Nutrient content or use",
    "Photosynthetic or physiological traits", "Fertility or hybrid performance",
]
TRAIT_SHORT = {
    "Yield": "Yield", "Plant or panicle architecture": "Architecture", "Grain quality or composition": "Grain quality",
    "Grain shape or weight": "Grain size and weight", "Drought or water response": "Drought and water",
    "Disease or pest resistance": "Disease and pest", "Salinity or ion response": "Salinity and ions",
    "Flowering or maturity": "Flowering and maturity", "Root or seedling traits": "Root and seedling",
    "Nutrient content or use": "Nutrient use", "Photosynthetic or physiological traits": "Physiology",
    "Fertility or hybrid performance": "Fertility and hybrid",
}
SCALES = [
    "Cultivar or breeding line", "Germplasm or wild relative", "Breeding or mapping population",
    "Gene, allele, QTL or locus", "Pathogen, pest or microbial partner",
    "Plant organ, tissue or developmental material", "Field environment or management",
    "Pathway or regulatory network",
]
SCALE_SHORT = {
    "Cultivar or breeding line": "Cultivar and line", "Germplasm or wild relative": "Wild relative and germplasm",
    "Breeding or mapping population": "Population", "Gene, allele, QTL or locus": "Gene and QTL",
    "Pathogen, pest or microbial partner": "Pathogen and pest", "Plant organ, tissue or developmental material": "Tissue and organ",
    "Field environment or management": "Field and environment", "Pathway or regulatory network": "Pathway and network",
}
METHODS = [
    "Field or multi-environment evaluation", "Crossing, selection or introgression", "Genetic mapping or association",
    "Functional genetics or gene editing", "Omics or molecular profiling", "Controlled treatment or laboratory experiment",
    "Phenotyping or physiological assay", "Computational or genomic resource", "Review or resource synthesis",
]
METHOD_SHORT = {
    "Field or multi-environment evaluation": "Field trials", "Crossing, selection or introgression": "Crossing and selection",
    "Genetic mapping or association": "QTL and GWAS", "Functional genetics or gene editing": "Functional genetics and editing",
    "Omics or molecular profiling": "Omics", "Controlled treatment or laboratory experiment": "Controlled and lab",
    "Phenotyping or physiological assay": "Phenotyping", "Computational or genomic resource": "Computational and resource",
    "Review or resource synthesis": "Review and resource",
}
FUNCTIONS = [
    "Disease resistance", "Abiotic stress response", "Grain and nutritional traits", "Yield and plant architecture",
    "Flowering and development", "Nutrient use", "Fertility and hybrid performance", "Physiology and signalling",
]
FUNCTION_SHORT = {
    "Disease resistance": "Disease resistance", "Abiotic stress response": "Abiotic stress",
    "Grain and nutritional traits": "Grain and nutrition", "Yield and plant architecture": "Yield and architecture",
    "Flowering and development": "Development", "Nutrient use": "Nutrient use",
    "Fertility and hybrid performance": "Fertility and hybrid", "Physiology and signalling": "Physiology and signalling",
}
PALETTE = ["#176B87", "#D95F59", "#4E9F73", "#E3A72F", "#8F6BB3", "#2C82C9", "#C66A2B", "#5C677D", "#BA4A7A", "#3C8DAD"]
INK = "#17242D"
MUTED = "#64747D"
GRID = "#D8E0E4"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 8, "xtick.labelsize": 6,
    "ytick.labelsize": 6, "axes.linewidth": 0.55, "axes.spines.top": False,
    "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
})


def read_rows() -> list[dict]:
    with ROWS_PATH.open(encoding="utf-8-sig", newline="") as stream:
        source = list(csv.DictReader(stream))
    rows = []
    for row in source:
        if row["question_validator"] != "valid":
            continue
        rows.append({
            "objectives": set(filter(None, row["objective_signals"].split("; "))),
            "traits": set(filter(None, row["trait_signals"].split("; "))),
            "scales": set(filter(None, row["biological_scale_signals"].split("; "))),
            "methods": set(filter(None, row["method_signals"].split("; "))),
            "genes": set(filter(None, row["question_gene_locus_symbols"].split("; "))),
            "year": int(row["verified_crossref_year"]) if row["verified_crossref_year"].isdigit() else None,
        })
    return rows


def counts(rows: list[dict], left: str, right: str) -> np.ndarray:
    left_order = {"objectives": OBJECTIVES, "traits": TRAITS, "scales": SCALES, "methods": METHODS}[left]
    right_order = {"objectives": OBJECTIVES, "traits": TRAITS, "scales": SCALES, "methods": METHODS}[right]
    matrix = np.zeros((len(left_order), len(right_order)), dtype=int)
    for i, a in enumerate(left_order):
        for j, b in enumerate(right_order):
            matrix[i, j] = sum(a in row[left] and b in row[right] for row in rows)
    return matrix


def save_pdf_png_svg(fig: plt.Figure, outbase: Path) -> None:
    outbase.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outbase.with_suffix(".pdf"), dpi=600, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(outbase.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(outbase.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.05)
    lines = outbase.with_suffix(".svg").read_text(encoding="utf-8").splitlines()
    outbase.with_suffix(".svg").write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")
    plt.close(fig)


def draw_bubble_matrix(ax: plt.Axes, matrix: np.ndarray, row_labels: list[str], col_labels: list[str], title: str) -> None:
    row_totals = matrix.sum(axis=1)
    percent = np.divide(matrix, row_totals[:, None], out=np.zeros_like(matrix, dtype=float), where=row_totals[:, None] > 0) * 100
    max_count = max(int(matrix.max()), 1)
    norm = Normalize(vmin=0, vmax=max(float(percent.max()), 1))
    cmap = plt.get_cmap("viridis")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            n = int(matrix[i, j])
            if n == 0:
                ax.scatter(j, i, s=5, facecolors="none", edgecolors="#D4DDE1", linewidths=0.25)
                continue
            ax.scatter(j, i, s=18 + 700 * math.sqrt(n / max_count), c=[cmap(norm(percent[i, j]))],
                       edgecolors="white", linewidths=0.45, zorder=3)
            if n >= max(20, int(max_count * 0.05)):
                ax.text(j, i, str(n), ha="center", va="center", fontsize=4.8,
                        color="white" if percent[i, j] > 42 else INK, zorder=4)
    ax.set_xlim(-0.7, matrix.shape[1] - 0.3)
    ax.set_ylim(matrix.shape[0] - 0.35, -0.65)
    ax.set_xticks(np.arange(len(col_labels)), col_labels, rotation=55, ha="right")
    ax.set_yticks(np.arange(len(row_labels)), row_labels)
    ax.grid(color=GRID, linewidth=0.35, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left", fontweight="bold", color=INK, pad=7)


def draw_alluvial(ax: plt.Axes, rows: list[dict]) -> None:
    obj_counts = Counter(o for row in rows for o in row["objectives"])
    scale_counts = Counter(s for row in rows for s in row["scales"])
    method_counts = Counter(m for row in rows for m in row["methods"])
    obj_order = sorted(OBJECTIVES, key=lambda x: obj_counts[x], reverse=True)
    scale_order = sorted(SCALES, key=lambda x: scale_counts[x], reverse=True)
    method_order = sorted(METHODS, key=lambda x: method_counts[x], reverse=True)
    x_positions = [0.02, 0.49, 0.96]
    axes_orders = [obj_order, scale_order, method_order]
    axis_counts = [obj_counts, scale_counts, method_counts]
    y_positions: list[dict[str, float]] = []
    for order, counter in zip(axes_orders, axis_counts):
        step = 1 / (len(order) + 1)
        y_positions.append({label: 0.96 - (i + 1) * step for i, label in enumerate(order)})
    def curve(x0, y0, x1, y1, width, color, alpha):
        dx = (x1 - x0) * 0.43
        verts = [(x0, y0 - width), (x0 + dx, y0 - width), (x1 - dx, y1 - width), (x1, y1 - width),
                 (x1, y1 + width), (x1 - dx, y1 + width), (x0 + dx, y0 + width), (x0, y0 + width), (x0, y0 - width)]
        codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
                 MplPath.LINETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4, MplPath.CLOSEPOLY]
        ax.add_patch(PathPatch(MplPath(verts, codes), facecolor=color, edgecolor="none", alpha=alpha, zorder=1))
    for oi, obj in enumerate(obj_order):
        pairs = [(scale, sum(obj in r["objectives"] and scale in r["scales"] for r in rows)) for scale in scale_order]
        pairs = [(s, n) for s, n in pairs if n > 0]
        for s, n in sorted(pairs, key=lambda p: p[1], reverse=True)[:5]:
            curve(x_positions[0] + 0.018, y_positions[0][obj], x_positions[1] - 0.018, y_positions[1][s],
                  0.004 + 0.018 * n / max(obj_counts[obj], 1), PALETTE[oi % len(PALETTE)], 0.20)
    for si, scale in enumerate(scale_order):
        pairs = [(method, sum(scale in r["scales"] and method in r["methods"] for r in rows)) for method in method_order]
        for method, n in sorted([(m, n) for m, n in pairs if n > 0], key=lambda p: p[1], reverse=True)[:5]:
            curve(x_positions[1] + 0.018, y_positions[1][scale], x_positions[2] - 0.018, y_positions[2][method],
                  0.003 + 0.014 * n / max(scale_counts[scale], 1), "#6D7B83", 0.14)
    for axis_index, (x, order, counter) in enumerate(zip(x_positions, axes_orders, axis_counts)):
        for label in order:
            y = y_positions[axis_index][label]
            color = PALETTE[obj_order.index(label) % len(PALETTE)] if axis_index == 0 else "#8AA8B4"
            ax.add_patch(Rectangle((x - 0.012, y - 0.018), 0.024, 0.036, facecolor=color, edgecolor="white", linewidth=0.5, zorder=3))
            short = OBJECTIVE_SHORT.get(label, SCALE_SHORT.get(label, METHOD_SHORT.get(label, label)))
            align = "right" if axis_index == 0 else "left"
            tx = x - 0.018 if axis_index == 0 else x + 0.018
            ax.text(tx, y, f"{short}  {counter[label]:,}", ha=align, va="center", fontsize=5.8, color=INK,
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.25})
    ax.text(x_positions[0], 1.04, "OBJECTIVES", ha="center", va="bottom", fontsize=7.0, fontweight="bold", color=INK)
    ax.text(x_positions[1], 1.04, "BIOLOGICAL SCALE", ha="center", va="bottom", fontsize=7.0, fontweight="bold", color=INK)
    ax.text(x_positions[2], 1.04, "METHOD SIGNAL", ha="center", va="bottom", fontsize=7.0, fontweight="bold", color=INK)
    ax.set_xlim(-0.13, 1.13)
    ax.set_ylim(-0.10, 1.08)
    ax.axis("off")
    ax.set_title("Cross-scale links from objective to biological object and method", loc="left", x=0.10,
                 fontweight="bold", color=INK, pad=7)


def save_figure6(rows: list[dict]) -> None:
    objective_order = sorted(OBJECTIVES, key=lambda x: sum(x in r["objectives"] for r in rows), reverse=True)
    trait_order = sorted(TRAITS, key=lambda x: sum(x in r["traits"] for r in rows), reverse=True)
    matrix = counts(rows, "objectives", "traits")
    matrix = matrix[[OBJECTIVES.index(x) for x in objective_order]][:, [TRAITS.index(x) for x in trait_order]]
    fig = plt.figure(figsize=(10.8, 12.2), facecolor="white")
    gs = fig.add_gridspec(3, 2, height_ratios=[1.18, 1.16, 0.82], hspace=0.30, wspace=0.44, top=0.94)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])
    ax_d = fig.add_subplot(gs[2, :])
    # The lower atlas spans the full plotting width; its vertical position is
    # separated from the heatmaps while remaining close to the cohort panel.
    top_left = 0.045
    top_right = 0.965
    c_position = ax_c.get_position()
    d_position = ax_d.get_position()
    ax_c.set_position([top_left, c_position.y0 - 0.025, top_right - top_left, c_position.height - 0.01])
    ax_d.set_position([top_left, d_position.y0 + 0.022, top_right - top_left, d_position.height + 0.005])

    trait_share = np.divide(matrix, matrix.sum(axis=1, keepdims=True), out=np.zeros_like(matrix, dtype=float), where=matrix.sum(axis=1, keepdims=True) > 0) * 100
    heat_a = ax_a.imshow(trait_share, aspect="auto", cmap="YlOrBr", vmin=0, vmax=max(1, float(trait_share.max())))
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            n = int(matrix[i, j])
            if n:
                ax_a.text(j, i, f"{n:,}\n{trait_share[i, j]:.0f}%", ha="center", va="center", fontsize=6.2,
                          color="white" if trait_share[i, j] > 48 else INK, linespacing=0.95)
    ax_a.set_xticks(np.arange(len(trait_order)), [TRAIT_SHORT[x] for x in trait_order], rotation=53, ha="right")
    ax_a.set_yticks(np.arange(len(objective_order)), [OBJECTIVE_SHORT[x] for x in objective_order])
    ax_a.set_title("Objective by focal trait", loc="left", fontweight="bold", color=INK, pad=7)
    ax_a.tick_params(axis="x", labelsize=4.8)
    ax_a.grid(False)
    fig.colorbar(heat_a, ax=ax_a, fraction=0.04, pad=0.025, label="Share within objective (%)")

    objective_matrix = counts(rows, "objectives", "objectives")
    objective_matrix = objective_matrix[[OBJECTIVES.index(x) for x in objective_order]][:, [OBJECTIVES.index(x) for x in objective_order]]
    objective_n = np.asarray([sum(obj in row["objectives"] for row in rows) for obj in objective_order])
    conditional = np.divide(objective_matrix, objective_n[:, None], out=np.zeros_like(objective_matrix, dtype=float), where=objective_n[:, None] > 0) * 100
    heat_b = ax_b.imshow(conditional, aspect="auto", cmap="PuBuGn", vmin=0, vmax=100)
    for i in range(conditional.shape[0]):
        for j in range(conditional.shape[1]):
            ax_b.text(j, i, f"{conditional[i, j]:.0f}%", ha="center", va="center", fontsize=6.4,
                      color="white" if conditional[i, j] > 60 else INK)
    ax_b.set_xticks(np.arange(len(objective_order)), [OBJECTIVE_SHORT[x] for x in objective_order], rotation=52, ha="right")
    ax_b.set_yticks(np.arange(len(objective_order)), [OBJECTIVE_SHORT[x] for x in objective_order])
    ax_b.set_title("Conditional overlap between objectives", loc="left", fontweight="bold", color=INK, pad=7)
    ax_b.tick_params(axis="x", labelsize=4.8)
    fig.colorbar(heat_b, ax=ax_b, fraction=0.04, pad=0.025, label="Co-occurring records (%)")

    draw_alluvial(ax_c, rows)

    periods = [(2000, 2004), (2005, 2009), (2010, 2014), (2015, 2019), (2020, 2024), (2025, 2026)]
    period_rows = [[r for r in rows if r["year"] is not None and start <= r["year"] <= end] for start, end in periods]
    annual_share = np.zeros((len(OBJECTIVES), len(periods)), dtype=float)
    for pi, subset in enumerate(period_rows):
        for oi, objective in enumerate(OBJECTIVES):
            annual_share[oi, pi] = 100 * sum(objective in r["objectives"] for r in subset) / len(subset) if subset else 0
    xvals = np.arange(len(periods))
    ordered_annual = annual_share[[OBJECTIVES.index(obj) for obj in objective_order]]
    heat_d = ax_d.imshow(ordered_annual, aspect="auto", cmap="YlOrRd", vmin=0,
                         vmax=max(1, float(np.nanmax(ordered_annual))))
    for i in range(ordered_annual.shape[0]):
        for j in range(ordered_annual.shape[1]):
            value = ordered_annual[i, j]
            ax_d.text(j, i, f"{value:.0f}%", ha="center", va="center", fontsize=5.7,
                      color="white" if value > 26 else INK, fontweight="bold")
    ax_d.set_xticks(xvals, [f"{a}–{b}\nn={len(subset):,}" for (a, b), subset in zip(periods, period_rows)])
    ax_d.set_yticks(np.arange(len(objective_order)), [OBJECTIVE_SHORT[obj] for obj in objective_order])
    ax_d.set_xlabel("Verified publication cohort")
    ax_d.set_ylabel("Objective signal")
    ax_d.set_title("Objective prevalence across verified publication cohorts", loc="left", fontweight="bold", color=INK, pad=7)
    ax_d.set_xticks(np.arange(-0.5, len(periods), 1), minor=True)
    ax_d.set_yticks(np.arange(-0.5, len(objective_order), 1), minor=True)
    ax_d.grid(which="minor", color="white", linewidth=0.8)
    ax_d.set_axisbelow(True)
    ax_d.spines["top"].set_visible(False)
    ax_d.spines["right"].set_visible(False)
    ax_d.tick_params(which="minor", bottom=False, left=False)
    fig.colorbar(heat_d, ax=ax_d, fraction=0.025, pad=0.018, label="Questions with objective signal (%)")
    for label, ax in (("A", ax_a), ("C", ax_c), ("D", ax_d)):
        position = ax.get_position()
        fig.text(0.045, position.y1 + 0.005, label, ha="left", va="bottom",
                 fontsize=10, fontweight="bold", color=INK)
    b_position = ax_b.get_position()
    fig.text(b_position.x0 - 0.035, b_position.y1 + 0.005, "B", ha="left", va="bottom",
             fontsize=10, fontweight="bold", color=INK)
    fig.suptitle("Biological structure of 6,238 rice-breeding questions", fontsize=12, fontweight="bold", color=INK, y=0.98)
    save_pdf_png_svg(fig, FIG_DIR / "figure2_biological_question_atlas")


def save_figure7(rows: list[dict]) -> None:
    gene_counts = Counter(g for r in rows for g in r["genes"])
    genes = [g for g, _ in gene_counts.most_common(30)]
    matrix = np.zeros((len(genes), len(OBJECTIVES)), dtype=int)
    for i, gene in enumerate(genes):
        gene_rows = [r for r in rows if gene in r["genes"]]
        for j, objective in enumerate(OBJECTIVES):
            matrix[i, j] = sum(objective in r["objectives"] for r in gene_rows)
    retention = {}
    with (PROFILE / "named_gene_locus_candidates.csv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            retention[row["symbol_candidate"]] = float(row["target_retention_percent"])
    function_by_gene = defaultdict(list)
    crosswalk = PROFILE / "gene_reference_crosswalk.csv"
    if crosswalk.exists():
        with crosswalk.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                function_by_gene[row["question_symbol_candidate"]] = [x for x in row["functional_categories"].split("; ") if x]
    fig = plt.figure(figsize=(10.5, 9.6), facecolor="white")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.52, 0.9], width_ratios=[1.46, 0.9], hspace=0.32, wspace=0.28)
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])
    max_count = max(int(matrix.max()), 1)
    for i, gene in enumerate(genes):
        for j, objective in enumerate(OBJECTIVES):
            n = int(matrix[i, j])
            if n == 0:
                ax_a.scatter(j, i, s=4, facecolors="none", edgecolors="#D4DDE1", linewidths=0.25)
            else:
                ax_a.scatter(j, i, s=15 + 950 * math.sqrt(n / max_count), c=[PALETTE[j % len(PALETTE)]], alpha=0.96,
                             edgecolors="white", linewidths=0.4, zorder=3)
                ax_a.text(j, i, str(n), ha="center", va="center", fontsize=6.8, color="white" if n >= 3 else INK, zorder=4)
    ax_a.set_yticks(np.arange(len(genes)), [f"{g}  n={gene_counts[g]}  retain={retention.get(g, 0):.0f}%" for g in genes])
    ax_a.set_xticks(np.arange(len(OBJECTIVES)), [OBJECTIVE_SHORT[o] for o in OBJECTIVES], rotation=43, ha="right")
    ax_a.set_ylim(len(genes) - 0.4, -0.7)
    ax_a.set_axisbelow(True)
    ax_a.grid(color=GRID, linewidth=0.3, zorder=0)
    ax_a.set_axisbelow(True)
    ax_a.set_title("Top 30 named gene and locus candidates across question objectives", loc="left", fontweight="bold", color=INK, pad=7)
    ax_a.text(-0.11, 1.02, "a", transform=ax_a.transAxes, fontsize=10, fontweight="bold", color=INK)

    objectives = OBJECTIVES
    functions = FUNCTIONS
    fm = np.zeros((len(functions), len(objectives)), dtype=int)
    with (PROFILE / "gene_function_by_question_objective.csv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["functional_category"] in functions and row["question_objective"] in objectives:
                fm[functions.index(row["functional_category"]), objectives.index(row["question_objective"])] = int(row["cooccurring_records"])
    im = ax_b.imshow(fm, aspect="auto", cmap="magma", vmin=0, vmax=max(1, int(fm.max())))
    ax_b.set_yticks(np.arange(len(functions)), [FUNCTION_SHORT[x] for x in functions])
    ax_b.set_xticks(np.arange(len(objectives)), [OBJECTIVE_SHORT[x] for x in objectives], rotation=47, ha="right")
    for i in range(fm.shape[0]):
        for j in range(fm.shape[1]):
            if fm[i, j]:
                ax_b.text(j, i, str(fm[i, j]), ha="center", va="center", fontsize=8.5,
                          fontweight="bold", color=INK if fm[i, j] > fm.max() * 0.45 else "white")
    ax_b.set_title("Curated function x question objective (n)", loc="left", fontweight="bold", color=INK, pad=7)
    ax_b.tick_params(axis="x", labelsize=4.7)

    x = np.asarray([gene_counts[g] for g in genes], dtype=float)
    y = np.asarray([retention.get(g, 0) for g in genes], dtype=float)
    breadth = np.asarray([sum(matrix[i, :] > 0) for i in range(len(genes))])
    ax_c.scatter(x, y, s=28 + 9 * breadth, c=breadth, cmap="viridis", edgecolors="white", linewidths=0.55, alpha=0.9)
    ax_c.set_xlabel("Question records mentioning candidate")
    ax_c.set_ylabel("Exact candidate retention in target (%)")
    ax_c.set_ylim(-4, 104)
    ax_c.set_axisbelow(True)
    ax_c.grid(color=GRID, linewidth=0.4, zorder=0)
    ax_c.set_title("Named-candidate carry-through", loc="left", fontweight="bold", color=INK, pad=7)
    cbar = fig.colorbar(im, ax=ax_b, fraction=0.028, pad=0.02)
    cbar.set_label("Question records (black = 0)", fontsize=5.5)
    ax_b.text(-0.11, 1.025, "b", transform=ax_b.transAxes, fontsize=10, fontweight="bold", color=INK)
    ax_c.text(-0.11, 1.025, "c", transform=ax_c.transAxes, fontsize=10, fontweight="bold", color=INK)
    fig.subplots_adjust(top=0.95, bottom=0.075, left=0.13, right=0.98)
    fig.suptitle("Gene and locus context with question-to-plan continuity", fontsize=12, fontweight="bold", color=INK, y=0.988)
    save_pdf_png_svg(fig, FIG_DIR / "figure3_gene_locus_profile")


def render_table9(rows: list[dict]) -> None:
    obj_order = sorted(OBJECTIVES, key=lambda x: sum(x in r["objectives"] for r in rows), reverse=True)
    trait_counts = {o: Counter(t for r in rows if o in r["objectives"] for t in r["traits"]) for o in obj_order}
    scale_counts = {o: Counter(s for r in rows if o in r["objectives"] for s in r["scales"]) for o in obj_order}
    method_counts = {o: Counter(m for r in rows if o in r["objectives"] for m in r["methods"]) for o in obj_order}
    years = [("2000–09", range(2000, 2010)), ("2010–19", range(2010, 2020)),
             ("2020–24", range(2020, 2025)), ("2025–26*", range(2025, 2027))]

    table_objective_short = {
        "Yield and plant architecture": "Yield and\narchitecture",
        "Germplasm and diversity": "Germplasm and\ndiversity",
        "Hybrid breeding and fertility": "Hybrid and\nfertility",
        "Flowering and development": "Flowering and\ndevelopment",
    }
    table_trait_short = {**TRAIT_SHORT, "Grain shape or weight": "Grain size and\nweight"}
    table_scale_short = {
        **SCALE_SHORT,
        "Germplasm or wild relative": "Wild germplasm",
        "Field environment or management": "Field setting",
        "Pathway or regulatory network": "Pathway network",
        "Cultivar or line": "Cultivar lines",
        "Tissue or organ": "Tissue organ",
        "Gene or QTL": "Gene or QTL",
    }
    table_method_short = {
        **METHOD_SHORT,
        "Crossing, selection or introgression": "Crossing",
        "Functional genetics or gene editing": "Functional\ngenetics",
        "Computational or genomic resource": "Computational",
        "Controlled treatment or laboratory experiment": "Lab experiment",
        "Review or resource synthesis": "Review",
    }

    def top(counter, mapping):
        return "\n".join(f"{mapping[k]}  {v:,}" for k, v in counter.most_common(2)) or "–"

    rows_out = []
    for obj in obj_order:
        subset = [r for r in rows if obj in r["objectives"]]
        den = len(subset)
        year_values = [
            f"{100 * sum(r['year'] in yr_range for r in subset) / den:.0f}%" if den else "–"
            for _, yr_range in years
        ]
        rows_out.append([
            table_objective_short.get(obj, OBJECTIVE_SHORT[obj]), f"{den:,}", f"{100 * den / len(rows):.1f}%",
            top(trait_counts[obj], table_trait_short), top(scale_counts[obj], table_scale_short),
            top(method_counts[obj], table_method_short), *year_values,
            "–", "–", "–", "–", "–", f"{sum(bool(r["genes"]) for r in subset):,}", "–",
        ])

    # Add lexical carry-through and mapped-gene counts from the released analysis tables.
    carry = {}
    carry_path = PROFILE / "objective_section_carrythrough.csv"
    if carry_path.exists():
        with carry_path.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                if row["proposal_section"] in {"Primary hypothesis", "Experimental design", "Measurements and endpoints", "Quantitative analysis", "Breeding/scientific decision"}:
                    carry[(row["objective"], row["proposal_section"])] = float(row["lexical_carrythrough_percent"])
    mapped_symbols = set()
    crosswalk = PROFILE / "gene_reference_crosswalk.csv"
    if crosswalk.exists():
        with crosswalk.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                if row["match_status"] == "unique_reference_identifier" and row["functional_categories"]:
                    mapped_symbols.add(row["question_symbol_candidate"])
    for row, obj in zip(rows_out, obj_order):
        subset = [r for r in rows if obj in r["objectives"]]
        row[6:10] = [
            f"{100 * sum(record['year'] in yr for record in subset) / len(subset):.0f}%" if subset else "–"
            for _, yr in years
        ]
        row[10:15] = [
            f"{carry.get((obj, section), 0):.0f}%"
            for section in ("Primary hypothesis", "Experimental design", "Measurements and endpoints", "Quantitative analysis", "Breeding/scientific decision")
        ]
        row[15] = f"{sum(bool(record["genes"]) for record in subset):,}"
        row[16] = f"{sum(bool(record["genes"] & mapped_symbols) for record in subset):,}"

    subheaders = [
        "Objective", "N", "Share", "Leading\ntraits", "Material\nscales", "Methods",
        "2000\n–09", "2010\n–19", "2020\n–24", "2025\n–26*", "Hyp.", "Des.",
        "Meas.", "Anal.", "Dec.", "Name", "Map",
    ]
    widths = [0.115, 0.035, 0.04, 0.13, 0.13, 0.12, 0.04, 0.04, 0.04, 0.04,
              0.04, 0.04, 0.04, 0.04, 0.04, 0.035, 0.035]
    fig, ax = plt.subplots(figsize=(10.4, 6.25), facecolor="white")
    ax.axis("off")
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    table_bbox = [0.01, 0.02, 0.98, 0.96]
    x0, y0, table_w, table_h = table_bbox
    row_weights = [0.75, 0.9] + [1.0] * len(rows_out)
    row_heights = [table_h * weight / sum(row_weights) for weight in row_weights]
    x_positions = [x0 + table_w * sum(widths[:i]) for i in range(len(widths) + 1)]
    y_top = y0 + table_h

    def draw_cell(x: float, y: float, width: float, height: float, text: str, *,
                  header: bool = False, bold: bool = False, align: str = "left") -> None:
        ax.add_patch(Rectangle(
            (x, y), width, height, transform=ax.transAxes,
            facecolor="#F3AD6B" if header else "white",
            edgecolor="#111111", linewidth=0.58 if header else 0.42,
        ))
        inset = 0.004 if align == "left" else 0.0
        ax.text(x + width / 2 if align == "center" else x + inset, y + height / 2, text,
                transform=ax.transAxes, ha=align, va="center", color="#111111",
                fontsize=5.8 if header else 5.85, fontweight="bold" if header or bold else "normal",
                linespacing=1.04)

    first_header_height, second_header_height = row_heights[:2]
    draw_cell(x_positions[0], y_top - first_header_height - second_header_height,
              table_w * widths[0], first_header_height + second_header_height,
              "Objective", header=True, align="center")
    groups = [(1, 3, "Volume"), (3, 6, "Content"), (6, 10, "Cohort"),
              (10, 15, "Plan"), (15, 17, "Genes")]
    for start, end, label in groups:
        draw_cell(x_positions[start], y_top - first_header_height,
                  x_positions[end] - x_positions[start], first_header_height,
                  label, header=True, align="center")
    second_header_top = y_top - first_header_height
    for column in range(1, len(widths)):
        draw_cell(x_positions[column], second_header_top - second_header_height,
                  table_w * widths[column], second_header_height,
                  subheaders[column], header=True, align="center")

    data_top = second_header_top - second_header_height
    for row_index, row in enumerate(rows_out):
        height = row_heights[row_index + 2]
        bottom = data_top - height
        for column, value in enumerate(row):
            draw_cell(x_positions[column], bottom, table_w * widths[column], height, value,
                      bold=(column == 0), align="left" if column in {0, 3, 4, 5} else "center")
        data_top = bottom
    outbase = FIG_DIR / "table2_biological_objective_profile"
    save_pdf_png_svg(fig, outbase)
    source = ROOT / "source_data/macro_biological_objective_atlas.csv"
    source.parent.mkdir(parents=True, exist_ok=True)
    with source.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(subheaders)
        writer.writerows(rows_out)


def main() -> None:
    rows = read_rows()
    if len(rows) != 6238:
        raise ValueError(f"Expected 6,238 validator-valid rows, found {len(rows)}")
    save_figure6(rows)
    save_figure7(rows)
    render_table9(rows)
    print(f"Wrote Figures 2-3 and Table 2 for {len(rows):,} valid questions")


if __name__ == "__main__":
    main()
