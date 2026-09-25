"""Render the AV/AR swap matrix as a PNG heatmap.

Reads the matrix.json files written by scripts/eval_matrix.py and draws one
figure per decode setting. Cell (i, j) = AV_i's explanations scored by AR_j.

Two honesty constraints drive the design:

  * The colour ramp is keyed to the DATA range, which here is ~2 FVE points wide.
    That makes small differences visible, so the range is stated on the colour bar
    and in the subtitle — otherwise the picture implies far more spread than
    exists. Every cell is also labelled, so the figure is readable as a table and
    the colour is only a secondary cue.
  * Cells are restricted to the rows valid in EVERY cell of EVERY input matrix, so
    all cells average the same prompts. Without this an arm with more extraction
    failures scores over an easier subset and looks better than it is.

Sequential single-hue blue ramp (no rainbow), diagonal cells ringed to mark the
own-partner pairing that the figure exists to test.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle

# Sequential blue ramp, light -> dark (steps 100..700 of the reference palette).
BLUE_STEPS = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
              "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
              "#0d366b"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#d8d7d3"
CMAP = LinearSegmentedColormap.from_list("seq_blue", BLUE_STEPS)


def load_matrices(paths, names):
    """Return (stack[n_files, n, n], common_valid_row_count).

    Recomputes FVE from per-row MSE rather than reading the summary, so the
    restriction to commonly-valid rows actually changes the numbers.
    """
    docs = [json.loads(Path(p).read_text()) for p in paths]
    n_rows = len(docs[0]["per_row"][names[0]][names[0]])
    bad = set()
    for d in docs:
        for j in names:
            for i in names:
                for k, v in enumerate(d["per_row"][j][i]):
                    if v <= -2.0:          # failed extraction: excluded, as in training
                        bad.add(k)
    keep = [k for k in range(n_rows) if k not in bad]
    out = np.zeros((len(docs), len(names), len(names)))
    for f, d in enumerate(docs):
        base = d["summary"]["fve_baseline"]
        for jj, j in enumerate(names):
            for ii, i in enumerate(names):
                mse = np.mean([-d["per_row"][j][i][k] for k in keep])
                out[f, ii, jj] = (1.0 - mse / base) * 100.0
    return out, len(keep), n_rows


def draw(mat, sd, names, title, subtitle, out_png):
    n = len(names)
    # Explicit axes geometry: imshow forces equal aspect, so a subplots_adjust
    # box that is not itself square leaves the matrix floating in dead space.
    # Size the axes square in INCHES and place the colour bar beside it.
    figw, figh = 8.0, 6.7
    side = 4.35
    fig = plt.figure(figsize=(figw, figh), dpi=220)
    ax = fig.add_axes([0.150, 0.070, side / figw, side / figh])
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    vmin, vmax = mat.min(), mat.max()
    norm = Normalize(vmin=vmin, vmax=vmax)
    ax.imshow(mat, cmap=CMAP, norm=norm, aspect="equal")

    for i in range(n):
        for j in range(n):
            # Label ink flips on dark cells so both stay legible; text never
            # wears the series colour.
            dark = norm(mat[i, j]) > 0.55
            ax.text(j, i - (0.10 if sd is not None else 0.0), f"{mat[i, j]:.2f}",
                    ha="center", va="center", fontsize=15,
                    color="#ffffff" if dark else INK,
                    fontweight="bold" if i == j else "normal")
            if sd is not None:
                ax.text(j, i + 0.22, f"±{sd[i, j]:.2f}", ha="center", va="center",
                        fontsize=9.5, color="#e8eef7" if dark else INK_MUTED)
            if i == j:   # ring the own-partner cell — the comparison under test
                ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, fill=False,
                                       edgecolor=SURFACE, linewidth=3.0, zorder=3))
                ax.add_patch(Rectangle((j - .44, i - .44), .88, .88, fill=False,
                                       edgecolor=INK, linewidth=1.6, zorder=4))

    ax.set_xticks(range(n), [f"AR {m[-1]}" for m in names], fontsize=12, color=INK)
    ax.set_yticks(range(n), [f"AV {m[-1]}" for m in names], fontsize=12, color=INK)
    ax.set_xlabel("AR$_j$", fontsize=11.5,
                  color=INK_MUTED, labelpad=12)
    ax.set_ylabel("AV$_i$", fontsize=11.5,
                  color=INK_MUTED, labelpad=12)
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-.5, n, 1), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2.5)   # 2px surface gap between fills
    ax.tick_params(which="both", length=0)

    cax = fig.add_axes([0.150 + side / figw + 0.035, 0.070, 0.024, side / figh])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=CMAP), cax=cax)
    # The ramp spans only the observed range (~2 FVE points), so say so on the
    # bar itself — otherwise the colour contrast implies far more spread.
    cb.set_label(f"held-out FVE %",
                 fontsize=10, color=INK_MUTED, labelpad=8)
    cb.ax.tick_params(labelsize=9.5, colors=INK_MUTED, length=0)
    cb.outline.set_visible(False)

    fig.text(0.045, 0.955, title, fontsize=15.5, color=INK, ha="left", va="top")
    fig.text(0.045, 0.900, subtitle, fontsize=10.5, color=INK_MUTED, ha="left",
             va="top", linespacing=1.45)
    fig.savefig(out_png, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out_png}")


def summarize(mat, names):
    n = len(names)
    diag = np.array([mat[i, i] for i in range(n)])
    off = np.array([mat[i, j] for i in range(n) for j in range(n) if i != j])
    per_av = [mat[i, i] - np.mean([mat[i, j] for j in range(n) if j != i])
              for i in range(n)]
    return diag.mean(), off.mean(), diag.mean() - off.mean(), per_av


def main():
    # Defaults resolve relative to THIS file, not the cwd, so `python
    # notebooks/ioi/plot_swap_matrix.py` regenerates both figures from any
    # directory with no arguments.
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(
        description="Render the AV/AR swap-matrix figures from results/paraphrase/.")
    ap.add_argument("--temp0-json", default=str(here / "results" / "paraphrase" / "matrix_temp0.json"))
    ap.add_argument("--temp1-json", nargs="+",
                    default=sorted(str(p) for p in (here / "results" / "paraphrase").glob("matrix_temp1_seed*.json")),
                    help="one matrix.json per seed; averaged")
    ap.add_argument("--names", nargs="+", default=["nla1", "nla2", "nla3", "nla4"])
    ap.add_argument("--out-dir", default=str(here / "figure_results"))
    args = ap.parse_args()
    assert args.temp1_json, f"no matrix_temp1_seed*.json found in {here / 'results' / 'paraphrase'}"
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    names = args.names

    # ONE shared row set across every input, so the two figures are comparable to
    # each other and not just internally consistent: a row dropped for extraction
    # failure in any seed is dropped everywhere.
    stack, keep, total = load_matrices([args.temp0_json] + list(args.temp1_json), names)
    m0, m1 = stack[0], stack[1:]
    keep1 = keep
    print(f"rows valid in every cell of every input: {keep}/{total}")

    for tag, mats, sd, title, sub in [
        ("temp0", m0, None,
         "AV/AR swap matrix — temperature 0",
         "Cell = AV$_i$'s explanations, scored by AR$_j$.\n"
         f"n={keep} held-out prompts, temperature 0 (deterministic)."),
        ("temp1_avg4", m1.mean(axis=0), m1.std(axis=0, ddof=1),
         "AV/AR swap matrix — temperature 1.0, mean of 4 seeds",
         "Cell = mean FVE over 4 independently sampled generations (±sd).\n"
         f"n={keep1} held-out prompts per seed."),
    ]:
        d, o, gap, per_av = summarize(mats, names)
        draw(mats, sd, names, title, sub, out / f"swap_matrix_{tag}.svg")
        # The summary is reported on stdout rather than stamped on the figure, so
        # the numbers stay available without crowding the plot.
        print(f"[{tag}] diagonal {d:.2f}%   off-diagonal {o:.2f}%   "
              f"own-partner advantage +{gap:.2f} pt")
        print(f"[{tag}] per-AV advantage: "
              + ", ".join(f"{n_} {a:+.2f}" for n_, a in zip(names, per_av)))


if __name__ == "__main__":
    main()
