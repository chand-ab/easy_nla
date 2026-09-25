"""Two panels, one figure: what the ARs read, and whether the arms are distinguishable.

Left  — the AV/AR swap matrix at temperature 0, with a fifth column giving each
        AV's own-partner FVE after its explanations are rewritten (medium
        paraphrase, fidelity-constrained). Source: plot_paraphrase.draw_combined.
Right — pairwise "which arm wrote this?" accuracy from the mean-pooled layer-20
        activations of the same explanations. Source: plot_activation_logreg,
        redrawn from its archived json (the nested CV takes ~25 min).

The two belong side by side because they answer the same question from opposite
directions on the SAME 127 prompts and the SAME layer. The left panel says the
ARs are near-interchangeable (~0.7 pt own-partner edge) and that most of what
they read survives rewriting. The right panel says a linear probe on the
activations still tells any two arms apart ~90% of the time. So the arms are
plainly different objects; the difference is just not in what the AR reads.

Two colour bars, not one: FVE % and accuracy % are different quantities on
different ranges, and a shared ramp would invite reading a 73 against a 93. They
share the ramp itself (plot_swap_matrix.CMAP) so the figure reads as one family,
and each bar states its own scale — the accuracy bar anchored at 50 = chance,
the FVE bar keyed to the data range.

Redraws in seconds from results/paraphrase/; no model or GPU needed. Writes
figure_results/paraphrase_identity.svg.
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

from paraphrase_effect import NAMES, cell_matrix, common_rows, load
from plot_paraphrase import CONDS
from plot_swap_matrix import CMAP, INK, INK_MUTED, SURFACE

HERE = Path(__file__).resolve().parent

# The right panel is a different quantity (classifier accuracy, not FVE), so it
# gets its own hue: a warm ramp with the same light -> dark progression as the
# blue one, so the two bars read as siblings, not as one scale.
AMBER_STEPS = ["#fbe3c4", "#f8d4a6", "#f5c488", "#f2b46b", "#eea350", "#e9923a",
               "#e08128", "#cf711c", "#b96317", "#a25613", "#8a4910", "#733c0c",
               "#5c3009"]
ACC_CMAP = LinearSegmentedColormap.from_list("seq_amber", AMBER_STEPS)


def swap_panel(ax, m_orig, m_para):
    """Original 4x4 + a 5th own-partner column for medium paraphrase + fidelity.

    The 5th column is not a fifth reader -- it is AV_i read by its OWN AR_i on
    rewritten text -- but it stays in the same grid rather than on separate axes,
    because the comparison the panel exists for runs along a ROW (72.8 -> 55.6)
    and a gap turns that into a jump between two charts. A white rule and the
    `own AR$_i$` tick carry the separation. The ring means one thing everywhere:
    an own-partner pairing.

    The norm is keyed to the DATA range -- anchoring at 0 would flatten a
    20-point drop into two near-identical shades of dark blue.
    """
    diag = np.array([m_para[i, i] for i in range(4)])
    full = np.column_stack([m_orig, diag])                    # (4, 5)
    norm = Normalize(vmin=full.min(), vmax=full.max())
    ax.imshow(full, cmap=CMAP, norm=norm, aspect="equal")

    for i in range(4):
        for j in range(5):
            dark = norm(full[i, j]) > 0.55
            ax.text(j, i, f"{full[i, j]:.1f}", ha="center", va="center", fontsize=11,
                    color="#ffffff" if dark else INK,
                    fontweight="bold" if (j == 4 or i == j) else "normal")
        # own-partner cells, in both conditions: the comparison under test
        for x in (i, 4):
            ax.add_patch(Rectangle((x - .5, i - .5), 1, 1, fill=False,
                                   edgecolor=SURFACE, linewidth=3.0, zorder=3))
            ax.add_patch(Rectangle((x - .44, i - .44), .88, .88, fill=False,
                                   edgecolor=INK, linewidth=1.6, zorder=4))
    ax.axvline(3.5, color=SURFACE, lw=3.5, zorder=5)   # condition boundary

    ax.set_xticks(range(5), [f"AR{k[-1]}" for k in NAMES] + ["own AR$_i$"],
                  fontsize=10.5, color=INK)
    ax.set_yticks(range(4), [f"AV{k[-1]}" for k in NAMES], fontsize=10.5, color=INK)
    return norm


def identity_panel(ax, acc):
    """Upper-triangular pairwise accuracy, arm i vs arm j.

    Only the upper triangle is defined -- the diagonal is an arm against itself
    and "tell A from B" is symmetric -- so arm 1 never appears as a column and
    arm 4 never as a row, and the untrimmed row/column are dropped rather than
    left as dead space.
    """
    A = acc[:-1, 1:]
    # Anchored at chance, not at the data minimum: the question is distance above
    # 50, and a data-keyed floor would paint a cell doing nothing as a low end.
    norm = Normalize(vmin=0.5, vmax=max(float(np.nanmax(A)), 0.55))
    cm = ACC_CMAP.copy()
    cm.set_bad(SURFACE)                 # the unused lower-left corner stays blank
    ax.imshow(np.ma.masked_invalid(A), cmap=cm, norm=norm, aspect="equal")

    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if np.isnan(A[i, j]):
                continue
            dark = norm(A[i, j]) > 0.55
            ax.text(j, i, f"{100 * A[i, j]:.1f}", ha="center", va="center",
                    fontsize=11, color="#ffffff" if dark else INK)

    ax.set_xticks(range(3), [f"NLA {a[-1]}" for a in NAMES[1:]],
                  fontsize=10.5, color=INK)
    ax.set_yticks(range(3), [f"NLA {a[-1]}" for a in NAMES[:-1]],
                  fontsize=10.5, color=INK)
    return norm


def style(ax, n_cols, n_rows):
    """Ticks on top, no spines, surface-coloured gutters between fills."""
    ax.set_facecolor(SURFACE)
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2.5)
    ax.tick_params(which="both", length=0)


def colorbar(fig, norm, x, y, h, figw, figh, label, ticks=None, ticklabels=None,
             cmap=CMAP):
    cax = fig.add_axes([x / figw, y / figh, 0.16 / figw, h / figh])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    if ticks is not None:
        cb.set_ticks(ticks, labels=ticklabels)
    cb.ax.tick_params(labelsize=9, colors=INK_MUTED, length=0)
    cb.outline.set_visible(False)
    # Label above the bar, horizontal: rotated beside the ticks it collides with
    # them at this bar width.
    fig.text(x / figw, (y + h + 0.10) / figh, label, fontsize=9, color=INK_MUTED,
             ha="left", va="bottom", linespacing=1.4)


def draw(m_orig, m_para, acc, out):
    # imshow forces an equal aspect, so both panels are sized square in INCHES
    # from one cell size and placed explicitly -- that keeps a left cell and a
    # right cell the same size, which is what makes the two read as one figure.
    cell = 0.68
    lw, lh = 5 * cell, 4 * cell
    rw, rh = 3 * cell, 3 * cell
    lx, cb1 = 0.62, 0.62 + 5 * cell + 0.22
    rx = cb1 + 0.16 + 1.05
    cb2 = rx + rw + 0.22
    figw = cb2 + 1.25                 # room for the colour-bar ticks and label
    # Vertical budget, top down: two-line column header + ticks (0.84, measured
    # from the panel top) / panels / bottom margin. No title and no notes: the
    # caption carries those. The right panel is one row shorter, so the two are
    # TOP-aligned -- their headers and ticks are the same objects and have to
    # sit on one line.
    ybot = 0.25
    ytop = ybot + lh
    figh = ytop + 0.84

    fig = plt.figure(figsize=(figw, figh), dpi=200)
    fig.patch.set_facecolor(SURFACE)

    axl = fig.add_axes([lx / figw, ybot / figh, lw / figw, lh / figh])
    axr = fig.add_axes([rx / figw, (ytop - rh) / figh, rw / figw, rh / figh])

    nl = swap_panel(axl, m_orig, m_para)
    nr = identity_panel(axr, acc)
    style(axl, 5, 4)
    style(axr, 3, 3)

    colorbar(fig, nl, cb1, ybot, lh, figw, figh, "held-out\nFVE %")
    ticks = [t for t in np.arange(0.5, 1.001, 0.1) if t <= nr.vmax + 1e-9]
    colorbar(fig, nr, cb2, ytop - rh, rh, figw, figh, "accuracy %\n50 = chance",
             ticks, [f"{100 * t:.0f}" for t in ticks], cmap=ACC_CMAP)

    # Column-group headers sit in DATA x, so each is centred over the columns it
    # covers and stays put if the geometry changes.
    tr = axl.get_xaxis_transform()
    axl.text(1.5, 1.10, "original", transform=tr, ha="center", va="bottom",
             fontsize=11, color=INK_MUTED)
    axl.text(4.0, 1.10, "medium paraphrase\n+fidelity", transform=tr, ha="center",
             va="bottom", fontsize=11, color=INK_MUTED, linespacing=1.3)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=HERE / "results" / "paraphrase")
    p.add_argument("--logreg-json", type=Path, default=None,
                   help="default: <data-dir>/activation_logreg_mean.json")
    p.add_argument("--hs-layer", type=int, default=21,
                   help="hidden_states index; NLA layer L lives at index L+1")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    out = args.out or HERE / "figure_results" / "paraphrase_identity.svg"
    jpath = args.logreg_json or args.data_dir / "activation_logreg_mean.json"

    # The row set is the one shared by ALL five paraphrase conditions, not just
    # the two drawn, so this panel stays comparable to the five-panel figure.
    mats_json = [load(args.data_dir / f) for _, f in CONDS]
    keep, n_total = common_rows(mats_json)
    print(f"[rows] {len(keep)}/{n_total} valid in every cell of every matrix")
    mats = [cell_matrix(d, keep) for d in mats_json]

    r = json.loads(jpath.read_text())
    assert r["arms"] == NAMES, f"{jpath} has arms {r['arms']}, expected {NAMES}"
    assert r["n_prompts"] == len(keep), \
        f"{jpath} scored {r['n_prompts']} prompts, matrices keep {len(keep)}"
    k = r["layers"].index(args.hs_layer)
    acc = np.array(r["acc"][k], dtype=float)

    draw(mats[0], mats[1], acc, out)


if __name__ == "__main__":
    main()
