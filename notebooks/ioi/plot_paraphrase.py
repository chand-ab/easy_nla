"""Plot the paraphrase result: the four swap matrices.

Numbers come from paraphrase_effect.py so the figures and its printed tables
cannot drift apart. Matplotlib defaults throughout.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from paraphrase_effect import NAMES, cell_matrix, common_rows, load

HERE = Path(__file__).resolve().parent
CONDS = [("original", "matrix_temp0.json"),
         ("medium paraphrase\n+fidelity", "matrix_temp0_para_medium_faithful.json"),
         ("heavy paraphrase\n+fidelity", "matrix_temp0_para_heavy_faithful.json"),
         ("medium paraphrase", "matrix_temp0_para_medium.json"),
         ("heavy paraphrase", "matrix_temp0_para_heavy.json")]
NOTES = (
    "\u2022  base model / layer: Qwen2.5-7B-Instruct, layer 20 "
    "(0-indexed block \u2014 hooks model.layers[20], = HF hidden_states[21])\n"
    "\u2022  n = 127 held-out prompts (1 of 128 dropped: failed extraction)\n"
    "\u2022  paraphrase model: google/gemma-3-27b-it, greedy, one rewrite per explanation\n"
    "\u2022  AV decode temperature = 0 (greedy)"
)



def draw_combined(m_orig, m_para, out):
    """One matrix: the original 4x4, plus a 5th column of own-partner FVE under
    medium paraphrase + fidelity.

    The 5th column is not a fifth reader -- it is AV_i read by its OWN AR_i on
    rewritten text -- but it lives in the same grid rather than on separate axes,
    because the comparison the figure exists for is along a ROW (72.8 -> 55.6),
    and a gap turns that into a jump between two charts. A white rule and its own
    header carry the separation; the `own AR$_i$` tick keeps it from reading as
    AR5. The ring means the same thing everywhere: an own-partner pairing.

    The norm is keyed to the DATA range -- a 0-anchored ramp would flatten a
    20-point drop into two near-identical shades of dark blue.
    """
    diag = np.array([m_para[i, i] for i in range(4)])
    full = np.column_stack([m_orig, diag])                    # (4, 5)
    vmin, vmax = full.min(), full.max()

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    im = ax.imshow(full, cmap="Blues", vmin=vmin, vmax=vmax)
    for i in range(4):
        for j in range(5):
            dark = full[i, j] > vmin + 0.55 * (vmax - vmin)
            ax.text(j, i, f"{full[i, j]:.1f}", ha="center", va="center", fontsize=9,
                    color="white" if dark else "black",
                    fontweight="bold" if (j == 4 or i == j) else "normal")
        # own-partner cells: the comparison the figure is for, in both conditions
        ax.add_patch(Rectangle((i - 0.44, i - 0.44), 0.88, 0.88, fill=False,
                               edgecolor="black", lw=1.8))
        ax.add_patch(Rectangle((4 - 0.44, i - 0.44), 0.88, 0.88, fill=False,
                               edgecolor="black", lw=1.8))
    ax.axvline(3.5, color="white", lw=3.0)   # condition boundary, not a gap

    ax.set_xticks(range(5), [f"AR{k[-1]}" for k in NAMES] + ["own AR$_i$"], fontsize=9)
    ax.set_yticks(range(4), [f"AV{k[-1]}" for k in NAMES], fontsize=9)
    # Headers sit in DATA x, so each is centred over the columns it covers and
    # stays put if the figure is resized.
    tr = ax.get_xaxis_transform()
    ax.text(1.5, 1.02, "original", transform=tr, ha="center", va="bottom", fontsize=11)
    ax.text(4.0, 1.02, "medium paraphrase\n+fidelity", transform=tr,
            ha="center", va="bottom", fontsize=11, linespacing=1.3)

    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.10, label="held-out FVE %")
    cb.ax.tick_params(labelsize=8)
    fig.text(0.02, 0.02, NOTES, ha="left", va="top", fontsize=7.5, linespacing=1.5)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=HERE / "results" / "paraphrase")
    p.add_argument("--out-dir", type=Path, default=HERE / "figure_results")
    args = p.parse_args()

    mats_json = [load(args.data_dir / f) for _, f in CONDS]
    keep, n_total = common_rows(mats_json)
    print(f"[rows] {len(keep)}/{n_total} valid in every cell of every matrix")
    mats = [cell_matrix(d, keep) for d in mats_json]

    # --- the four matrices, one shared scale so the collapse is visible ---
    fig, axes = plt.subplots(1, len(CONDS), figsize=(3.3 * len(CONDS), 3.6))
    vmin, vmax = min(0, min(m.min() for m in mats)), max(m.max() for m in mats)
    for ax, (label, _), m in zip(axes, CONDS, mats):
        im = ax.imshow(m, cmap="Blues", vmin=vmin, vmax=vmax)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{m[i, j]:.1f}", ha="center", va="center", fontsize=8,
                        color="white" if m[i, j] > 0.55 * vmax else "black")
        ax.set_xticks(range(4), [f"AR{k[-1]}" for k in NAMES], fontsize=8)
        ax.set_yticks(range(4), [f"AV{k[-1]}" for k in NAMES], fontsize=8)
        for i in range(4):      # own-partner cells: the comparison the figure is for
            ax.add_patch(Rectangle((i - 0.44, i - 0.44), 0.88, 0.88, fill=False,
                                   edgecolor="black", lw=1.8))
        ax.set_title(label, fontsize=10)
    fig.colorbar(im, ax=axes, fraction=0.02, label="held-out FVE %")
    # Provenance, so the panel reads on its own. The layer convention is the
    # "Indexing convention" note in nla/models.py — layer_index=K hooks
    # model.layers[K], whose output is HF's hidden_states[K+1], so "layer 20" is a
    # 0-indexed block index.
    fig.text(0.125, 0.02, NOTES, ha="left", va="top", fontsize=8, linespacing=1.5)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "paraphrase_swap_matrices.svg"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")

    # --- single matrix: original + one own-partner column for medium+fidelity ---
    # Same `keep` row set as the five-panel figure, so the two are comparable.
    draw_combined(mats[0], mats[1],
                  args.out_dir / "paraphrase_swap_matrix_medium.svg")


if __name__ == "__main__":
    main()
