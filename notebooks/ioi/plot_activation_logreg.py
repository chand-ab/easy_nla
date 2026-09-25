"""Pairwise logistic classification of NLA identity from explanation activations.

A single 4-class "which of the four arms wrote this?" number cannot say WHICH
arms are confusable, and the whole point of the OFAT design is that each pair
isolates a different source of training randomness:

    1 vs 2      rollout stochasticity alone (same SFT, same RL data order)
    3 vs 1/2    SFT LoRA init, on top of the rollout floor
    4 vs 1/2    RL data order, on top of the rollout floor
    3 vs 4      init vs order

So the six upper-triangular cells are six separate experiments, and a flat matrix
means something quite different from one where only the NLA-3 column is hot.

Protocol, per pair (this is the part that decides whether the numbers mean
anything):

  * L2 logistic regression on the raw 3584-dim mean-pooled activation.
    Regularisation is not optional here — 3584 dimensions against ~200 training
    points, and an unpenalised fit separates ANY labelling, including random ones.
  * 5-fold cross-validation GROUPED BY PROMPT. Both arms in a pair explain the
    same document, and document content is by far the dominant source of variance
    in these activations, so a row-wise split lets the classifier recognise a
    document it was fitted on rather than an arm.
  * C is chosen by an INNER grouped 5-fold on each outer training fold only —
    nested CV, so no held-out point influences the model that scores it.

Chance is 50%. The observed cells sit 40-48 points above it, far outside the range
fold noise can produce at this sample size.

Centring is global. Document centring (subtracting each prompt's across-arm mean)
is NOT available here and would be meaningless: within a pair the two centred
points are exact negatives, so any linear rule through the origin separates them
perfectly, and one point literally determines the other across a fold boundary.

Read the mean-pooled numbers. `--pool last` is offered for completeness but the
last-token activation is dominated by the explanation's final punctuation
character.

Needs scikit-learn (unlike the sibling figure scripts, which hand-roll their
numerics): nested CV over a C grid is a lot of surface to reimplement, and a
hand-rolled solver that quietly fails to converge would look exactly like a
negative result.

Runtime is ~25 min for the default three layers, dominated by the inner C search
(5 outer x 5 inner x 9 grid points x 6 pairs x 3 layers). Results are archived to
results/paraphrase/activation_logreg_<pool>.json; --from-json redraws in seconds.
"""

import argparse
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from sklearn.linear_model import LogisticRegression

# Same sequential blue ramp as plot_swap_matrix.py (steps 100..700), so the two
# matrix figures read as one family.
BLUE_STEPS = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
              "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
              "#0d366b"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
CMAP = LinearSegmentedColormap.from_list("seq_blue", BLUE_STEPS)

# The activations come from collect_explanation_activations.py, which reads the
# GREEDY generation caches by construction (see its --gen-dir help). Verified for
# the shipped npz by row identity: its 127 rows per arm are exactly the cross-arm
# intersection of valid explanations in results/paraphrase/generations/temp0 — including the
# idx-94 nla3 token-cap failure that the temp1 seeds do not reproduce.
DECODE = "temperature = 0"

# What each pair isolates under the OFAT design. Printed with the results so the
# matrix is read as six experiments rather than six numbers.
FACTOR = {
    ("nla1", "nla2"): "rollout noise floor",
    ("nla1", "nla3"): "SFT init (+ rollout)",
    ("nla1", "nla4"): "RL data order (+ rollout)",
    ("nla2", "nla3"): "SFT init",
    ("nla2", "nla4"): "RL data order",
    ("nla3", "nla4"): "SFT init vs RL data order",
}


def nan_to_null(a):
    """Array -> nested lists with NaN as None, so the archive is valid JSON.

    Ten of the sixteen cells are never computed — the diagonal is an arm against
    itself, and the lower triangle mirrors the upper because "tell A from B" is
    symmetric. numpy carries those as NaN, but json.dump writes bare `NaN`, which
    Python and jq accept as an extension and JSON.parse rejects outright. `null`
    is the standard spelling, and np.array(..., dtype=float) turns it back into
    NaN on load, so --from-json is unaffected.
    """
    return [[None if np.isnan(v) else float(v) for v in row] for row in a]


def grouped_folds(groups, n_splits):
    """Deterministic grouped K-fold: returns a boolean test mask per fold.

    Interleaved rather than shuffled, so reruns and layers see identical splits
    and the numbers are comparable across panels.
    """
    ug = np.unique(groups)
    return [np.isin(groups, ug[i::n_splits]) for i in range(n_splits)]


def fit_predict(Xtr, ytr, Xte, C):
    """Fit on the training rows, predict the test rows.

    Scaling is a single scalar — the training rows' RMS distance from their mean —
    which puts the features at O(1) for the solver without touching the L2
    geometry. Per-feature standardisation would instead reweight the penalty
    across dimensions, which is a modelling choice this figure has no reason to
    make.
    """
    mu = Xtr.mean(0, keepdims=True)
    s = np.sqrt(((Xtr - mu) ** 2).sum(1).mean())
    lr = LogisticRegression(C=C, max_iter=5000).fit((Xtr - mu) / s, ytr)
    return lr.predict((Xte - mu) / s)


def pick_C(X, y, groups, grid, n_splits):
    """Inner grouped CV over the C grid, run on one outer training fold."""
    inner = grouped_folds(groups, n_splits)
    accs = [np.mean([(fit_predict(X[~t], y[~t], X[t], C) == y[t]).mean()
                     for t in inner]) for C in grid]
    return grid[int(np.argmax(accs))]


def evaluate_pair(X, y, groups, grid, folds_n):
    """Nested-CV accuracy for one pair. Returns (accuracy, per-fold C)."""
    folds = grouped_folds(groups, folds_n)
    Cs, pred = [], np.empty(len(y), dtype=y.dtype)
    for te in folds:
        C = pick_C(X[~te], y[~te], groups[~te], grid, folds_n)
        pred[te] = fit_predict(X[~te], y[~te], X[te], C)
        Cs.append(C)
    return float((pred == y).mean()), Cs


def draw(acc, names, layers, pool, n_prompts, out_png, decode=DECODE):
    # Only the upper triangle is defined, so arm 1 never appears as a column and
    # the last arm never as a row. Dropping both keeps the panel to the cells that
    # exist — an untrimmed n x n leaves a blank row and column of dead space.
    rows, cols = names[:-1], names[1:]
    m = len(rows)
    npan = len(layers)
    # imshow forces an equal aspect, so the axes are sized square in INCHES and
    # placed explicitly — a non-square subplots box would leave the matrix
    # floating in dead space (same reasoning as plot_swap_matrix.py).
    side, gap = 2.9, 0.95
    figw, figh = side * npan + gap * (npan - 1) + 1.95, side + 2.0
    # The header is a fixed run of text, so a one- or two-panel figure has to be
    # widened to fit it rather than letting it run off the canvas. Costs some
    # empty space to the right of the colour bar in those cases.
    figw = max(figw, 9.6)
    fig = plt.figure(figsize=(figw, figh), dpi=200)
    fig.patch.set_facecolor(SURFACE)

    # The ramp is anchored at 50% (chance) rather than at the data minimum: the
    # question is distance above chance, and a data-keyed floor would paint a
    # cell that is doing nothing as though it were the low end of a real range.
    norm = Normalize(vmin=0.5, vmax=max(float(np.nanmax(acc)), 0.55))
    cm = CMAP.copy()
    cm.set_bad(SURFACE)              # the unused lower-left corner stays blank

    for k, L in enumerate(layers):
        left = (0.80 + k * (side + gap)) / figw
        ax = fig.add_axes([left, 0.075, side / figw, side / figh])
        ax.set_facecolor(SURFACE)
        A = acc[k][:-1, 1:]
        ax.imshow(np.ma.masked_invalid(A), cmap=cm, norm=norm, aspect="equal")

        for i in range(m):
            for j in range(m):
                if np.isnan(A[i, j]):
                    continue
                # Label ink flips on dark fills so both stay legible.
                dark = norm(A[i, j]) > 0.55
                ax.text(j, i, f"{100 * A[i, j]:.1f}", ha="center", va="center",
                        fontsize=16, color="#ffffff" if dark else INK)

        ax.set_xticks(range(m), [f"NLA {a[-1]}" for a in cols], fontsize=11, color=INK)
        ax.set_yticks(range(m), [f"NLA {a[-1]}" for a in rows], fontsize=11, color=INK)
        ax.xaxis.set_label_position("top")
        ax.xaxis.tick_top()
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_xticks(np.arange(-.5, m, 1), minor=True)
        ax.set_yticks(np.arange(-.5, m, 1), minor=True)
        ax.grid(which="minor", color=SURFACE, linewidth=2.5)   # 2px surface gap
        ax.tick_params(which="both", length=0)
        ax.set_title(f"Layer {L - 1}", fontsize=12.5, color=INK, pad=28)

    cx = (0.80 + (npan - 1) * (side + gap) + side + 0.28) / figw
    cax = fig.add_axes([cx, 0.075, 0.20 / figw, side / figh])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=CMAP), cax=cax)
    cb.ax.tick_params(labelsize=9, colors=INK_MUTED, length=0)
    ticks = [t for t in np.arange(0.5, 1.001, 0.1) if t <= norm.vmax + 1e-9]
    cb.set_ticks(ticks, labels=[f"{100 * t:.0f}" for t in ticks])
    cb.outline.set_visible(False)
    # Label sits above the bar, horizontal: rotated beside the ticks it collides
    # with them at this bar width.
    fig.text(cx, 0.075 + side / figh + 0.022, "accuracy %\n50 = chance",
             fontsize=9, color=INK_MUTED, ha="left", va="bottom", linespacing=1.4)

    fig.text(0.035, 0.965, "Pairwise NLA identification accuracy from mean of verbalization activations",
             fontsize=15, color=INK, ha="left", va="top")
    fig.text(0.035, 0.900,
             f"Cell (i, j) = 5-fold CV pooled-accuracy, "
             f"L2 logistic regression on {pool}-pooled activations.\n"
             f"n={n_prompts} prompts x 2 NLAs = {2 * n_prompts} points per cell; "
             f"5-fold grouped by prompt, C by nested inner CV.\n"
             f"{decode}",
             fontsize=9.6, color=INK_MUTED, ha="left", va="top", linespacing=1.5)
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out_png}")


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", default=str(here / "results" / "paraphrase" / "activations" / "greedy_base_subset.npz"),
                    help="from collect_explanation_activations.py; not shipped (24 MB)")
    ap.add_argument("--layers", nargs="+", type=int, default=[6, 14, 21],
                    help="hidden_states indices (NLA layer L lives at index L+1)")
    ap.add_argument("--arms", nargs="+", default=["nla1", "nla2", "nla3", "nla4"])
    ap.add_argument("--pool", choices=("mean", "last"), default="mean",
                    help="last-token activations are punctuation-contaminated")
    ap.add_argument("--folds", type=int, default=5)
    # Wide enough that the selected C is interior rather than pinned at the top.
    # Accuracy is on a plateau above C~10 (checked out to 1e6: the six cells move
    # by <0.5 pt), so the exact pick is not load-bearing — the grid mainly has to
    # be able to reach the strong-shrinkage end if a pair has no signal to fit.
    ap.add_argument("--C-grid", nargs="+", type=float,
                    default=list(np.logspace(-4, 4, 9)))
    ap.add_argument("--out", default=None)
    ap.add_argument("--json", default=None,
                    help="where the scored matrices are written (default "
                         "results/paraphrase/activation_logreg_<pool>.json)")
    ap.add_argument("--from-json", action="store_true",
                    help="redraw from a previous run's json instead of "
                         "recomputing; the nested CV takes ~25 min")
    args = ap.parse_args()

    jpath = Path(args.json) if args.json else \
        here / "results" / "paraphrase" / f"activation_logreg_{args.pool}.json"

    if args.from_json:
        r = json.loads(jpath.read_text())
        out = Path(args.out) if args.out else here / "figure_results" / f"activation_logreg_{r['pool']}.svg"
        draw(np.array(r["acc"], dtype=float), r["arms"], r["layers"], r["pool"],
             r["n_prompts"], out, r.get("decode", DECODE))
        return

    d = np.load(args.npz, allow_pickle=False)
    hs_index = d["hs_index"].tolist()
    arm, idx = d["arm"], d["idx"]
    for L in args.layers:
        if L not in hs_index:
            raise SystemExit(f"layer hs[{L}] not in file (has {hs_index}); "
                             f"reslice from the full 29-layer npz on the box")

    # Paired prompts only, defined over ALL requested arms rather than per pair, so
    # every cell of the matrix is computed on the same documents and the six
    # numbers are comparable to each other.
    common = set.intersection(*(set(idx[arm == a].tolist()) for a in args.arms))
    names = list(args.arms)
    n = len(names)
    print(f"{len(common)} paired prompts | {args.pool}-pooled | "
          f"{args.folds}-fold grouped by prompt | C grid "
          f"{', '.join(f'{c:g}' for c in args.C_grid)}")

    acc = np.full((len(args.layers), n, n), np.nan)
    for k, L in enumerate(args.layers):
        ci = hs_index.index(L)
        print(f"\nlayer {L - 1}  (hs[{L}])")
        for i, j in itertools.combinations(range(n), 2):
            a, b = names[i], names[j]
            keep = np.array([r for r in range(len(arm))
                             if arm[r] in (a, b) and idx[r] in common])
            X = d[args.pool][keep][:, ci, :].astype(np.float64)
            y = (arm[keep] == b).astype(np.int64)
            a_, Cs = evaluate_pair(X, y, idx[keep], args.C_grid, args.folds)
            acc[k, i, j] = a_
            print(f"  {a} vs {b:<5} acc {100 * a_:5.1f}%   "
                  f"C={'/'.join(f'{c:g}' for c in Cs)}"
                  f"   [{FACTOR.get((a, b), '')}]", flush=True)

    # Archived next to the swap matrices, so a layout change can be redrawn with
    # --from-json rather than paying for the nested CV again.
    jpath.parent.mkdir(parents=True, exist_ok=True)
    jpath.write_text(json.dumps(
        {"pool": args.pool, "arms": names, "layers": list(args.layers),
         "n_prompts": len(common), "folds": args.folds,
         "C_grid": list(args.C_grid),
         "acc": [nan_to_null(m) for m in acc]}, indent=1))
    print(f"\nwrote {jpath}")

    out = Path(args.out) if args.out else here / "figure_results" / f"activation_logreg_{args.pool}.svg"
    draw(acc, names, args.layers, args.pool, len(common), out)


if __name__ == "__main__":
    main()
