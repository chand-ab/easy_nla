"""Draw the two IOI figures from the archives in results/. No GPU, no model load.

    ioi_patch_conditions_grid.png    ioi_intervention.py's six conditions —
                                     4 layer panels, 5 shot counts per bar group
    ioi_text_intervention_grid.png   ioi_text_intervention.py's edits —
                                     4 layer rows x 2 groups (A: rows the round
                                     trip got right; B: rows it got wrong)

In both, each bar GROUP is one condition and the five bars inside it are
0/2/4/6/8-shot, one hue per shot count, the same five hues everywhere — so colour
means exactly one thing across the two figures.

Both are scored with `answer_tok1` by default: whether the patched token starts
the target name. `--metric answer` uses the first word of the six-token
continuation instead and writes a suffixed file.

    python notebooks/ioi/plot_ioi_grids.py                  # both figures
    python notebooks/ioi/plot_ioi_grids.py --figure text    # one of them
    python notebooks/ioi/plot_ioi_grids.py --metric answer

Needs matplotlib and numpy. Figures land in notebooks/ioi/figures/ (gitignored).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# The condition list and metric names come from the script that measured them,
# so this file cannot drift from what the archives contain.
from ioi_intervention import CONDITIONS, METRICS as PATCH_METRICS, PLOT_SKIP  # noqa: E402

RESULTS = HERE / "results"
OUT_DIR = HERE / "figures"

# Gemma-3-12B decoder depth, for the "block k of N" label only.
N_BLOCKS = 48
SURFACE = "white"
INK = "black"
INK_MUTED = "#444444"

LAYERS = (24, 32, 40, 47)
SHOTS = (0, 2, 4, 6, 8)
# One hue per shot count, in fixed order — a categorical scale, not a ramp: five
# steps of a single hue are indistinguishable at this bar width. Three of the
# five sit under 3:1 contrast against white, so the condition labels under
# EVERY panel are load-bearing, not decoration.
SHOT_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")
assert len(SHOT_COLORS) == len(SHOTS)

DEFAULT_METRIC = "answer_tok1"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Score interval for a proportion.

    Not the textbook p +- z*sqrt(p(1-p)/n): that evaluates the standard error AT
    the observed p, so an all-miss condition returns a zero-width [0, 0] and
    claims certainty. Wilson inverts the test instead, asking which true rates
    are consistent with k of n, and stays inside [0, 1] at both extremes.
    """
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z / d * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, c - h), min(1.0, c + h)


def cluster_wilson(p, rows_used, z=1.96):
    """Wilson interval with the ROW as the unit of sampling, not the draw.

    A pooled arm takes three substitutions on the SAME prompt, the SAME
    activation and the SAME explanation, so its three draws are not three
    independent trials. Pinning the effective n at the row count is the
    worst-case correction: it cannot come out anti-conservative, and it
    estimates nothing, which matters at p = 0 where a design effect is
    undefined. Single-draw bars are unaffected (rows_used == n).
    """
    return wilson(p * rows_used, rows_used, z)


# ==========================================================================
# figure 1: the patch conditions
# ==========================================================================
def _patch_archive(shots: int, layer: int) -> dict:
    p = RESULTS / f"ioi_patch_conditions_{shots}shot_L{layer}.json"
    assert p.exists(), f"no archive at {p} — run ioi_intervention.py for this cell first"
    a = json.loads(p.read_text())
    assert a.get("k_positions", 1) == 1, f"{p.name} is a k-ladder archive; this figure draws k=1"
    return a


def _patch_panel(ax, archives: dict[int, dict], metric: str, *, chance_label: bool) -> None:
    """One layer: condition groups x 5 shot bars."""
    # `in summary` as well as the skip list: an archive written before a
    # condition existed still draws, minus that bar.
    shown = [c for c in CONDITIONS if c[0] not in PLOT_SKIP
             and all(c[0] in a["summary"] for a in archives.values())]
    labels = [lab for _, lab in shown]
    _, chance = PATCH_METRICS[metric]

    x = np.arange(len(shown), dtype=float)
    width = 0.155
    ax.set_facecolor(SURFACE)

    for si, shots in enumerate(SHOTS):
        a = archives[shots]
        n = a["n"]
        vals = [a["summary"][k][metric] for k, _ in shown]
        # Every metric here is a mean of per-prompt booleans, so k is recoverable
        # and the same interval applies to all of them.
        ci = [wilson(round(v * n), n) for v in vals]
        # max(0, ...): a condition at exactly 1.000 makes hi - v come out as
        # -1e-16 in floating point, and matplotlib rejects a negative yerr.
        err = [[max(0.0, v - lo) for v, (lo, _) in zip(vals, ci)],
               [max(0.0, hi - v) for v, (_, hi) in zip(vals, ci)]]
        pos = x + (si - (len(SHOTS) - 1) / 2) * width
        # 0.88: a hairline of surface between neighbouring bars, so five filled
        # rectangles read as five bars rather than one striped block.
        ax.bar(pos, vals, width=width * 0.88, color=SHOT_COLORS[si], zorder=3)
        ax.errorbar(pos, vals, yerr=err, fmt="none", ecolor=INK_MUTED,
                    elinewidth=0.8, capsize=1.8, capthick=0.8, zorder=5)

    if chance is not None:
        ax.axhline(chance, color=INK_MUTED, lw=1.0, ls=(0, (4, 3)), zorder=2)
        if chance_label:
            ax.text(len(shown) - 0.52, chance + 0.02, "chance",
                    fontsize=8, color=INK_MUTED, ha="right", va="bottom")

    # Headroom for the legend: at L24 the mismatch floor reaches 1.00, so the
    # legend needs clear air above the tallest possible bar.
    ax.set_ylim(0, 1.42)
    ax.set_yticks(np.arange(0, 1.01, 0.25))
    ax.set_xticks(x)
    # Labelled on every panel, not just the bottom row: three of the five hues
    # are below 3:1 against white, so identity must not rest on colour alone.
    ax.set_xticklabels(labels, fontsize=8.8, color=INK)
    ax.set_xlim(-0.6, len(shown) - 0.4)
    ax.tick_params(axis="y", labelsize=9)
    ax.tick_params(axis="x", length=0, pad=6)
    ax.grid(axis="y", color="#dddddd", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    # Separator between condition groups: with 30 bars in a panel, the gap alone
    # is not enough to tell "the next shot" from "the next condition".
    for xi in x[:-1]:
        ax.axvline(xi + 0.5, color="#e8e8e8", lw=0.8, zorder=1)

    layer = archives[SHOTS[0]]["layer"]
    last = layer + 1 == N_BLOCKS   # 1-BASED in the title, 0-indexed in the file name
    ax.set_title(f"layer {layer + 1} / {N_BLOCKS}"
                 + ("  (the last decoder block)" if last else ""),
                 fontsize=12, color=INK, pad=9)

    # The legend does double duty: it is the shot-count key AND the per-cell
    # reconstruction numbers, which are properties of the (layer, shots) cell
    # rather than of any single bar.
    handles, labs = [], []
    for si, shots in enumerate(SHOTS):
        r = archives[shots]["recon"]
        fve = f"{r['fve']:+.2f}" if r["fve"] is not None else "  n/a"
        handles.append(Patch(facecolor=SHOT_COLORS[si], edgecolor="none"))
        labs.append(f"{shots}-shot    FVE {fve}    cos {r['cos']:.3f}")
    ax.legend(handles=handles, labels=labs, loc="upper right", fontsize=7.6,
              handlelength=1.0, handleheight=0.9, handletextpad=0.6,
              labelspacing=0.32, borderpad=0.5, framealpha=0.94,
              edgecolor="#cccccc", labelcolor=INK_MUTED)


def draw_patch(metric: str, out_png: Path) -> None:
    cells = {L: {s: _patch_archive(s, L) for s in SHOTS} for L in LAYERS}
    print(f"patch grid: read {len(LAYERS) * len(SHOTS)} archives")

    fig = plt.figure(figsize=(16.4, 11.0), dpi=200)
    fig.patch.set_facecolor(SURFACE)

    # Hand-placed axes rather than subplots: the header and footnote blocks are
    # fixed-height prose, and a tight_layout pass would push the grid around
    # every time a line is added to either.
    boxes = [[0.055, 0.556, 0.425, 0.214], [0.550, 0.556, 0.425, 0.214],
             [0.055, 0.210, 0.425, 0.214], [0.550, 0.210, 0.425, 0.214]]
    for i, (L, box) in enumerate(zip(LAYERS, boxes)):
        ax = fig.add_axes(box)
        _patch_panel(ax, cells[L], metric, chance_label=i == 0)
        if i % 2 == 0:
            ax.set_ylabel(PATCH_METRICS[metric][0].split(" — ")[0].lower(),
                          fontsize=9.5, color=INK_MUTED, labelpad=6)

    flat = [a for cell in cells.values() for a in cell.values()]
    fig.text(0.030, 0.977,
             "IOI round-trip accuracy, with controls — 4 layers x 5 shot counts",
             fontsize=16, color=INK, ha="left", va="top")

    drawn = {a["n"] + a["n_dropped"] for a in flat}
    # Pooling runs that attempted different prompt sets into one figure is the
    # failure this catches.
    assert len(drawn) == 1, f"runs drew different subsets: {sorted(drawn)}"
    ns = [a["n"] for a in flat]
    setup = [
        f"model = {flat[0]['base_ckpt']}",
        "temperature = 0  (greedy)",
        f"metric = {PATCH_METRICS[metric][0]}",
        "patched position = the final token only",
        f"each bar = the 200 prompts in ioi_1.jsonl, minus the ones whose "
        f"verbalization never closed its </explanation> tag (n = {min(ns)}–{max(ns)} per bar)",
    ]
    for i, line in enumerate(setup):
        fig.text(0.033, 0.936 - i * 0.028, "•", fontsize=9, color=INK_MUTED, va="top")
        fig.text(0.050, 0.936 - i * 0.028, line, fontsize=9.6, color=INK_MUTED,
                 ha="left", va="top")

    notes = [
        "bars within a group = 0 / 2 / 4 / 6 / 8-shot (legend)",
        "scored at the patched token only — a later token cannot rescue a patch that failed there, which with h intact costs nothing (identity = 0.99)",
        "h_j = a name-disjoint prompt's activation (arbitrary — the next such row in the shuffled sample order) — the mismatch donor",
        "FVE / mean cos in each legend = fraction of variance explained and mean cos(h, AR(AV(h))) for THAT layer and shot count",
        "error bars = 95% Wilson intervals over prompts (sampling error; decoding is greedy)",
        "prompt = 'Answer with a single word only. No punctuation, no explanation.'  [+ k demonstrations 'Q: … A: <name>']  |  Q: <query>  A:",
    ]
    for i, line in enumerate(notes):
        fig.text(0.033, 0.150 - i * 0.024, "•", fontsize=8.6, color=INK_MUTED, va="top")
        fig.text(0.050, 0.150 - i * 0.024, line, fontsize=8.8, color=INK_MUTED,
                 ha="left", va="top")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out_png}")


# ==========================================================================
# figure 2: the text edits
# ==========================================================================
# A bar may pool several variant keys: swap_in1..3 are one condition drawn
# three times with a different name per row, so the index carries no meaning
# and separate bars would invent one. (keys, label).
#
# Column A. `→ S` alone cannot distinguish "S is special because the IOI
# structure makes it the competing name" from "the token merely has to be
# somewhere in the context": `→ other, in prompt` is a demo name with no role in
# the final question, and if it steers as well as S, role is not what matters.
# Each `→ other` arm is followed by its own ×3 bar, so the pair reads as one
# comparison. ×2 is in the archives but not drawn: it lands inside ×3's interval
# at every layer.
#
# The replace arms (right of the dashed rule) ask the same three questions of a
# different edit: the explanation is thrown away and replaced by one sentence
# naming that person. Same rows, same targets. No arrow in their labels: `→ X`
# is this panel's substitution operator, and these arms substitute nothing.
PANEL_A = [(("orig",), "unedited"),
           (("swap_s",), "→ S"),
           (("swap_in1", "swap_in2", "swap_in3"), "→ other,\nin prompt"),
           (("swap_in1_x3", "swap_in2_x3", "swap_in3_x3"), "in prompt,\n×3 copies"),
           (("swap_out1", "swap_out2", "swap_out3"), "→ other,\nnot in prompt"),
           (("swap_out1_x3", "swap_out2_x3", "swap_out3_x3"), "not in prompt,\n×3 copies"),
           (("replace_s",), "replace:\nS sentence"),
           (("replace_in1", "replace_in2", "replace_in3"),
            "replace:\nin-prompt\nsentence"),
           (("replace_out1", "replace_out2", "replace_out3"),
            "replace:\nnot-in-prompt\nsentence")]
PANEL_B = [(("orig",), "unedited"),
           (("append_none",), "append:\nno name"),
           (("append_s",), "append:\nS sentence"),
           (("append_gold",), "append:\ngold sentence"),
           (("append2_gold",), "append ×2:\ngold sentence"),
           (("prepend_gold",), "prepend:\ngold sentence"),
           (("replace_gold",), "replace:\ngold sentence"),
           (("replace_bare_gold",), "replace:\nIO name only")]

# metric -> (how the panel subtitle phrases it, field prefix on a scored dict).
TEXT_METRICS = {
    "answer_tok1": ("does the patched token start {t}?", "tok1_is_"),
    "answer": ("is the first word {t}?", "pred_is_"),
}
# (field suffix, how the subtitle names the target) per column.
TARGET_A = ("injected", "the substituted name")
TARGET_B = ("gold", "the IO name")

COLUMNS = [
    # (spec, group key, target, column heading)
    (PANEL_A, "A", TARGET_A, "A  ·  rows that round trip got right\nand name the IO at the same time"),
    (PANEL_B, "B", TARGET_B, "B  ·  rows it got wrong"),
]


def family(keys):
    """Which edit a bar's variants perform. Panel A mixes two; panel B does not."""
    return "replace" if keys[0].startswith("replace") else "edit"


def rate(vs, metric, target):
    """(k, n) for one bar: how many of these scored dicts hit the target."""
    prefix = TEXT_METRICS[metric][1]
    # A bar can legitimately have NO draws — `→ other, in prompt` at 0-shot,
    # where the prompt holds only the IO and S and the pool it draws from is
    # empty. That is absent, not zero, and the caller draws nothing for it.
    if not vs:
        return 0, 0
    # `unedited` has no substituted name; there the name in the text IS the
    # gold, so the two fields coincide and the fallback is not a fudge.
    f = prefix + target if prefix + target in vs[0] else prefix + "gold"
    return sum(bool(v.get(f, False)) for v in vs), len(vs)


def _text_cell(shots: int, layer: int) -> dict:
    """One (layer, shots) archive, split into the two panels' row sets."""
    p = RESULTS / f"ioi_text_intervention_{shots}shot_L{layer}.json"
    assert p.exists(), f"no archive at {p} — run ioi_text_intervention.py for this cell first"
    d = json.loads(p.read_text())
    assert d.get("k_positions", 1) == 1, f"{p.name} is a k-ladder archive; this figure draws k=1"
    # Panel A only plots rows the edit could touch: where the AV never wrote the
    # IO name the substitution is a no-op, and averaging over no-ops reads as a
    # failure to steer. The drop IS a finding, so the count rides in the legend
    # as "n = kept of n_a", where n_a is every row the round trip got right.
    return {
        "A": [r for r in d["records"] if r["group"] == "A" and r.get("n_gold_in_expl", 0) > 0],
        "B": [r for r in d["records"] if r["group"] == "B"],
        "n_a_total": d["n_a"],
        "base_ckpt": d["base_ckpt"],
    }


def _bar_rows(rows: list, keys: tuple) -> list:
    """The scored dicts behind one bar — empty if this arm never ran these keys."""
    return [r["variants"][k] for r in rows for k in keys if k in r["variants"]]


def _text_panel(ax, spec, cells: dict[int, dict], group: str, metric: str, target: str,
                mark_family: bool = False) -> None:
    """One (layer, group) cell: len(spec) condition groups x 5 shot bars."""
    labels = [lab for _, lab in spec]
    x = np.arange(len(spec), dtype=float)
    width = 0.155
    ax.set_facecolor(SURFACE)

    legend = []
    for si, shots in enumerate(SHOTS):
        rows = cells[shots][group]
        pos = x + (si - (len(SHOTS) - 1) / 2) * width
        n_rows = len(rows)
        # A group can be legitimately EMPTY: B is "the rows the patch got wrong",
        # and at L24 4/6/8-shot there are none; A is "the rows whose explanation
        # named the IO", and at L24 there are none at any shot count. Absent is
        # not zero, so draw NOTHING here and say so in the legend.
        if not rows:
            n_txt = (f"n = 0 of {cells[shots]['n_a_total']}" if group == "A"
                     else "n = 0")
            legend.append((si, shots, n_txt + " — no rows, no bars"))
            continue

        vals, err = [], [[], []]
        for keys, _ in spec:
            vs = _bar_rows(rows, keys)
            k, n = rate(vs, metric, target)
            if n == 0:
                # NaN keeps the slot and draws nothing — including no baseline
                # sliver below, which is what marks a real 0.00.
                vals.append(np.nan)
                err[0].append(0.0)
                err[1].append(0.0)
                continue
            p = k / n
            # Rows, not draws: the `→ other` arms pool three draws per prompt.
            rows_used = sum(1 for r in rows if any(kk in r["variants"] for kk in keys))
            lo, hi = cluster_wilson(p, rows_used)
            vals.append(p)
            err[0].append(max(0.0, p - lo))
            err[1].append(max(0.0, hi - p))

        ax.bar(pos, vals, width=width * 0.88, color=SHOT_COLORS[si], zorder=3)
        # A bar at exactly 0.00 draws no rectangle at all, and with 45 bars in a
        # panel there is no room for per-bar value labels. This sliver sits at
        # the base of every bar that HAS data — hidden under the tall ones, and
        # the whole of a zero one.
        drawn = ~np.isnan(np.asarray(vals, dtype=float))
        ax.hlines(np.zeros(int(drawn.sum())),
                  pos[drawn] - width * 0.44, pos[drawn] + width * 0.44,
                  color=SHOT_COLORS[si], lw=1.8, zorder=4)
        ax.errorbar(pos, vals, yerr=err, fmt="none", ecolor=INK_MUTED,
                    elinewidth=0.8, capsize=1.8, capthick=0.8, zorder=5)

        # n is a property of the (layer, shots) cell, not of a bar: every group
        # here draws from the same rows.
        if group == "A":
            n_txt = f"n = {n_rows} of {cells[shots]['n_a_total']}"
        else:
            n_txt = f"n = {n_rows}"
        legend.append((si, shots, n_txt))

    # A whole panel can be empty — at L24 the AV never writes the IO name, so
    # column A has nothing to substitute at any shot count.
    if all(not cells[s][group] for s in SHOTS):
        ax.text(0.5, 0.42,
                "no rows where the AV wrote the IO name —\nevery substitution "
                "would be a no-op edit"
                if group == "A" else
                "no rows in this group —\nnothing the patch got wrong",
                ha="center", va="center", fontsize=10, color=INK_MUTED,
                linespacing=1.6, transform=ax.transAxes)

    ax.set_ylim(0, 1.42)
    ax.set_yticks(np.arange(0, 1.01, 0.25))
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.2, color=INK)
    ax.set_xlim(-0.6, len(spec) - 0.4)
    ax.tick_params(axis="y", labelsize=8.5)
    ax.tick_params(axis="x", length=0, pad=5)
    ax.grid(axis="y", color="#dddddd", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for xi in x[:-1]:
        i = int(xi)
        # Column A only: column B holds `replace_gold` too, and a rule there
        # would split append/prepend from replace — a distinction that column's
        # question does not make.
        if mark_family and family(spec[i + 1][0]) != family(spec[i][0]):
            # Heavier and dashed: every hue is spoken for by the shot count, so
            # this rule is the ONLY thing separating "edit in place" from
            # "replace outright".
            ax.axvline(xi + 0.5, color="#999999", lw=1.2, ls=(0, (4, 3)), zorder=2)
        else:
            ax.axvline(xi + 0.5, color="#e8e8e8", lw=0.8, zorder=1)

    # The legend does double duty: it is the shot-count key AND the per-cell row
    # counts, which vary by more than an order of magnitude down a column.
    handles = [Patch(facecolor=SHOT_COLORS[si], edgecolor="none") for si, _, _ in legend]
    labs = [f"{shots}-shot    {n}" for _, shots, n in legend]
    ax.legend(handles=handles, labels=labs, loc="upper right", fontsize=7.0,
              handlelength=1.0, handleheight=0.9, handletextpad=0.6,
              labelspacing=0.30, borderpad=0.5, framealpha=0.94,
              edgecolor="#cccccc", labelcolor=INK_MUTED, ncols=1)


def draw_text(metric: str, out_png: Path) -> None:
    cells = {L: {s: _text_cell(s, L) for s in SHOTS} for L in LAYERS}
    print(f"text grid: read {len(LAYERS) * len(SHOTS)} archives")

    fig = plt.figure(figsize=(21.3, 17.5), dpi=200)
    fig.patch.set_facecolor(SURFACE)

    # Hand-placed axes rather than subplots: the header and footnote blocks are
    # fixed-height prose. Column widths are proportional to their bar counts, so
    # a bar is the same width on both sides of the figure and the two columns
    # read as one scale.
    LEFT, RIGHT, GAP = 0.062, 0.962, 0.066
    nbars = [len(spec) for spec, *_ in COLUMNS]
    unit = (RIGHT - LEFT - GAP) / sum(nbars)
    cols = [(LEFT, unit * nbars[0]),
            (LEFT + unit * nbars[0] + GAP, unit * nbars[1])]
    height, pitch, top = 0.128, 0.176, 0.738
    how = TEXT_METRICS[metric][0]

    for ri, L in enumerate(LAYERS):
        bottom = top - ri * pitch
        for ci, (spec, group, (target, against), heading) in enumerate(COLUMNS):
            xl, w = cols[ci]
            ax = fig.add_axes([xl, bottom, w, height])
            _text_panel(ax, spec, cells[L], group, metric, target,
                        mark_family=(group == "A"))
            if ci == 0:
                ax.set_ylabel("accuracy", fontsize=9, color=INK_MUTED, labelpad=6)
            last = L + 1 == N_BLOCKS   # 1-BASED in the label, 0-indexed in the file name
            fig.text(xl, bottom + height + 0.008,
                     f"layer {L + 1} / {N_BLOCKS}"
                     + ("  (the last decoder block)" if last else ""),
                     fontsize=11, color=INK, ha="left", va="bottom")
            if ri == 0:
                # Top-anchored: A's heading wraps to two lines and B's does not,
                # and it is the FIRST line of each that has to sit on one rule.
                fig.text(xl, 0.9295, heading, fontsize=12.5, color=INK,
                         ha="left", va="top", linespacing=1.35)
                fig.text(xl, 0.8980, "accuracy = " + how.format(t=against),
                         fontsize=9, color=INK_MUTED, ha="left", va="top")

    fig.text(0.030, 0.988,
             "IOI text intervention — edit the explanation, does the answer follow?"
             "   ·   4 layers x 5 shot counts",
             fontsize=16.5, color=INK, ha="left", va="top")

    setup = [
        f"model = {cells[LAYERS[0]][SHOTS[0]]['base_ckpt']}     temperature = 0  (greedy)"
        f"     metric = {how.format(t='the target name')[:-1]}"
        + ("  —  the only token the patch determines" if metric == "answer_tok1" else
           "  —  of a six-token continuation, only the first of which the patch determines"),
        "patched position = the final token only",
    ]
    for i, line in enumerate(setup):
        fig.text(0.033, 0.962 - i * 0.017, "•", fontsize=8.6, color=INK_MUTED, va="top")
        fig.text(0.048, 0.962 - i * 0.017, line, fontsize=9.4, color=INK_MUTED,
                 ha="left", va="top")

    # Wrapped by hand to each column's width. Seven body lines is the ceiling
    # before the block collides with the closing bullets.
    terms_a = [
        "Column A terms",
        "The first six groups replace the IO name wherever it appears in the AV's verbalization; the last three",
        "(right of the dashed rule) throw that verbalization away and leave one sentence naming the person instead —",
        "the same template column B calls a gold sentence, with a different name; same rows, same targets",
        "n = rows naming the IO, of rows the round trip got right.  IO name — the answer, the name the AV wrote",
        "→ S — replace IO by the prompt's other name.  → other, in prompt — a few-shot demo name, never IO or S;",
        "        not in prompt — absent from it. Both pool 3 draws",
        "×3 copies — the same name as the bar to its left, three times at every site",
    ]
    terms_b = [
        "Column B terms",
        'gold sentence — Final token "to" is a preposition mid-sentence,',
        '        immediately expecting "<IO name>" as the answer.',
        "IO name only — that bare name alone, with no sentence around it",
        "no name — the same sentence with the quoted name deleted",
        "S sentence — the same sentence, with S in place of the IO name",
    ]
    terms_top = 0.174
    for (xl, _), block in zip(cols, (terms_a, terms_b)):
        fig.text(xl, terms_top, block[0], fontsize=9.4, color=INK, va="top")
        fig.text(xl, terms_top - 0.020, "\n".join(block[1:]), fontsize=8.6,
                 color=INK_MUTED, va="top", linespacing=1.55)

    notes = [
        "bars within a group = 0 / 2 / 4 / 6 / 8-shot (legend)",
        "the A / B split is by `answer`, the first-word metric — so here an `unedited` bar off 1.00 / 0.00 "
        "is the two metrics disagreeing, not a bug"
        if metric != "answer" else
        "the A / B split is by this same metric, so the `unedited` bars are 1.00 / 0.00 by construction — "
        "reference points, not results",
        "error bars = 95% Wilson intervals over ROWS, not draws — the two → other arms "
        "pool three substitutions per prompt, which are not three independent trials",
        "n ≤ 5 rows — the interval spans most of [0, 1]; not a measurement",
    ]
    for i, line in enumerate(notes):
        fig.text(0.033, 0.068 - i * 0.0145, "•", fontsize=8.4, color=INK_MUTED, va="top")
        fig.text(0.048, 0.068 - i * 0.0145, line, fontsize=8.8, color=INK_MUTED,
                 ha="left", va="top")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out_png}")


# ==========================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--figure", choices=("patch", "text", "both"), default="both")
    ap.add_argument("--metric", choices=tuple(PATCH_METRICS), default=DEFAULT_METRIC,
                    help="the text figure supports answer_tok1 and answer only")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    suffix = "" if args.metric == DEFAULT_METRIC else f"_{args.metric}"
    if args.figure in ("patch", "both"):
        draw_patch(args.metric, args.out_dir / f"ioi_patch_conditions_grid{suffix}.png")
    if args.figure in ("text", "both"):
        if args.metric not in TEXT_METRICS:
            ap.error(f"the text figure has no metric {args.metric!r}; "
                     f"pick one of {tuple(TEXT_METRICS)}")
        draw_text(args.metric, args.out_dir / f"ioi_text_intervention_grid{suffix}.png")


if __name__ == "__main__":
    main()
