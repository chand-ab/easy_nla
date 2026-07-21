"""Plot NLA training curves from the trainer logs (PNG).

We run with --no-wandb, so the text logs are the only record. This parses them
directly rather than depending on a metrics backend.

    python scripts/plot_curves.py --sft-log <sft.log> --rl-log <rl.log> \
        --out-dir <dir> --tag qwen25

Log shapes it reads:
  SFT (one file holds AV then AR, split on the stage banner; rank 0 only prints)
    step NNNN | loss L | ... | ppl P | ent E          (AV)
    step NNNN | loss L | ... | FVE F%                 (AR)
    [heldout@N] val_loss L | val_ppl P                (AV)
    [heldout@N] mse M | FVE F%                        (AR)
  RL (every rank prints, so per-step values are averaged across ranks — each
     rank holds a different shard of the global batch, so the mean is the
     batch-level number; rank 0 alone would be a 1/8 sample)
    step NNNN | r R | FVE F% | ar_mse M | kl K | ent E | ext P% | t Ts
    [eval@N] reward R | FVE F% | ext P%
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Reference palette, slots 1-2 in fixed order (light mode). Documented as
# pre-validated: the first three slots clear the all-pairs CVD/normal-vision
# floors in both modes. Never cycle or re-order these.
C_PRIMARY = "#2a78d6"   # slot 1, blue
C_SECOND = "#eb6834"    # slot 2, orange
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#dcdcd8"

STEP_RE = re.compile(r"^step (\d+) \|(.*)$")
FIELD_RE = {
    "loss": re.compile(r"\bloss ([-\d.]+)"),
    "ppl": re.compile(r"\bppl ([-\d.]+)"),
    "fve": re.compile(r"\bFVE ([-\d.]+)%"),
    "r": re.compile(r"\br ([-\d.]+)"),
    "kl": re.compile(r"\bkl ([-\d.]+)"),
    "ent": re.compile(r"\bent ([-\d.]+)"),
    "ext": re.compile(r"\bext ([\d.]+)%"),
}
HELD_AV_RE = re.compile(r"\[heldout@(\d+)\].*?val_ppl ([\d.]+)")
HELD_AR_RE = re.compile(r"\[heldout@(\d+)\] mse [\d.]+ \| FVE ([-\d.]+)%")
EVAL_RE = re.compile(r"\[eval@(\d+)\] reward ([-\d.]+) \| FVE ([-\d.]+)% \| ext ([\d.]+)%")


def _style(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=11, loc="left", pad=8)
    ax.set_xlabel(xlabel, color=INK_2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8)


def parse_steps(lines, fields):
    """step -> {field: [values across ranks]} for the requested fields."""
    out = defaultdict(lambda: defaultdict(list))
    for ln in lines:
        m = STEP_RE.match(ln)
        if not m:
            continue
        step, rest = int(m.group(1)), m.group(2)
        for f in fields:
            fm = FIELD_RE[f].search(rest)
            if fm:
                out[step][f].append(float(fm.group(1)))
    steps = sorted(out)
    return steps, {f: np.array([np.mean(out[s][f]) if out[s][f] else np.nan
                                for s in steps]) for f in fields}


def rolling(y, w):
    """Centered rolling mean; the raw per-step series is too noisy to read."""
    if len(y) < w:
        return y
    k = np.ones(w) / w
    pad = w // 2
    ypad = np.concatenate([np.full(pad, y[0]), y, np.full(w - 1 - pad, y[-1])])
    return np.convolve(ypad, k, mode="valid")


def plot_sft(sft_log, out, tag):
    lines = Path(sft_log).read_text(errors="ignore").splitlines()
    # The file holds AV then AR back to back; split on the AR banner.
    cut = next((i for i, l in enumerate(lines) if "AR SFT — starting" in l), len(lines))
    av_lines, ar_lines = lines[:cut], lines[cut:]

    av_steps, av = parse_steps(av_lines, ["ppl"])
    ar_steps, ar = parse_steps(ar_lines, ["fve"])
    av_h = [(int(a), float(b)) for a, b in HELD_AV_RE.findall("\n".join(av_lines))]
    ar_h = [(int(a), float(b)) for a, b in HELD_AR_RE.findall("\n".join(ar_lines))]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1), facecolor=SURFACE)

    ax = axes[0]
    ax.plot(av_steps, av["ppl"], color=C_PRIMARY, lw=0.7, alpha=0.22)
    ax.plot(av_steps, rolling(av["ppl"], 51), color=C_PRIMARY, lw=2,
            label="train perplexity")
    if av_h:
        hx, hy = zip(*av_h)
        ax.plot(hx, hy, color=C_SECOND, lw=2, marker="o", ms=5,
                label="held-out val perplexity")
        ax.annotate(f"{hy[-1]:.3f}", (hx[-1], hy[-1]), textcoords="offset points",
                    xytext=(-6, 10), ha="right", color=INK, fontsize=9)
    ax.set_yscale("log")
    ax.set_ylim(3, 100)
    _style(ax, "AV (verbalizer) SFT — perplexity", "step", "perplexity (log scale)")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_2)

    ax = axes[1]
    ax.plot(ar_steps, ar["fve"], color=C_PRIMARY, lw=0.7, alpha=0.22)
    ax.plot(ar_steps, rolling(ar["fve"], 51), color=C_PRIMARY, lw=2,
            label="train FVE")
    if ar_h:
        hx, hy = zip(*ar_h)
        ax.plot(hx, hy, color=C_SECOND, lw=2, marker="o", ms=5,
                label="held-out FVE")
        ax.annotate(f"{hy[-1]:.1f}%", (hx[-1], hy[-1]), textcoords="offset points",
                    xytext=(-6, -14), ha="right", color=INK, fontsize=9)
    ax.set_ylim(0, 80)
    _style(ax, "AR (reconstructor) SFT — fraction of variance explained",
           "step", "FVE (%)")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_2, loc="lower right")

    fig.suptitle(f"{tag} — SFT warm-start", color=INK, fontsize=13,
                 x=0.008, ha="left", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    print(f"wrote {out}  (AV {len(av_steps)} steps, AR {len(ar_steps)} steps)")


def plot_rl_fve(rl_log, out, tag):
    text = Path(rl_log).read_text(errors="ignore")
    steps, f = parse_steps(text.splitlines(), ["fve"])
    ev = [(int(a), float(r), float(v), float(e)) for a, r, v, e in EVAL_RE.findall(text)]

    fig, ax = plt.subplots(figsize=(9, 4.6), facecolor=SURFACE)
    ax.plot(steps, f["fve"], color=C_PRIMARY, lw=0.7, alpha=0.22)
    ax.plot(steps, rolling(f["fve"], 21), color=C_PRIMARY, lw=2,
            label="train FVE (rollout batch)")
    if ev:
        ex, _, ey, _ = zip(*ev)
        ax.plot(ex, ey, color=C_SECOND, lw=2, marker="o", ms=4,
                label="held-out eval FVE")
        ax.annotate(f"{ey[-1]:.1f}%", (ex[-1], ey[-1]), textcoords="offset points",
                    xytext=(-8, 8), ha="right", color=INK, fontsize=10)
        ax.annotate(f"{ey[0]:.1f}%  (SFT warm-start)", (ex[0], ey[0]),
                    textcoords="offset points", xytext=(10, -4), color=INK_2, fontsize=9)
    _style(ax, "RL (GRPO) — reconstruction quality", "step", "FVE (%)")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_2, loc="lower right")
    fig.suptitle(f"{tag} — RL", color=INK, fontsize=13, x=0.008, ha="left", y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    print(f"wrote {out}  ({len(steps)} steps, {len(ev)} evals)")


def plot_rl_diag(rl_log, out, tag):
    lines = Path(rl_log).read_text(errors="ignore").splitlines()
    steps, d = parse_steps(lines, ["r", "kl", "ent", "ext"])
    # One axis per measure — reward, KL, entropy and extraction-rate have
    # unrelated scales, so they get small multiples, never a shared/dual axis.
    panels = [
        ("r", "GRPO reward  (-MSE, higher is better)", "reward", None),
        ("kl", "KL from the SFT reference policy", "KL", None),
        ("ent", "policy entropy", "nats", None),
        ("ext", "extraction rate  (parseable <explanation>)", "%", (90, 101)),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.6), facecolor=SURFACE)
    for ax, (key, title, ylab, ylim) in zip(axes.ravel(), panels):
        ax.plot(steps, d[key], color=C_PRIMARY, lw=0.7, alpha=0.22)
        ax.plot(steps, rolling(d[key], 21), color=C_PRIMARY, lw=2)
        if ylim:
            ax.set_ylim(*ylim)
        _style(ax, title, "step", ylab)
    fig.suptitle(f"{tag} — RL diagnostics", color=INK, fontsize=13,
                 x=0.008, ha="left", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    print(f"wrote {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sft-log", required=True)
    p.add_argument("--rl-log", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--tag", default="run")
    a = p.parse_args()
    od = Path(a.out_dir); od.mkdir(parents=True, exist_ok=True)
    slug = a.tag.lower().replace(" ", "_").replace("/", "_")
    plot_sft(a.sft_log, od / f"{slug}_sft.png", a.tag)
    plot_rl_fve(a.rl_log, od / f"{slug}_rl_fve.png", a.tag)
    plot_rl_diag(a.rl_log, od / f"{slug}_rl_diagnostics.png", a.tag)


if __name__ == "__main__":
    main()
