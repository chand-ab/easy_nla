"""Does the swap matrix survive paraphrase? Compare matrix.json files row-wise.

The original matrix is flat: every AR reads every AV within ~0.7 FVE pt of its
own partner. That is consistent with two different stories, and the paraphrase
pass separates them:

  * The ARs read CONTENT. Rewriting an explanation into ~25-37% word overlap
    keeps what it says, so FVE should hold up and the diagonal advantage should
    stay whatever it was.
  * The ARs read SURFACE FORM that the arms happen to share. Then rewriting
    destroys the signal and every cell craters.

The own-partner advantage is the second question, and it points the other way:
if the +0.68 pt diagonal edge is carried by an arm's private vocabulary
(`within` vs `inside`), paraphrasing should ERASE it while leaving the overall
level intact. So the informative outcome is two numbers moving independently.

Every number here is recomputed from per-row MSE over the rows valid in EVERY
cell of EVERY input matrix — never read from a summary — so all conditions
average the same prompts. Confidence intervals are cluster-bootstrapped over
DOCUMENTS, not rows: the eval set is 128 prompts from 124 distinct docs, and
rows sharing a doc are not independent draws.
"""

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
NAMES = ["nla1", "nla2", "nla3", "nla4"]


def load(path):
    return json.loads(Path(path).read_text())


def common_rows(docs):
    """Row indices valid in every cell of every matrix (-2.0 = failed extraction)."""
    n = len(docs[0]["per_row"][NAMES[0]][NAMES[0]])
    bad = {k for d in docs for j in NAMES for i in NAMES
           for k, v in enumerate(d["per_row"][j][i]) if v <= -2.0}
    return [k for k in range(n) if k not in bad], n


def cell_matrix(d, keep):
    """FVE % per cell, rows = AV writer, cols = AR reader."""
    base = d["summary"]["fve_baseline"]
    m = np.zeros((4, 4))
    for ii, i in enumerate(NAMES):
        for jj, j in enumerate(NAMES):
            m[ii, jj] = (1.0 - np.mean([-d["per_row"][j][i][k] for k in keep]) / base) * 100.0
    return m


def per_row_stack(d, keep):
    """(4, 4, n_keep) of NEGATIVE reward = MSE, for bootstrapping."""
    return np.array([[[-d["per_row"][j][i][k] for k in keep] for j in NAMES]
                     for i in NAMES])


def advantage(mse, base, mask_diag):
    """Diagonal mean minus off-diagonal mean, in FVE points, over given rows."""
    fve = (1.0 - mse.mean(axis=2) / base) * 100.0
    return fve[mask_diag].mean() - fve[~mask_diag].mean()


def bootstrap(d, keep, doc_ids, reps, rng):
    """Cluster bootstrap over documents. Returns (diag, offdiag, advantage) draws."""
    base = d["summary"]["fve_baseline"]
    stack = per_row_stack(d, keep)                      # (4, 4, n)
    docs = np.array([doc_ids[k] for k in keep])
    uniq = np.unique(docs)
    members = [np.flatnonzero(docs == u) for u in uniq]
    eye = np.eye(4, dtype=bool)
    out = np.zeros((reps, 3))
    for b in range(reps):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([members[p] for p in pick])
        fve = (1.0 - stack[:, :, idx].mean(axis=2) / base) * 100.0
        out[b] = (fve[eye].mean(), fve[~eye].mean(), fve[eye].mean() - fve[~eye].mean())
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--matrix", action="append", required=True, metavar="LABEL=PATH",
                   help="Repeatable. First one is treated as the baseline.")
    p.add_argument("--gen-dir", type=Path, default=HERE / "results/paraphrase/generations/temp0",
                   help="Source of doc_id per row, for clustering the bootstrap")
    p.add_argument("--reps", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json-out", type=Path, default=None)
    args = p.parse_args()

    labels, paths = zip(*(m.split("=", 1) for m in args.matrix))
    mats = [load(pp) for pp in paths]
    keep, n_total = common_rows(mats)
    print(f"[rows] {len(keep)}/{n_total} valid in every cell of all "
          f"{len(mats)} matrices\n")

    # doc_id per row: the generation caches are written in eval-row order, the
    # same order per_row is indexed by.
    gen = json.loads((args.gen_dir / "nla1.json").read_text())
    assert len(gen) == n_total, f"{args.gen_dir} has {len(gen)} rows, matrices have {n_total}"
    doc_ids = [r["doc_id"] for r in gen]
    print(f"[cluster] {len(set(doc_ids[k] for k in keep))} distinct docs over "
          f"{len(keep)} rows\n")

    rng = np.random.default_rng(args.seed)
    eye = np.eye(4, dtype=bool)
    report = {}
    base_fve = None
    for label, d in zip(labels, mats):
        m = cell_matrix(d, keep)
        draws = bootstrap(d, keep, doc_ids, args.reps, rng)
        lo, hi = np.percentile(draws[:, 2], [2.5, 97.5])
        rec = {
            "diagonal": float(m[eye].mean()),
            "off_diagonal": float(m[~eye].mean()),
            "advantage": float(m[eye].mean() - m[~eye].mean()),
            "advantage_ci95": [float(lo), float(hi)],
            "diag_is_row_max": int(sum(m[i].argmax() == i for i in range(4))),
            "cells": m.tolist(),
        }
        if base_fve is None:
            base_fve = m
        else:
            rec["mean_fve_change"] = float((m - base_fve).mean())
        report[label] = rec

        print(f"=== {label} ===")
        print("        " + "".join(f"{('AR=' + j):>10s}" for j in NAMES))
        for ii, i in enumerate(NAMES):
            print(f"AV={i:5s}" + "".join(f"{m[ii, jj]:>10.2f}" for jj in range(4)))
        print(f"  diagonal {rec['diagonal']:.2f} | off-diagonal {rec['off_diagonal']:.2f} "
              f"| advantage {rec['advantage']:+.2f} pt, 95% CI [{lo:+.2f}, {hi:+.2f}]")
        print(f"  diagonal cell is its row's max in {rec['diag_is_row_max']}/4 rows")
        if "mean_fve_change" in rec:
            print(f"  mean FVE vs {labels[0]}: {rec['mean_fve_change']:+.2f} pt")
        print()

    # Paired contrast: does paraphrase move the advantage? Same bootstrap draws
    # cannot be reused across matrices (independent rng), so redraw jointly.
    if len(mats) > 1:
        print("=== paired contrast (same resampled docs for both matrices) ===")
        docs = np.array([doc_ids[k] for k in keep])
        uniq = np.unique(docs)
        members = [np.flatnonzero(docs == u) for u in uniq]
        stacks = [per_row_stack(d, keep) for d in mats]
        bases = [d["summary"]["fve_baseline"] for d in mats]
        rng2 = np.random.default_rng(args.seed + 1)
        for t in range(1, len(mats)):
            d_adv, d_lvl = np.zeros(args.reps), np.zeros(args.reps)
            for b in range(args.reps):
                pick = rng2.integers(0, len(uniq), len(uniq))
                idx = np.concatenate([members[p] for p in pick])
                f0 = (1.0 - stacks[0][:, :, idx].mean(axis=2) / bases[0]) * 100.0
                ft = (1.0 - stacks[t][:, :, idx].mean(axis=2) / bases[t]) * 100.0
                a0 = f0[eye].mean() - f0[~eye].mean()
                at = ft[eye].mean() - ft[~eye].mean()
                d_adv[b], d_lvl[b] = at - a0, (ft - f0).mean()
            for nm, dr in (("advantage", d_adv), ("mean FVE", d_lvl)):
                lo, hi = np.percentile(dr, [2.5, 97.5])
                print(f"  {labels[t]} - {labels[0]}: {nm} {dr.mean():+.2f} pt, "
                      f"95% CI [{lo:+.2f}, {hi:+.2f}]")
                report.setdefault(labels[t], {})[f"delta_{nm.replace(' ', '_')}"] = {
                    "mean": float(dr.mean()), "ci95": [float(lo), float(hi)]}

    if args.json_out:
        args.json_out.write_text(json.dumps(report, indent=1))
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
