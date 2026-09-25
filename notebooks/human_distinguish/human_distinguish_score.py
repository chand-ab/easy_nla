"""Score a human distinguish run against the hidden key.

One number: how many of the 42 trials had the four explanations split correctly.
Chance is 1/3 because there are exactly three ways to partition four items into
two pairs, so 14 correct is what guessing gives and 20 is the pre-registered
threshold (the smallest count that guessing reaches less than 5% of the time).

The per-pair column is 7 trials wide. It is printed because it is free, not
because it means anything -- guessing produces a 5/7 pair about one run in six,
and there are six pairs.

Usage:
    python human_distinguish_score.py results.json [--key human_distinguish_key.json]
"""

import argparse
import collections
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHANCE = 1 / 3


def binom_sf(k, n, p):
    """P(X >= k) for X ~ Binomial(n, p). Exact; n is 42, so the sum is cheap."""
    return sum(math.comb(n, i) * p**i * (1 - p)**(n - i) for i in range(k, n + 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", type=Path, help="results.json exported from the page")
    ap.add_argument("--key", type=Path, default=HERE / "human_distinguish_key.json")
    a = ap.parse_args()

    res = json.loads(a.results.read_text())
    key = json.loads(a.key.read_text())
    if res["build"]["stamp"] != key["build"]["stamp"]:
        raise SystemExit(f"build mismatch: results {res['build']['stamp']} "
                         f"vs key {key['build']['stamp']} -- wrong key file?")

    truth = {r["t"]: r for r in key["key"]}
    ans = res["answers"]
    n = len(ans)
    if n != key["build"]["n"]:
        print(f"NOTE: {n} of {key['build']['n']} trials answered\n")

    hits = 0
    per_pair = collections.defaultdict(lambda: [0, 0])
    reasons = []
    for r in ans:
        k = truth[r["t"]]
        ok = r["pick"] == k["correct"]
        hits += ok
        cell = per_pair[tuple(k["pair"])]
        cell[0] += ok
        cell[1] += 1
        if r.get("why"):
            reasons.append((ok, k["pair"], r["why"]))

    thr = next(k for k in range(n + 1) if binom_sf(k, n, CHANCE) <= 0.05)
    verdict = "DISTINGUISHABLE" if hits >= thr else "not distinguishable"
    print(f"{hits}/{n} correct ({hits/n:.0%}).  Guessing gives {n/3:.0f}/{n}, "
          f"threshold is {thr}/{n}.")
    print(f"-> {verdict}")
    print("\nby pair (7 trials each -- noise, not a result)")
    for (i, j), (k, m) in sorted(per_pair.items()):
        print(f"  {i} vs {j:<6} {k}/{m}")
    if reasons:
        print("\nreasons given  (ok = split was right)")
        for ok, (i, j), why in reasons:
            print(f"  {'ok' if ok else '  '}  {i}/{j}  {why}")


if __name__ == "__main__":
    main()
