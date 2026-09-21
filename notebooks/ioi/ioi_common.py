"""Shared IOI plumbing: the dataset, the prompts, and how names are tokenized.

Used by `ioi_intervention.py`, `ioi_text_intervention.py` and (indirectly)
`plot_ioi_grids.py`. Stdlib only, plus whatever tokenizer the caller hands in,
so the plot script can import it on a laptop without torch. Kept as its own
module because prompt construction and name parsing are the rules every IOI
script has to match exactly.
"""

from __future__ import annotations

import json
import re
import string
from pathlib import Path

HERE = Path(__file__).resolve().parent
IOI_DATA = HERE / "datasets" / "ioi_1.jsonl"

# Every mecha_ioi template introduces the pair as "X and Y" and the gold is
# always one of the two, so this one pattern recovers both names with no
# stoplist to maintain.
_PAIR = re.compile(r"\b([A-Z][a-z]+) and ([A-Z][a-z]+)\b")

_ICL_INSTRUCTION = "Answer with a single word only. No punctuation, no explanation."


def load_json_pairs(path: Path) -> list[dict]:
    """One {"input": ..., "output": ...} object per line."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def names_in(row: dict) -> set[str]:
    m = _PAIR.search(row["input"])
    assert m, f"no 'X and Y' pair in: {row['input']!r}"
    return set(m.groups())


def io_and_s(row: dict) -> tuple[str, str]:
    """IO = the gold (received the object). S = the repeated subject, i.e. the
    distractor the IOI circuit has to suppress."""
    io = row["output"]
    (s,) = names_in(row) - {io}
    return io, s


def build_icl_prompt(pairs, query_idx, n_shots, rng) -> str:
    """Q/A in-context prompt. Demos are excluded by shared NAME, not by input
    string, so no demo can leak either of the query's two names."""
    q_names = names_in(pairs[query_idx])
    pool = [i for i in range(len(pairs))
            if i != query_idx and not (names_in(pairs[i]) & q_names)]
    assert len(pool) >= n_shots, f"pool {len(pool)} < {n_shots} shots at query {query_idx}"
    lines = [f"Q: {pairs[i]['input']}\nA: {pairs[i]['output']}"
             for i in rng.sample(pool, n_shots)]
    lines.append(f"Q: {pairs[query_idx]['input']}\nA:")
    return _ICL_INSTRUCTION + "\n\n" + "\n\n".join(lines)


def build_target_prompt(pairs, query_idx, n_shots, rng, style: str = "qa") -> str:
    """The text whose final-token activation gets verbalized / patched.

    `style` only decides what a ZERO-shot prompt looks like; any n_shots > 0 is
    the Q/A format by construction.

      qa   — the instruction + a single "Q: ...\\nA:" turn, no demonstrations.
             The default, and the format the original NLA authors' IOI runs
             use. The choice matters: it moves the final token from " to" to
             ":" and changes both the activation and the accuracy the patch is
             measured against.
      bare — the raw IOI sentence, no instruction wrapper. A cleaner probe of
             the IOI circuit, but not comparable to the published numbers.
    """
    if n_shots == 0 and style == "bare":
        return pairs[query_idx]["input"]
    return build_icl_prompt(pairs, query_idx, n_shots, rng)


def disjoint_partner(pairs, idxs: list[int], i: int) -> int:
    """Index of the mismatch donor for position `i` of `idxs`.

    Walks the sample order from i+1 and takes the first row sharing NEITHER name
    with row `idxs[i]`. Sharing a name is not a cosmetic problem: if the donor's
    gold happens to be our gold, a patch that destroyed the state entirely can
    still be scored correct, which flatters exactly the control that is supposed
    to be a floor.
    """
    mine = names_in(pairs[idxs[i]])
    n = len(idxs)
    for k in range(1, n):
        j = idxs[(i + k) % n]
        if not (names_in(pairs[j]) & mine):
            return j
    raise AssertionError(f"no name-disjoint donor for row {idxs[i]} among {n} samples")


def first_token_ids(tok, name: str) -> tuple[int, ...]:
    """Every first-token id that means "the answer starts HERE".

    Gemma tokenizes ' Jerry' and 'Jerry' as different ids, and which one the
    model emits depends on what precedes it — after 'A:' it is the spaced form,
    at the start of a line the bare one. Both are the name beginning at this
    position, so a first-token metric that accepts only the spaced form scores a
    correct immediate answer as wrong (484 rows across this sweep).
    """
    ids = {first_token_id(tok, name)}
    bare = tok.encode(name, add_special_tokens=False)
    assert bare, f"empty tokenization for {name!r}"
    ids.add(bare[0])
    return tuple(sorted(ids))


def strip_word(text: str) -> str:
    parts = text.strip().split()
    return parts[0].strip(string.punctuation) if parts else ""


def first_token_id(tok, name: str) -> int:
    """Id of the first token of ' <Name>' — the position the IOI logit diff reads.

    Gemma splits several of these names ('Dianne' -> ' Dian' + 'ne'), so a
    top-1-token string comparison scores a CORRECT prediction as wrong. Compare
    at the first token instead, which is what the name is identified by here.
    """
    ids = tok.encode(" " + name, add_special_tokens=False)
    assert ids, f"empty tokenization for {name!r}"
    return ids[0]
