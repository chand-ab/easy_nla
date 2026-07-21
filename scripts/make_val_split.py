"""Carve a doc-disjoint train/val pair out of an NLA parquet.

`train_sft.py --heldout-parquet` wants a SEPARATE file, and some published
datasets (e.g. `ceselder/qwen3-8b-nla-L24-finefineweb-100k`) ship only a single
shuffled split. Splitting on a row boundary would be worthless here for the
reason `nla/val_split.py` documents: the file is row-shuffled, so each document's
~10 rows are scattered uniformly and a row-index cut leaves ~zero documents fully
unseen. The metric would read as held-out while being almost entirely train docs.

So reuse the trainers' own criterion — `val_split.is_val_doc`, a crc32 bucket on
doc_id — to put every row of a document on exactly one side. That is the same
function `train_rl_vllm` uses for its auto-split, so an SFT held-out set built
here and the RL held-out evals agree on which documents are val.

    python scripts/make_val_split.py --in data/av_sft_shuf.full.parquet --val-rows 2000
        -> data/av_sft_shuf.full.train.parquet + .val.parquet (+ sidecars)
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from nla.val_split import is_val_doc, val_doc_permille

SIDECAR_SUFFIX = ".nla_meta.yaml"


def _write(table: pa.Table, idx: np.ndarray, out: str, src_sidecar: Path, chunk: int) -> None:
    writer = None
    try:
        for s in range(0, len(idx), chunk):
            part = table.take(pa.array(idx[s : s + chunk]))
            if writer is None:
                writer = pq.ParquetWriter(out, part.schema, compression="zstd")
            writer.write_table(part)
    finally:
        if writer is not None:
            writer.close()
    if src_sidecar.exists():
        shutil.copy2(src_sidecar, out + SIDECAR_SUFFIX)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--train-out", default="")
    p.add_argument("--val-out", default="")
    p.add_argument("--val-rows", type=int, default=2000,
                   help="approximate target size of the val split")
    p.add_argument("--chunk", type=int, default=20_000,
                   help="rows per write batch (the activation column is wide)")
    args = p.parse_args()

    stem = args.inp[: -len(".parquet")] if args.inp.endswith(".parquet") else args.inp
    train_out = args.train_out or f"{stem}.train.parquet"
    val_out = args.val_out or f"{stem}.val.parquet"

    table = pq.read_table(args.inp)
    doc_ids = table.column("doc_id").to_pylist()
    permille = val_doc_permille(args.val_rows, table.num_rows)
    is_val = np.fromiter((is_val_doc(d, permille) for d in doc_ids), dtype=bool,
                         count=len(doc_ids))
    val_idx = np.nonzero(is_val)[0]
    train_idx = np.nonzero(~is_val)[0]

    n_docs = len(set(doc_ids))
    n_val_docs = len({d for d, v in zip(doc_ids, is_val) if v})
    print(f"{table.num_rows} rows / {n_docs} docs -> permille={permille}")
    print(f"  train {len(train_idx)} rows ({n_docs - n_val_docs} docs) -> {train_out}")
    print(f"  val   {len(val_idx)} rows ({n_val_docs} docs) -> {val_out}")
    assert len(val_idx) > 0, "val split is empty — raise --val-rows"

    sidecar = Path(args.inp + SIDECAR_SUFFIX)
    _write(table, train_idx, train_out, sidecar, args.chunk)
    _write(table, val_idx, val_out, sidecar, args.chunk)
    print("done")


if __name__ == "__main__":
    main()
