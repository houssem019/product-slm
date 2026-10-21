"""Sample product titles from Amazon Products 2023 into fixed train/val/test splits.

Usage: uv run python -m pslm.data.sample [--train 600 --val 50 --test 150]

The source parquet is ~1.7 GB, mostly precomputed embeddings. DuckDB reads only the columns we need over HTTP, which
is a few MB, and caches them in data/raw/ (gitignored). The splits are small and committed, so labels stay pinned to
fixed titles.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb

SOURCE = "https://huggingface.co/datasets/milistu/AMAZON-Products-2023/resolve/refs%2Fconvert%2Fparquet/default/train"
SHARDS = [f"{SOURCE}/{i:04d}.parquet" for i in range(4)]
RAW = Path("data/raw/amazon_titles.parquet")
OUT = Path("data/titles")
SEED = "pslm-v1"
MAX_STRATUM_SHARE = 0.2  # no source category may exceed 20% of the sample (Amazon Home is ~35% of the source)


def fetch_raw(con: duckdb.DuckDBPyConnection) -> None:
    if RAW.exists():
        return
    RAW.parent.mkdir(parents=True, exist_ok=True)
    con.execute("INSTALL httpfs; LOAD httpfs;")
    files = ", ".join(f"'{u}'" for u in SHARDS)
    con.execute(
        f"COPY (SELECT parent_asin, title, main_category FROM read_parquet([{files}])) TO '{RAW.as_posix()}'"
    )


def sample(con: duckdb.DuckDBPyConnection, n: int) -> list[dict]:
    cap = max(1, int(n * MAX_STRATUM_SHARE))
    rows = con.execute(
        f"""
        WITH clean AS (
            SELECT parent_asin AS id,
                   trim(regexp_replace(title, '\\s+', ' ', 'g')) AS title,
                   coalesce(main_category, 'none') AS source_category
            FROM '{RAW.as_posix()}'
            WHERE title IS NOT NULL
        ),
        filtered AS (
            -- English-ish, not truncated to nothing, not a wall of keywords
            SELECT * FROM clean
            WHERE length(title) BETWEEN 15 AND 220
              AND NOT regexp_matches(title, '[^\\x20-\\x7E]')
        ),
        dedup AS (
            SELECT * FROM filtered QUALIFY row_number() OVER (PARTITION BY lower(title) ORDER BY id) = 1
        ),
        ranked AS (
            SELECT *, hash(id || '{SEED}') AS h,
                   row_number() OVER (PARTITION BY source_category ORDER BY hash(id || '{SEED}')) AS rn
            FROM dedup
        )
        SELECT id, title, source_category FROM ranked WHERE rn <= {cap} ORDER BY h LIMIT {n}
        """
    ).fetchall()
    return [{"id": r[0], "title": r[1], "source_category": r[2]} for r in rows]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", type=int, default=600)
    p.add_argument("--val", type=int, default=50)
    p.add_argument("--test", type=int, default=150)
    a = p.parse_args()

    con = duckdb.connect()
    fetch_raw(con)
    rows = sample(con, a.train + a.val + a.test)
    # Hash order is already random, so contiguous slices are random splits. Test comes first so growing the train
    # set later never moves a title out of test.
    splits = {
        "test": rows[: a.test],
        "val": rows[a.test : a.test + a.val],
        "train": rows[a.test + a.val :],
    }
    for name, part in splits.items():
        write_jsonl(OUT / f"{name}.jsonl", part)
        print(f"{name}: {len(part)}")


if __name__ == "__main__":
    main()
