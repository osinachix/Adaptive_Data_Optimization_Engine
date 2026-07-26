"""Reproducible sample-data generator for ADOE examples.

Run:

    poetry run python examples/make_samples.py

Regenerates every file under examples/sample_data/ from fixed seeds, so
output is byte-identical across runs (safe to regenerate, safe to diff).
Each dataset function below documents what it's designed to exercise; see
also the module docstring in the accompanying README-style summary this
script prints when run.
"""

import datetime
import random
from pathlib import Path

import polars as pl

_OUTPUT_DIR = Path(__file__).parent / "sample_data"


# --- (a) core dataset: everything numeric_downcast and dictionary_encoding
# --- are designed to find, plus a boolean and a datetime column that
# --- neither optimizer touches (no optimizer for those kinds exists yet).


def make_core_dataset(rows: int = 2_000, seed: int = 1) -> pl.DataFrame:
    rng = random.Random(seed)
    countries = [
        "US",
        "CA",
        "MX",
        "GB",
        "FR",
        "DE",
        "ES",
        "IT",
        "NG",
        "KE",
        "IN",
        "JP",
    ]
    statuses = ["active", "inactive", "pending", "suspended"]
    start = datetime.datetime(2023, 1, 1)

    return pl.DataFrame(
        {
            # Small ID: references a pool of ~300 customers, not unique
            # per row - fits Int16, unlike a row-count-scale sequential id.
            "customer_id": [rng.randint(1, 300) for _ in range(rows)],
            "age": [rng.randint(18, 95) for _ in range(rows)],
            "visit_count": [rng.randint(0, 40) for _ in range(rows)],
            "country": [rng.choice(countries) for _ in range(rows)],
            "status": [rng.choice(statuses) for _ in range(rows)],
            "is_active": [rng.choice([True, False]) for _ in range(rows)],
            "signup_at": [
                start + datetime.timedelta(minutes=rng.randint(0, 900_000))
                for _ in range(rows)
            ],
        },
        schema={
            "customer_id": pl.Int64,
            "age": pl.Int64,
            "visit_count": pl.Int64,
            "country": pl.String,
            "status": pl.String,
            "is_active": pl.Boolean,
            "signup_at": pl.Datetime,
        },
    )


# --- (c) already optimal: nothing here should show meaningful savings.


def make_already_optimal_dataset(rows: int = 1_000, seed: int = 2) -> pl.DataFrame:
    rng = random.Random(seed)
    return pl.DataFrame(
        {
            # Near-unique per row: dictionary encoding's dictionary would
            # cost about as much as the raw strings, plus per-row code
            # overhead on top, so it should not be proposed.
            "transaction_uuid": [
                "".join(rng.choices("0123456789abcdef", k=16)) for _ in range(rows)
            ],
            # Exceeds Int32's range (~2.15B), so it needs the full Int64
            # it already has - no smaller dtype fits.
            "amount_micros": [
                rng.randint(1_000_000, 9_999_999_999) for _ in range(rows)
            ],
            # No float optimizer exists yet, so this is untouched
            # regardless - included for completeness, not as a claim.
            "rate": [round(rng.uniform(0.01, 9.99), 4) for _ in range(rows)],
        },
        schema={
            "transaction_uuid": pl.String,
            "amount_micros": pl.Int64,
            "rate": pl.Float64,
        },
    )


# --- (d) wide table: many columns, mixing optimizable and not.


def make_wide_dataset(
    rows: int = 500,
    small_int_cols: int = 20,
    string_cols: int = 20,
    bool_cols: int = 10,
    big_num_cols: int = 10,
    seed: int = 3,
) -> pl.DataFrame:
    rng = random.Random(seed)
    data: dict[str, list[object]] = {}
    schema: dict[str, pl.DataType] = {}

    for i in range(small_int_cols):
        data[f"small_int_{i}"] = [rng.randint(0, 100) for _ in range(rows)]
        schema[f"small_int_{i}"] = pl.Int64()
    for i in range(string_cols):
        categories = [f"cat_{j}" for j in range(4)]
        data[f"category_{i}"] = [rng.choice(categories) for _ in range(rows)]
        schema[f"category_{i}"] = pl.String()
    for i in range(bool_cols):
        data[f"flag_{i}"] = [rng.choice([True, False]) for _ in range(rows)]
        schema[f"flag_{i}"] = pl.Boolean()
    for i in range(big_num_cols):
        data[f"big_num_{i}"] = [rng.randint(10**9, 10**10) for _ in range(rows)]
        schema[f"big_num_{i}"] = pl.Int64()

    return pl.DataFrame(data, schema=schema)


# --- (e) missing values + one malformed row.


def make_missing_and_malformed_csv(path: Path, rows: int = 500, seed: int = 4) -> None:
    """Polars can't write a ragged (non-rectangular) CSV from a
    DataFrame - a DataFrame is inherently rectangular - so this writes a
    valid file first and then hand-injects one malformed line, at a fixed,
    documented position, via plain text I/O."""
    rng = random.Random(seed)
    countries = ["US", "CA", "MX", "GB", "FR"]

    df = pl.DataFrame(
        {
            "id": list(range(1, rows + 1)),
            "age": [
                None if rng.random() < 0.08 else rng.randint(18, 95)
                for _ in range(rows)
            ],
            "country": [
                None if rng.random() < 0.05 else rng.choice(countries)
                for _ in range(rows)
            ],
            "note": [f"row-{i}" for i in range(1, rows + 1)],
        },
        schema={
            "id": pl.Int64,
            "age": pl.Int64,
            "country": pl.String,
            "note": pl.String,
        },
    )
    df.write_csv(path)

    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    # Row 10 (line index 10: 1 header line + rows 1-9 before it) gets an
    # extra trailing field, making it ragged - documented position so
    # it's easy to find when inspecting the file.
    malformed_index = 10
    original = lines[malformed_index].rstrip("\n").rstrip("\r")
    lines[malformed_index] = f"{original},EXTRA_FIELD\n"
    path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    core = make_core_dataset()
    core.write_csv(_OUTPUT_DIR / "core_dataset.csv")
    core.write_parquet(_OUTPUT_DIR / "core_dataset.parquet")
    core.write_ndjson(_OUTPUT_DIR / "core_dataset.jsonl")

    make_already_optimal_dataset().write_csv(_OUTPUT_DIR / "already_optimal.csv")
    make_wide_dataset().write_csv(_OUTPUT_DIR / "wide_table.csv")
    make_missing_and_malformed_csv(_OUTPUT_DIR / "missing_and_malformed.csv")

    print(f"Wrote sample data to {_OUTPUT_DIR}:")
    for generated in sorted(_OUTPUT_DIR.iterdir()):
        print(f"  {generated.name} ({generated.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
