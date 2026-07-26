# ADOE: Adaptive Data Optimization Engine

**A streaming-first engine that profiles structured datasets and applies safe,
explainable optimizations to cut storage and memory footprint, lossless by default.**

> **Status: in development.** Building toward version 1.0 one milestone at a time. See
> the [roadmap](docs/ADOE-guide.md#11-implementation-roadmap) for full milestone detail;
> the summary below is what's actually working right now, not what's intended.
>
> - **Engine**: streaming reader for CSV, Parquet, and NDJSON (`.json`/`.jsonl`); a
>   read-only streaming reader for Excel (`.xlsx`, first worksheet only — no Excel
>   *writer* yet). Streaming profiler, schema analyzer, planner, and a plugin registry.
>   Two optimizers ship today: `numeric_downcast` and `dictionary_encoding`. Validator
>   and Zstandard-compressed export (CSV/Parquet/JSON) are both implemented.
> - **Modes**: `lossless` is fully real (exact reconstruction, enforced by the
>   validator). `balanced`/`aggressive` are wired through the whole engine but produce
>   byte-for-byte identical output to `lossless` today, because no shipped optimizer is
>   actually lossy yet — see [Optimization modes](#optimization-modes).
> - **CLI**: all five commands (`profile`, `optimize`, `validate`, `benchmark`,
>   `report`) work end to end and are packaged (`adoe` installs as a console script via
>   Poetry/pipx).
> - **GUI**: a working Streamlit app (upload, pick mode, optimize, view report,
>   download) — see [Run the web app](#run-the-web-app). Same engine calls as the CLI,
>   no separate logic.
> - **Not yet built**: five of the planned optimizers (boolean, string, sparse,
>   duplicate, datetime), an Excel writer, and a PyPI release.

---

## What it does

Unlike a plain compression tool, ADOE looks at a dataset's structure, data types,
distributions and redundancy *before* choosing how to optimize it. It profiles the data,
generates an explainable optimization plan, applies the plan safely, validates the
result against the original, and produces a report of everything it did.

Everything runs in a stream: ADOE never loads a whole dataset into memory, so it handles
files larger than available RAM.

The command-line interface (below) is the primary way to run it, for files on disk and
scripting. A Streamlit web app (upload a file in the browser, pick a mode, download the
result) is also available — see [Run the web app](#run-the-web-app).

## Design commitments

These are enforced throughout the engine, not aspirational:

- **Streaming first.** No stage loads the full dataset; all processing is chunk-based.
- **Lossless by default.** The default mode reconstructs the original exactly. Lossy
  modes exist but must be explicitly requested.
- **Validate before export.** Nothing is written until it has been verified against the
  chosen mode.
- **Explainable.** Every transformation is recorded and accountable in the report.
- **Extensible.** New optimizers are plugins; adding one never touches the core.

## Non-goals

ADOE does not guarantee a fixed compression ratio, does not replace a database or a
distributed compute engine, and cannot improve data that is already optimally encoded.

---

## Architecture

```
Reader → Chunk Manager → Profiler → Schema Analyzer → Optimization Planner
       → Optimization Engine → Validator → Exporter → Report Generator
```

Full architecture and per-component detail: [`docs/ADOE-guide.md`](docs/ADOE-guide.md).

## Requirements & dependencies

- **Python 3.12 or newer** (`requires-python = ">=3.12,<4.0"` in `pyproject.toml`).
  Developed and verified on Windows; nothing in the codebase is OS-specific (paths go
  through `pathlib.Path` throughout), so Linux/macOS are expected to work but haven't
  been explicitly verified by this project yet.
- **Poetry** for dependency management and packaging (installation instructions:
  [python-poetry.org](https://python-poetry.org/docs/#installation)).
- **pipx**, only if you want the end-user install path below rather than a source
  checkout ([pipx.pypa.io](https://pipx.pypa.io/stable/installation/)).

Runtime dependencies (installed automatically by `poetry install`; every one of these
is required today, including for CLI-only use — there is currently no optional/"GUI
extras" split, so `streamlit`/`openpyxl` install even if you only ever use the CLI):

| Package | Used for |
|---|---|
| `polars` | The dataframe engine: streaming reads, dtype casts, all in-memory data handling |
| `pyarrow` | Parquet reading/writing (row-group streaming) |
| `duckdb` | Declared as part of the data-processing stack; **not actually used by any code path yet** — reserved, not dead weight by design, but honestly unused today |
| `typer` | The CLI framework |
| `streamlit` | The web GUI |
| `openpyxl` | Streaming (read-only) `.xlsx` reading, via its `read_only` row iterator |
| `pydantic-settings` | Centralized config (flags > env vars > `adoe.toml` > defaults) |
| `structlog` | Structured (JSON) logging throughout the engine and CLI |
| `zstandard` | Optional output compression (`--compress`) |

Development-only dependencies (installed by `poetry install`, used for testing/quality
checks, not needed to just run `adoe` or the GUI): `pytest`, `ruff`, `black`, `mypy`.

---

## Installation

**End users**

Not published to PyPI yet, so `pipx install adoe` doesn't work today. Until it is,
build the wheel yourself and install that instead (this works today; verified against
this exact repo state):

```bash
poetry install
poetry build
pipx install dist/adoe-0.1.0-py3-none-any.whl
```

This gets you an `adoe` command on your PATH, isolated in its own environment, with no
source checkout required afterward.

**Developers** (from a checkout of this repository):

```bash
poetry install
poetry run adoe --help
```

## Quickstart

Two commands, on the one sample file checked into this repo
([`examples/sample_orders.csv`](examples/sample_orders.csv): a generated 50,000-row
order log — order id, customer age, quantity, region, status, unit price).

Profile it first, without changing anything:

```bash
adoe profile examples/sample_orders.csv --log-level WARNING
```

```
Rows: 50000
  order_id (integer, Int64): null_ratio=0.00% cardinality~49940 duplicate%=0.1
  customer_age (integer, Int64): null_ratio=0.00% cardinality~74 duplicate%=99.9
  quantity (integer, Int64): null_ratio=0.00% cardinality~20 duplicate%=100.0
  region (string, String): null_ratio=0.00% cardinality~4 duplicate%=100.0
  status (string, String): null_ratio=0.00% cardinality~4 duplicate%=100.0
  unit_price (float, Float64): null_ratio=0.00% cardinality~32133 duplicate%=35.7
```

Then optimize it, lossless by default, saving the execution report alongside the output:

```bash
adoe optimize examples/sample_orders.csv --out output.parquet --report report.json --log-level WARNING
```

```
Wrote output.parquet (parquet).
Bytes (optimized columns only): 1824993 -> 700000 (validation passed)
File size:                      1626124 -> 847114 (47.9% smaller)
Report saved to report.json.
```

`order_id`/`customer_age`/`quantity` were downcast to smaller integer types and
`region`/`status` to dictionary-encoded categoricals — losslessly, confirmed by the
validator before anything was written. See [Optimization modes](#optimization-modes) for
why the two size lines differ, and why they'd differ even more (or barely at all) on a
different dataset shape.

## Command-line usage

Beyond the two Quickstart commands, `adoe` has three more:

Inspect a previously saved report without re-running anything:

```bash
adoe report report.json
```

```
Mode:              lossless
Validation:        passed
Bytes (optimized): 1824993 -> 700000
File size:         1626124 -> 847114 (47.9% smaller)
Results:           10 optimizer applications recorded
  - customer_age: Int64 -> Int8; order_id: Int64 -> Int32; quantity: Int64 -> Int8 (lossless=True)
  - region: String -> Categorical; status: String -> Categorical (lossless=True)
  ... (8 more entries: one pair per chunk, same detail each time for this file)
```

Independently re-validate an existing output against its original input:

```bash
adoe validate examples/sample_orders.csv output.parquet --log-level WARNING
```

```
Validation passed (50000 rows, mode=lossless).
```

Measure profiling throughput and peak memory (elapsed time and throughput will vary by
machine; the shape won't):

```bash
adoe benchmark examples/sample_orders.csv --log-level WARNING
```

```
Rows:              50000
Elapsed:           4.597s
Throughput:        10877 rows/s
Peak memory (CLI): 0.86 MB
```

Every command accepts `--rows-per-chunk`, `--log-level`, and (`optimize`/`validate`)
`--mode`; run `adoe <command> --help` for the full option list. Configuration falls back
to `ADOE_*` environment variables and an `adoe.toml` file when a flag is omitted; see
below for the exact precedence and format.

The CLI isn't limited to CSV input either, despite every example above using it: `adoe
profile`/`adoe optimize` accept CSV, Parquet, NDJSON (`.json`/`.jsonl`), or Excel
(`.xlsx`) directly, the same formats the GUI accepts — e.g. `adoe profile
data.parquet` or `adoe profile data.xlsx` work exactly like `adoe profile data.csv`.
One asymmetry to know about: there's no Excel *writer*, so `adoe optimize input.xlsx
--out output.xlsx` fails with "cannot infer export format" — point `--out` at
`.csv`/`.parquet`/`.json` instead when the input is Excel.

## Configuration

Three sources, resolved in this order (highest priority first) — an explicit CLI flag
always wins, an omitted flag falls through:

1. **CLI flags** you actually pass (e.g. `--mode balanced`).
2. **Environment variables**, prefixed `ADOE_`: `ADOE_MODE`, `ADOE_ROWS_PER_CHUNK`,
   `ADOE_LOG_LEVEL`.
3. **`adoe.toml`** in the current directory:
   ```toml
   mode = "balanced"
   rows_per_chunk = 5000
   log_level = "DEBUG"
   ```
4. Built-in defaults (`lossless`, `10000`, `INFO`) if nothing else is set.

All three fields are optional in `adoe.toml`/the environment — set only the ones you
want to override. This precedence is verified directly in
[`tests/test_settings.py`](tests/test_settings.py), not just described here.

## Run the web app

```bash
poetry run streamlit run gui/app.py
```

This opens a browser tab (defaults to `http://localhost:8501`) with a single page:

1. Upload a dataset (CSV, Parquet, NDJSON `.json`/`.jsonl`, or Excel `.xlsx`).
2. Pick an optimization mode (`lossless` is the default).
3. Click **Optimize**.
4. The page shows the profile summary, the optimization report (validation status,
   bytes before/after, every optimizer application), and a **Download optimized file**
   button for the result.

Excel input is read-only today (there's no Excel writer yet): an uploaded `.xlsx` is
optimized and offered back as Parquet instead, with a note in the UI explaining why.

The GUI is a second door into the same engine the CLI uses, not a separate
implementation: it calls `run_profile`/`run_optimize` from `engine/pipeline.py`
directly and contains no profiling, optimization, or validation logic of its own (see
invariant I6 in [`CLAUDE.md`](CLAUDE.md)) — so a result optimized through the browser is
identical to the same file run through `adoe optimize`.

---

## Sample data

Everything above uses [`examples/sample_orders.csv`](examples/sample_orders.csv) (a
generated 50,000-row order log), but [`examples/sample_data/`](examples/sample_data/)
has more, each built to exercise something specific:

| File | Purpose |
|---|---|
| `core_dataset.csv` / `.jsonl` / `.parquet` | The same 2,000-row dataset in three formats — try `adoe profile` on each to compare how format affects what's detectable (e.g. NDJSON round-trips `signup_at` as a string; Parquet keeps it as a real datetime) |
| `already_optimal.csv` | Nothing here should show meaningful savings — confirms the planner doesn't force a change just to report one |
| `wide_table.csv` | Many columns, a mix of optimizable and not |
| `missing_and_malformed.csv` | A ragged row (wrong field count) — confirms `adoe profile`/`optimize` fail with a clear, actionable error instead of a raw traceback |

Regenerate all of them deterministically (fixed random seeds, byte-identical output
every run):

```bash
poetry run python examples/make_samples.py
```

`examples/real_data.csv` and its derived `real_data_optimized*` files, if present in
your working copy, are **not** part of this repository (excluded via `.gitignore`) —
they're a real-world dataset a user of this project downloaded separately for testing
at real scale (~200MB, exceeds GitHub's 100MB file limit anyway). A fresh clone won't
have them; supply your own large CSV/Parquet/JSON/Excel file at that path to reproduce
that kind of test.

---

## Optimization modes

| Mode | Behaviour |
|---|---|
| `lossless` (default) | Exact reconstruction guaranteed. |
| `balanced` | Limited, configurable precision reduction. |
| `aggressive` | Maximum optimization with acceptable information loss. |

The mode is threaded all the way through today (it changes how strictly the validator
checks reconstruction). What it doesn't yet change is the optimization plan itself: both
shipped optimizers (`numeric_downcast`, `dictionary_encoding`) are unconditionally
lossless, so `balanced` and `aggressive` currently produce byte-for-byte identical output
to `lossless` — verified above. They'll diverge once a genuinely lossy optimizer ships.

---

## Project layout

```
adoe/
├── core/         streaming framework, chunk manager, shared types
├── engine/       optimization planner and execution pipeline
├── optimizers/   individual optimizer plugins
├── plugins/      plugin discovery and registration
├── cli/          Typer command-line interface
├── gui/          Streamlit web app (see "Run the web app" above)
├── config/       configuration loading and precedence
├── tests/        unit, integration, regression, performance
├── benchmarks/   benchmark datasets and harness
├── docs/         architecture guide
└── examples/     runnable usage examples (see "Sample data" above)
```

---

## Development

Everything below assumes `poetry install` has already been run.

Run the full test suite:

```bash
poetry run pytest
```

`203 passed, 2 skipped` is the expected baseline (the 2 skips are slow stress tests,
opt-in only):

```bash
poetry run pytest --run-stress
```

Lint, format-check, and type-check (all three must be clean before a commit, per
[`CLAUDE.md`](CLAUDE.md)):

```bash
poetry run ruff check .
poetry run black --check .
poetry run mypy .
```

Run the benchmark harness (execution time, peak memory, size reduction, throughput,
and validation success rate across small/medium/large synthetic datasets):

```bash
poetry run python benchmarks/harness.py
```

---

## Contributing

New optimizers are added as plugins without modifying the core engine. See the
[plugin chapter](docs/ADOE-guide.md#7-optimizer-plugins) for the plugin contract. Every
optimizer must be deterministic, streaming-compatible where possible, and shipped with
unit tests.

## License

MIT.
