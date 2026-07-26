# Agent Instructions: Adaptive Data Optimization Engine (ADOE)

This file drives the entire implementation. Read it fully before writing any code, and
re-read the relevant section at the start of each milestone. `docs/ADOE-guide.md` holds
the chapter-level architecture; this file holds the rules, the interfaces, and the build
order you actually implement against.

---

## 1. What this project is

A modular, streaming-first platform that profiles structured datasets and applies safe,
explainable optimizations to reduce storage and memory footprint, lossless by default.
Open source, single-machine, CLI-driven.

The value of the project is its guarantees: it never loads a full dataset into memory,
never loses data by accident, and can explain every transformation it made. Protect
those guarantees over features.

---

## 2. Invariants (never violate)

### I1. Streaming first
No component loads an entire dataset into memory. All processing consumes chunks from the
`ChunkManager`. The only exception is an algorithm that provably cannot stream; it must be
justified, documented, and confirmed with me before implementation. "Simpler in memory"
is not a justification.

### I2. Lossless by default
The default mode reconstructs the input exactly. Balanced and aggressive modes may lose
information only when explicitly selected. An optimizer that cannot guarantee exact
reconstruction does not run in lossless mode. Ever.

### I3. Validate before export
No output path is written until the validator confirms it against the selected mode
(schema, row count, column integrity, and lossless comparison where required). A failed
validation aborts export and reports why; it never writes partial output.

### I4. Explainable
Every optimizer records what it changed, on which columns, and why, into the execution
report. A transformation the report cannot account for is a bug.

### I5. Plugins never touch the core
A new optimizer is a plugin, registered through the plugin registry. Adding one must never
require editing the engine, planner, validator, or another optimizer. If it seems to,
stop and raise it.

### I6. Interfaces are doors, not logic
The CLI and the GUI are two doors into one engine. Neither contains optimization,
profiling, validation, or export logic of its own; both import and call the same engine
entry points. If the GUI needs something the engine does not expose, that is a gap in the
engine to fix, not logic to add in the GUI. This keeps CLI and GUI permanently in sync: a
fix in the engine is a fix in both doors at once. The core engine must remain importable
and runnable with no CLI and no web framework present.

---

## 3. Build order (one milestone, then stop)

Build one milestone, review it, meet its acceptance criteria, run its tests, commit, then
start the next. Do not scaffold ahead. The dependency order is not negotiable.

| # | Milestone | Depends on |
|---|---|---|
| M1 | Repo skeleton, tooling, config loading | none |
| M2 | Core types + `ChunkManager` + streaming contract | M1 |
| M3 | `Reader` protocol + CSV reader (end to end, tested) | M2 |
| M4 | Parquet, JSON, Excel readers | M3 |
| M5 | Profiler + statistics merge + schema analyzer | M3 |
| M6 | Optimization profile + planner | M5 |
| M7 | Optimizer protocol + registry + one worked optimizer | M6 |
| M8 | Remaining core optimizers (one at a time) | M7 |
| M9 | Validator + exporter + report generator | M7 |
| M10 | CLI + structured logging | M9 |
| M11 | Benchmark harness + performance/stress tests | M10 |
| M12 | Sample data + real-dataset testing (via CLI) | M10 |
| M13 | Packaging and install (`adoe` as a command) | M10 |
| M14 | Web GUI (Streamlit), wrapping the engine | M13 |

Two hard rules on order: build M2 before any reader; build one reader fully (with tests)
before the next. Do not start optimizers (M7) until the profiler (M5/M6) produces a real
optimization profile from real chunked data.

**Interface order is also fixed.** The CLI (M10) is the first and primary door into the
engine. The web GUI (M14) is built last, after the engine is packaged, and wraps the same
engine the CLI calls. Do not build the GUI before the CLI, and do not build them in
parallel.

---

## 4. Core interfaces (implement these signatures)

These are the contracts the whole engine is built on. Implement them as written; if one
needs to change, that is a design question to raise, not a detail to vary silently.

### 4.1 The normalized chunk

Every reader emits chunks of this shape, so downstream code is format-independent.

```python
# core/types.py
from dataclasses import dataclass
import polars as pl

@dataclass(frozen=True)
class Chunk:
    """One streamed batch of rows in the common internal representation."""
    data: pl.DataFrame          # the rows in this chunk
    index: int                  # 0-based position in the stream
    is_last: bool               # True for the final chunk
    source_schema: "Schema"     # column names and dtypes, fixed for the stream

@dataclass(frozen=True)
class Schema:
    """Column names and Polars dtypes, established once per dataset."""
    columns: dict[str, pl.DataType]
```

### 4.2 Reader protocol

```python
# core/reader.py
from typing import Protocol, Iterator

class Reader(Protocol):
    """Streams a dataset as a sequence of row batches. Format-specific
    implementations live in core/readers/. Never reads the whole file into
    memory."""

    def open(self) -> None:
        """Acquire the file handle and read enough to establish the schema."""

    def schema(self) -> Schema:
        """Return the dataset schema. Valid only after open()."""

    def read_batches(self, rows_per_batch: int) -> Iterator[pl.DataFrame]:
        """Yield row batches of at most rows_per_batch rows, streaming."""

    def close(self) -> None:
        """Release the file handle. Idempotent."""
```

### 4.3 ChunkManager

Sits between readers and everything downstream. Owns chunk sizing and stream metadata so
no other component decides it.

```python
# core/chunk_manager.py
from typing import Iterator

class ChunkManager:
    """Turns a Reader's batches into a stream of Chunk objects with correct
    index and is_last flags. The single source of chunk boundaries; readers
    never emit Chunk objects directly."""

    def __init__(self, reader: Reader, rows_per_chunk: int) -> None: ...

    def stream(self) -> Iterator[Chunk]:
        """Open the reader, yield Chunk objects until exhausted, close the
        reader even if iteration is abandoned (use try/finally)."""
```

### 4.4 The optimizer protocol (the most important interface)

Every optimizer implements this. It is what keeps the engine extensible without core
edits. Specify and follow it exactly.

```python
# optimizers/base.py
from typing import Protocol
from dataclasses import dataclass

@dataclass(frozen=True)
class OptimizationDecision:
    """An optimizer's judgement about one dataset, produced from the profile
    before any data is touched. Deterministic given the same profile."""
    applicable: bool                 # does this optimizer apply at all?
    columns: list[str]               # which columns it would act on
    rationale: str                   # human-readable why, for the report
    estimated_saving_bytes: int      # planner uses this to order the plan

@dataclass(frozen=True)
class OptimizationResult:
    """What an optimizer actually did to one chunk, for the report."""
    columns_changed: list[str]
    bytes_before: int
    bytes_after: int
    lossless: bool                   # did this operation preserve exact values?
    detail: str                      # specifics, e.g. "int64 -> int16 on col 'age'"

class Optimizer(Protocol):
    name: str
    """Unique identifier, e.g. 'numeric_downcast'."""

    def evaluate(self, profile: "OptimizationProfile") -> OptimizationDecision:
        """Decide whether and where this optimizer applies, from the profile
        alone. No data access. Deterministic."""

    def apply(self, chunk: Chunk, decision: OptimizationDecision) -> tuple[Chunk, OptimizationResult]:
        """Transform one chunk per the decision. Returns the new chunk and a
        record of what was done. Must be pure with respect to the chunk it is
        given: same input, same output, no shared state between chunks unless
        that state is explicitly threaded and documented."""

    def is_lossless(self, decision: OptimizationDecision) -> bool:
        """True only if apply() guarantees exact reconstruction for this
        decision. The engine refuses to run a non-lossless optimizer in
        lossless mode."""
```

### 4.5 Plugin registry

```python
# plugins/registry.py
class OptimizerRegistry:
    """Discovers and holds available optimizers. Optimizers register here;
    the engine asks here. Adding an optimizer means registering it, never
    editing the engine."""

    def register(self, optimizer: Optimizer) -> None: ...
    def all(self) -> list[Optimizer]: ...
    def get(self, name: str) -> Optimizer | None: ...
```

---

## 5. The hard part: streaming statistics (M5)

The profiler must produce dataset-level metrics without holding the dataset. This is the
component most likely to be built wrong by quietly accumulating rows. It must not.

The rule: maintain a fixed-size **running accumulator** per column, updated one chunk at a
time, from which final metrics are derived at the end. Never retain rows.

```python
# core/profiler.py
from dataclasses import dataclass, field

@dataclass
class ColumnAccumulator:
    """Fixed-size running state for one column across all chunks. Its memory
    footprint must not grow with the number of rows seen."""
    count: int = 0
    nulls: int = 0
    minimum: object | None = None
    maximum: object | None = None
    # cardinality: use an approximate sketch (e.g. HyperLogLog) or a capped
    # exact set that spills to 'approximate above N distinct'. Do NOT keep an
    # unbounded set of all distinct values.
    ...

    def update(self, column_chunk) -> None:
        """Fold one chunk's worth of a column into the running state. O(chunk),
        not O(dataset)."""

    def finalize(self) -> "ColumnStats":
        """Derive the reported metrics once the stream is exhausted."""
```

Metrics to collect: column dtype, null ratio, min, max, cardinality (approximate is
acceptable and expected), duplicate percentage, average string length, a small
distribution summary. Each must be computable from a bounded accumulator. If a metric
cannot be, either approximate it or drop it, and say so in the report; do not buffer rows
to compute it.

Test this explicitly: a test that feeds a large dataset in small chunks and asserts peak
memory stays roughly flat is the proof the streaming profiler works. A test on a tiny
in-memory frame does not prove it.

---

## 6. One worked optimizer (M7 template)

Build `numeric_downcast` first, fully, as the template every other optimizer copies.

- **evaluate:** from the profile, find integer/float columns whose observed min/max fit a
  smaller dtype (e.g. int64 whose range fits int16). Return those columns, a rationale,
  and an estimated saving.
- **apply:** cast those columns in the chunk to the smaller dtype.
- **is_lossless:** True when the smaller dtype represents every value in range exactly
  (always true for integer downcast within range; for float downcast, only under balanced
  or aggressive mode, never lossless).
- **tests:** a value at the boundary of the smaller type; a value just outside it (must
  not be selected in lossless mode); round-trip equality after downcast; report content
  correct.

Once this one is complete, tested, and reviewed, the remaining optimizers (dictionary
encoding, boolean, string, sparse, duplicate, datetime, compression) follow the same
five-part shape: evaluate from profile, apply per chunk, declare losslessness, report,
test.

---

## 7. When to ask vs. act

**Act:** implementing the current milestone, reading files, fixing a failing test,
running tests, writing benchmarks.

**Ask first:** any in-memory operation on a full dataset; any optimizer that cannot
guarantee lossless reconstruction; coupling a plugin to the core; adding a dependency
beyond the declared stack; introducing the API before the CLI is done; changing any
interface in section 4.

When ambiguous, ask. One question is cheaper than a wrong direction.

---

## 8. Scope (do not build until the core is done)

**In scope for v1:** the engine, the CLI (primary interface), and a Streamlit web GUI
(secondary interface, built last). The GUI is upload-file, pick-mode, optimize, show
report, download result, and nothing more; it calls the engine, per invariant I6.

**Out of scope for v1:** REST API (until the CLI and GUI are done and a genuine
programmatic-network need exists); distributed execution (Spark, Ray); cloud-storage
connectors; AI-assisted recommendations. These are future work in the guide's Chapter 12
and stay there. If a milestone seems to need one, that is a scoping question.

---

## 9. Technology stack (do not substitute without asking)

| Concern | Choice |
|---|---|
| Language | Python 3.12+ |
| Data processing | Polars, PyArrow, DuckDB |
| CLI | Typer |
| Web GUI | Streamlit (M14 only) |
| Config | Pydantic Settings |
| Logging | Structlog |
| Compression | Zstandard |
| Testing | Pytest |
| Tooling | Ruff, Black, MyPy |
| Packaging | Poetry |

---

## 10. Code standards

- Python 3.12+, full type hints, MyPy must pass.
- Ruff and Black clean before every commit.
- One module, one responsibility. Composition over inheritance.
- Every reader, optimizer and validator independently testable in isolation.
- Docstrings on public interfaces: what it does, what it takes, what it returns.
- Structured logging via Structlog; no bare `print()` in library code.
- Non-zero exit and a clear stderr message on failure.
- No em-dashes in project prose; use semicolons, parentheses, or separate sentences.

---

## 11. Testing (a milestone is not done until these pass)

- Unit tests for every module and optimizer.
- Integration tests for complete workflows.
- Regression tests for every fixed defect, each with a docstring naming the bug.
- **Streaming tests:** feed data in small chunks and assert memory stays bounded. This is
  the test that proves the core promise; a small in-memory test does not.
- Performance and stress tests on representative datasets (M11).
- Round-trip tests for every lossless optimizer: optimize, then reconstruct, then assert
  exact equality with the original.

---

## 12. Documentation

`README.md` is the public face; every command in it must run. `docs/ADOE-guide.md` is the
architecture reference; update it when a design decision changes. Keep both accurate to
the code, and mark anything not yet implemented as such rather than describing intent as
done.

---

## 13. Bottom line

Streaming-first and lossless-by-default are the whole point; protect them over features.
Keep plugins decoupled, validate before writing, and build one milestone at a time. A
smaller engine that never loads a full dataset and never silently loses data is worth more
than a complete one that breaks either promise.
