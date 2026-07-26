from dataclasses import dataclass

import polars as pl


@dataclass(frozen=True)
class Chunk:
    """One streamed batch of rows in the common internal representation."""

    data: pl.DataFrame  # the rows in this chunk
    index: int  # 0-based position in the stream
    is_last: bool  # True for the final chunk
    source_schema: "Schema"  # column names and dtypes, fixed for the stream


@dataclass(frozen=True)
class Schema:
    """Column names and Polars dtypes, established once per dataset."""

    columns: dict[str, pl.DataType]
