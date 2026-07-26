import dataclasses
from typing import Any

import polars as pl
import pytest

from core.types import Chunk, Schema


def test_schema_holds_column_name_to_dtype_mapping() -> None:
    schema = Schema(columns={"id": pl.Int64(), "name": pl.String()})

    assert schema.columns == {"id": pl.Int64(), "name": pl.String()}


def test_schema_is_frozen() -> None:
    schema: Any = Schema(columns={"id": pl.Int64()})

    with pytest.raises(dataclasses.FrozenInstanceError):
        schema.columns = {"id": pl.Float64()}


def test_chunk_holds_data_index_is_last_and_schema() -> None:
    schema = Schema(columns={"id": pl.Int64()})
    data = pl.DataFrame({"id": [1, 2, 3]})

    chunk = Chunk(data=data, index=0, is_last=True, source_schema=schema)

    assert chunk.data.equals(data)
    assert chunk.index == 0
    assert chunk.is_last is True
    assert chunk.source_schema == schema


def test_chunk_is_frozen() -> None:
    schema = Schema(columns={"id": pl.Int64()})
    chunk: Any = Chunk(
        data=pl.DataFrame({"id": [1]}), index=0, is_last=True, source_schema=schema
    )

    with pytest.raises(dataclasses.FrozenInstanceError):
        chunk.index = 1
