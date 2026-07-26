from collections.abc import Callable
from pathlib import Path

from core.reader import Reader
from core.readers.csv_reader import CSVReader
from core.readers.excel_reader import ExcelReader
from core.readers.json_reader import JSONReader
from core.readers.parquet_reader import ParquetReader

# JSON here means NDJSON specifically (one JSON object per line), the
# only JSON variant that can be streamed rather than parsed as a whole -
# see JSONReader's docstring. .xlsx reads only the first/active
# worksheet - see ExcelReader's docstring.
_READER_BY_SUFFIX: dict[str, Callable[[str | Path], Reader]] = {
    ".csv": CSVReader,
    ".parquet": ParquetReader,
    ".json": JSONReader,
    ".jsonl": JSONReader,
    ".xlsx": ExcelReader,
}


class ReaderFactory:
    """Selects the Reader implementation for a file, based on its
    extension. Adding a format means registering it here; callers never
    need to know which concrete Reader class backs a given file type."""

    @staticmethod
    def for_path(path: str | Path) -> Reader:
        """Return a Reader for path, chosen by its file extension. A
        trailing .zst (Zstandard-compressed CSV/JSON, as produced by
        `adoe optimize --compress`) is stripped before dispatch, then
        the reader is constructed against the *original* path - Polars'
        CSV/NDJSON scanners auto-detect and transparently decompress
        Zstandard streams by their magic bytes, so CSVReader needs no
        changes to read foo.csv.zst correctly; only the format-dispatch
        here needed to know to look past the .zst."""
        original = Path(path)
        detection_path = (
            original.with_suffix("") if original.suffix == ".zst" else original
        )
        suffix = detection_path.suffix.lower()
        try:
            reader_cls = _READER_BY_SUFFIX[suffix]
        except KeyError:
            supported = ", ".join(sorted(_READER_BY_SUFFIX))
            raise ValueError(
                f"no reader registered for file type {suffix!r} ({path}); "
                f"supported types: {supported}"
            ) from None
        return reader_cls(original)
