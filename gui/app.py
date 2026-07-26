"""Streamlit web GUI (M14): upload a file, pick a mode, optimize, see the
profile and report, download the result.

This module is a "door" into the engine, exactly like cli/main.py - it
contains no profiling, optimization, or validation logic of its own
(invariant I6). Every action below calls the same engine.pipeline entry
points the CLI calls (run_profile, run_optimize); this file only adapts
Streamlit's upload/download widgets to the file paths those functions
expect, and formats their already-computed results for display - the same
kind of presentation-only work cli/main.py does with typer.echo.

Run with:

    poetry run streamlit run gui/app.py
"""

import tempfile
from pathlib import Path

import polars as pl
import streamlit as st

from config.settings import ADOESettings
from core.mode import OptimizationMode
from engine.exporter import ExportError, ExportFormat
from engine.pipeline import run_optimize, run_profile
from optimizers.defaults import build_default_registry

# Same exceptions cli/main.py treats as a clean, reportable failure rather
# than a crash: a missing file, an unsupported format, or a malformed row.
_DATA_ERRORS = (FileNotFoundError, ValueError, pl.exceptions.ComputeError)

# ExcelReader can read .xlsx (see core/readers/excel_reader.py), but the
# Exporter has no Excel writer - only CSV/Parquet/JSON export exist. An
# uploaded .xlsx is therefore optimized and offered back as Parquet
# instead of round-tripping to .xlsx.
_INPUT_ONLY_SUFFIXES = {".xlsx"}
_FALLBACK_FORMAT_FOR_INPUT_ONLY = ExportFormat.PARQUET

_SETTINGS = ADOESettings()

st.set_page_config(page_title="ADOE", page_icon="\U0001f4e6")
st.title("Adaptive Data Optimization Engine")
st.caption(
    "Upload a dataset, choose a mode, and optimize it, lossless by default. "
    "Streaming, validated, and explainable - the same engine the CLI uses."
)

uploaded_file = st.file_uploader(
    "Dataset to optimize", type=["csv", "parquet", "json", "jsonl", "xlsx"]
)

mode = st.selectbox(
    "Optimization mode",
    options=list(OptimizationMode),
    format_func=lambda m: (
        f"{m.value} (default)" if m is OptimizationMode.LOSSLESS else m.value
    ),
)
# options is a fixed, non-empty list, so selectbox always returns a member
# of it; only the type stub's signature (T | None) is more permissive.
assert mode is not None

if uploaded_file is not None:
    input_suffix = Path(uploaded_file.name).suffix.lower()
    if input_suffix in _INPUT_ONLY_SUFFIXES:
        export_format: ExportFormat | None = _FALLBACK_FORMAT_FOR_INPUT_ONLY
        output_suffix = f".{_FALLBACK_FORMAT_FOR_INPUT_ONLY.value}"
        st.info(
            "ADOE can read .xlsx but doesn't export it yet - this file will "
            "be optimized and offered back as Parquet."
        )
    else:
        export_format = ExportFormat.from_suffix(uploaded_file.name)
        output_suffix = input_suffix

    if export_format is None:
        st.error(f"Unsupported file type: '{input_suffix}'.")
    elif st.button("Optimize", type="primary"):
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_path = Path(tmp_dir) / f"input{input_suffix}"
            input_path.write_bytes(uploaded_file.getvalue())
            output_path = Path(tmp_dir) / f"optimized{output_suffix}"

            try:
                with st.spinner("Profiling..."):
                    profile_outcome = run_profile(input_path, _SETTINGS.rows_per_chunk)
                with st.spinner("Optimizing..."):
                    outcome = run_optimize(
                        input_path,
                        output_path,
                        export_format,
                        mode,
                        _SETTINGS.rows_per_chunk,
                        build_default_registry(),
                    )
            except _DATA_ERRORS as exc:
                st.session_state.pop("result", None)
                st.error(f"Could not process file: {exc}")
            except ExportError as exc:
                st.session_state.pop("result", None)
                st.error(f"Export refused: {exc}")
            else:
                st.session_state["result"] = {
                    "profile_outcome": profile_outcome,
                    "report": outcome.report,
                    "output_bytes": output_path.read_bytes(),
                    "download_name": f"optimized_{Path(uploaded_file.name).stem}"
                    f"{output_suffix}",
                }

# Rendered from session_state, not inline with the button click above: a
# download_button click reruns this whole script, and the "Optimize"
# button's own clicked state would otherwise be False again on that
# rerun, making the results disappear the instant download was clicked.
if "result" in st.session_state:
    result = st.session_state["result"]
    profile_outcome = result["profile_outcome"]
    report = result["report"]

    st.subheader("Profile")
    st.write(f"Rows: {profile_outcome.row_count}")
    st.dataframe(
        [
            {
                "column": name,
                "kind": profile_outcome.schema_analysis.columns[name].value,
                "dtype": str(stats.dtype),
                "null_ratio": f"{stats.null_ratio:.2%}",
                "cardinality (approx.)": stats.cardinality_estimate,
                "duplicate %": f"{stats.duplicate_percentage:.1f}",
            }
            for name, stats in profile_outcome.dataset_profile.columns.items()
        ],
        width="stretch",
        hide_index=True,
    )

    st.subheader("Optimization report")
    st.write(f"Mode: {report.mode.value}")
    if report.validation.passed:
        st.success("Validation passed")
    else:
        st.error("Validation FAILED")
        for reason in report.validation.reasons:
            st.write(f"- {reason}")

    left, right = st.columns(2)
    left.metric(
        "Bytes (optimized columns only)",
        f"{report.total_bytes_after:,}",
        delta=f"{report.total_bytes_after - report.total_bytes_before:,}",
        delta_color="inverse",
    )
    if report.input_file_bytes is not None and report.output_file_bytes is not None:
        reduction = (
            1 - report.output_file_bytes / report.input_file_bytes
            if report.input_file_bytes
            else 0.0
        )
        right.metric(
            "File size",
            f"{report.output_file_bytes:,} bytes",
            delta=f"-{reduction:.1%}",
            delta_color="inverse",
        )

    st.write(f"{len(report.results)} optimizer applications recorded:")
    for optimization_result in report.results:
        st.write(
            f"- {optimization_result.detail} "
            f"(lossless={optimization_result.lossless})"
        )

    st.download_button(
        "Download optimized file",
        data=result["output_bytes"],
        file_name=result["download_name"],
    )
