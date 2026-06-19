"""Result conversion and model-option HDF5 export helpers."""

from __future__ import annotations

from typing import Any

from fullflow.HDF5 import write_model_option_results


def format_records(records: list[dict[str, Any]], return_type: str):
    """Convert raw network records to the requested return type."""
    return_type = return_type.lower()

    if return_type in {"dict", "records"}:
        return records

    raise ValueError("return_type must be 'dict'.")


def save_model_option_results(
    results: dict[str, list[dict[str, Any]]],
    filename: str,
    *,
    model_name: str,
    network_name: str | None = None,
) -> None:
    """Save model-option sweep results to one HDF5 file."""
    write_model_option_results(
        filename,
        results,
        model_name=model_name,
        network_name=network_name,
    )
