"""Result conversion and model-option file export helpers.

``Network.save()`` already handles ordinary single-run exports. This module adds
small utilities needed by model-option sweeps, where the solver may need to
return or save multiple result tables keyed by option name.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def format_records(records: list[dict[str, Any]], return_type: str):
    """Convert raw network records to the requested return type."""
    return_type = return_type.lower()

    if return_type == "dict":
        return records

    if return_type == "dataframe":
        import pandas as pd
        return pd.DataFrame(records)

    raise ValueError("return_type must be 'dict' or 'dataframe'")


def safe_sheet_name(name: str) -> str:
    """Return an Excel-safe sheet name for a model option."""
    for char in "\\/*[]:?":
        name = name.replace(char, "_")
    return name[:31]


def save_model_option_results(results: dict[str, list[dict[str, Any]]], filename: str) -> None:
    """Save model-option sweep results.

    JSON stores one object keyed by option name. XLSX/XLS stores one sheet per
    option. CSV cannot store multiple sheets, so one file is written per option
    using ``<base>_<option>.csv``.
    """
    path = Path(filename)
    extension = path.suffix.lower().lstrip(".")

    if extension == "json":
        import json
        path.write_text(json.dumps(results, indent=4))
        return

    if extension in {"xlsx", "xls"}:
        import pandas as pd
        with pd.ExcelWriter(path) as writer:
            for option_name, records in results.items():
                pd.DataFrame(records).to_excel(
                    writer,
                    sheet_name=safe_sheet_name(option_name),
                    index=False,
                )
        return

    if extension == "csv":
        import pandas as pd
        base = path.with_suffix("")
        for option_name, records in results.items():
            pd.DataFrame(records).to_csv(
                f"{base}_{option_name}.csv",
                index=False,
            )
        return

    raise ValueError("Unsupported file extension. Use .csv, .json, or .xlsx")
