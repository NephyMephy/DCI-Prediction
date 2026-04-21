from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Tuple

import pandas as pd

from .data import clean_dci_data, load_and_prepare_data

if TYPE_CHECKING:
    from .model import DCIModelBundle


def filter_training_data(
    frame: pd.DataFrame,
    year_min: int | None = None,
    year_max: int | None = None,
    class_pattern: str | None = None,
    max_rows: int | None = None,
) -> pd.DataFrame:
    filtered = frame.copy()

    if year_min is not None and "year" in filtered.columns:
        filtered = filtered.loc[pd.to_numeric(filtered["year"], errors="coerce") >= year_min]

    if year_max is not None and "year" in filtered.columns:
        filtered = filtered.loc[pd.to_numeric(filtered["year"], errors="coerce") <= year_max]

    if class_pattern and "class" in filtered.columns:
        filtered = filtered.loc[
            filtered["class"].astype(str).str.contains(class_pattern, case=False, regex=True, na=False)
        ]

    if max_rows is not None and max_rows > 0 and len(filtered) > max_rows:
        if "competition_date" in filtered.columns:
            filtered = filtered.sort_values("competition_date").tail(max_rows)
        else:
            filtered = filtered.tail(max_rows)

    return filtered.reset_index(drop=True)


def prepare_bundle(
    data_dir: str | Path = "data",
    year_min: int | None = None,
    year_max: int | None = None,
    class_pattern: str | None = None,
    max_rows: int | None = None,
) -> DCIModelBundle:
    from .model import build_dci_model

    data_bundle = load_and_prepare_data(data_dir)
    cleaned = clean_dci_data(data_bundle.merged)
    cleaned = filter_training_data(
        cleaned,
        year_min=year_min,
        year_max=year_max,
        class_pattern=class_pattern,
        max_rows=max_rows,
    )
    model_bundle = build_dci_model(cleaned, data_dir=str(data_dir))
    return model_bundle


def fit_bundle(
    data_dir: str | Path = "data",
    year_min: int | None = None,
    year_max: int | None = None,
    class_pattern: str | None = None,
    max_rows: int | None = None,
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 4,
    cores: int | None = None,
) -> Tuple[DCIModelBundle, Any]:
    from .model import sample_dci_model

    bundle = prepare_bundle(
        data_dir=data_dir,
        year_min=year_min,
        year_max=year_max,
        class_pattern=class_pattern,
        max_rows=max_rows,
    )
    trace = sample_dci_model(bundle, draws=draws, tune=tune, chains=chains, cores=cores)
    return bundle, trace
