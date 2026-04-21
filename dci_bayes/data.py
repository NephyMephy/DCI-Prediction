from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
import json
import re

import numpy as np
import pandas as pd


STANDARD_COLUMN_MAP = {
    "date": "date",
    "year": "year",
    "month": "month",
    "day": "day",
    "month_name": "month_name",
    "season": "season",
    "city": "city",
    "state": "state",
    "location": "location",
    "corps_count": "corps_count",
    "corps": "corps",
    "class": "class",
    "place": "place",
    "score": "score",
    "music": "music",
    "visual": "visual",
    "general_effect": "general_effect",
    "general_effect_score": "general_effect",
    "general effect": "general_effect",
    "penalties": "penalties",
    "event_name": "event_name",
}

CAPTION_COLUMNS = ["general_effect", "visual", "music"]


@dataclass
class DataBundle:
    data_dir: Path
    csv_files: List[Path]
    show_file: Optional[Path]
    raw_frames: Dict[str, pd.DataFrame]
    merged: pd.DataFrame
    schema_report: Dict[str, Dict[str, Any]]
    column_roles: Dict[str, str]
    corps_name_map: Dict[str, str]


def _standardize_column_name(name: str) -> str:
    normalized = re.sub(r"[^0-9a-zA-Z]+", "_", str(name).strip().lower())
    return normalized.strip("_")


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = {}
    for column in df.columns:
        key = _standardize_column_name(column)
        renamed[column] = STANDARD_COLUMN_MAP.get(key, key)
    return df.rename(columns=renamed)


def discover_csv_files(data_dir: Path) -> List[Path]:
    return sorted(data_dir.rglob("*.csv"))


def infer_column_roles(columns: Iterable[str]) -> Dict[str, str]:
    roles: Dict[str, str] = {}
    for column in columns:
        lower = _standardize_column_name(column)
        if lower in {"corps", "team", "group", "ensemble"}:
            roles[column] = "corps_id"
        elif lower in {"date", "show_date", "competition_date"}:
            roles[column] = "date"
        elif lower in {"year", "season_year"}:
            roles[column] = "year"
        elif lower in {"score", "total_score", "total"}:
            roles[column] = "total_score"
        elif lower in {"general_effect", "ge", "effect"}:
            roles[column] = "general_effect"
        elif lower in {"visual", "visual_score"}:
            roles[column] = "visual"
        elif lower in {"music", "music_score"}:
            roles[column] = "music"
        elif lower in {"place", "placement", "rank"}:
            roles[column] = "placement"
        elif lower in {"city", "state", "location", "class", "season", "month", "day", "month_name", "corps_count", "event_name", "penalties"}:
            roles[column] = lower
    return roles


def inspect_csv(path: Path, preview_rows: int = 5) -> Dict[str, Any]:
    preview = pd.read_csv(path, nrows=preview_rows)
    preview = _standardize_columns(preview)
    full = pd.read_csv(path)
    full = _standardize_columns(full)

    schema: Dict[str, Any] = {
        "path": str(path),
        "rows": int(full.shape[0]),
        "columns": list(full.columns),
        "dtypes": {column: str(dtype) for column, dtype in full.dtypes.items()},
        "missing": {column: int(full[column].isna().sum()) for column in full.columns},
        "preview": preview.head(preview_rows).to_dict(orient="records"),
    }
    print(f"\n[CSV] {path.name}")
    print(f"Rows: {schema['rows']}")
    print(f"Columns: {schema['columns']}")
    print("Preview:")
    print(preview.head(preview_rows).to_string(index=False))
    return schema


def load_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = _standardize_columns(frame)
    frame["source_file"] = path.name
    return frame


def load_corps_name_map(data_dir: Path) -> Dict[str, str]:
    logo_path = data_dir / "corps_logos" / "logo_dictionary.csv"
    if not logo_path.exists():
        return {}

    try:
        logo_df = pd.read_csv(logo_path)
    except pd.errors.EmptyDataError:
        return {}

    logo_df = _standardize_columns(logo_df)
    if "corps" not in logo_df.columns:
        return {}

    corps_map: Dict[str, str] = {}
    for value in logo_df["corps"].dropna().astype(str):
        cleaned = re.sub(r"\s+", " ", value).strip()
        corps_map[cleaned.lower()] = cleaned
    return corps_map


def standardize_corps_name(value: Any, corps_name_map: Dict[str, str]) -> Any:
    if pd.isna(value):
        return value
    cleaned = re.sub(r"\s+", " ", str(value)).strip()
    key = cleaned.lower()
    return corps_name_map.get(key, cleaned)


def parse_competition_date(frame: pd.DataFrame) -> pd.Series:
    if "date" not in frame.columns:
        return pd.Series(pd.NaT, index=frame.index)

    date_series = frame["date"].astype(str)
    if date_series.str.contains(r"^\d{4}-\d{2}-\d{2}$", regex=True).any():
        return pd.to_datetime(date_series, errors="coerce")

    if "year" in frame.columns:
        combined = frame["year"].astype(str).str.strip() + "-" + date_series.str.strip()
        parsed = pd.to_datetime(combined, format="%Y-%m/%d", errors="coerce")
        if parsed.notna().any():
            return parsed
        parsed = pd.to_datetime(combined, format="%Y-%m-%d", errors="coerce")
        if parsed.notna().any():
            return parsed

    parsed = pd.to_datetime(date_series, errors="coerce")
    if parsed.notna().any():
        return parsed

    return pd.Series(pd.NaT, index=frame.index)


def load_and_prepare_data(data_dir: str | Path) -> DataBundle:
    data_dir = Path(data_dir).resolve()
    csv_files = discover_csv_files(data_dir)
    show_file = data_dir / "shows" / "master_shows_list.csv"
    raw_frames: Dict[str, pd.DataFrame] = {}
    schema_report: Dict[str, Dict[str, Any]] = {}

    print(f"Discovered {len(csv_files)} CSV files under {data_dir}")
    for path in csv_files:
        try:
            raw_frames[path.name] = load_csv(path)
            schema_report[path.name] = inspect_csv(path)
        except Exception as exc:
            print(f"Skipping {path.name}: {exc}")
            continue

    corps_name_map = load_corps_name_map(data_dir)
    merged_frames: List[pd.DataFrame] = []
    for path in csv_files:
        if path.name == show_file.name or path.parent.name != "years":
            continue
        frame = raw_frames.get(path.name)
        if frame is not None:
            merged_frames.append(frame)

    if merged_frames:
        merged = pd.concat(merged_frames, ignore_index=True, sort=False)
    else:
        merged = pd.DataFrame()

    if show_file.exists():
        try:
            show_frame = load_csv(show_file)
            if "competition_date" not in show_frame.columns:
                show_frame["competition_date"] = parse_competition_date(show_frame)
            if "competition_date" not in merged.columns:
                merged["competition_date"] = parse_competition_date(merged)
            merged = merged.merge(
                show_frame,
                on=[
                    column
                    for column in ["competition_date", "city", "state"]
                    if column in merged.columns and column in show_frame.columns
                ],
                how="left",
                suffixes=("", "_show"),
            )
        except Exception as exc:
            print(f"Could not merge show metadata: {exc}")

    if not merged.empty and "corps" in merged.columns:
        merged["corps"] = merged["corps"].apply(lambda value: standardize_corps_name(value, corps_name_map))

    if not merged.empty and "date" in merged.columns:
        merged["competition_date"] = parse_competition_date(merged)

    column_roles = infer_column_roles(merged.columns)
    print("Column roles inferred:", column_roles)
    print("Caption columns present:", [column for column in CAPTION_COLUMNS if column in merged.columns])

    return DataBundle(
        data_dir=data_dir,
        csv_files=csv_files,
        show_file=show_file if show_file.exists() else None,
        raw_frames=raw_frames,
        merged=merged,
        schema_report=schema_report,
        column_roles=column_roles,
        corps_name_map=corps_name_map,
    )


def clean_dci_data(frame: pd.DataFrame) -> pd.DataFrame:
    cleaned = frame.copy()
    if "corps" in cleaned.columns:
        cleaned["corps"] = cleaned["corps"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    if "class" in cleaned.columns:
        cleaned["class"] = cleaned["class"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()

    if "score" in cleaned.columns:
        cleaned["score"] = pd.to_numeric(cleaned["score"], errors="coerce")
    for column in [column for column in CAPTION_COLUMNS if column in cleaned.columns]:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    if "place" in cleaned.columns:
        cleaned["place"] = pd.to_numeric(cleaned["place"], errors="coerce")
    if "year" in cleaned.columns:
        cleaned["year"] = pd.to_numeric(cleaned["year"], errors="coerce").astype("Int64")

    if "competition_date" not in cleaned.columns:
        cleaned["competition_date"] = parse_competition_date(cleaned)

    cleaned = cleaned.dropna(subset=[column for column in ["corps", "score", "competition_date"] if column in cleaned.columns])
    cleaned = cleaned.drop_duplicates(subset=[column for column in ["competition_date", "city", "state", "corps", "score"] if column in cleaned.columns])
    cleaned = cleaned.sort_values([column for column in ["competition_date", "corps", "place"] if column in cleaned.columns]).reset_index(drop=True)
    return cleaned
