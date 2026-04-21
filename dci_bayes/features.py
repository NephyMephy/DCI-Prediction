from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple
import json
import re

import numpy as np
import pandas as pd

from .data import CAPTION_COLUMNS, parse_competition_date
from .preprocessing import ScoreScaler


def build_corps_metadata(data_dir: str | Path) -> pd.DataFrame:
    data_dir = Path(data_dir).resolve()
    corps_dir = data_dir / "corps"
    rows = []
    if not corps_dir.exists():
        return pd.DataFrame(columns=["corps", "championship_count", "finalist_count", "caption_award_count", "source_file"])

    for path in sorted(corps_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text())
        except Exception:
            continue

        corps_name = path.stem
        if isinstance(payload, dict):
            corps_name = str(payload.get("name", corps_name))
            championships = payload.get("championships", []) or []
            finalists = payload.get("finalists", []) or payload.get("shows", []) or []
            caption_awards = payload.get("captionAwards", []) or payload.get("caption_awards", []) or []
            rows.append(
                {
                    "corps": corps_name,
                    "championship_count": len(championships),
                    "finalist_count": len(finalists),
                    "caption_award_count": len(caption_awards),
                    "source_file": path.name,
                }
            )

    return pd.DataFrame(rows)


def engineer_features(frame: pd.DataFrame, score_scaler: ScoreScaler | None = None) -> Tuple[pd.DataFrame, ScoreScaler]:
    cleaned = frame.copy()
    if "competition_date" not in cleaned.columns:
        cleaned["competition_date"] = parse_competition_date(cleaned)
    cleaned["competition_date"] = pd.to_datetime(cleaned["competition_date"], errors="coerce")
    cleaned = cleaned.dropna(subset=["competition_date"]).copy()

    if "year" not in cleaned.columns:
        cleaned["year"] = cleaned["competition_date"].dt.year
    cleaned["year"] = pd.to_numeric(cleaned["year"], errors="coerce").astype("Int64")

    cleaned = cleaned.sort_values(["competition_date", "corps" if "corps" in cleaned.columns else "competition_date"]).reset_index(drop=True)
    cleaned["year_index"] = pd.factorize(cleaned["year"], sort=True)[0]

    if "corps" in cleaned.columns:
        cleaned["corps_index"] = pd.factorize(cleaned["corps"], sort=True)[0]
    else:
        cleaned["corps_index"] = 0

    yearly_min_dates = cleaned.groupby("year")["competition_date"].transform("min")
    cleaned["days_since_season_start"] = (cleaned["competition_date"] - yearly_min_dates).dt.days.astype(float)
    cleaned["days_since_season_start"] = cleaned["days_since_season_start"].fillna(0.0)

    if "corps" in cleaned.columns:
        cleaned["corps_show_index"] = cleaned.groupby(["corps", "year"]).cumcount()
        cleaned["corps_total_show_index"] = cleaned.groupby("corps").cumcount()
    else:
        cleaned["corps_show_index"] = np.arange(len(cleaned))
        cleaned["corps_total_show_index"] = np.arange(len(cleaned))

    latest_date = cleaned["competition_date"].max()
    cleaned["days_from_latest"] = (latest_date - cleaned["competition_date"]).dt.days.astype(float)
    cleaned["days_from_latest"] = cleaned["days_from_latest"].fillna(0.0)

    cleaned["early_season_uncertainty"] = np.exp(-cleaned["days_since_season_start"].clip(lower=0) / 35.0)
    cleaned["time_decay_weight"] = np.exp(-cleaned["days_from_latest"].clip(lower=0) / 3650.0)
    cleaned["time_decay_weight"] = cleaned["time_decay_weight"].clip(lower=0.05, upper=1.0)

    if score_scaler is None:
        score_scaler = ScoreScaler().fit(cleaned["score"] if "score" in cleaned.columns else [])
    if "score" in cleaned.columns:
        cleaned["score_scaled"] = score_scaler.transform(cleaned["score"].to_numpy(dtype=float))
    else:
        cleaned["score_scaled"] = np.nan

    for column in [column for column in CAPTION_COLUMNS if column in cleaned.columns]:
        cleaned[f"{column}_scaled"] = cleaned[column]

    return cleaned, score_scaler
