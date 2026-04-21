from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .model import DCIModelBundle


@dataclass
class PredictionResult:
    corps_id: str
    prediction_mean: float
    prediction_median: float
    lower_95: float
    upper_95: float
    samples: np.ndarray
    next_show_date: pd.Timestamp | None


def _select_posterior_samples(trace: Any, variable: str, **selectors: Any) -> np.ndarray:
    posterior = trace.posterior[variable]
    if selectors:
        posterior = posterior.sel(**selectors)
    stacked = posterior.stack(sample=("chain", "draw"))
    return stacked.to_numpy()


def _median_gap_days(frame: pd.DataFrame, corps_id: str) -> float:
    corps_frame = frame.loc[frame["corps"] == corps_id].sort_values("competition_date")
    if len(corps_frame) < 2:
        return 7.0
    deltas = corps_frame["competition_date"].diff().dropna().dt.days
    deltas = deltas[deltas > 0]
    if deltas.empty:
        return 7.0
    return float(deltas.median())


def _predict_from_date(
    corps_id: str,
    bundle: DCIModelBundle,
    trace: Any,
    reference_date: pd.Timestamp,
    n_samples: int = 1000,
) -> PredictionResult:
    frame = bundle.frame
    if corps_id not in bundle.corps_lookup:
        raise KeyError(f"Unknown corps_id: {corps_id}")

    corps_frame = frame.loc[frame["corps"] == corps_id].sort_values("competition_date")
    if corps_frame.empty:
        raise ValueError(f"No historical rows found for corps '{corps_id}'.")

    gap_days = _median_gap_days(frame, corps_id)
    next_show_date = pd.to_datetime(reference_date) + pd.to_timedelta(gap_days, unit="D")
    next_year = int(next_show_date.year)
    known_years = sorted(bundle.year_lookup.keys())
    selected_year = next_year if next_year in bundle.year_lookup else known_years[-1]

    days_since_season_start = float((next_show_date - pd.Timestamp(year=next_year, month=1, day=1)).days)
    day_center = frame["days_since_season_start"].mean()
    day_scale = frame["days_since_season_start"].std(ddof=0) or 1.0
    day_scaled = (days_since_season_start - day_center) / day_scale
    early_uncertainty = float(np.exp(-max(days_since_season_start, 0.0) / 35.0))

    global_intercept = _select_posterior_samples(trace, "global_intercept")
    corps_base = _select_posterior_samples(trace, "corps_base", corps=corps_id)
    corps_time_slope = _select_posterior_samples(trace, "corps_time_slope", corps=corps_id)
    corps_year_state = _select_posterior_samples(trace, "corps_year_state", corps=corps_id, year=selected_year)
    obs_sigma_base = _select_posterior_samples(trace, "obs_sigma_base")
    year_rw_sigma = _select_posterior_samples(trace, "year_rw_sigma")

    if next_year > known_years[-1]:
        year_step = np.random.default_rng(42).normal(0.0, year_rw_sigma)
    else:
        year_step = 0.0

    mu_scaled = global_intercept + corps_base + corps_year_state + corps_time_slope * day_scaled + year_step
    sigma_scaled = obs_sigma_base * (1.0 + 0.75 * early_uncertainty)
    rng = np.random.default_rng(42)
    base_samples = rng.normal(mu_scaled, sigma_scaled)
    if n_samples > 0:
        choice_idx = rng.choice(len(base_samples), size=n_samples, replace=True)
        samples_scaled = base_samples[choice_idx]
    else:
        samples_scaled = base_samples
    samples = bundle.scaler.inverse_transform(samples_scaled)

    return PredictionResult(
        corps_id=corps_id,
        prediction_mean=float(np.mean(samples)),
        prediction_median=float(np.median(samples)),
        lower_95=float(np.quantile(samples, 0.025)),
        upper_95=float(np.quantile(samples, 0.975)),
        samples=samples,
        next_show_date=next_show_date,
    )


def predict_next_show(corps_id: str, bundle: DCIModelBundle, trace: Any, n_samples: int = 1000) -> PredictionResult:
    frame = bundle.frame
    corps_frame = frame.loc[frame["corps"] == corps_id].sort_values("competition_date")
    if corps_frame.empty:
        raise ValueError(f"No historical rows found for corps '{corps_id}'.")
    reference_date = pd.to_datetime(corps_frame.iloc[-1]["competition_date"])
    return _predict_from_date(corps_id, bundle, trace, reference_date=reference_date, n_samples=n_samples)


def simulate_season_progression(
    corps_ids: Iterable[str],
    bundle: DCIModelBundle,
    trace: Any,
    horizon: int = 10,
    n_samples: int = 1000,
) -> pd.DataFrame:
    corps_ids = list(corps_ids)
    rows: List[Dict[str, Any]] = []

    for corps_id in corps_ids:
        current_frame = bundle.frame.loc[bundle.frame["corps"] == corps_id].sort_values("competition_date")
        if current_frame.empty:
            continue
        current_date = pd.to_datetime(current_frame.iloc[-1]["competition_date"])
        for step in range(1, horizon + 1):
            prediction = _predict_from_date(corps_id, bundle, trace, reference_date=current_date, n_samples=n_samples)
            current_date = prediction.next_show_date if prediction.next_show_date is not None else current_date
            rows.append(
                {
                    "corps_id": corps_id,
                    "step": step,
                    "show_date": prediction.next_show_date,
                    "predicted_score": prediction.prediction_mean,
                    "lower_95": prediction.lower_95,
                    "upper_95": prediction.upper_95,
                }
            )

    return pd.DataFrame(rows)
