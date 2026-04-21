from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

for cache_dir in [Path.home() / "Library" / "Caches" / "arviz", Path.home() / ".cache" / "arviz"]:
    cache_dir.mkdir(parents=True, exist_ok=True)

try:
    import pymc as pm
    import pytensor.tensor as pt
except ModuleNotFoundError:  # pragma: no cover - handled at runtime with a clearer error
    pm = None
    pt = None

from .features import build_corps_metadata, engineer_features
from .preprocessing import ScoreScaler


@dataclass
class DCIModelBundle:
    frame: pd.DataFrame
    scaler: ScoreScaler
    corps_lookup: Dict[str, int]
    year_lookup: Dict[int, int]
    show_lookup: Dict[str, int]
    model: Optional[pm.Model] = None
    corps_metadata: Optional[pd.DataFrame] = None


@dataclass
class ModelHyperParams:
    corps_scale_prior: float = 1.0
    year_rw_sigma_prior: float = 0.5
    drift_strength: float = 1.0
    time_slope_prior: float = 0.25
    obs_sigma_prior: float = 1.0
    early_uncertainty_weight: float = 0.75
    panel_noise_model: bool = False
    caption_noise_scaling: bool = False
    show_sigma_prior: float = 0.35
    realism_penalty_strength: float = 0.0
    sigma_floor: float = 0.08
    corps_volatility_term: bool = True
    corps_sigma_prior: float = 0.25
    corps_specific_drift: bool = True
    corps_drift_prior: float = 0.35


def _build_show_key(frame: pd.DataFrame) -> pd.Series:
    parts = []
    if "competition_date" in frame.columns:
        parts.append(pd.to_datetime(frame["competition_date"], errors="coerce").astype(str))
    else:
        parts.append(pd.Series(["unknown-date"] * len(frame), index=frame.index, dtype="object"))

    for column in ["city", "state", "event_name"]:
        if column in frame.columns:
            value = frame[column].astype(str).str.strip().str.lower()
            value = value.str.replace(r"[^a-z0-9]+", "-", regex=True).str.strip("-")
            parts.append(value)

    combined = parts[0]
    for value in parts[1:]:
        combined = combined + "|" + value
    return combined


def _build_lookup(series: pd.Series) -> Dict[Any, int]:
    values = pd.Series(series.dropna().unique()).sort_values().tolist()
    return {value: index for index, value in enumerate(values)}


def build_dci_model(
    frame: pd.DataFrame,
    score_scaler: ScoreScaler | None = None,
    data_dir: str | None = None,
    model_hyperparams: ModelHyperParams | None = None,
) -> DCIModelBundle:
    if pm is None or pt is None:
        raise RuntimeError(
            "PyMC and PyTensor are required to build the Bayesian model. "
            "Install them in a Python 3.11/3.12 environment with compatible wheels."
        )

    params = model_hyperparams or ModelHyperParams()
    engineered, score_scaler = engineer_features(frame, score_scaler=score_scaler)
    if engineered.empty:
        raise ValueError("No rows available to build the model.")

    if "score_scaled" not in engineered.columns:
        raise ValueError("Score scaling failed; score_scaled column is missing.")

    corps_lookup = _build_lookup(engineered["corps"] if "corps" in engineered.columns else pd.Series(["all"]))
    year_lookup = _build_lookup(engineered["year"].astype(int))
    show_keys = _build_show_key(engineered).fillna("unknown-show").astype(str)
    show_codes, show_uniques = pd.factorize(show_keys, sort=True)
    show_lookup = {value: index for index, value in enumerate(show_uniques.tolist())}

    corps_metadata = None
    corps_prior_mean = np.zeros(len(corps_lookup), dtype=float)
    if data_dir is not None:
        corps_metadata = build_corps_metadata(data_dir)
        if not corps_metadata.empty and "corps" in corps_metadata.columns:
            metadata_lookup = corps_metadata.set_index("corps")
            for corps_name, corps_index in corps_lookup.items():
                if corps_name in metadata_lookup.index:
                    row = metadata_lookup.loc[corps_name]
                    championship_count = float(row.get("championship_count", 0.0) or 0.0)
                    finalist_count = float(row.get("finalist_count", 0.0) or 0.0)
                    corps_prior_mean[corps_index] = 0.05 * np.log1p(championship_count + 0.5 * finalist_count)

    engineered["corps_idx"] = engineered["corps"].map(corps_lookup).astype(int) if "corps" in engineered.columns else 0
    engineered["year_idx"] = engineered["year"].astype(int).map(year_lookup).astype(int)
    engineered["show_key"] = show_keys
    engineered["show_idx"] = pd.Series(show_codes, index=engineered.index).astype(int)
    obs_count = len(engineered)

    coords = {
        "corps": list(corps_lookup.keys()),
        "year": list(year_lookup.keys()),
        "show": list(show_lookup.keys()),
        "obs_id": np.arange(obs_count),
    }

    with pm.Model(coords=coords) as model:
        corps_idx = pm.Data("corps_idx", engineered["corps_idx"].to_numpy(dtype="int64"), dims="obs_id")
        year_idx = pm.Data("year_idx", engineered["year_idx"].to_numpy(dtype="int64"), dims="obs_id")
        show_idx = pm.Data("show_idx", engineered["show_idx"].to_numpy(dtype="int64"), dims="obs_id")
        days_since_season_start = pm.Data(
            "days_since_season_start",
            engineered["days_since_season_start"].to_numpy(dtype=float),
            dims="obs_id",
        )
        early_season_uncertainty = pm.Data(
            "early_season_uncertainty",
            engineered["early_season_uncertainty"].to_numpy(dtype=float),
            dims="obs_id",
        )
        time_decay_weight = pm.Data(
            "time_decay_weight",
            engineered["time_decay_weight"].to_numpy(dtype=float),
            dims="obs_id",
        )
        score_obs = pm.Data("score_obs", engineered["score_scaled"].to_numpy(dtype=float), dims="obs_id")

        global_intercept = pm.Normal("global_intercept", mu=0.0, sigma=1.0)
        corps_scale = pm.HalfNormal("corps_scale", sigma=params.corps_scale_prior)
        corps_base = pm.Normal("corps_base", mu=corps_prior_mean, sigma=corps_scale, dims="corps")

        year_rw_sigma = pm.HalfNormal("year_rw_sigma", sigma=params.year_rw_sigma_prior * params.drift_strength)
        if params.corps_specific_drift:
            corps_drift_scale = pm.HalfNormal("corps_drift_scale", sigma=params.corps_drift_prior, dims="corps")
            year_increment_sigma = year_rw_sigma * (1.0 + corps_drift_scale[:, None])
        else:
            year_increment_sigma = year_rw_sigma
        year_increments = pm.Normal("year_increments", mu=0.0, sigma=year_increment_sigma, dims=("corps", "year"))
        corps_year_state = pm.Deterministic("corps_year_state", pt.cumsum(year_increments, axis=1), dims=("corps", "year"))

        corps_time_slope = pm.Normal("corps_time_slope", mu=0.0, sigma=params.time_slope_prior, dims="corps")
        obs_sigma_base = pm.HalfNormal("obs_sigma_base", sigma=params.obs_sigma_prior)

        if params.panel_noise_model:
            show_sigma_offset = pm.Normal("show_sigma_offset", mu=0.0, sigma=params.show_sigma_prior, dims="show")
            sigma_show_multiplier = pm.Deterministic("sigma_show_multiplier", pt.exp(show_sigma_offset), dims="show")
            sigma_show = obs_sigma_base * sigma_show_multiplier[show_idx]
        else:
            sigma_show = obs_sigma_base

        if params.corps_volatility_term:
            corps_sigma_offset = pm.Normal("corps_sigma_offset", mu=0.0, sigma=params.corps_sigma_prior, dims="corps")
            sigma_corps_multiplier = pm.Deterministic("sigma_corps_multiplier", pt.exp(corps_sigma_offset), dims="corps")
            sigma_corps = sigma_show * sigma_corps_multiplier[corps_idx]
        else:
            sigma_corps = sigma_show

        if params.caption_noise_scaling:
            sigma_ge_scale = pm.HalfNormal("sigma_ge_scale", sigma=0.5)
            sigma_visual_scale = pm.HalfNormal("sigma_visual_scale", sigma=0.5)
            sigma_music_scale = pm.HalfNormal("sigma_music_scale", sigma=0.5)
            pm.Deterministic("caption_sigma_scales", pt.stack([sigma_ge_scale, sigma_visual_scale, sigma_music_scale]))
            caption_scale = (sigma_ge_scale + sigma_visual_scale + sigma_music_scale) / 3.0
        else:
            caption_scale = 1.0

        time_scaled = (days_since_season_start - pt.mean(days_since_season_start)) / (pt.std(days_since_season_start) + 1e-6)
        mu = (
            global_intercept
            + corps_base[corps_idx]
            + corps_year_state[corps_idx, year_idx]
            + corps_time_slope[corps_idx] * time_scaled
        )
        sigma = sigma_corps * caption_scale * (1.0 + params.early_uncertainty_weight * early_season_uncertainty)
        sigma = pm.math.maximum(sigma, params.sigma_floor)

        if params.realism_penalty_strength > 0.0:
            sorted_idx = np.argsort(engineered["competition_date"].to_numpy(dtype="datetime64[ns]"))
            mu_sorted = mu[sorted_idx]
            mu_jump = mu_sorted[1:] - mu_sorted[:-1]
            realism_penalty = -params.realism_penalty_strength * pt.sum(pt.square(pt.clip(pt.abs(mu_jump) - 0.35, 0.0, np.inf)))
            pm.Potential("realism_penalty", realism_penalty)

        weighted_logp = pm.logp(pm.Normal.dist(mu=mu, sigma=sigma), score_obs)
        pm.Potential("weighted_score_likelihood", (time_decay_weight * weighted_logp).sum())

        pm.Deterministic("score_mu", mu, dims="obs_id")
        pm.Deterministic("score_sigma", sigma, dims="obs_id")

    return DCIModelBundle(
        frame=engineered,
        scaler=score_scaler,
        corps_lookup=corps_lookup,
        year_lookup=year_lookup,
        show_lookup=show_lookup,
        model=model,
        corps_metadata=corps_metadata,
    )


def sample_dci_model(
    bundle: DCIModelBundle,
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 4,
    cores: int | None = None,
    target_accept: float = 0.9,
    random_seed: int = 42,
) -> Any:
    if pm is None:
        raise RuntimeError(
            "PyMC is required to sample the Bayesian model. Install it in a compatible environment first."
        )

    if bundle.model is None:
        raise ValueError("Model has not been built.")

    with bundle.model:
        trace = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=cores,
            target_accept=target_accept,
            random_seed=random_seed,
            return_inferencedata=True,
        )
    return trace
