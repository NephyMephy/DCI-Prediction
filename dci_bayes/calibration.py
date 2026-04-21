from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple
import json
import logging

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

from .data import clean_dci_data, load_and_prepare_data
from .model import DCIModelBundle, ModelHyperParams, build_dci_model, sample_dci_model


@dataclass
class HoldoutResult:
    holdout_year: int
    best_iteration: int
    best_metrics: Dict[str, float]
    scaling: Dict[str, Any]
    predictions_path: str
    metrics_path: str
    best_hyperparams: Dict[str, Any]
    predictions_frame: pd.DataFrame | None = None
    train_frame: pd.DataFrame | None = None
    calibration_frame: pd.DataFrame | None = None


@dataclass
class MultiYearValidationResult:
    years: List[int]
    per_year: Dict[str, Dict[str, Any]]
    aggregate_metrics: Dict[str, float]
    overfit_gap: Dict[str, float]
    output_path: str


def _ensure_output_dir(output_dir: str | Path) -> Path:
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    return out


def _setup_logger(output_dir: Path, logger_name: str = "dci.calibration") -> logging.Logger:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(output_dir / "calibration.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def _split_train_calibration(frame: pd.DataFrame, holdout_year: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    year_values = pd.to_numeric(frame["year"], errors="coerce")
    train_data = frame.loc[year_values != holdout_year].copy()
    calibration_data = frame.loc[year_values == holdout_year].copy()
    sort_cols = [column for column in ["competition_date", "city", "state", "corps", "place"] if column in frame.columns]
    if sort_cols:
        train_data = train_data.sort_values(sort_cols).reset_index(drop=True)
        calibration_data = calibration_data.sort_values(sort_cols).reset_index(drop=True)
    return train_data, calibration_data


def _make_show_key(frame: pd.DataFrame) -> pd.Series:
    date_part = pd.to_datetime(frame["competition_date"], errors="coerce").astype(str)
    city_part = frame["city"].astype(str) if "city" in frame.columns else pd.Series([""] * len(frame), index=frame.index)
    state_part = frame["state"].astype(str) if "state" in frame.columns else pd.Series([""] * len(frame), index=frame.index)
    return date_part + "|" + city_part + "|" + state_part


def _caption_split_from_total(total_samples: np.ndarray, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    weights = rng.dirichlet(alpha=np.array([12.0, 12.0, 12.0]), size=len(total_samples))
    return {
        "pred_ge_samples": total_samples * weights[:, 0],
        "pred_visual_samples": total_samples * weights[:, 1],
        "pred_music_samples": total_samples * weights[:, 2],
    }


def _estimate_sigma_for_context(trace: Any, show_key: str, corps_id: str) -> np.ndarray:
    posterior = trace.posterior
    sigma_base = posterior["obs_sigma_base"].stack(sample=("chain", "draw")).to_numpy()

    sigma = sigma_base.copy()

    if "sigma_show_multiplier" in posterior:
        show_values = list(posterior["sigma_show_multiplier"].coords["show"].values.tolist())
        if show_key in show_values:
            multiplier = posterior["sigma_show_multiplier"].sel(show=show_key).stack(sample=("chain", "draw")).to_numpy()
        else:
            multiplier = posterior["sigma_show_multiplier"].mean(dim="show").stack(sample=("chain", "draw")).to_numpy()
        sigma = sigma * multiplier

    if "sigma_corps_multiplier" in posterior:
        corps_values = list(posterior["sigma_corps_multiplier"].coords["corps"].values.tolist())
        if corps_id in corps_values:
            corps_multiplier = posterior["sigma_corps_multiplier"].sel(corps=corps_id).stack(sample=("chain", "draw")).to_numpy()
        else:
            corps_multiplier = posterior["sigma_corps_multiplier"].mean(dim="corps").stack(sample=("chain", "draw")).to_numpy()
        sigma = sigma * corps_multiplier

    if "caption_sigma_scales" in posterior:
        caption_scale = posterior["caption_sigma_scales"].mean(dim="caption_sigma_scales_dim_0").stack(sample=("chain", "draw")).to_numpy()
        sigma = sigma * caption_scale

    return np.maximum(sigma, 0.08)


def _predict_for_corps(
    row: pd.Series,
    trace: Any,
    year_lookup: Dict[int, int],
    train_frame: pd.DataFrame,
    scaler_mean: float,
    scaler_std: float,
    latent_state_scaled: Dict[str, float],
    rng: np.random.Generator,
    n_samples: int,
    bundle: DCIModelBundle,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    posterior = trace.posterior
    corps_id = str(row["corps"])
    show_date = pd.to_datetime(row["competition_date"])
    show_key = str(row["show_key"])

    known_corps = set(posterior["corps_base"].coords["corps"].values.tolist())
    known_years = sorted(year_lookup.keys())
    selected_year = show_date.year if show_date.year in year_lookup else known_years[-1]

    day_center = float(train_frame["days_since_season_start"].mean())
    day_scale = float(train_frame["days_since_season_start"].std(ddof=0) or 1.0)
    days_since_season_start = float((show_date - pd.Timestamp(year=show_date.year, month=1, day=1)).days)
    day_scaled = (days_since_season_start - day_center) / day_scale
    early_uncertainty = float(np.exp(-max(days_since_season_start, 0.0) / 35.0))

    stacked_global = posterior["global_intercept"].stack(sample=("chain", "draw")).to_numpy()
    sigma_scaled = _estimate_sigma_for_context(trace, show_key, corps_id)

    if corps_id in known_corps:
        corps_base = posterior["corps_base"].sel(corps=corps_id).stack(sample=("chain", "draw")).to_numpy()
        corps_time_slope = posterior["corps_time_slope"].sel(corps=corps_id).stack(sample=("chain", "draw")).to_numpy()
        corps_year_state = (
            posterior["corps_year_state"]
            .sel(corps=corps_id, year=selected_year)
            .stack(sample=("chain", "draw"))
            .to_numpy()
        )
        mu_scaled = stacked_global + corps_base + corps_year_state + corps_time_slope * day_scaled
    else:
        mu_scaled = stacked_global

    previous_state = latent_state_scaled.get(corps_id)
    if previous_state is not None:
        mu_scaled = 0.7 * mu_scaled + 0.3 * previous_state

    if show_date.year > known_years[-1]:
        year_rw_sigma = posterior["year_rw_sigma"].stack(sample=("chain", "draw")).to_numpy()
        mu_scaled = mu_scaled + rng.normal(0.0, year_rw_sigma)

    total_scaled_samples = rng.normal(mu_scaled, sigma_scaled * (1.0 + 0.75 * early_uncertainty))
    if n_samples > 0:
        idx = rng.choice(len(total_scaled_samples), size=n_samples, replace=True)
        total_scaled_samples = total_scaled_samples[idx]

    total_samples = total_scaled_samples * scaler_std + scaler_mean
    caption_samples = _caption_split_from_total(total_samples, rng)
    latent_state_scaled[corps_id] = float(np.mean(total_scaled_samples))

    return (
        total_samples,
        caption_samples["pred_ge_samples"],
        caption_samples["pred_visual_samples"],
        caption_samples["pred_music_samples"],
    )


def rolling_predict_season(
    train_frame: pd.DataFrame,
    calibration_frame: pd.DataFrame,
    bundle: DCIModelBundle,
    trace: Any,
    n_samples: int = 500,
    online_update: bool = True,
    random_seed: int = 42,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)
    rows: List[Dict[str, Any]] = []

    season = calibration_frame.copy()
    season = season.sort_values([column for column in ["competition_date", "city", "state", "corps"] if column in season.columns])
    season["show_key"] = _make_show_key(season)

    latent_state_scaled: Dict[str, float] = {}

    for show_key, show_df in season.groupby("show_key", sort=True):
        show_date = pd.to_datetime(show_df["competition_date"].iloc[0])
        if logger:
            logger.info("Rolling prediction for show %s (%d corps)", show_key, len(show_df))

        for _, row in show_df.iterrows():
            corps_id = str(row["corps"])
            total_samples, ge_samples, visual_samples, music_samples = _predict_for_corps(
                row=row,
                trace=trace,
                year_lookup=bundle.year_lookup,
                train_frame=bundle.frame,
                scaler_mean=bundle.scaler.mean_,
                scaler_std=bundle.scaler.std_,
                latent_state_scaled=latent_state_scaled,
                rng=rng,
                n_samples=n_samples,
                bundle=bundle,
            )

            observed_total = float(row["score"])
            row_payload = {
                "show_key": show_key,
                "show_date": show_date,
                "year": int(show_date.year),
                "season_week": int(max((show_date.dayofyear - 1) // 7 + 1, 1)),
                "days_since_season_start": float((show_date - pd.Timestamp(year=show_date.year, month=1, day=1)).days),
                "corps": corps_id,
                "observed_total": observed_total,
                "predicted_total": float(np.mean(total_samples)),
                "pred_total_lower_95": float(np.quantile(total_samples, 0.025)),
                "pred_total_upper_95": float(np.quantile(total_samples, 0.975)),
                "pred_ge": float(np.mean(ge_samples)),
                "pred_visual": float(np.mean(visual_samples)),
                "pred_music": float(np.mean(music_samples)),
                "observed_place": float(row["place"]) if pd.notna(row.get("place")) else np.nan,
                "sample_std": float(np.std(total_samples, ddof=0)),
                "show_loglik": float(np.mean(norm.logpdf(observed_total, loc=total_samples, scale=max(np.std(total_samples), 1e-3)))),
            }
            if "general_effect" in row.index and pd.notna(row["general_effect"]):
                row_payload["observed_ge"] = float(row["general_effect"])
            if "visual" in row.index and pd.notna(row["visual"]):
                row_payload["observed_visual"] = float(row["visual"])
            if "music" in row.index and pd.notna(row["music"]):
                row_payload["observed_music"] = float(row["music"])

            rows.append(row_payload)

        if online_update:
            pass

    frame = pd.DataFrame(rows)
    frame = frame.sort_values(["show_date", "show_key", "corps"]).reset_index(drop=True)
    return frame


def fit_score_scaling(predictions: pd.DataFrame) -> Dict[str, float]:
    frame = predictions.dropna(subset=["observed_total", "predicted_total"]).copy()
    if frame.empty or frame["predicted_total"].std(ddof=0) == 0:
        return {"mode": "global", "alpha": 1.0, "beta": 0.0}

    alpha, beta = np.polyfit(frame["predicted_total"].to_numpy(), frame["observed_total"].to_numpy(), 1)
    return {"mode": "global", "alpha": float(alpha), "beta": float(beta)}


def fit_time_dependent_scaling(predictions: pd.DataFrame, method: str = "piecewise") -> Dict[str, Any]:
    frame = predictions.dropna(subset=["observed_total", "predicted_total", "season_week"]).copy()
    if frame.empty:
        return {"mode": "global", "alpha": 1.0, "beta": 0.0}

    if method == "week_linear":
        frame["week_norm"] = (frame["season_week"] - frame["season_week"].min()) / (
            max(frame["season_week"].max() - frame["season_week"].min(), 1)
        )
        x1 = frame["predicted_total"].to_numpy()
        x2 = frame["predicted_total"].to_numpy() * frame["week_norm"].to_numpy()
        x3 = frame["week_norm"].to_numpy()
        design = np.column_stack([x1, x2, x3, np.ones(len(frame))])
        y = frame["observed_total"].to_numpy()
        coeffs, *_ = np.linalg.lstsq(design, y, rcond=None)
        return {
            "mode": "week_linear",
            "alpha0": float(coeffs[0]),
            "alpha1": float(coeffs[1]),
            "beta1": float(coeffs[2]),
            "beta0": float(coeffs[3]),
        }

    quantiles = frame["season_week"].quantile([0.3, 0.7]).tolist()
    q1, q2 = quantiles[0], quantiles[1]

    segments = {
        "early": frame.loc[frame["season_week"] <= q1],
        "mid": frame.loc[(frame["season_week"] > q1) & (frame["season_week"] <= q2)],
        "late": frame.loc[frame["season_week"] > q2],
    }
    params: Dict[str, Dict[str, float]] = {}
    for name, seg in segments.items():
        if len(seg) < 8 or seg["predicted_total"].std(ddof=0) == 0:
            params[name] = {"alpha": 1.0, "beta": 0.0}
        else:
            alpha, beta = np.polyfit(seg["predicted_total"].to_numpy(), seg["observed_total"].to_numpy(), 1)
            params[name] = {"alpha": float(alpha), "beta": float(beta)}

    return {
        "mode": "piecewise",
        "q1": float(q1),
        "q2": float(q2),
        "segments": params,
    }


def _apply_scaling(predictions: pd.DataFrame, scaling: Dict[str, Any] | None) -> pd.DataFrame:
    frame = predictions.copy()
    if scaling is None:
        return frame

    mode = scaling.get("mode", "global")
    if mode == "global":
        alpha = float(scaling.get("alpha", 1.0))
        beta = float(scaling.get("beta", 0.0))
        for col in ["predicted_total", "pred_total_lower_95", "pred_total_upper_95"]:
            frame[col] = alpha * frame[col] + beta
        return frame

    if mode == "week_linear":
        week_min = float(frame["season_week"].min())
        week_max = float(frame["season_week"].max())
        week_norm = (frame["season_week"] - week_min) / max(week_max - week_min, 1.0)
        alpha_t = float(scaling["alpha0"]) + float(scaling["alpha1"]) * week_norm
        beta_t = float(scaling["beta0"]) + float(scaling["beta1"]) * week_norm
        for col in ["predicted_total", "pred_total_lower_95", "pred_total_upper_95"]:
            frame[col] = alpha_t * frame[col] + beta_t
        return frame

    if mode == "piecewise":
        q1 = float(scaling["q1"])
        q2 = float(scaling["q2"])
        segments = scaling["segments"]
        bucket = np.where(frame["season_week"] <= q1, "early", np.where(frame["season_week"] <= q2, "mid", "late"))
        for name in ["early", "mid", "late"]:
            mask = bucket == name
            alpha = float(segments[name]["alpha"])
            beta = float(segments[name]["beta"])
            for col in ["predicted_total", "pred_total_lower_95", "pred_total_upper_95"]:
                frame.loc[mask, col] = alpha * frame.loc[mask, col] + beta
        return frame

    return frame


def compute_calibration_metrics(predictions: pd.DataFrame, scaling: Dict[str, Any] | None = None) -> Dict[str, float]:
    frame = _apply_scaling(predictions, scaling)

    residual = frame["observed_total"] - frame["predicted_total"]
    mae = float(np.mean(np.abs(residual)))

    per_show_spearman: List[float] = []
    finals_cutoff_hits: List[float] = []
    for _, show in frame.groupby("show_key"):
        if len(show) < 2:
            continue
        rho = spearmanr(show["predicted_total"], show["observed_total"]).correlation
        if np.isfinite(rho):
            per_show_spearman.append(float(rho))

        obs_window = set(show.sort_values("observed_total", ascending=False).head(14).tail(5)["corps"].tolist())
        pred_window = set(show.sort_values("predicted_total", ascending=False).head(14).tail(5)["corps"].tolist())
        if obs_window:
            finals_cutoff_hits.append(len(obs_window & pred_window) / len(obs_window))

    mean_spearman = float(np.mean(per_show_spearman)) if per_show_spearman else np.nan
    finals_cutoff_accuracy = float(np.mean(finals_cutoff_hits)) if finals_cutoff_hits else np.nan

    last_show_key = frame.sort_values("show_date")["show_key"].iloc[-1]
    final_show = frame.loc[frame["show_key"] == last_show_key].copy()
    pred_top12 = set(final_show.sort_values("predicted_total", ascending=False).head(12)["corps"].tolist())
    obs_top12 = set(final_show.sort_values("observed_total", ascending=False).head(12)["corps"].tolist())
    top12_accuracy = float(len(pred_top12 & obs_top12) / max(1, len(obs_top12)))

    coverage95 = float(
        np.mean(
            (frame["observed_total"] >= frame["pred_total_lower_95"])
            & (frame["observed_total"] <= frame["pred_total_upper_95"])
        )
    )

    sigma = frame["sample_std"].replace(0, np.nan).fillna(frame["sample_std"].mean())
    pit = norm.cdf(frame["observed_total"], loc=frame["predicted_total"], scale=sigma)
    pit_mean = float(np.nanmean(pit))
    calibration_error = float(abs(coverage95 - 0.95) + abs(pit_mean - 0.5))

    return {
        "mean_spearman_per_show": mean_spearman,
        "top12_accuracy": top12_accuracy,
        "finals_cutoff_accuracy": finals_cutoff_accuracy,
        "mae_total": mae,
        "coverage95": coverage95,
        "pit_mean": pit_mean,
        "calibration_error": calibration_error,
    }


def _adjust_hyperparams(current: ModelHyperParams, metrics: Dict[str, float], predictions: pd.DataFrame) -> ModelHyperParams:
    next_params = ModelHyperParams(**asdict(current))

    observed_std = float(predictions["observed_total"].std(ddof=0))
    pred_std = float(predictions["predicted_total"].std(ddof=0))

    if np.isfinite(observed_std) and np.isfinite(pred_std) and observed_std > 0:
        ratio = pred_std / observed_std
        if ratio < 0.85:
            next_params.corps_scale_prior *= 1.08
            next_params.year_rw_sigma_prior *= 1.08
            next_params.drift_strength *= 1.05
        elif ratio > 1.15:
            next_params.corps_scale_prior *= 0.92
            next_params.year_rw_sigma_prior *= 0.92
            next_params.drift_strength *= 0.95

    coverage = metrics.get("coverage95", np.nan)
    if np.isfinite(coverage):
        if coverage < 0.92:
            next_params.obs_sigma_prior *= 1.10
        elif coverage > 0.97:
            next_params.obs_sigma_prior *= 0.93

    cutoff = metrics.get("finals_cutoff_accuracy", np.nan)
    if np.isfinite(cutoff) and cutoff < 0.5:
        next_params.time_slope_prior *= 1.05

    return next_params


def _gaussian_holdout_loglik(predictions: pd.DataFrame, scaling: Dict[str, Any] | None = None) -> float:
    frame = _apply_scaling(predictions, scaling)
    sigma = frame["sample_std"].replace(0, np.nan).fillna(frame["sample_std"].mean()).clip(lower=0.05)
    ll = norm.logpdf(frame["observed_total"], loc=frame["predicted_total"], scale=sigma)
    return float(np.sum(ll))


def compare_noise_models(
    train_data: pd.DataFrame,
    calibration_data: pd.DataFrame,
    data_dir: str,
    draws: int,
    tune: int,
    chains: int,
    cores: int,
    n_samples: int,
    random_seed: int,
    logger: logging.Logger,
) -> Dict[str, Any]:
    candidates = {
        "constant": ModelHyperParams(panel_noise_model=False, caption_noise_scaling=False),
        "panel_aware": ModelHyperParams(panel_noise_model=True, caption_noise_scaling=True),
    }
    results: Dict[str, Any] = {}

    for name, params in candidates.items():
        logger.info("Noise model comparison: fitting %s", name)
        bundle = build_dci_model(frame=train_data, data_dir=data_dir, model_hyperparams=params)
        trace = sample_dci_model(
            bundle,
            draws=max(20, draws // 2),
            tune=max(20, tune // 2),
            chains=1,
            cores=1,
            random_seed=random_seed,
        )
        preds = rolling_predict_season(
            train_frame=train_data,
            calibration_frame=calibration_data,
            bundle=bundle,
            trace=trace,
            n_samples=max(80, n_samples // 2),
            online_update=True,
            random_seed=random_seed,
            logger=logger,
        )
        scaling = fit_time_dependent_scaling(preds, method="piecewise")
        ll = _gaussian_holdout_loglik(preds, scaling=scaling)
        metrics = compute_calibration_metrics(preds, scaling=scaling)
        results[name] = {
            "loglik": ll,
            "metrics": metrics,
            "params": asdict(params),
        }

    best_name = max(results.keys(), key=lambda key: results[key]["loglik"])
    return {"best": best_name, "candidates": results}


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    return value


def run_year_holdout_experiment(
    data_dir: str | Path = "data",
    year: int = 2025,
    train_year_min: int | None = None,
    train_year_max: int | None = None,
    class_pattern: str = "World Class",
    draws: int = 400,
    tune: int = 400,
    chains: int = 2,
    cores: int = 1,
    n_samples: int = 500,
    max_iterations: int = 4,
    mae_tolerance: float = 0.02,
    random_seed: int = 42,
    output_dir: str | Path = "outputs/calibration",
    scaling_mode: str = "piecewise",
    include_frames: bool = False,
    enable_noise_model_selection: bool = True,
) -> HoldoutResult:
    out_dir = _ensure_output_dir(Path(output_dir) / f"holdout_{year}")
    logger = _setup_logger(out_dir)

    logger.info("Loading and splitting data for holdout year %d", year)
    data_bundle = load_and_prepare_data(data_dir)
    cleaned = clean_dci_data(data_bundle.merged)
    train_data, calibration_data = _split_train_calibration(cleaned, holdout_year=year)

    if train_year_min is not None:
        train_data = train_data.loc[pd.to_numeric(train_data["year"], errors="coerce") >= train_year_min]
    if train_year_max is not None:
        train_data = train_data.loc[pd.to_numeric(train_data["year"], errors="coerce") <= train_year_max]
    if class_pattern and "class" in train_data.columns:
        train_data = train_data.loc[train_data["class"].astype(str).str.contains(class_pattern, case=False, regex=True, na=False)]
    if class_pattern and "class" in calibration_data.columns:
        calibration_data = calibration_data.loc[
            calibration_data["class"].astype(str).str.contains(class_pattern, case=False, regex=True, na=False)
        ]

    if calibration_data.empty:
        raise ValueError(f"No calibration rows found for holdout year {year} and class pattern '{class_pattern}'.")

    logger.info("Train rows: %d | Calibration rows: %d", len(train_data), len(calibration_data))

    params = ModelHyperParams(panel_noise_model=False, caption_noise_scaling=False)
    if enable_noise_model_selection:
        noise_selection = compare_noise_models(
            train_data=train_data,
            calibration_data=calibration_data,
            data_dir=str(data_dir),
            draws=draws,
            tune=tune,
            chains=chains,
            cores=cores,
            n_samples=n_samples,
            random_seed=random_seed,
            logger=logger,
        )
        best_noise = noise_selection["best"]
        logger.info("Selected noise model: %s", best_noise)
        if best_noise == "panel_aware":
            params.panel_noise_model = True
            params.caption_noise_scaling = True
    else:
        noise_selection = {"best": "constant", "candidates": {}}

    history: List[Dict[str, Any]] = []
    best_iteration = 0
    best_primary_rank = float("-inf")
    best_metrics: Dict[str, float] = {}
    best_scaling: Dict[str, Any] = {"mode": "global", "alpha": 1.0, "beta": 0.0}
    best_predictions = pd.DataFrame()
    best_params = asdict(params)

    prev_mae = None
    prev_spearman = None

    for iteration in range(1, max_iterations + 1):
        logger.info("Iteration %d | params=%s", iteration, asdict(params))

        model_bundle = build_dci_model(
            frame=train_data,
            data_dir=str(data_dir),
            model_hyperparams=params,
        )
        trace = sample_dci_model(
            model_bundle,
            draws=draws,
            tune=tune,
            chains=chains,
            cores=cores,
            random_seed=random_seed,
        )

        predictions = rolling_predict_season(
            train_frame=train_data,
            calibration_frame=calibration_data,
            bundle=model_bundle,
            trace=trace,
            n_samples=n_samples,
            online_update=True,
            random_seed=random_seed,
            logger=logger,
        )

        if scaling_mode == "week_linear":
            scaling = fit_time_dependent_scaling(predictions, method="week_linear")
        elif scaling_mode == "piecewise":
            scaling = fit_time_dependent_scaling(predictions, method="piecewise")
        else:
            scaling = fit_score_scaling(predictions)

        metrics = compute_calibration_metrics(predictions, scaling=scaling)
        holdout_loglik = _gaussian_holdout_loglik(predictions, scaling=scaling)

        logger.info("Iteration %d metrics: %s", iteration, metrics)
        history.append(
            {
                "iteration": iteration,
                "params": asdict(params),
                "metrics": metrics,
                "scaling": scaling,
                "holdout_loglik": holdout_loglik,
            }
        )

        rank_objective = float(np.nan_to_num(metrics.get("mean_spearman_per_show", np.nan), nan=-1.0)) + float(
            np.nan_to_num(metrics.get("top12_accuracy", np.nan), nan=0.0)
        )
        if rank_objective > best_primary_rank:
            best_primary_rank = rank_objective
            best_iteration = iteration
            best_metrics = metrics
            best_scaling = scaling
            best_predictions = predictions.copy()
            best_params = asdict(params)

        if prev_mae is not None and prev_spearman is not None:
            mae_delta = abs(metrics["mae_total"] - prev_mae)
            spearman_delta = metrics.get("mean_spearman_per_show", np.nan) - prev_spearman
            if mae_delta < mae_tolerance and (not np.isfinite(spearman_delta) or spearman_delta <= 1e-3):
                logger.info("Stopping early at iteration %d due to stabilization.", iteration)
                break

        prev_mae = metrics["mae_total"]
        prev_spearman = metrics.get("mean_spearman_per_show", np.nan)
        params = _adjust_hyperparams(params, metrics, predictions)

    if best_predictions.empty:
        raise RuntimeError("Holdout experiment did not produce predictions.")

    calibrated = _apply_scaling(best_predictions, best_scaling)
    best_predictions["predicted_total_calibrated"] = calibrated["predicted_total"]

    predictions_path = out_dir / f"predictions_holdout_{year}.csv"
    metrics_path = out_dir / f"metrics_holdout_{year}.json"

    best_predictions.to_csv(predictions_path, index=False)

    payload = {
        "holdout_year": year,
        "best_iteration": best_iteration,
        "best_metrics": best_metrics,
        "best_scaling": best_scaling,
        "best_hyperparams": best_params,
        "noise_model_selection": noise_selection,
        "history": history,
    }
    metrics_path.write_text(json.dumps(_to_jsonable(payload), indent=2))

    logger.info("Wrote predictions to %s", predictions_path)
    logger.info("Wrote metrics to %s", metrics_path)

    return HoldoutResult(
        holdout_year=year,
        best_iteration=best_iteration,
        best_metrics=best_metrics,
        scaling=best_scaling,
        predictions_path=str(predictions_path),
        metrics_path=str(metrics_path),
        best_hyperparams=best_params,
        predictions_frame=best_predictions if include_frames else None,
        train_frame=train_data if include_frames else None,
        calibration_frame=calibration_data if include_frames else None,
    )


def run_multi_year_validation(
    years: Iterable[int] = (2023, 2024, 2025),
    data_dir: str | Path = "data",
    train_year_min: int | None = None,
    train_year_max: int | None = None,
    class_pattern: str = "World Class",
    draws: int = 300,
    tune: int = 300,
    chains: int = 2,
    cores: int = 1,
    n_samples: int = 400,
    max_iterations: int = 3,
    random_seed: int = 42,
    output_dir: str | Path = "outputs/calibration",
    scaling_mode: str = "piecewise",
) -> MultiYearValidationResult:
    out_dir = _ensure_output_dir(output_dir)
    per_year: Dict[str, Dict[str, Any]] = {}

    for year in years:
        result = run_year_holdout_experiment(
            data_dir=data_dir,
            year=int(year),
            train_year_min=train_year_min,
            train_year_max=train_year_max,
            class_pattern=class_pattern,
            draws=draws,
            tune=tune,
            chains=chains,
            cores=cores,
            n_samples=n_samples,
            max_iterations=max_iterations,
            random_seed=random_seed,
            output_dir=output_dir,
            scaling_mode=scaling_mode,
            include_frames=False,
            enable_noise_model_selection=True,
        )
        per_year[str(year)] = {
            "best_iteration": result.best_iteration,
            "best_metrics": result.best_metrics,
            "scaling": result.scaling,
            "predictions_path": result.predictions_path,
            "metrics_path": result.metrics_path,
        }

    metric_names = ["mean_spearman_per_show", "top12_accuracy", "finals_cutoff_accuracy", "mae_total", "calibration_error"]
    aggregate_metrics: Dict[str, float] = {}
    for metric in metric_names:
        vals = [float(per_year[str(year)]["best_metrics"].get(metric, np.nan)) for year in years]
        vals = [value for value in vals if np.isfinite(value)]
        aggregate_metrics[f"avg_{metric}"] = float(np.mean(vals)) if vals else np.nan
        aggregate_metrics[f"std_{metric}"] = float(np.std(vals, ddof=0)) if vals else np.nan

    overfit_gap = {
        "rank_metric_range": float(
            (np.nanmax([per_year[str(year)]["best_metrics"].get("mean_spearman_per_show", np.nan) for year in years])
            - np.nanmin([per_year[str(year)]["best_metrics"].get("mean_spearman_per_show", np.nan) for year in years]))
        ),
        "mae_range": float(
            (np.nanmax([per_year[str(year)]["best_metrics"].get("mae_total", np.nan) for year in years])
            - np.nanmin([per_year[str(year)]["best_metrics"].get("mae_total", np.nan) for year in years]))
        ),
    }

    payload = {
        "years": [int(year) for year in years],
        "per_year": per_year,
        "aggregate_metrics": aggregate_metrics,
        "overfit_gap": overfit_gap,
    }
    output_path = out_dir / "multi_year_validation.json"
    output_path.write_text(json.dumps(_to_jsonable(payload), indent=2))

    return MultiYearValidationResult(
        years=[int(year) for year in years],
        per_year=per_year,
        aggregate_metrics=aggregate_metrics,
        overfit_gap=overfit_gap,
        output_path=str(output_path),
    )
