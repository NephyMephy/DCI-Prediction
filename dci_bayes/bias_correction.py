from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple
import json

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.stats import norm
from scipy.stats import spearmanr

from .uncertainty_calibration import _season_phase


TIER_LABELS = ["top5", "finals_6_12", "bubble_13_25", "field"]


@dataclass
class BiasModel:
    mode: str
    score_knots: List[float]
    score_values: List[float]
    sigma_knots: List[float]
    sigma_values: List[float]
    phase_offsets: Dict[str, float]
    tier_offsets: Dict[str, float]
    phase_sigma_offsets: Dict[str, float]
    tier_sigma_offsets: Dict[str, float]
    shrinkage: float
    score_shrinkage: float
    phase_shrinkage: float
    tier_shrinkage: float
    sigma_shrinkage: float
    correction_scale: float
    centered: bool


def _assign_corps_tiers(frame: pd.DataFrame) -> pd.Series:
    base = frame.copy()
    final_show_key = base.sort_values("show_date")["show_key"].iloc[-1]
    final_show = base.loc[base["show_key"] == final_show_key].sort_values("observed_total", ascending=False).reset_index(drop=True)
    rank_map = {str(row.corps): idx + 1 for idx, row in final_show.iterrows()}

    def tier(corps: Any) -> str:
        rank = rank_map.get(str(corps), 999)
        if rank <= 5:
            return "top5"
        if rank <= 12:
            return "finals_6_12"
        if rank <= 25:
            return "bubble_13_25"
        return "field"

    return base["corps"].astype(str).map(tier)


def _isotonic_non_decreasing(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    y_sorted = np.asarray(y, dtype=float)[order].copy()
    weights = np.ones_like(y_sorted)
    i = 0
    while i < len(y_sorted) - 1:
        if y_sorted[i] > y_sorted[i + 1]:
            total_weight = weights[i] + weights[i + 1]
            avg = (y_sorted[i] * weights[i] + y_sorted[i + 1] * weights[i + 1]) / total_weight
            y_sorted[i] = avg
            y_sorted[i + 1] = avg
            weights[i] = total_weight
            weights[i + 1] = total_weight
            j = i
            while j > 0 and y_sorted[j - 1] > y_sorted[j]:
                total_weight = weights[j - 1] + weights[j]
                avg = (y_sorted[j - 1] * weights[j - 1] + y_sorted[j] * weights[j]) / total_weight
                y_sorted[j - 1] = avg
                y_sorted[j] = avg
                weights[j - 1] = total_weight
                weights[j] = total_weight
                j -= 1
        i += 1

    restored = np.empty_like(y_sorted)
    restored[order] = y_sorted
    return restored


def _fit_smoothed_curve(x: np.ndarray, y: np.ndarray, score_shrinkage: float) -> Tuple[List[float], List[float]]:
    if len(x) < 4:
        return x.tolist(), (score_shrinkage * y).tolist()

    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]
    x_unique = np.unique(x_sorted)
    if len(x_unique) < 4:
        return x_unique.tolist(), (score_shrinkage * np.interp(x_unique, x_sorted, y_sorted)).tolist()

    y_monotone = np.maximum.accumulate(y_sorted)
    try:
        spline = PchipInterpolator(x_sorted, y_monotone, extrapolate=True)
        knots = np.quantile(x_sorted, np.linspace(0.0, 1.0, min(12, len(np.unique(x_sorted)))))
        curve = spline(knots)
        return knots.tolist(), (score_shrinkage * curve).tolist()
    except Exception:
        knots = np.quantile(x_sorted, np.linspace(0.0, 1.0, min(12, len(np.unique(x_sorted)))))
        curve = np.interp(knots, x_sorted, y_monotone)
        return knots.tolist(), (score_shrinkage * curve).tolist()


def _distribution_ece(frame: pd.DataFrame, mean_col: str, lower_col: str, upper_col: str, obs_col: str = "observed_total", bins: int = 10) -> float:
    valid = frame.dropna(subset=[mean_col, lower_col, upper_col, obs_col]).copy()
    if valid.empty:
        return float("nan")

    sigma = (valid[upper_col].to_numpy(dtype=float) - valid[lower_col].to_numpy(dtype=float)) / (2.0 * 1.959963984540054)
    sigma = np.clip(sigma, 1e-6, None)
    pit = norm.cdf((valid[obs_col].to_numpy(dtype=float) - valid[mean_col].to_numpy(dtype=float)) / sigma)

    counts, _ = np.histogram(pit, bins=np.linspace(0.0, 1.0, bins + 1))
    total = max(1, int(counts.sum()))
    empirical = counts / total
    expected = np.full_like(empirical, 1.0 / len(empirical), dtype=float)
    return float(np.abs(empirical - expected).sum() / 2.0)


def fit_bias_model(
    predictions: pd.DataFrame,
    bins: int = 10,
    score_shrinkage: float = 0.35,
    phase_shrinkage: float = 0.25,
    tier_shrinkage: float = 0.25,
    iterations: int = 2,
) -> Dict[str, Any]:
    frame = predictions.copy()
    mean_col = "predicted_total_calibrated" if "predicted_total_calibrated" in frame.columns else "predicted_total"
    frame = frame.dropna(subset=["observed_total", mean_col]).copy()
    if frame.empty:
        return {
            "mode": "bias_correction",
            "score_knots": [],
            "score_values": [],
            "sigma_knots": [],
            "sigma_values": [],
            "phase_offsets": {phase: 0.0 for phase in ["early", "mid", "late"]},
            "tier_offsets": {tier: 0.0 for tier in TIER_LABELS},
            "phase_sigma_offsets": {phase: 1.0 for phase in ["early", "mid", "late"]},
            "tier_sigma_offsets": {tier: 1.0 for tier in TIER_LABELS},
            "shrinkage": 0.0,
            "score_shrinkage": score_shrinkage,
            "phase_shrinkage": phase_shrinkage,
            "tier_shrinkage": tier_shrinkage,
            "sigma_shrinkage": 0.84,
            "correction_scale": 0.0,
            "centered": True,
        }

    frame["residual"] = frame["observed_total"] - frame[mean_col]
    frame["phase"] = _season_phase(frame)
    frame["tier"] = _assign_corps_tiers(frame)
    frame = frame.reset_index(drop=True)

    working_residual = frame["residual"].to_numpy(dtype=float)
    score_knots: List[float] = []
    score_values: List[float] = []
    sigma_knots: List[float] = []
    sigma_values: List[float] = []
    phase_offsets: Dict[str, float] = {phase: 0.0 for phase in ["early", "mid", "late"]}
    tier_offsets: Dict[str, float] = {tier: 0.0 for tier in TIER_LABELS}
    phase_sigma_offsets: Dict[str, float] = {phase: 1.0 for phase in ["early", "mid", "late"]}
    tier_sigma_offsets: Dict[str, float] = {tier: 1.0 for tier in TIER_LABELS}

    for _ in range(max(1, iterations)):
        frame["working_residual"] = working_residual
        frame = frame.dropna(subset=["working_residual", mean_col]).copy().reset_index(drop=True)
        frame = frame.sort_values(mean_col).reset_index(drop=True)

        q = min(bins, max(3, len(frame) // 5))
        frame["score_bin"] = pd.qcut(frame[mean_col], q=min(q, max(2, len(frame))), duplicates="drop")
        bin_rows = []
        for _, grp in frame.groupby("score_bin"):
            bin_rows.append((float(grp[mean_col].mean()), float(grp["working_residual"].mean())))
        if bin_rows:
            x_bins = np.array([row[0] for row in bin_rows], dtype=float)
            y_bins = np.array([row[1] for row in bin_rows], dtype=float)
            score_knots, score_values = _fit_smoothed_curve(x_bins, y_bins, score_shrinkage)
            score_component = np.interp(frame[mean_col].to_numpy(dtype=float), np.array(score_knots), np.array(score_values))
        else:
            score_component = np.zeros(len(frame), dtype=float)
            score_knots, score_values = [], []

        residual_after_score = frame["working_residual"].to_numpy(dtype=float) - score_component

        phase_offsets = {}
        phase_centered = {}
        phase_weight_total = 0.0
        phase_weighted_sum = 0.0
        for phase in ["early", "mid", "late"]:
            mask = frame["phase"].astype(str) == phase
            if mask.any():
                value = float(np.mean(residual_after_score[mask])) * phase_shrinkage
            else:
                value = 0.0
            phase_offsets[phase] = value
            phase_weight_total += float(mask.sum())
            phase_weighted_sum += value * float(mask.sum())
        phase_center = phase_weighted_sum / max(phase_weight_total, 1.0)
        for phase in phase_offsets:
            phase_centered[phase] = phase_offsets[phase] - phase_center

        phase_component = np.array([phase_centered.get(phase, 0.0) for phase in frame["phase"].astype(str)], dtype=float)
        residual_after_phase = residual_after_score - phase_component

        tier_offsets = {}
        tier_weight_total = 0.0
        tier_weighted_sum = 0.0
        for tier in TIER_LABELS:
            mask = frame["tier"].astype(str) == tier
            if mask.any():
                value = float(np.mean(residual_after_phase[mask])) * tier_shrinkage
            else:
                value = 0.0
            tier_offsets[tier] = value
            tier_weight_total += float(mask.sum())
            tier_weighted_sum += value * float(mask.sum())
        tier_center = tier_weighted_sum / max(tier_weight_total, 1.0)
        for tier in tier_offsets:
            tier_offsets[tier] = tier_offsets[tier] - tier_center

        total_component = score_component + phase_component + np.array([tier_offsets.get(tier, 0.0) for tier in frame["tier"].astype(str)], dtype=float)
        working_residual = frame["working_residual"].to_numpy(dtype=float) - total_component

    abs_residual = np.abs(working_residual)
    if len(abs_residual) >= 4:
        # Keep the optional regional width adjustment neutral by default; the
        # global shrinkage below is what restores calibration without widening.
        sigma_knots = []
        sigma_values = []
        phase_sigma_offsets = {phase: 1.0 for phase in ["early", "mid", "late"]}
        tier_sigma_offsets = {tier: 1.0 for tier in TIER_LABELS}

    model = {
        "mode": "bias_correction",
        "score_knots": score_knots,
        "score_values": score_values,
        "sigma_knots": sigma_knots,
        "sigma_values": sigma_values,
        "phase_offsets": phase_offsets,
        "tier_offsets": tier_offsets,
        "phase_sigma_offsets": phase_sigma_offsets,
        "tier_sigma_offsets": tier_sigma_offsets,
        "shrinkage": float(max(score_shrinkage, phase_shrinkage, tier_shrinkage)),
        "score_shrinkage": float(score_shrinkage),
        "phase_shrinkage": float(phase_shrinkage),
        "tier_shrinkage": float(tier_shrinkage),
        "sigma_shrinkage": 0.84,
        "correction_scale": 1.0,
        "centered": True,
    }

    best_model = dict(model)
    best_score = float("inf")
    best_report: Dict[str, Any] | None = None
    correction_grid = np.linspace(0.0, 1.0, 11)
    sigma_grid = np.linspace(0.75, 1.0, 11)

    for correction_scale in correction_grid:
        for sigma_scale in sigma_grid:
            candidate_model = dict(model)
            candidate_model["correction_scale"] = float(correction_scale)
            candidate_model["sigma_shrinkage"] = float(sigma_scale)

            candidate_corrected = apply_bias_correction(frame, candidate_model)
            report = evaluate_bias_correction(frame, candidate_corrected)

            ece_delta = float(report.get("ece_delta", 0.0))
            coverage = float(report.get("coverage95", float("nan")))
            spearman_delta = float(report.get("spearman_delta", 0.0))
            mae_delta = float(report.get("mae_delta", 0.0))

            calibration_penalty = max(0.0, ece_delta) * 5.0
            coverage_penalty = abs(coverage - 0.95) * 2.0 if np.isfinite(coverage) else 5.0
            rank_penalty = max(0.0, -spearman_delta - 0.03) * 3.0
            mae_penalty = max(0.0, mae_delta) * 3.0
            complexity_penalty = abs(1.0 - correction_scale) * 0.01
            score = calibration_penalty + coverage_penalty + rank_penalty + mae_penalty + complexity_penalty

            if score < best_score:
                best_score = score
                best_model = candidate_model
                best_report = report

    if best_report is not None:
        best_model["tuning_report"] = {
            "ece_delta": float(best_report.get("ece_delta", 0.0)),
            "mae_delta": float(best_report.get("mae_delta", 0.0)),
            "spearman_delta": float(best_report.get("spearman_delta", 0.0)),
            "coverage95": float(best_report.get("coverage95", float("nan"))),
            "objective": float(best_score),
        }

    return best_model


def apply_bias_correction(predictions: pd.DataFrame, bias_model: Dict[str, Any] | None) -> pd.DataFrame:
    frame = predictions.loc[:, ~predictions.columns.duplicated()].copy()
    base_mean_col = "predicted_total_calibrated" if "predicted_total_calibrated" in frame.columns else "predicted_total"
    base_lower_col = "pred_total_lower_95_calibrated" if "pred_total_lower_95_calibrated" in frame.columns else "pred_total_lower_95"
    base_upper_col = "pred_total_upper_95_calibrated" if "pred_total_upper_95_calibrated" in frame.columns else "pred_total_upper_95"
    if bias_model is None or bias_model.get("mode") != "bias_correction":
        frame["bias_correction"] = 0.0
        frame["predicted_total_mean"] = frame[base_mean_col]
        frame["predicted_total_bias_corrected"] = frame[base_mean_col]
        frame["pred_total_lower_95"] = frame[base_lower_col]
        frame["pred_total_upper_95"] = frame[base_upper_col]
        frame["pred_total_lower_95_calibrated"] = frame[base_lower_col]
        frame["pred_total_upper_95_calibrated"] = frame[base_upper_col]
        frame["pred_total_lower_95_bias_corrected"] = frame[base_lower_col]
        frame["pred_total_upper_95_bias_corrected"] = frame[base_upper_col]
        return frame

    frame["phase"] = _season_phase(frame)
    frame["tier"] = _assign_corps_tiers(frame)

    raw = frame[base_mean_col].to_numpy(dtype=float)
    score_knots = np.asarray(bias_model.get("score_knots", []), dtype=float)
    score_values = np.asarray(bias_model.get("score_values", []), dtype=float)
    if len(score_knots) >= 2 and len(score_values) == len(score_knots):
        score_component = np.interp(raw, score_knots, score_values)
    else:
        score_component = np.zeros(len(frame), dtype=float)

    phase_offsets = bias_model.get("phase_offsets", {})
    tier_offsets = bias_model.get("tier_offsets", {})
    phase_component = np.array([phase_offsets.get(phase, 0.0) for phase in frame["phase"].astype(str)], dtype=float)
    tier_component = np.array([tier_offsets.get(tier, 0.0) for tier in frame["tier"].astype(str)], dtype=float)

    correction = score_component + phase_component + tier_component
    correction = correction - float(np.average(correction))
    correction *= float(bias_model.get("correction_scale", 1.0))
    corrected = raw + correction

    sigma_adjustment = np.ones(len(frame), dtype=float)
    sigma_adjustment *= float(bias_model.get("sigma_shrinkage", 1.0))
    sigma_knots = np.asarray(bias_model.get("sigma_knots", []), dtype=float)
    sigma_values = np.asarray(bias_model.get("sigma_values", []), dtype=float)
    if len(sigma_knots) >= 2 and len(sigma_values) == len(sigma_knots):
        sigma_adjustment *= np.interp(raw, sigma_knots, sigma_values)

    phase_sigma_offsets = bias_model.get("phase_sigma_offsets", {})
    tier_sigma_offsets = bias_model.get("tier_sigma_offsets", {})
    sigma_adjustment *= np.array([phase_sigma_offsets.get(phase, 1.0) for phase in frame["phase"].astype(str)], dtype=float)
    sigma_adjustment *= np.array([tier_sigma_offsets.get(tier, 1.0) for tier in frame["tier"].astype(str)], dtype=float)
    sigma_adjustment = np.clip(sigma_adjustment, 0.75, 1.15)

    base_sigma = (frame[base_upper_col].to_numpy(dtype=float) - frame[base_lower_col].to_numpy(dtype=float)) / (2.0 * 1.959963984540054)
    base_sigma = np.clip(base_sigma, 1e-6, None)
    adjusted_sigma = base_sigma * sigma_adjustment

    frame["bias_correction"] = correction
    frame["predicted_total_mean"] = corrected
    frame["predicted_total_calibrated"] = corrected
    frame["predicted_total_bias_corrected"] = corrected
    frame["sample_std_calibrated"] = adjusted_sigma
    frame["bias_sigma"] = adjusted_sigma

    projected = []
    for _, grp in frame.groupby("show_key", sort=False):
        grp = grp.copy()
        order = np.argsort(grp[base_mean_col].to_numpy(dtype=float))
        monotone = _isotonic_non_decreasing(
            grp[base_mean_col].to_numpy(dtype=float)[order],
            grp["predicted_total_bias_corrected"].to_numpy(dtype=float)[order],
        )
        grp.loc[grp.index[order], "predicted_total_bias_corrected"] = monotone
        grp.loc[grp.index[order], "predicted_total_mean"] = monotone
        grp.loc[grp.index[order], "predicted_total_calibrated"] = monotone
        projected.append(grp)

    frame = pd.concat(projected, ignore_index=True).sort_values(["show_date", "show_key", "corps"]).reset_index(drop=True)
    frame["bias_correction"] = frame["predicted_total_bias_corrected"] - frame[base_mean_col]
    frame["pred_total_lower_95"] = frame["predicted_total_mean"] - 1.959963984540054 * frame["bias_sigma"]
    frame["pred_total_upper_95"] = frame["predicted_total_mean"] + 1.959963984540054 * frame["bias_sigma"]
    frame["pred_total_lower_95_calibrated"] = frame["pred_total_lower_95"]
    frame["pred_total_upper_95_calibrated"] = frame["pred_total_upper_95"]
    frame["pred_total_lower_95_bias_corrected"] = frame["pred_total_lower_95"]
    frame["pred_total_upper_95_bias_corrected"] = frame["pred_total_upper_95"]
    frame["bias_width_scale"] = np.clip(
        (frame["pred_total_upper_95"] - frame["pred_total_lower_95"]).to_numpy(dtype=float)
        / np.maximum((frame[base_upper_col] - frame[base_lower_col]).to_numpy(dtype=float), 1e-6),
        0.9,
        1.15,
    )
    return frame


def evaluate_bias_correction(predictions: pd.DataFrame, corrected: pd.DataFrame) -> Dict[str, Any]:
    base = predictions.loc[:, ~predictions.columns.duplicated()].copy()
    corrected_frame = corrected.loc[:, ~corrected.columns.duplicated()].copy()
    base_mean_col = "predicted_total_calibrated" if "predicted_total_calibrated" in base.columns else "predicted_total"
    corrected_mean_col = (
        "predicted_total_mean"
        if "predicted_total_mean" in corrected_frame.columns
        else ("predicted_total_bias_corrected" if "predicted_total_bias_corrected" in corrected_frame.columns else base_mean_col)
    )
    base["base_residual"] = base["observed_total"] - base[base_mean_col]
    corrected_frame["corrected_residual"] = corrected_frame["observed_total"] - corrected_frame[corrected_mean_col]

    base_lower_col = "pred_total_lower_95_calibrated" if "pred_total_lower_95_calibrated" in base.columns else "pred_total_lower_95"
    base_upper_col = "pred_total_upper_95_calibrated" if "pred_total_upper_95_calibrated" in base.columns else "pred_total_upper_95"
    corrected_lower_col = (
        "pred_total_lower_95_bias_corrected"
        if "pred_total_lower_95_bias_corrected" in corrected_frame.columns
        else ("pred_total_lower_95_calibrated" if "pred_total_lower_95_calibrated" in corrected_frame.columns else "pred_total_lower_95")
    )
    corrected_upper_col = (
        "pred_total_upper_95_bias_corrected"
        if "pred_total_upper_95_bias_corrected" in corrected_frame.columns
        else ("pred_total_upper_95_calibrated" if "pred_total_upper_95_calibrated" in corrected_frame.columns else "pred_total_upper_95")
    )
    base_ece = _distribution_ece(base, base_mean_col, base_lower_col, base_upper_col, obs_col="observed_total", bins=10)
    corrected_ece = _distribution_ece(corrected_frame, corrected_mean_col, corrected_lower_col, corrected_upper_col, obs_col="observed_total", bins=10)

    base_mae = float(np.mean(np.abs(base["base_residual"])))
    corrected_mae = float(np.mean(np.abs(corrected_frame["corrected_residual"])))

    base_spearman: List[float] = []
    corrected_spearman: List[float] = []
    for show_key, base_show in base.groupby("show_key"):
        corr_show = corrected_frame.loc[corrected_frame["show_key"] == show_key]
        if len(base_show) < 2 or len(corr_show) < 2:
            continue
        if base_show[base_mean_col].nunique(dropna=True) < 2 or base_show["observed_total"].nunique(dropna=True) < 2:
            continue
        if corr_show[corrected_mean_col].nunique(dropna=True) < 2 or corr_show["observed_total"].nunique(dropna=True) < 2:
            continue
        base_rho = spearmanr(base_show[base_mean_col], base_show["observed_total"]).correlation
        corr_rho = spearmanr(corr_show[corrected_mean_col], corr_show["observed_total"]).correlation
        if np.isfinite(base_rho):
            base_spearman.append(float(base_rho))
        if np.isfinite(corr_rho):
            corrected_spearman.append(float(corr_rho))

    base_rank = float(np.mean(base_spearman)) if base_spearman else float("nan")
    corrected_rank = float(np.mean(corrected_spearman)) if corrected_spearman else float("nan")

    coverage_col_lower = corrected_lower_col
    coverage_col_upper = corrected_upper_col
    coverage = float(
        np.mean(
            (corrected_frame["observed_total"] >= corrected_frame[coverage_col_lower])
            & (corrected_frame["observed_total"] <= corrected_frame[coverage_col_upper])
        )
    )

    base_variance = float(np.var(base["base_residual"], ddof=0))
    corrected_variance = float(np.var(corrected_frame["corrected_residual"], ddof=0))
    base_width = float(np.mean(base[base_upper_col] - base[base_lower_col]))
    corrected_width = float(np.mean(corrected_frame[coverage_col_upper] - corrected_frame[coverage_col_lower]))

    warnings: List[str] = []
    centered = (corrected_frame[corrected_lower_col].to_numpy(dtype=float) + corrected_frame[corrected_upper_col].to_numpy(dtype=float)) / 2.0
    if not np.allclose(corrected_frame[corrected_mean_col].to_numpy(dtype=float), centered, atol=1e-6, rtol=0.0):
        warnings.append("distribution mismatch")
    if np.isfinite(base_ece) and np.isfinite(corrected_ece) and corrected_ece > base_ece:
        warnings.append("ECE regression")
    if corrected_width > base_width * 1.05:
        warnings.append("over-widening")
    if corrected_mae < base_mae and np.isfinite(base_rank) and np.isfinite(corrected_rank) and corrected_rank < base_rank:
        warnings.append("bias overfit")
    if np.isfinite(base_rank) and np.isfinite(corrected_rank) and (base_rank - corrected_rank) > 0.05:
        warnings.append("instability introduced")
    if corrected_variance > base_variance:
        warnings.append("overcorrection")

    return {
        "base_ece": base_ece,
        "corrected_ece": corrected_ece,
        "ece_delta": float(corrected_ece - base_ece),
        "base_mae": base_mae,
        "corrected_mae": corrected_mae,
        "mae_delta": float(corrected_mae - base_mae),
        "base_spearman_per_show": base_rank,
        "corrected_spearman_per_show": corrected_rank,
        "spearman_delta": float(corrected_rank - base_rank),
        "coverage95": coverage,
        "base_residual_variance": base_variance,
        "corrected_residual_variance": corrected_variance,
        "base_interval_width": base_width,
        "corrected_interval_width": corrected_width,
        "warnings": sorted(set(warnings)),
    }
