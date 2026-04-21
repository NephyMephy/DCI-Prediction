from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .calibration import HoldoutResult, run_multi_year_validation, run_year_holdout_experiment


@dataclass
class DiagnosticsResult:
    summary: Dict[str, Any]
    summary_path: str
    calibration_plot_path: str
    latent_plot_path: str
    tier_plot_path: str


def _ensure_plot_dir(output_dir: str | Path) -> Path:
    out = Path(output_dir).resolve()
    plot_dir = out / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    return plot_dir


def _safe_stats(error: pd.Series) -> Dict[str, float]:
    if len(error) == 0:
        return {"mae": float("nan"), "bias": float("nan"), "variance": float("nan")}
    return {
        "mae": float(np.mean(np.abs(error))),
        "bias": float(np.mean(error)),
        "variance": float(np.var(error, ddof=0)),
    }


def _season_phase(frame: pd.DataFrame) -> pd.Series:
    frame = frame.sort_values("show_date").copy()
    unique_shows = frame[["show_key", "show_date"]].drop_duplicates().sort_values("show_date").reset_index(drop=True)
    if unique_shows.empty:
        return pd.Series(["mid"] * len(frame), index=frame.index)

    show_count = len(unique_shows)
    early_cut = max(int(np.ceil(show_count * 0.30)), 1)
    late_cut = max(int(np.floor(show_count * 0.70)), 1)
    show_to_rank = {row.show_key: i for i, row in unique_shows.iterrows()}

    def classify(show_key: str) -> str:
        rank = show_to_rank.get(show_key, 0)
        if rank < early_cut:
            return "early"
        if rank >= late_cut:
            return "late"
        return "mid"

    return frame["show_key"].astype(str).map(classify)


def _assign_corps_tiers(frame: pd.DataFrame) -> pd.Series:
    base = frame.copy()
    final_show_key = base.sort_values("show_date")["show_key"].iloc[-1]
    final_show = base.loc[base["show_key"] == final_show_key].sort_values("observed_total", ascending=False).reset_index(drop=True)
    final_show["final_rank"] = np.arange(1, len(final_show) + 1)

    rank_map = dict(zip(final_show["corps"].astype(str), final_show["final_rank"]))

    def tier(corps: str) -> str:
        rank = rank_map.get(str(corps), 999)
        if rank <= 5:
            return "top5"
        if rank <= 12:
            return "finals_6_12"
        if rank <= 25:
            return "bubble_13_25"
        return "field"

    return base["corps"].astype(str).map(tier)


def temporal_error_breakdown(predictions: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    frame = predictions.copy()
    frame["phase"] = _season_phase(frame)
    frame["error"] = frame["predicted_total_calibrated"] - frame["observed_total"]

    payload: Dict[str, Dict[str, float]] = {}
    for phase, grp in frame.groupby("phase"):
        payload[str(phase)] = _safe_stats(grp["error"])  # bias > 0 overpredict, < 0 underpredict
    return payload


def structural_error_breakdown(predictions: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    frame = predictions.copy()
    frame["tier"] = _assign_corps_tiers(frame)
    frame["error"] = frame["predicted_total_calibrated"] - frame["observed_total"]

    payload: Dict[str, Dict[str, float]] = {}
    for tier, grp in frame.groupby("tier"):
        payload[str(tier)] = _safe_stats(grp["error"])  # bias > 0 overpredict, < 0 underpredict
    return payload


def caption_error_breakdown(predictions: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    frame = predictions.copy()
    payload: Dict[str, Dict[str, float]] = {}
    caption_map = [
        ("ge", "pred_ge", "observed_ge"),
        ("visual", "pred_visual", "observed_visual"),
        ("music", "pred_music", "observed_music"),
    ]
    for name, pred_col, obs_col in caption_map:
        if pred_col not in frame.columns or obs_col not in frame.columns:
            payload[name] = {"mae": float("nan"), "bias": float("nan"), "variance": float("nan")}
            continue
        error = frame[pred_col] - frame[obs_col]
        payload[name] = _safe_stats(error.dropna())
    return payload


def calibration_curve_analysis(predictions: pd.DataFrame, bins: int = 10) -> Dict[str, Any]:
    frame = predictions.dropna(subset=["predicted_total_calibrated", "observed_total"]).copy()
    frame = frame.sort_values("predicted_total_calibrated").reset_index(drop=True)
    frame["bin"] = pd.qcut(frame.index, q=min(bins, max(2, len(frame))), duplicates="drop")

    rows: List[Dict[str, float]] = []
    for _, grp in frame.groupby("bin"):
        pred_mean = float(grp["predicted_total_calibrated"].mean())
        obs_mean = float(grp["observed_total"].mean())
        rows.append(
            {
                "pred_mean": pred_mean,
                "obs_mean": obs_mean,
                "abs_gap": float(abs(pred_mean - obs_mean)),
                "count": int(len(grp)),
            }
        )

    bins_df = pd.DataFrame(rows)
    if bins_df.empty:
        ece = float("nan")
    else:
        ece = float((bins_df["abs_gap"] * bins_df["count"]).sum() / max(1, bins_df["count"].sum()))

    frame["magnitude"] = frame["predicted_total_calibrated"]
    frame["error"] = frame["predicted_total_calibrated"] - frame["observed_total"]
    mag_rho = spearmanr(frame["magnitude"], frame["error"]).correlation

    return {
        "ece": ece,
        "bias_vs_score_magnitude_spearman": float(mag_rho) if np.isfinite(mag_rho) else float("nan"),
        "bins": rows,
    }


def uncertainty_audit(predictions: pd.DataFrame) -> Dict[str, Any]:
    frame = predictions.copy()
    covered = (frame["observed_total"] >= frame["pred_total_lower_95"]) & (frame["observed_total"] <= frame["pred_total_upper_95"])
    coverage = float(np.mean(covered))

    frame["phase"] = _season_phase(frame)
    phase_coverage = {
        str(phase): float(np.mean((grp["observed_total"] >= grp["pred_total_lower_95"]) & (grp["observed_total"] <= grp["pred_total_upper_95"])))
        for phase, grp in frame.groupby("phase")
    }

    width = frame["pred_total_upper_95"] - frame["pred_total_lower_95"]
    abs_error = np.abs(frame["predicted_total_calibrated"] - frame["observed_total"])
    narrow_threshold = float(np.nanquantile(width, 0.25))
    high_error_threshold = float(np.nanquantile(abs_error, 0.75))
    overconfidence = (width <= narrow_threshold) & (abs_error >= high_error_threshold)

    return {
        "coverage95": coverage,
        "phase_coverage": phase_coverage,
        "early_uncertainty_collapse": bool(phase_coverage.get("early", coverage) < 0.90),
        "overconfidence_rate": float(np.mean(overconfidence)),
    }


def latent_drift_diagnostics(predictions: pd.DataFrame) -> Dict[str, Any]:
    frame = predictions.copy().sort_values(["show_date", "corps"])
    frame["latent_proxy"] = frame["predicted_total_calibrated"]

    corps_payload: Dict[str, Dict[str, Any]] = {}
    warnings: List[str] = []

    rank_std_by_show = frame.groupby("show_key").apply(
        lambda g: g.sort_values("latent_proxy", ascending=False).reset_index(drop=True).assign(rank=np.arange(1, len(g) + 1))["rank"].std(ddof=0)
    )
    rank_instability = float(rank_std_by_show.std(ddof=0)) if len(rank_std_by_show) > 1 else 0.0

    for corps, grp in frame.groupby("corps"):
        grp = grp.sort_values("show_date")
        if len(grp) < 2:
            continue
        diff = grp["latent_proxy"].diff().dropna()
        volatility = float(diff.std(ddof=0)) if len(diff) else 0.0
        slope = float((grp["latent_proxy"].iloc[-1] - grp["latent_proxy"].iloc[0]) / max(len(grp) - 1, 1))
        corps_payload[str(corps)] = {
            "volatility": volatility,
            "trend_per_show": slope,
            "shows": int(len(grp)),
        }

    all_vol = np.array([entry["volatility"] for entry in corps_payload.values()]) if corps_payload else np.array([0.0])
    mean_vol = float(np.mean(all_vol))

    if mean_vol < 0.15:
        warnings.append("drift too slow")
    if mean_vol > 1.75:
        warnings.append("drift too jagged")
    if rank_instability > 1.2:
        warnings.append("rank instability detected")

    early = frame.loc[_season_phase(frame) == "early"]
    late = frame.loc[_season_phase(frame) == "late"]
    if not early.empty and not late.empty:
        if float(early["predicted_total_calibrated"].std(ddof=0)) < 0.65 * float(late["predicted_total_calibrated"].std(ddof=0)):
            warnings.append("early convergence bias")

    return {
        "per_corps": corps_payload,
        "rank_instability": rank_instability,
        "warnings": sorted(set(warnings)),
    }


def _plot_calibration_curve(curve: Dict[str, Any], plot_path: Path) -> None:
    bins = curve.get("bins", [])
    pred = [entry["pred_mean"] for entry in bins]
    obs = [entry["obs_mean"] for entry in bins]

    plt.figure(figsize=(8, 6))
    if pred and obs:
        plt.plot(pred, obs, marker="o", label="Model")
        low = min(min(pred), min(obs))
        high = max(max(pred), max(obs))
        plt.plot([low, high], [low, high], linestyle="--", color="gray", label="Ideal")
    plt.xlabel("Mean Predicted Score")
    plt.ylabel("Mean Actual Score")
    plt.title("Calibration Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=160)
    plt.close()


def _plot_error_by_tier(structural: Dict[str, Dict[str, float]], plot_path: Path) -> None:
    tiers = list(structural.keys())
    maes = [structural[tier].get("mae", np.nan) for tier in tiers]

    plt.figure(figsize=(8, 6))
    plt.bar(tiers, maes)
    plt.ylabel("MAE")
    plt.title("Error by Corps Tier")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=160)
    plt.close()


def _plot_latent_trajectories(predictions: pd.DataFrame, plot_path: Path, top_n: int = 12) -> None:
    frame = predictions.copy().sort_values(["show_date", "corps"])
    latest = frame.groupby("corps")["predicted_total_calibrated"].mean().sort_values(ascending=False).head(top_n)
    selected = set(latest.index.tolist())

    plt.figure(figsize=(11, 7))
    for corps, grp in frame.groupby("corps"):
        if corps not in selected:
            continue
        grp = grp.sort_values("show_date")
        plt.plot(grp["show_date"], grp["predicted_total_calibrated"], alpha=0.75, linewidth=1.5, label=str(corps))

    plt.title("Latent Trajectories (Proxy from Calibrated Predictions)")
    plt.xlabel("Show Date")
    plt.ylabel("Calibrated Predicted Score")
    plt.legend(loc="best", fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=160)
    plt.close()


def run_diagnostics(
    holdout_year: int = 2025,
    data_dir: str | Path = "data",
    class_pattern: str = "World Class",
    draws: int = 250,
    tune: int = 250,
    chains: int = 2,
    cores: int = 1,
    n_samples: int = 400,
    max_iterations: int = 3,
    output_dir: str | Path = "outputs/diagnostics",
    plot_calibration: bool = True,
    latent_analysis: bool = True,
    compare_years: bool = False,
) -> DiagnosticsResult:
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = _ensure_plot_dir(out_dir)

    holdout: HoldoutResult = run_year_holdout_experiment(
        data_dir=data_dir,
        year=holdout_year,
        class_pattern=class_pattern,
        draws=draws,
        tune=tune,
        chains=chains,
        cores=cores,
        n_samples=n_samples,
        max_iterations=max_iterations,
        output_dir=out_dir / "holdout",
        scaling_mode="piecewise",
        include_frames=True,
        enable_noise_model_selection=True,
    )

    if holdout.predictions_frame is None:
        preds = pd.read_csv(holdout.predictions_path)
        if "show_date" in preds.columns:
            preds["show_date"] = pd.to_datetime(preds["show_date"], errors="coerce")
    else:
        preds = holdout.predictions_frame.copy()

    if "predicted_total_calibrated" not in preds.columns:
        preds["predicted_total_calibrated"] = preds["predicted_total"]

    temporal = temporal_error_breakdown(preds)
    structural = structural_error_breakdown(preds)
    captions = caption_error_breakdown(preds)
    calibration_curve = calibration_curve_analysis(preds, bins=10)
    uncertainty = uncertainty_audit(preds)
    latent = latent_drift_diagnostics(preds) if latent_analysis else {"warnings": [], "per_corps": {}}

    year_compare = None
    if compare_years:
        year_compare_result = run_multi_year_validation(
            years=(2023, 2024, 2025),
            data_dir=data_dir,
            class_pattern=class_pattern,
            draws=max(150, draws // 2),
            tune=max(150, tune // 2),
            chains=chains,
            cores=cores,
            n_samples=max(250, n_samples // 2),
            max_iterations=max(2, max_iterations - 1),
            output_dir=out_dir / "multi_year",
            scaling_mode="piecewise",
        )
        year_compare = {
            "aggregate_metrics": year_compare_result.aggregate_metrics,
            "overfit_gap": year_compare_result.overfit_gap,
            "output_path": year_compare_result.output_path,
        }

    calibration_plot_path = plot_dir / "calibration_curve.png"
    tier_plot_path = plot_dir / "error_by_tier.png"
    latent_plot_path = plot_dir / "latent_trajectories.png"

    if plot_calibration:
        _plot_calibration_curve(calibration_curve, calibration_plot_path)
    _plot_error_by_tier(structural, tier_plot_path)
    _plot_latent_trajectories(preds, latent_plot_path)

    summary = {
        "holdout_year": holdout_year,
        "best_iteration": holdout.best_iteration,
        "best_metrics": holdout.best_metrics,
        "scaling": holdout.scaling,
        "error_decomposition": {
            "temporal": temporal,
            "structural": structural,
            "caption": captions,
        },
        "calibration_curve": calibration_curve,
        "uncertainty_audit": uncertainty,
        "latent_drift": latent,
        "rank_primary": {
            "mean_spearman_per_show": holdout.best_metrics.get("mean_spearman_per_show"),
            "top12_accuracy": holdout.best_metrics.get("top12_accuracy"),
            "finals_cutoff_accuracy": holdout.best_metrics.get("finals_cutoff_accuracy"),
        },
        "secondary": {
            "mae_total": holdout.best_metrics.get("mae_total"),
            "calibration_error": holdout.best_metrics.get("calibration_error"),
        },
        "warnings": sorted(
            set(latent.get("warnings", []))
            | ({"uncertainty collapse early"} if uncertainty.get("early_uncertainty_collapse") else set())
            | ({"overconfidence detected"} if uncertainty.get("overconfidence_rate", 0.0) > 0.15 else set())
        ),
    }
    if year_compare is not None:
        summary["cross_year_validation"] = year_compare

    summary_path = out_dir / "diagnostics_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))

    return DiagnosticsResult(
        summary=summary,
        summary_path=str(summary_path),
        calibration_plot_path=str(calibration_plot_path),
        latent_plot_path=str(latent_plot_path),
        tier_plot_path=str(tier_plot_path),
    )
