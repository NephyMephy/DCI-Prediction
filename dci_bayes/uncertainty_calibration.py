from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.stats import norm
from scipy.stats import spearmanr


Z_95 = 1.959963984540054


def _season_phase(frame: pd.DataFrame) -> pd.Series:
    base = frame.sort_values("show_date").copy()
    if "season_week" in base.columns:
        q1, q2 = base["season_week"].quantile([0.3, 0.7]).tolist()
        phase = np.where(base["season_week"] <= q1, "early", np.where(base["season_week"] <= q2, "mid", "late"))
        return pd.Series(phase, index=base.index)

    unique_shows = base[["show_key", "show_date"]].drop_duplicates().sort_values("show_date").reset_index(drop=True)
    if unique_shows.empty:
        return pd.Series(["mid"] * len(base), index=base.index)

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

    return base["show_key"].astype(str).map(classify)


def _ece(frame: pd.DataFrame, pred_col: str, obs_col: str, bins: int = 10) -> float:
    valid = frame.dropna(subset=[pred_col, obs_col]).sort_values(pred_col).reset_index(drop=True)
    if valid.empty:
        return float("nan")

    valid["bin"] = pd.qcut(valid.index, q=min(bins, max(2, len(valid))), duplicates="drop")
    rows = []
    for _, grp in valid.groupby("bin"):
        rows.append(
            {
                "abs_gap": abs(float(grp[pred_col].mean()) - float(grp[obs_col].mean())),
                "count": int(len(grp)),
            }
        )

    payload = pd.DataFrame(rows)
    return float((payload["abs_gap"] * payload["count"]).sum() / max(1, payload["count"].sum()))


def _isotonic_fit(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]

    levels = y_sorted.astype(float).copy()
    weights = np.ones_like(levels)

    i = 0
    while i < len(levels) - 1:
        if levels[i] > levels[i + 1]:
            total_w = weights[i] + weights[i + 1]
            avg = (levels[i] * weights[i] + levels[i + 1] * weights[i + 1]) / total_w
            levels[i] = avg
            levels[i + 1] = avg
            weights[i] = total_w
            weights[i + 1] = total_w
            j = i
            while j > 0 and levels[j - 1] > levels[j]:
                total_w = weights[j - 1] + weights[j]
                avg = (levels[j - 1] * weights[j - 1] + levels[j] * weights[j]) / total_w
                levels[j - 1] = avg
                levels[j] = avg
                weights[j - 1] = total_w
                weights[j] = total_w
                j -= 1
        i += 1

    return x_sorted, levels


def fit_uncertainty_model(
    predictions: pd.DataFrame,
    target_coverage: float = 0.95,
    sigma_floor: float = 0.8,
) -> Dict[str, Any]:
    frame = predictions.copy()
    mean_col = "predicted_total_calibrated" if "predicted_total_calibrated" in frame.columns else "predicted_total"
    frame = frame.dropna(subset=["observed_total", mean_col, "sample_std"]).copy()
    if frame.empty:
        return {
            "mode": "uncertainty_scale",
            "target_coverage": target_coverage,
            "global_scale": 1.0,
            "phase_scale": {"early": 1.0, "mid": 1.0, "late": 1.0},
            "score_curve": [1.0, 0.0, 0.0],
            "sigma_floor": sigma_floor,
        }

    z = float(norm.ppf((1 + target_coverage) / 2))
    if not np.isfinite(z) or z <= 0:
        z = Z_95

    frame["phase"] = _season_phase(frame)
    residual = (frame["observed_total"] - frame[mean_col]).abs()
    base_sigma = frame["sample_std"].clip(lower=1e-3)

    target = float(np.quantile(residual, target_coverage))
    denom = float(z * np.median(base_sigma)) if np.median(base_sigma) > 0 else 1.0
    global_scale = float(max(target / max(denom, 1e-6), 1.0))

    phase_scale: Dict[str, float] = {}
    for phase in ["early", "mid", "late"]:
        grp = frame.loc[frame["phase"] == phase]
        if len(grp) < 8:
            phase_scale[phase] = 1.0
            continue
        p_res = (grp["observed_total"] - grp[mean_col]).abs()
        p_sig = grp["sample_std"].clip(lower=1e-3)
        num = float(np.quantile(p_res, target_coverage))
        den = float(z * np.median(p_sig))
        phase_scale[phase] = float(np.clip(num / max(den, 1e-6), 0.8, 4.0))

    frame["pred_norm"] = (frame[mean_col] - frame[mean_col].mean()) / (frame[mean_col].std(ddof=0) or 1.0)
    bin_table = frame.copy()
    bin_table["bin"] = pd.qcut(bin_table["pred_norm"], q=min(8, max(2, len(bin_table))), duplicates="drop")
    curve_pts = []
    for _, grp in bin_table.groupby("bin"):
        g_res = (grp["observed_total"] - grp[mean_col]).abs()
        g_sig = grp["sample_std"].clip(lower=1e-3)
        num = float(np.quantile(g_res, target_coverage))
        den = float(z * np.median(g_sig))
        curve_pts.append((float(grp["pred_norm"].mean()), float(np.clip(num / max(den, 1e-6), 0.6, 5.0))))

    if len(curve_pts) >= 3:
        x = np.array([p[0] for p in curve_pts])
        y = np.array([p[1] for p in curve_pts])
        coeffs = np.polyfit(x, y, deg=2)
    else:
        coeffs = np.array([0.0, 0.0, 1.0])

    return {
        "mode": "uncertainty_scale",
        "target_coverage": target_coverage,
        "global_scale": global_scale,
        "phase_scale": phase_scale,
        "score_curve": [float(coeffs[0]), float(coeffs[1]), float(coeffs[2])],
        "sigma_floor": float(sigma_floor),
    }


def fit_nonlinear_calibration(predictions: pd.DataFrame, method: str = "spline") -> Dict[str, Any]:
    frame = predictions.dropna(subset=["predicted_total", "observed_total"]).copy()
    if frame.empty:
        return {"mode": "identity"}

    x = frame["predicted_total"].to_numpy(dtype=float)
    y = frame["observed_total"].to_numpy(dtype=float)

    if method == "isotonic":
        x_iso, y_iso = _isotonic_fit(x, y)
        return {
            "mode": "isotonic",
            "x": x_iso.tolist(),
            "y": y_iso.tolist(),
        }

    bins = min(20, max(6, len(frame) // 15))
    binned = frame.sort_values("predicted_total").reset_index(drop=True)
    binned["bin"] = pd.qcut(binned.index, q=min(bins, max(2, len(binned))), duplicates="drop")

    knot_x: List[float] = []
    knot_y: List[float] = []
    for _, grp in binned.groupby("bin"):
        knot_x.append(float(grp["predicted_total"].mean()))
        knot_y.append(float(grp["observed_total"].mean()))

    if len(knot_x) < 4:
        alpha, beta = np.polyfit(x, y, 1)
        return {"mode": "linear", "alpha": float(alpha), "beta": float(beta)}

    order = np.argsort(knot_x)
    kx = np.array(knot_x)[order]
    ky = np.array(knot_y)[order]

    ky_monotone = np.maximum.accumulate(ky)
    if not np.all(np.diff(kx) > 1e-6):
        x_iso, y_iso = _isotonic_fit(x, y)
        return {
            "mode": "isotonic",
            "x": x_iso.tolist(),
            "y": y_iso.tolist(),
        }

    return {
        "mode": "spline",
        "x": kx.tolist(),
        "y": ky_monotone.tolist(),
    }


def apply_calibration(
    predictions: pd.DataFrame,
    calibration_model: Dict[str, Any] | None,
    uncertainty_model: Dict[str, Any] | None,
) -> pd.DataFrame:
    frame = predictions.copy()
    if "season_week" not in frame.columns:
        frame["season_week"] = 1

    raw_mean = frame["predicted_total"].to_numpy(dtype=float)

    if calibration_model is None or calibration_model.get("mode") == "identity":
        mean_cal = raw_mean
    else:
        mode = calibration_model.get("mode")
        if mode == "linear":
            mean_cal = calibration_model["alpha"] * raw_mean + calibration_model["beta"]
        elif mode == "spline":
            spline = PchipInterpolator(calibration_model["x"], calibration_model["y"], extrapolate=True)
            mean_cal = spline(raw_mean)
        elif mode == "isotonic":
            mean_cal = np.interp(raw_mean, np.array(calibration_model["x"]), np.array(calibration_model["y"]))
        else:
            mean_cal = raw_mean

    frame["predicted_total_calibrated"] = mean_cal

    base_sigma = frame["sample_std"].to_numpy(dtype=float)
    if uncertainty_model is None:
        sigma_cal = np.clip(base_sigma, 0.8, None)
    else:
        phase = _season_phase(frame).reindex(frame.index).fillna("mid")
        phase_scale = np.array([uncertainty_model.get("phase_scale", {}).get(p, 1.0) for p in phase], dtype=float)
        global_scale = float(uncertainty_model.get("global_scale", 1.0))

        norm = (mean_cal - np.mean(mean_cal)) / (np.std(mean_cal) or 1.0)
        c2, c1, c0 = uncertainty_model.get("score_curve", [0.0, 0.0, 1.0])
        score_scale = np.clip(c2 * norm * norm + c1 * norm + c0, 0.6, 5.0)

        sigma_floor = float(uncertainty_model.get("sigma_floor", 0.8))
        sigma_cal = np.maximum(base_sigma * global_scale * phase_scale * score_scale, sigma_floor)

    frame["sample_std_calibrated"] = sigma_cal
    frame["pred_total_lower_95_calibrated"] = frame["predicted_total_calibrated"] - Z_95 * sigma_cal
    frame["pred_total_upper_95_calibrated"] = frame["predicted_total_calibrated"] + Z_95 * sigma_cal
    return frame


def evaluate_calibrated_model(predictions: pd.DataFrame) -> Dict[str, Any]:
    frame = predictions.copy()

    coverage = float(
        np.mean(
            (frame["observed_total"] >= frame["pred_total_lower_95_calibrated"])
            & (frame["observed_total"] <= frame["pred_total_upper_95_calibrated"])
        )
    )

    per_show_spearman: List[float] = []
    finals_cutoff_hits: List[float] = []
    for _, show in frame.groupby("show_key"):
        if len(show) < 2:
            continue
        rho = spearmanr(show["predicted_total_calibrated"], show["observed_total"]).correlation
        if np.isfinite(rho):
            per_show_spearman.append(float(rho))

        obs_window = set(show.sort_values("observed_total", ascending=False).head(14).tail(5)["corps"].tolist())
        pred_window = set(show.sort_values("predicted_total_calibrated", ascending=False).head(14).tail(5)["corps"].tolist())
        if obs_window:
            finals_cutoff_hits.append(len(obs_window & pred_window) / len(obs_window))

    mean_spearman = float(np.mean(per_show_spearman)) if per_show_spearman else float("nan")
    finals_cutoff_accuracy = float(np.mean(finals_cutoff_hits)) if finals_cutoff_hits else float("nan")

    last_show_key = frame.sort_values("show_date")["show_key"].iloc[-1]
    final_show = frame.loc[frame["show_key"] == last_show_key].copy()
    pred_top12 = set(final_show.sort_values("predicted_total_calibrated", ascending=False).head(12)["corps"].tolist())
    obs_top12 = set(final_show.sort_values("observed_total", ascending=False).head(12)["corps"].tolist())
    top12_accuracy = float(len(pred_top12 & obs_top12) / max(1, len(obs_top12)))

    mae = float(np.mean(np.abs(frame["predicted_total_calibrated"] - frame["observed_total"])))
    ece = _ece(frame, pred_col="predicted_total_calibrated", obs_col="observed_total", bins=10)

    warnings: List[str] = []
    if coverage < 0.85:
        warnings.append("overconfident posterior")
    if coverage > 0.98:
        warnings.append("over-dispersed posterior")

    by_bin = frame.copy().sort_values("predicted_total_calibrated").reset_index(drop=True)
    by_bin["bin"] = pd.qcut(by_bin.index, q=min(8, max(2, len(by_bin))), duplicates="drop")
    rows = []
    for _, grp in by_bin.groupby("bin"):
        rows.append(
            {
                "pred": float(grp["predicted_total_calibrated"].mean()),
                "res": float((grp["observed_total"] - grp["predicted_total_calibrated"]).mean()),
            }
        )
    bin_df = pd.DataFrame(rows)
    if len(bin_df) >= 4:
        x = bin_df["pred"].to_numpy()
        y = bin_df["res"].to_numpy()
        x_norm = (x - np.mean(x)) / (np.std(x) or 1.0)
        coeff = np.polyfit(x_norm, y, 2)
        if abs(float(coeff[0])) > 0.08:
            warnings.append("nonlinear bias present")

    phase = _season_phase(frame)
    early_w = (frame.loc[phase == "early", "pred_total_upper_95_calibrated"] - frame.loc[phase == "early", "pred_total_lower_95_calibrated"]).mean()
    late_w = (frame.loc[phase == "late", "pred_total_upper_95_calibrated"] - frame.loc[phase == "late", "pred_total_lower_95_calibrated"]).mean()
    if np.isfinite(early_w) and np.isfinite(late_w) and early_w > 0 and late_w / early_w < 0.35:
        warnings.append("late-season collapse")

    return {
        "coverage95": coverage,
        "ece": ece,
        "mean_spearman_per_show": mean_spearman,
        "top12_accuracy": top12_accuracy,
        "finals_cutoff_accuracy": finals_cutoff_accuracy,
        "mae_total": mae,
        "warnings": sorted(set(warnings)),
    }
