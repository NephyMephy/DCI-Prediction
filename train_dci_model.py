from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    from dci_bayes.pipeline import fit_bundle
    from dci_bayes.predict import predict_next_show, simulate_season_progression
    from dci_bayes.calibration import run_multi_year_validation, run_year_holdout_experiment
    from dci_bayes.bias_correction import apply_bias_correction, evaluate_bias_correction, fit_bias_model
    from dci_bayes.diagnostics import run_diagnostics
    from dci_bayes.uncertainty_calibration import (
        apply_calibration,
        evaluate_calibrated_model,
        fit_nonlinear_calibration,
        fit_uncertainty_model,
    )

    parser = argparse.ArgumentParser(description="Train a hierarchical Bayesian DCI score model.")
    parser.add_argument("--data-dir", default="data", help="Path to the data directory.")
    parser.add_argument("--year-min", type=int, default=None, help="Optional minimum year to include.")
    parser.add_argument("--year-max", type=int, default=None, help="Optional maximum year to include.")
    parser.add_argument(
        "--class-pattern",
        default=None,
        help="Optional regex filter for class names, e.g. 'World Class'.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap on number of most recent rows for faster experiments.",
    )
    parser.add_argument("--draws", type=int, default=1000, help="Posterior draws.")
    parser.add_argument("--tune", type=int, default=1000, help="Warmup/tuning steps.")
    parser.add_argument("--chains", type=int, default=4, help="Number of MCMC chains.")
    parser.add_argument("--cores", type=int, default=None, help="Number of CPU cores to use.")
    parser.add_argument("--corps-id", default=None, help="Optional corps id to predict after fitting.")
    parser.add_argument("--simulate-horizon", type=int, default=0, help="Optional season simulation horizon.")
    parser.add_argument(
        "--run-holdout-calibration",
        action="store_true",
        help="Run year holdout calibration experiment instead of direct training.",
    )
    parser.add_argument("--holdout-year", type=int, default=2025, help="Holdout year for calibration mode.")
    parser.add_argument("--max-iterations", type=int, default=4, help="Calibration refinement iterations.")
    parser.add_argument(
        "--calibration-output-dir",
        default="outputs/calibration",
        help="Output directory for calibration artifacts.",
    )
    parser.add_argument(
        "--run-diagnostics",
        action="store_true",
        help="Run diagnostics module with calibration-aware analysis and plots.",
    )
    parser.add_argument(
        "--compare-years",
        action="store_true",
        help="Run 2023/2024/2025 cross-year validation in diagnostics mode.",
    )
    parser.add_argument(
        "--plot-calibration",
        action="store_true",
        help="Generate calibration curve plot in diagnostics mode.",
    )
    parser.add_argument(
        "--latent-analysis",
        action="store_true",
        help="Enable latent drift analysis and warnings in diagnostics mode.",
    )
    parser.add_argument(
        "--fix-uncertainty",
        action="store_true",
        help="Fit a separate uncertainty calibration model for posterior interval correction.",
    )
    parser.add_argument(
        "--fit-calibration-layer",
        action="store_true",
        help="Fit nonlinear holdout calibration layer for mean prediction correction.",
    )
    parser.add_argument(
        "--recalibrate-holdout",
        action="store_true",
        help="Run holdout experiment then apply uncertainty/nonlinear calibration and report metrics.",
    )
    parser.add_argument(
        "--fit-bias-layer",
        action="store_true",
        help="Fit a residual bias correction layer on calibrated holdout predictions.",
    )
    parser.add_argument(
        "--apply-bias-correction",
        action="store_true",
        help="Apply the fitted residual bias correction layer to mean predictions only.",
    )
    args = parser.parse_args()

    if args.recalibrate_holdout:
        holdout = run_year_holdout_experiment(
            data_dir=args.data_dir,
            year=args.holdout_year,
            train_year_min=args.year_min,
            train_year_max=args.year_max,
            class_pattern=args.class_pattern or "World Class",
            draws=args.draws,
            tune=args.tune,
            chains=args.chains,
            cores=args.cores or 1,
            max_iterations=args.max_iterations,
            output_dir=args.calibration_output_dir,
            scaling_mode="piecewise",
            include_frames=True,
        )

        if holdout.predictions_frame is None:
            raise RuntimeError("Holdout predictions frame is unavailable for recalibration.")

        predictions = holdout.predictions_frame.copy()
        if "show_date" in predictions.columns:
            predictions["show_date"] = predictions["show_date"].astype("datetime64[ns]")

        calibration_model = {"mode": "identity"}
        if args.fit_calibration_layer:
            calibration_model = fit_nonlinear_calibration(predictions, method="spline")

        uncertainty_model = None
        if args.fix_uncertainty:
            uncertainty_model = fit_uncertainty_model(predictions, target_coverage=0.95, sigma_floor=0.8)

        calibrated = apply_calibration(
            predictions=predictions,
            calibration_model=calibration_model,
            uncertainty_model=uncertainty_model,
        )
        report = evaluate_calibrated_model(calibrated)

        bias_model = None
        bias_corrected = calibrated
        bias_report = None
        if args.fit_bias_layer or args.apply_bias_correction:
            bias_model = fit_bias_model(calibrated)
            bias_corrected = apply_bias_correction(calibrated, bias_model)
            bias_report = evaluate_bias_correction(calibrated, bias_corrected)

        output_root = Path(args.calibration_output_dir).resolve() / f"holdout_{args.holdout_year}"
        output_root.mkdir(parents=True, exist_ok=True)
        calibrated_path = output_root / "recalibrated_predictions.csv"
        report_path = output_root / "recalibration_report.json"
        calibrated.to_csv(calibrated_path, index=False)
        bias_corrected_path = output_root / "bias_corrected_predictions.csv"
        bias_report_path = output_root / "bias_correction_report.json"
        if bias_model is not None:
            bias_corrected.to_csv(bias_corrected_path, index=False)
        report_payload = {
            "holdout_year": args.holdout_year,
            "base_metrics": holdout.best_metrics,
            "recalibrated_metrics": report,
            "calibration_model": calibration_model,
            "uncertainty_model": uncertainty_model,
        }
        if bias_model is not None:
            report_payload["bias_model"] = bias_model
            report_payload["bias_correction_metrics"] = bias_report
            bias_report_path.write_text(json.dumps(report_payload, indent=2, default=str))
        report_path.write_text(json.dumps(report_payload, indent=2, default=str))

        print("Holdout recalibration complete.")
        print(f"Recalibrated metrics: {report}")
        print(f"Recalibrated predictions: {calibrated_path}")
        print(f"Recalibration report: {report_path}")
        if bias_report is not None:
            print(f"Bias-corrected metrics: {bias_report}")
            print(f"Bias-corrected predictions: {bias_corrected_path}")
            print(f"Bias correction report: {bias_report_path}")
        return

    if args.run_diagnostics:
        result = run_diagnostics(
            holdout_year=args.holdout_year,
            data_dir=args.data_dir,
            class_pattern=args.class_pattern or "World Class",
            draws=args.draws,
            tune=args.tune,
            chains=args.chains,
            cores=args.cores or 1,
            n_samples=500,
            max_iterations=args.max_iterations,
            output_dir=args.calibration_output_dir,
            plot_calibration=args.plot_calibration,
            latent_analysis=args.latent_analysis,
            compare_years=args.compare_years,
        )
        print("Diagnostics complete.")
        print(f"Summary file: {result.summary_path}")
        print(f"Calibration curve: {result.calibration_plot_path}")
        print(f"Latent trajectories: {result.latent_plot_path}")
        print(f"Error by tier: {result.tier_plot_path}")
        return

    if args.compare_years:
        result = run_multi_year_validation(
            years=(2023, 2024, 2025),
            data_dir=args.data_dir,
            train_year_min=args.year_min,
            train_year_max=args.year_max,
            class_pattern=args.class_pattern or "World Class",
            draws=args.draws,
            tune=args.tune,
            chains=args.chains,
            cores=args.cores or 1,
            n_samples=500,
            max_iterations=args.max_iterations,
            output_dir=args.calibration_output_dir,
            scaling_mode="piecewise",
        )
        print("Cross-year validation complete.")
        print(f"Output file: {result.output_path}")
        print(f"Aggregate metrics: {result.aggregate_metrics}")
        print(f"Overfit gap: {result.overfit_gap}")
        return

    if args.run_holdout_calibration:
        result = run_year_holdout_experiment(
            data_dir=args.data_dir,
            year=args.holdout_year,
            train_year_min=args.year_min,
            train_year_max=args.year_max,
            class_pattern=args.class_pattern or "World Class",
            draws=args.draws,
            tune=args.tune,
            chains=args.chains,
            cores=args.cores or 1,
            max_iterations=args.max_iterations,
            output_dir=args.calibration_output_dir,
            scaling_mode="piecewise",
        )
        print("Calibration holdout complete.")
        print(f"Best iteration: {result.best_iteration}")
        print(f"Best metrics: {result.best_metrics}")
        print(f"Scaling: {result.scaling}")
        print(f"Predictions file: {result.predictions_path}")
        print(f"Metrics file: {result.metrics_path}")
        return

    bundle, trace = fit_bundle(
        data_dir=args.data_dir,
        year_min=args.year_min,
        year_max=args.year_max,
        class_pattern=args.class_pattern,
        max_rows=args.max_rows,
        draws=args.draws,
        tune=args.tune,
        chains=args.chains,
        cores=args.cores,
    )

    print("Model fit complete.")
    print(
        "Filters:",
        {
            "year_min": args.year_min,
            "year_max": args.year_max,
            "class_pattern": args.class_pattern,
            "max_rows": args.max_rows,
        },
    )
    print(f"Rows used: {len(bundle.frame)}")
    print(f"Corps: {len(bundle.corps_lookup)}")
    print(f"Years: {len(bundle.year_lookup)}")

    if args.corps_id:
        prediction = predict_next_show(args.corps_id, bundle, trace)
        print(
            f"Next-show prediction for {args.corps_id}: "
            f"mean={prediction.prediction_mean:.2f}, "
            f"median={prediction.prediction_median:.2f}, "
            f"95% CI=({prediction.lower_95:.2f}, {prediction.upper_95:.2f})"
        )

    if args.simulate_horizon > 0:
        corps_ids = list(bundle.corps_lookup.keys())[:10]
        simulation = simulate_season_progression(corps_ids, bundle, trace, horizon=args.simulate_horizon)
        print(simulation.head())


if __name__ == "__main__":
    main()
