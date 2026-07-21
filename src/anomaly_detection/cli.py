"""Command-line interface.

Two verbs matching the two things you actually do: ``train`` fits a detector
and saves a scorable bundle, ``score`` applies a saved bundle to new data.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import sys

from anomaly_detection import __version__
from anomaly_detection.config import CFG, Config

DEFAULT_COMMAND = "train"


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser."""
    parser = argparse.ArgumentParser(
        prog="anomaly-detection",
        description=(
            "LSTM-based anomaly detection for entity time series. Train a "
            "detector on simulated or real data, then score new data with it."
        ),
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    subparsers = parser.add_subparsers(dest="command", metavar="{train,score}")

    _add_train_parser(subparsers)
    _add_score_parser(subparsers)

    return parser


def _add_common(parser: argparse.ArgumentParser) -> None:
    """Add options shared by every subcommand."""
    parser.add_argument(
        "--quiet", action="store_true", help="only log warnings and errors"
    )


def _add_train_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "train",
        help="fit a detector and save a scorable bundle",
        description=(
            "Train on a CSV/Parquet file (--input-csv) or on simulated data. "
            "Input without a label column runs unsupervised: no metrics, but "
            "ranked alerts are still produced."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    data = parser.add_argument_group("input")
    data.add_argument(
        "--input-csv",
        dest="input_path",
        default=CFG.input_path,
        help="CSV or Parquet file to train on; omit to simulate",
    )
    data.add_argument(
        "--id-column",
        default=CFG.id_column,
        help="column identifying one entity's series",
    )
    data.add_argument(
        "--month-column",
        default=CFG.month_column,
        help="period column, used for chronological ordering",
    )
    data.add_argument(
        "--label-column",
        default=CFG.label_column,
        help="ground-truth column; absent means unsupervised",
    )
    data.add_argument(
        "--allow-gaps",
        action="store_true",
        default=CFG.allow_gaps,
        help="accept non-consecutive periods instead of failing",
    )

    sim = parser.add_argument_group("simulation (ignored with --input-csv)")
    sim.add_argument("--n-companies", type=int, default=CFG.n_companies)
    sim.add_argument("--n-months", type=int, default=CFG.n_months)
    sim.add_argument(
        "--n-anomalous-companies", type=int, default=CFG.n_anomalous_companies
    )
    sim.add_argument(
        "--anomaly-kind",
        choices=("univariate", "multivariate"),
        default=CFG.anomaly_kind,
        help="univariate: a turnover spike; multivariate: a broken lead-lag "
        "relationship, invisible in any single series",
    )

    split = parser.add_argument_group("splitting")
    split.add_argument(
        "--time-steps", type=int, default=CFG.time_steps, help="sliding window length"
    )
    split.add_argument("--test-size", type=float, default=CFG.test_size)
    split.add_argument("--val-size", type=float, default=CFG.val_size)

    training = parser.add_argument_group("training")
    training.add_argument("--epochs", type=int, default=CFG.epochs)
    training.add_argument("--batch-size", type=int, default=CFG.batch_size)
    training.add_argument("--lstm-units", type=int, default=CFG.lstm_units)
    training.add_argument("--dense-units", type=int, default=CFG.dense_units)
    training.add_argument("--learning-rate", type=float, default=CFG.learning_rate)

    _add_detection_args(parser)

    misc = parser.add_argument_group("output")
    misc.add_argument("--seed", type=int, default=CFG.seed)
    misc.add_argument(
        "--output-dir",
        default=CFG.output_dir,
        help="directory for the model, scaler, and metrics",
    )
    misc.add_argument(
        "--alerts-path",
        default=CFG.alerts_path,
        help="write ranked alerts here (.csv or .parquet)",
    )
    misc.add_argument(
        "--no-save", action="store_true", help="skip writing artifacts to disk"
    )
    _add_common(parser)


def _add_score_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "score",
        help="apply a saved bundle to new data",
        description=(
            "Load a model, scaler, and threshold saved by `train`, score new "
            "data, and write a ranked alert table. No labels required."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-dir", required=True, help="directory holding a bundle saved by `train`"
    )
    parser.add_argument(
        "--input-csv",
        dest="input_path",
        required=True,
        help="CSV or Parquet file to score",
    )
    parser.add_argument(
        "--alerts-path",
        default="alerts.csv",
        help="where to write ranked alerts (.csv or .parquet)",
    )
    parser.add_argument("--id-column", default=CFG.id_column)
    parser.add_argument("--month-column", default=CFG.month_column)
    parser.add_argument("--label-column", default=CFG.label_column)
    parser.add_argument("--allow-gaps", action="store_true", default=CFG.allow_gaps)

    _add_detection_args(parser)
    _add_common(parser)


def _add_detection_args(parser: argparse.ArgumentParser) -> None:
    """Add threshold and alert options, shared by both subcommands."""
    detection = parser.add_argument_group("detection")
    detection.add_argument(
        "--threshold-percentile",
        type=float,
        default=CFG.threshold_percentile,
        help="percentile of held-out error used as the cutoff",
    )
    detection.add_argument(
        "--alert-budget",
        type=int,
        default=CFG.alert_budget,
        help=(
            "flag this many highest-scoring entities instead of using a "
            "percentile; bounds false positives directly when anomalies are rare"
        ),
    )
    detection.add_argument(
        "--no-dedupe",
        dest="dedupe_alerts",
        action="store_false",
        default=CFG.dedupe_alerts,
        help="keep every window rather than the best one per entity",
    )


def config_from_args(args: argparse.Namespace) -> Config:
    """Build a :class:`Config` from parsed arguments."""
    overrides = {
        field.name: getattr(args, field.name)
        for field in dataclasses.fields(Config)
        if hasattr(args, field.name)
    }
    return dataclasses.replace(CFG, **overrides)


def _run_train(config: Config, args: argparse.Namespace) -> int:
    from anomaly_detection.process.train import run

    artifacts = run(config, save=not args.no_save)
    if args.quiet:
        # Unlabelled runs have no metrics, so the alert table is the output.
        if artifacts.result is not None:
            print(artifacts.result.summary())
        elif artifacts.alerts is not None:
            print(artifacts.alerts.to_string(index=False))
    return 0


def _run_score(config: Config, args: argparse.Namespace) -> int:
    from anomaly_detection.input.io import load_and_prepare
    from anomaly_detection.output.alerts import build_alerts, write_alerts
    from anomaly_detection.output.inference import load_artifacts, score_frame

    # argparse marks --input-csv required for `score`, but Config types it as
    # optional because `train` can simulate instead.
    if config.input_path is None:
        raise ValueError("score requires --input-csv")

    artifacts = load_artifacts(args.model_dir)
    logging.getLogger(__name__).info(
        "Loaded bundle: %s features, %s time steps, threshold %.6g",
        len(artifacts.features),
        artifacts.time_steps,
        artifacts.threshold,
    )

    df, report = load_and_prepare(
        config.input_path,
        features=list(artifacts.features),
        time_steps=artifacts.time_steps,
        id_column=config.id_column,
        month_column=config.month_column,
        label_column=config.label_column,
        rate_of_change_from=config.rate_of_change_from,
        allow_gaps=config.allow_gaps,
    )
    logging.getLogger(__name__).info("Input\n%s", report.summary())

    groups, scores, labels = score_frame(
        artifacts,
        df,
        id_column=config.id_column,
        label_column=config.label_column,
        month_column=config.month_column,
    )

    alerts = build_alerts(
        groups,
        scores,
        threshold=artifacts.threshold,
        budget=config.alert_budget,
        dedupe=config.dedupe_alerts,
        labels=labels if report.labelled else None,
    )
    written = write_alerts(alerts, args.alerts_path)

    print(f"{int(alerts['flagged'].sum())} of {len(alerts)} alerts above threshold")
    print(alerts.head(10).to_string(index=False))
    print(f"\nwritten to {written}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the CLI.

    Returns:
        ``0`` on success, ``2`` on invalid configuration or input, ``1`` on
        an unexpected pipeline failure.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 2

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
    )

    try:
        config = config_from_args(args)
    except ValueError as exc:
        print(f"error: invalid configuration: {exc}", file=sys.stderr)
        return 2

    # Imported lazily so `--help` stays fast; TensorFlow is slow to import.
    from anomaly_detection.input.io import DataQualityError

    try:
        if args.command == "score":
            return _run_score(config, args)
        return _run_train(config, args)
    except (DataQualityError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except (ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
