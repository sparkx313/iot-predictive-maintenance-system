"""Fit and apply an unsupervised vibration-window anomaly detector."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

MODEL_FORMAT_VERSION = 1
FEATURE_COLUMNS = [
    "ac_rms",
    "peak_absolute",
    "crest_factor",
    "kurtosis",
    "dominant_frequency_hz",
]
REQUIRED_COLUMNS = {"sensor_id", "timestamp", "vibration"}


def parse_contamination(value: str) -> str | float:
    if value.lower() == "auto":
        return "auto"
    try:
        rate = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use 'auto' or a value greater than 0 and at most 0.5.") from exc
    if not np.isfinite(rate) or not 0 < rate <= 0.5:
        raise argparse.ArgumentTypeError("Contamination must be 'auto' or greater than 0 and at most 0.5.")
    return rate


def window_features(samples: np.ndarray, sample_rate_hz: float) -> dict[str, float]:
    centered = samples - samples.mean()
    ac_rms = float(np.sqrt(np.mean(centered**2)))
    peak = float(np.max(np.abs(centered)))
    standard_deviation = float(np.std(centered))
    kurtosis = (
        float(np.mean((centered / standard_deviation) ** 4))
        if standard_deviation > 0
        else 0.0
    )

    spectrum = np.abs(np.fft.rfft(centered))
    frequencies = np.fft.rfftfreq(len(centered), d=1.0 / sample_rate_hz)
    if len(spectrum) > 1 and np.any(spectrum[1:] > 0):
        dominant_frequency = float(frequencies[1 + int(np.argmax(spectrum[1:]))])
    else:
        dominant_frequency = 0.0

    return {
        "ac_rms": ac_rms,
        "peak_absolute": peak,
        "crest_factor": peak / ac_rms if ac_rms > 0 else 0.0,
        "kurtosis": kurtosis,
        "dominant_frequency_hz": dominant_frequency,
    }


def extract_windows(
    input_file: Path, window_samples: int, sample_rate_hz: float
) -> tuple[pd.DataFrame, int]:
    if not input_file.is_file():
        raise ValueError(f"Sensor CSV does not exist: {input_file}")
    data = pd.read_csv(input_file)
    missing = REQUIRED_COLUMNS - set(data.columns)
    if missing:
        raise ValueError("CSV is missing required column(s): " + ", ".join(sorted(missing)))
    if data.empty:
        raise ValueError("Sensor CSV contains no rows.")
    if window_samples < 2:
        raise ValueError("Window size must be at least 2 samples.")
    if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
        raise ValueError("Sample rate must be a positive finite number of hertz.")

    sensor_ids = data["sensor_id"].astype("string").str.strip()
    timestamps = pd.to_datetime(data["timestamp"], errors="coerce", utc=True)
    vibration = pd.to_numeric(data["vibration"], errors="coerce")
    valid = (
        sensor_ids.notna()
        & sensor_ids.ne("")
        & timestamps.notna()
        & vibration.notna()
        & np.isfinite(vibration)
    )
    if not valid.all():
        raise ValueError(f"Found {int((~valid).sum())} row(s) with an invalid sensor ID, timestamp, or vibration value.")

    data = pd.DataFrame(
        {"sensor_id": sensor_ids, "timestamp": timestamps, "vibration": vibration}
    )
    records: list[dict[str, Any]] = []
    discarded_samples = 0
    for sensor_id, sensor_data in data.groupby("sensor_id", sort=False):
        sensor_data = sensor_data.sort_values("timestamp", kind="stable")
        values = sensor_data["vibration"].to_numpy(dtype=np.float64)
        times = sensor_data["timestamp"].to_numpy()
        complete_windows = len(values) // window_samples
        discarded_samples += len(values) - complete_windows * window_samples
        for window_index in range(complete_windows):
            start = window_index * window_samples
            end = start + window_samples
            record: dict[str, Any] = {
                "sensor_id": str(sensor_id),
                "window_start": pd.Timestamp(times[start]).isoformat(),
                "window_end": pd.Timestamp(times[end - 1]).isoformat(),
                "sample_count": window_samples,
            }
            record.update(window_features(values[start:end], sample_rate_hz))
            records.append(record)

    if len(records) < 2:
        raise ValueError(
            "At least two complete windows are needed to fit an anomaly detector; "
            "check the sample count and --window-samples value."
        )
    return pd.DataFrame(records), discarded_samples


def fit_model(args: argparse.Namespace) -> int:
    if args.input.resolve() == args.model.resolve():
        raise ValueError("Model output must not overwrite the input CSV.")
    windows, discarded = extract_windows(args.input, args.window_samples, args.sample_rate_hz)
    detector = IsolationForest(
        n_estimators=200,
        contamination=args.contamination,
        random_state=args.seed,
    )
    detector.fit(windows[FEATURE_COLUMNS])
    bundle = {
        "format_version": MODEL_FORMAT_VERSION,
        "detector": detector,
        "window_samples": args.window_samples,
        "sample_rate_hz": args.sample_rate_hz,
        "feature_columns": FEATURE_COLUMNS,
        "contamination": args.contamination,
    }
    args.model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.model)
    print(f"Fit anomaly detector on {len(windows)} window(s); discarded {discarded} trailing sample(s).")
    print(f"Saved model to {args.model}")
    return 0


def score_model(args: argparse.Namespace) -> int:
    if args.input.resolve() == args.output.resolve():
        raise ValueError("Report output must not overwrite the input CSV.")
    if args.model.resolve() == args.output.resolve():
        raise ValueError("Report output must not overwrite the model artifact.")
    if not args.model.is_file():
        raise ValueError(f"Model artifact does not exist: {args.model}")
    bundle = joblib.load(args.model)
    if not isinstance(bundle, dict) or bundle.get("format_version") != MODEL_FORMAT_VERSION:
        raise ValueError("Unsupported vibration model artifact format.")
    windows, discarded = extract_windows(
        args.input, int(bundle["window_samples"]), float(bundle["sample_rate_hz"])
    )
    features = windows[bundle["feature_columns"]]
    predictions = bundle["detector"].predict(features)
    windows["anomaly_score"] = -bundle["detector"].decision_function(features)
    windows["is_anomaly"] = predictions == -1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    windows.to_csv(args.output, index=False)
    print(f"Scored {len(windows)} window(s); flagged {int(windows['is_anomaly'].sum())} anomalous window(s).")
    print(f"Discarded {discarded} trailing sample(s); wrote {args.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit or apply an Isolation Forest to measured vibration windows."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    fit_parser = commands.add_parser("fit", help="Fit and save a model from measured vibration data.")
    fit_parser.add_argument("--input", required=True, type=Path, help="CSV with sensor_id, timestamp, vibration columns.")
    fit_parser.add_argument("--window-samples", required=True, type=int, help="Samples per non-overlapping window.")
    fit_parser.add_argument("--sample-rate-hz", required=True, type=float, help="Measured sample rate in hertz.")
    fit_parser.add_argument("--contamination", type=parse_contamination, default="auto", help="Isolation Forest threshold setting (default: auto).")
    fit_parser.add_argument("--seed", type=int, default=42)
    fit_parser.add_argument("--model", type=Path, default=Path("vibration_model.joblib"))
    fit_parser.set_defaults(run=fit_model)

    score_parser = commands.add_parser("score", help="Score windows with a saved model.")
    score_parser.add_argument("--input", required=True, type=Path, help="CSV with the same sensor schema.")
    score_parser.add_argument("--model", required=True, type=Path)
    score_parser.add_argument("--output", type=Path, default=Path("vibration_anomalies.csv"))
    score_parser.set_defaults(run=score_model)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        return args.run(args)
    except (OSError, ValueError, KeyError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
