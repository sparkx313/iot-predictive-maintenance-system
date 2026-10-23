# Vibration Window Anomaly Monitor

A local tool for fitting and applying an Isolation Forest to measured vibration data. It does not generate example waveforms or claim to diagnose a specific mechanical fault.

## Input CSV

The input needs three columns:

| Column | Meaning |
| --- | --- |
| `sensor_id` | Identifier for the vibration sensor |
| `timestamp` | Timestamp for the sample; ISO 8601 is recommended |
| `vibration` | Numeric reading from one vibration axis, in a consistent unit |

Rows are sorted by timestamp within each sensor, then split into non-overlapping windows of the configured sample count. The sample rate must be measured and consistent for all sensors in a run. Incomplete trailing windows are discarded and counted in the command output.

## Fit and score

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Fit a model using your measured baseline data:

```bash
python main.py fit --input path/to/vibration.csv --window-samples 1024 --sample-rate-hz 25600 --model vibration_model.joblib
```

Score another CSV using that model:

```bash
python main.py score --input path/to/new_vibration.csv --model vibration_model.joblib --output vibration_anomalies.csv
```

The model uses AC RMS, absolute peak, crest factor, kurtosis, and dominant FFT frequency for each window. The output contains those features, a signed anomaly score (larger means more unusual), and an `is_anomaly` flag. `--contamination` on the `fit` command controls the Isolation Forest threshold; its default is `auto`.

## Scope

This is unsupervised window-level anomaly screening. An unusual window is not a confirmed bearing fault, diagnosis, maintenance alert, or remaining-useful-life estimate. Validate thresholds and results against labeled sensor data and domain knowledge before operational use.
