# Predictive Maintenance System

Industrial IoT predictive maintenance using sensor fusion and ML.

## Pipeline
Sensors -> FFT Feature Engineering -> ML Models -> Alerts

## Features
- FFT Vibration Analysis (bearing fault detection)
- CUSUM Anomaly Detection
- RUL Estimation (Remaining Useful Life)
- Multi-sensor Fusion (accelerometer + temperature + current)

## Fault Detection
| Fault Type | Frequency | Severity |
|------------|-----------|----------|
| Imbalance | 1x RPM | Medium |
| Misalignment | 2x RPM | High |
| Bearing Outer Race (BPFO) | Variable | Critical |
