# IDS Model Drift Monitor

Simple MLOps-for-security sketch: train a proxy IDS model on a baseline window, then watch later weeks for feature PSI and prediction-rate shift.

## Run

```bash
pip install -r requirements.txt
python generate_windows.py
python monitor_drift.py
```

## What gets flagged

A window raises `drift_alert` if:

- |pred_rate − baseline_rate| > 0.1, or
- any feature PSI > 0.2

Thresholds are demo defaults — tune them against your own quiet periods.

## Outputs

`outputs/drift_report.json`, `outputs/window_metrics.csv`

In the sample data, `week1` stays calm and `week2` is intentionally drifted.

## License

MIT
