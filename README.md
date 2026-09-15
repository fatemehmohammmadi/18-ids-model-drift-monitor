# 18 — IDS Model Drift Monitor

Monitor **prediction drift / data drift** for an IDS-style model using PSI and prediction-distribution shift — essential MLOps-for-security demo.

## Run

```bash
pip install -r requirements.txt
python generate_windows.py
python monitor_drift.py
```

## Sample outputs

- `outputs/drift_report.json`
- `outputs/window_metrics.csv`
