"""Compute PSI + prediction-rate drift between baseline and later windows."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

DATA = Path(__file__).parent / "data"
OUT = Path(__file__).parent / "outputs"
FEATS = ["src_bytes", "failed_logins", "same_srv_rate"]


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    # population stability index vs the baseline window
    # (+eps so empty bins don't nuke the log)
    qs = np.linspace(0, 100, bins + 1)
    cuts = np.unique(np.percentile(expected, qs))
    if len(cuts) < 3:
        return 0.0
    e_counts = np.histogram(expected, bins=cuts)[0] + 1e-6
    a_counts = np.histogram(actual, bins=cuts)[0] + 1e-6
    e_perc = e_counts / e_counts.sum()
    a_perc = a_counts / a_counts.sum()
    return float(np.sum((a_perc - e_perc) * np.log(a_perc / e_perc)))


def main() -> None:
    for f in ("baseline.csv", "week1.csv", "week2.csv"):
        if not (DATA / f).exists():
            raise SystemExit("Run generate_windows.py first.")
    OUT.mkdir(parents=True, exist_ok=True)

    base = pd.read_csv(DATA / "baseline.csv")
    # synthetic labels for a simple IDS proxy model
    y = ((base["failed_logins"] > 2) | (base["same_srv_rate"] > 0.7)).astype(int)
    scaler = StandardScaler()
    Xb = scaler.fit_transform(base[FEATS])
    clf = LogisticRegression(max_iter=1000).fit(Xb, y)
    base_rate = float(clf.predict(Xb).mean())

    rows = []
    report = {"baseline_attack_rate": round(base_rate, 4), "windows": {}}
    for name in ("week1", "week2"):
        w = pd.read_csv(DATA / f"{name}.csv")
        Xw = scaler.transform(w[FEATS])
        pred_rate = float(clf.predict(Xw).mean())
        feature_psi = {c: round(psi(base[c].to_numpy(), w[c].to_numpy()), 4) for c in FEATS}
        pred_shift = abs(pred_rate - base_rate)
        drifted = pred_shift > 0.1 or max(feature_psi.values()) > 0.2
        entry = {
            "pred_attack_rate": round(pred_rate, 4),
            "pred_rate_shift": round(pred_shift, 4),
            "feature_psi": feature_psi,
            "drift_alert": drifted,
        }
        report["windows"][name] = entry
        rows.append({"window": name, **{f"psi_{k}": v for k, v in feature_psi.items()}, "pred_rate": pred_rate, "drift_alert": drifted})

    pd.DataFrame(rows).to_csv(OUT / "window_metrics.csv", index=False)
    (OUT / "drift_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
