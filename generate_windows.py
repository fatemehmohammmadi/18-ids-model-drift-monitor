"""Create baseline + drifted feature windows for an IDS-like model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RNG = np.random.default_rng(818)
OUT = Path(__file__).parent / "data"


def make_window(n: int, drift: bool = False) -> pd.DataFrame:
    src_bytes = RNG.lognormal(8.0 + (1.5 if drift else 0.0), 0.5, n)
    failed = RNG.poisson(0.5 + (4 if drift else 0.0), n)
    same_srv = RNG.uniform(0.1, 0.5 if not drift else 0.95, n)
    return pd.DataFrame({"src_bytes": src_bytes, "failed_logins": failed, "same_srv_rate": same_srv})


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    make_window(1000, False).to_csv(OUT / "baseline.csv", index=False)
    # week1 similar, week2 drifted
    make_window(800, False).to_csv(OUT / "week1.csv", index=False)
    make_window(800, True).to_csv(OUT / "week2.csv", index=False)
    print(f"Wrote baseline/week1/week2 under {OUT}")


if __name__ == "__main__":
    main()
