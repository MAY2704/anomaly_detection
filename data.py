import numpy as np
import pandas as pd

def generate_data(n_cp: int, n_m: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    rows = []
    for i in range(n_cp):
        base = np.linspace(1e7, 2e7, n_m)
        t = base * rng.normal(1.0, 0.05, n_m)
        a = t * rng.normal(5.0, 0.1, n_m)
        rows.append(pd.DataFrame({
            "CP_ID": f"CP_{i:04d}",
            "MONTH": pd.date_range("2022-01-01", periods=n_m, freq="M"),
            "TURNOVER": t,
            "ASSETS": a,
            "IS_TRUE_ANOMALY": 0
        }))
    df = pd.concat(rows).reset_index(drop=True)

    turnover_90th = df["TURNOVER"].quantile(0.90)
    candidate_cps = np.random.choice(df["CP_ID"].unique(), 40, replace=False)

    for cp in candidate_cps:
        g = df[df["CP_ID"] == cp]
        for off in (-1, -2, -3):
            if len(g) > abs(off):
                idx = g.index[off]
                prev = idx - 1
                if prev in df.index and df.loc[prev, "TURNOVER"] > turnover_90th:
                    if np.random.rand() < 0.5:
                        f = np.random.uniform(1.7, 3.0)
                    else:
                        f = np.random.uniform(0.01, 0.30)
                    df.loc[idx, "TURNOVER"] = max(0.0, df.loc[prev, "TURNOVER"] * f)
                    df.loc[idx, "IS_TRUE_ANOMALY"] = 1

    df["TURNOVER_ROC"] = df.groupby("CP_ID")["TURNOVER"].pct_change().fillna(0).clip(-5, 5)
    return df