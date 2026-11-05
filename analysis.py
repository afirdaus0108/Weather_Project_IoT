"""Forecast-accuracy analysis for the TFB2093 weather project.

Joins every stored forecast to the observation at its target time, then computes
error metrics per parameter and per lead time. Pure pandas, so it can be reused in
the dashboard, in a notebook, or for the numbers quoted in the report.
"""
import sqlite3

import numpy as np
import pandas as pd

# parameter name -> (label, unit)
PARAMS = {
    "temperature": ("Temperature", "°C"),
    "humidity": ("Humidity", "%"),
    "precipitation": ("Rainfall", "mm"),
    "wind_speed": ("Wind speed", "km/h"),
}


def load_tables(db_path="weather.db"):
    con = sqlite3.connect(db_path)
    obs = pd.read_sql("SELECT * FROM observations", con, parse_dates=["ts"])
    fc = pd.read_sql("SELECT * FROM forecasts", con,
                     parse_dates=["fetched_at", "target_ts"])
    con.close()
    return obs.sort_values("ts"), fc


def build_comparison(obs, fc):
    """One row per (forecast, matching observation). Columns: <param>_fc, <param>_obs,
    <param>_err (= forecast - observed)."""
    o = obs.rename(columns={"ts": "target_ts"})
    cols = list(PARAMS)
    merged = fc.merge(o, on="target_ts", suffixes=("_fc", "_obs"))
    for p in cols:
        merged[f"{p}_err"] = merged[f"{p}_fc"] - merged[f"{p}_obs"]
    return merged


def error_by_lead(cmp_df, param, max_lead=48):
    """MAE, RMSE and bias for one parameter at each lead time."""
    d = cmp_df[(cmp_df.lead_hours >= 0) & (cmp_df.lead_hours <= max_lead)]
    g = d.groupby("lead_hours")[f"{param}_err"]
    out = pd.DataFrame({
        "MAE": g.apply(lambda e: e.abs().mean()),
        "RMSE": g.apply(lambda e: np.sqrt((e ** 2).mean())),
        "Bias": g.mean(),
        "n": g.count(),
    })
    return out.reset_index()


def summary_table(cmp_df, leads=(1, 6, 12, 24)):
    """Compact table for the report: MAE / RMSE / bias per parameter at key horizons."""
    rows = []
    for p, (label, unit) in PARAMS.items():
        for lead in leads:
            e = cmp_df.loc[cmp_df.lead_hours == lead, f"{p}_err"].dropna()
            if len(e) == 0:
                continue
            rows.append({
                "Parameter": f"{label} ({unit})", "Lead (h)": lead,
                "MAE": e.abs().mean(), "RMSE": np.sqrt((e ** 2).mean()),
                "Bias": e.mean(), "n": len(e),
            })
    return pd.DataFrame(rows)


def rain_verification(cmp_df, prob_threshold=50, rain_mm=0.1, lead=None):
    """Treat rain as a yes/no event.
    Forecast says rain  : precip_prob >= prob_threshold
    Observed rain       : observed precipitation >= rain_mm
    Returns contingency counts and standard scores (accuracy, POD, FAR, Brier)."""
    d = cmp_df.dropna(subset=["precip_prob", "precipitation_obs"])
    if lead is not None:
        d = d[d.lead_hours == lead]
    if d.empty:
        return None
    f = d.precip_prob >= prob_threshold
    o = d.precipitation_obs >= rain_mm
    hits = int((f & o).sum())
    misses = int((~f & o).sum())
    false_alarms = int((f & ~o).sum())
    correct_neg = int((~f & ~o).sum())
    n = len(d)
    return {
        "hits": hits, "misses": misses,
        "false_alarms": false_alarms, "correct_negatives": correct_neg, "n": n,
        "accuracy": (hits + correct_neg) / n,
        "POD": hits / (hits + misses) if hits + misses else np.nan,
        "FAR": false_alarms / (hits + false_alarms) if hits + false_alarms else np.nan,
        "Brier": float(((d.precip_prob / 100 - o.astype(float)) ** 2).mean()),
    }


def forecast_vs_actual(cmp_df, param, lead):
    """Time series for plotting: what was forecast `lead` hours ahead vs what happened."""
    d = cmp_df[cmp_df.lead_hours == lead].sort_values("target_ts")
    return d[["target_ts", f"{param}_fc", f"{param}_obs"]].rename(
        columns={f"{param}_fc": "Forecast", f"{param}_obs": "Actual"})


def collection_stats(obs):
    if obs.empty:
        return {"readings": 0, "first": None, "last": None, "hours": 0, "expected": 0, "gaps": 0}
    first, last = obs.ts.min(), obs.ts.max()
    span_h = int((last - first).total_seconds() // 3600) + 1
    return {
        "readings": len(obs), "first": first, "last": last,
        "hours": span_h, "expected": span_h, "gaps": max(span_h - len(obs), 0),
    }
