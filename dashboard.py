"""Interactive dashboard - TFB2093 IoT weather project (Seri Iskandar, Perak).
Run locally:  streamlit run dashboard.py
Reads weather.db (filled hourly by collector.py) and uses analysis.py for the maths.
"""
import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analysis import (PARAMS, build_comparison, collection_stats, error_by_lead,
                      forecast_vs_actual, load_tables, rain_verification,
                      summary_table)

DB = "weather.db"

st.set_page_config(page_title="Seri Iskandar Weather Monitor", page_icon="🌦️", layout="wide")


@st.cache_data(show_spinner=False)
def get_data(path, mtime):
    obs, fc = load_tables(path)
    return obs, fc, build_comparison(obs, fc)


if not os.path.exists(DB):
    st.error("weather.db not found. Run collector.py at least once first.")
    st.stop()

obs, fc, cmp_df = get_data(DB, os.path.getmtime(DB))
stats = collection_stats(obs)

# ---------------------------------------------------------------- sidebar
st.sidebar.header("Controls")
param = st.sidebar.selectbox("Parameter", list(PARAMS), format_func=lambda p: PARAMS[p][0])
label, unit = PARAMS[param]
lead = st.sidebar.slider("Forecast lead time (hours ahead)", 1, 48, 6)
threshold = st.sidebar.slider("Rain alert threshold (probability %)", 10, 90, 50, step=10)
st.sidebar.caption("Data source: Open-Meteo API. 'Actual' values are Open-Meteo current "
                   "conditions (model-based analysis), logged hourly.")

# ---------------------------------------------------------------- header
st.title("🌦️ Seri Iskandar Weather Monitor")
st.caption("Hourly observations and 48-hour forecasts from Open-Meteo, with forecast accuracy "
           "measured against what actually happened.")

latest = obs.iloc[-1]
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Temperature now", f"{latest.temperature:.1f} °C")
c2.metric("Humidity now", f"{latest.humidity:.0f} %")
c3.metric("Wind now", f"{latest.wind_speed:.1f} km/h")
c4.metric("Hours collected", f"{stats['hours']}", f"{stats['readings']} readings")
c5.metric("Collection started", stats["first"].strftime("%d %b %H:%M"))

tab_live, tab_cmp, tab_acc, tab_rain, tab_data = st.tabs(
    ["Now & next 48h", "Forecast vs actual", "Accuracy", "Rain check", "Data"])

# ---------------------------------------------------------------- tab 1
with tab_live:
    snap = fc[fc.fetched_at == fc.fetched_at.max()].sort_values("target_ts")
    soon = snap[snap.lead_hours.between(0, 3)]
    peak = soon.precip_prob.max() if len(soon) else 0
    if peak >= threshold:
        t_peak = soon.loc[soon.precip_prob.idxmax(), "target_ts"]
        st.warning(f"☔ Rain likely in the next 3 hours: up to {peak:.0f}% chance "
                   f"(peak around {t_peak:%H:%M}).")
    else:
        st.success(f"No rain expected in the next 3 hours (max chance {peak:.0f}%).")

    past = obs[obs.ts >= obs.ts.max() - pd.Timedelta(hours=48)]
    fig = go.Figure()
    fig.add_scatter(x=past.ts, y=past[param], name="Observed", line=dict(width=3))
    fig.add_scatter(x=snap.target_ts, y=snap[param], name="Latest forecast",
                    line=dict(dash="dash", width=3))
    fig.update_layout(title=f"{label}: last 48h observed + next 48h forecast",
                      yaxis_title=f"{label} ({unit})", hovermode="x unified",
                      legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig)

    fig2 = go.Figure(go.Bar(x=snap.target_ts, y=snap.precip_prob, name="Rain probability"))
    fig2.add_hline(y=threshold, line_dash="dot", annotation_text="alert threshold")
    fig2.update_layout(title="Chance of rain, next 48 hours", yaxis_title="%",
                       yaxis_range=[0, 100])
    st.plotly_chart(fig2)

# ---------------------------------------------------------------- tab 2
with tab_cmp:
    st.subheader(f"{label}: forecast made {lead}h ahead vs what actually happened")
    ts = forecast_vs_actual(cmp_df, param, lead)
    if ts.empty:
        st.info("Not enough data yet for this lead time. Check back after more hours are collected.")
    else:
        fig = go.Figure()
        fig.add_scatter(x=ts.target_ts, y=ts.Actual, name="Actual", line=dict(width=3))
        fig.add_scatter(x=ts.target_ts, y=ts.Forecast, name="Forecast",
                        line=dict(dash="dash", width=3))
        fig.update_layout(yaxis_title=f"{label} ({unit})", hovermode="x unified",
                          legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig)

        lo, hi = float(min(ts.Actual.min(), ts.Forecast.min())), float(max(ts.Actual.max(), ts.Forecast.max()))
        sc = go.Figure()
        sc.add_scatter(x=ts.Actual, y=ts.Forecast, mode="markers", name="Hours")
        sc.add_scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="Perfect forecast",
                       line=dict(dash="dot"))
        sc.update_layout(title="Forecast vs actual (closer to the line = better)",
                         xaxis_title=f"Actual ({unit})", yaxis_title=f"Forecast ({unit})")
        st.plotly_chart(sc)

# ---------------------------------------------------------------- tab 3
with tab_acc:
    st.subheader(f"How does {label.lower()} accuracy change with lead time?")
    el = error_by_lead(cmp_df, param)
    if el.empty:
        st.info("No matched forecast/observation pairs yet.")
    else:
        fig = go.Figure()
        fig.add_scatter(x=el.lead_hours, y=el.MAE, name="MAE", mode="lines+markers")
        fig.add_scatter(x=el.lead_hours, y=el.RMSE, name="RMSE", mode="lines+markers")
        fig.add_scatter(x=el.lead_hours, y=el.Bias, name="Bias (forecast - actual)",
                        mode="lines+markers", line=dict(dash="dot"))
        fig.update_layout(xaxis_title="Lead time (hours ahead)", yaxis_title=unit,
                          hovermode="x unified", legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig)
        st.caption("MAE = average size of the error. RMSE = like MAE but punishes big misses. "
                   "Positive bias means the forecast runs too high.")

    st.subheader("Summary at key horizons (all parameters)")
    st.dataframe(summary_table(cmp_df).round(2), hide_index=True)

# ---------------------------------------------------------------- tab 4
with tab_rain:
    st.subheader("Did it rain when the forecast said it would?")
    st.caption(f"Forecast 'rain' = probability ≥ {threshold}%.  Observed 'rain' = at least 0.1 mm.")
    scope = st.radio("Evaluate", ["All lead times", f"Only {lead}h ahead"], horizontal=True)
    rv = rain_verification(cmp_df, threshold, lead=None if scope == "All lead times" else lead)
    if rv is None:
        st.info("Not enough data yet.")
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Overall accuracy", f"{rv['accuracy']:.0%}")
        m2.metric("Rain caught (POD)", "n/a" if pd.isna(rv["POD"]) else f"{rv['POD']:.0%}")
        m3.metric("False alarm ratio", "n/a" if pd.isna(rv["FAR"]) else f"{rv['FAR']:.0%}")
        m4.metric("Brier score", f"{rv['Brier']:.3f}", help="0 = perfect, lower is better")
        table = pd.DataFrame(
            [[rv["hits"], rv["false_alarms"]], [rv["misses"], rv["correct_negatives"]]],
            index=["Forecast: rain", "Forecast: no rain"],
            columns=["Observed: rain", "Observed: no rain"])
        st.dataframe(table)
        if rv["hits"] + rv["misses"] == 0:
            st.info("No rain has been observed in the matched hours yet, so rain-catching "
                    "skill can't be measured. It will fill in as data grows.")

# ---------------------------------------------------------------- tab 5
with tab_data:
    st.subheader("Collection health")
    st.write(f"First reading: **{stats['first']}**  |  Latest: **{stats['last']}**  |  "
             f"Missed hours: **{stats['gaps']}**")
    st.subheader("Observations")
    st.dataframe(obs.sort_values("ts", ascending=False), hide_index=True)
    st.download_button("Download observations (CSV)", obs.to_csv(index=False),
                       "observations.csv", "text/csv")
    st.download_button("Download forecast-vs-actual pairs (CSV)", cmp_df.to_csv(index=False),
                       "forecast_vs_actual.csv", "text/csv")
