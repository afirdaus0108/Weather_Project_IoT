"""Weather collector for Seri Iskandar, Perak (TFB2093 IoT project).

Each run:
  1. saves the current observation  -> table `observations`
  2. saves a 48h forecast snapshot  -> table `forecasts` (tagged with lead time)
Run it hourly (cron / GitHub Actions / Task Scheduler). Uses only the standard library.
"""
import json
import sqlite3
import urllib.parse
import urllib.request
from datetime import datetime

LAT, LON = 4.3667, 100.9667  # Seri Iskandar
DB = "weather.db"
VARS = "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m"

SCHEMA = """
CREATE TABLE IF NOT EXISTS observations (
    ts TEXT PRIMARY KEY,            -- local time (Asia/Kuala_Lumpur)
    temperature REAL, humidity REAL, precipitation REAL, wind_speed REAL
);
CREATE TABLE IF NOT EXISTS forecasts (
    fetched_at TEXT, target_ts TEXT, lead_hours INTEGER,
    temperature REAL, humidity REAL, precipitation REAL,
    precip_prob REAL, wind_speed REAL,
    PRIMARY KEY (fetched_at, target_ts)
);
"""


def fetch():
    params = {
        "latitude": LAT,
        "longitude": LON,
        "current": VARS,
        "hourly": VARS + ",precipitation_probability",
        "forecast_hours": 48,
        "timezone": "Asia/Kuala_Lumpur",
    }
    url = "https://api.open-meteo.com/v1/forecast?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.load(r)


def main():
    data = fetch()
    cur, hr = data["current"], data["hourly"]
    con = sqlite3.connect(DB)
    con.executescript(SCHEMA)

    # 1) observation (rounded to the hour so it joins cleanly with forecast targets)
    obs_ts = cur["time"][:13] + ":00"
    con.execute(
        "INSERT OR REPLACE INTO observations VALUES (?,?,?,?,?)",
        (obs_ts, cur["temperature_2m"], cur["relative_humidity_2m"],
         cur["precipitation"], cur["wind_speed_10m"]),
    )

    # 2) forecast snapshot
    fetched = datetime.fromisoformat(obs_ts)
    for i, t in enumerate(hr["time"]):
        lead = round((datetime.fromisoformat(t) - fetched).total_seconds() / 3600)
        con.execute(
            "INSERT OR REPLACE INTO forecasts VALUES (?,?,?,?,?,?,?,?)",
            (obs_ts, t, lead, hr["temperature_2m"][i],
             hr["relative_humidity_2m"][i], hr["precipitation"][i],
             hr["precipitation_probability"][i], hr["wind_speed_10m"][i]),
        )
    con.commit()
    con.close()
    print("saved", obs_ts)


if __name__ == "__main__":
    main()
