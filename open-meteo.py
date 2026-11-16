
import requests
import pandas as pd

def haal_live_weer_op():

    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": 52.43,
        "longitude": 6.23,
        "minutely_15": [
            "temperature_2m",
            "relative_humidity_2m",
            "shortwave_radiation"
        ],
        "forecast_days": 1,
        "timezone": "Europe/Amsterdam"
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()

    data = response.json()

    df = pd.DataFrame(data["minutely_15"])
    df["timestamp"] = pd.to_datetime(df["time"])

    return df.drop(columns=["time"])


df_weather = haal_live_weer_op()

