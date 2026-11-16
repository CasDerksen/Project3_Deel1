import pandas as pd
import requests


def weer_data_ophalen(start, einde, timezone="UTC"):
    """Haal historische uurdata op bij Open-Meteo."""

    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": 52.33,
        "longitude": 6.11,
        "start_date": start,
        "end_date": einde,
        "hourly": [
            "temperature_2m",
            "shortwave_radiation",
            "relative_humidity_2m"
        ],
        "timezone": timezone
    }

    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()

    df = pd.DataFrame(r.json()["hourly"])
    df["time"] = pd.to_datetime(df["time"])

    return df.set_index("time")


def naar_5_minuten(df):
    """Uurdata naar 5 minuten, lineair interpoleren."""

    return df.resample("5min").interpolate(method="time")
