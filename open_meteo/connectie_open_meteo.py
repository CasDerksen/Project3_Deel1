from pathlib import Path

import pandas as pd
import requests

# locatie van de Aardehuizen in Olst
LATITUDE = 52.33
LONGITUDE = 6.11

# opgehaalde weerdata wordt hier bewaard, zodat niet elk notebook opnieuw de API aanroept
CACHE_DIR = Path(__file__).resolve().parents[1] / "prepared_data"


def _met_cache(naam, start, einde, timezone, vernieuw, ophalen):
    """Geef de bewaarde weerdata terug als die de hele periode dekt, haal anders op bij de API."""

    start = pd.Timestamp(str(start))
    laatste_uur = pd.Timestamp(str(einde)) + pd.Timedelta(hours=23)
    pad = CACHE_DIR / f"open_meteo_{naam}_{timezone.replace('/', '-')}.parquet"

    if pad.exists() and not vernieuw:
        df = pd.read_parquet(pad)

        if df.index.min() <= start and df.index.max() >= laatste_uur:
            return df.loc[start:laatste_uur]

    df = ophalen()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(pad)

    return df


def weer_data_ophalen(start, einde, timezone="UTC", vernieuw=False):
    """Haal historische uurdata op bij Open-Meteo.

    De data wordt bewaard in prepared_data. Met vernieuw=True wordt opnieuw opgehaald.
    """

    def ophalen():
        url = "https://archive-api.open-meteo.com/v1/archive"

        params = {
            "latitude": LATITUDE,
            "longitude": LONGITUDE,
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

    return _met_cache("weer", start, einde, timezone, vernieuw, ophalen)


def weer_voorspelling_ophalen(start, einde, timezone="UTC", vernieuw=False):
    """Haal historische weersvoorspellingen op met een lead time van 24 uur.

    De data wordt bewaard in prepared_data. Met vernieuw=True wordt opnieuw opgehaald.
    """

    def ophalen():
        url = "https://previous-runs-api.open-meteo.com/v1/forecast"

        params = {
            "latitude": LATITUDE,
            "longitude": LONGITUDE,
            "start_date": start,
            "end_date": einde,
            "hourly": [
                "temperature_2m_previous_day1",
                "shortwave_radiation_previous_day1",
                "relative_humidity_2m_previous_day1"
            ],
            "timezone": timezone
        }

        r = requests.get(url, params=params, timeout=60)
        r.raise_for_status()

        df = pd.DataFrame(r.json()["hourly"])
        df["time"] = pd.to_datetime(df["time"])

        df = df.rename(columns={
            "temperature_2m_previous_day1": "temperature_2m",
            "shortwave_radiation_previous_day1": "shortwave_radiation",
            "relative_humidity_2m_previous_day1": "relative_humidity_2m"
        })

        return df.set_index("time")

    return _met_cache("voorspelling", start, einde, timezone, vernieuw, ophalen)


def naar_5_minuten(df):
    """Uurdata naar 5 minuten, lineair interpoleren."""

    return df.resample("5min").interpolate(method="time")
