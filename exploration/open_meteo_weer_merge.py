from open_meteo.connectie_open_meteo import weer_data_ophalen, naar_5_minuten
from open_meteo.tijd_features import tijd_features_aanmaken
from pathlib import Path
import pandas as pd

def laad_data():

    # energie data inladen
    project_dir = Path(__file__).resolve().parents[1]
    energie_pad = project_dir / "prepared_data" / "forecast_targets_long.parquet"

    energie = pd.read_parquet(energie_pad)

    # energie timestamp naar UTC voor het mergen
    energie["timestamp"] = pd.to_datetime(
        energie["timestamp"], utc=True
    )

    # het begin en het einde van de energy data gebruiken als kaders
    start = energie["timestamp"].min().date()
    einde = (
        energie["timestamp"].max() + pd.Timedelta(days=1)
    ).date()

    # weer data ophalen
    weer = weer_data_ophalen(start, einde, timezone="UTC")

    # weer data omzetten naar 5 minuten interval
    weer = naar_5_minuten(weer)

    # weer data ook UTC tijd
    weer = weer.tz_localize("UTC")

    # ook een tijd kolom bij de weer data voor het mergen
    weer = weer.rename_axis("timestamp").reset_index()

    # mergen van de weer en de energy data
    data = pd.merge(
        energie,
        weer,
        on="timestamp",
        how="left",
        validate="many_to_one"
    )

    # nederlandse tijd toevoegen om later tijd features te maken zodat het weer klopt
    data["timestamp_nl"] = data["timestamp"].dt.tz_convert(
        "Europe/Amsterdam"
    )

    return data

def haal_modeldata_op():

    # Energie en weer data ophalen en combineren
    data = laad_data()

    # Tijdfeatures toevoegen
    data = tijd_features_aanmaken(data)

    return data