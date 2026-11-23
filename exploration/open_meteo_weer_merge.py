from open_meteo_weer_merge import weer_data_ophalen, naar_5_minuten, weer_voorspelling_ophalen
from open_meteo.tijd_features import tijd_features_aanmaken
from pathlib import Path
import pandas as pd


def _weer_op_5_minuten(start, einde, voorspelling=False):

    # gemeten weer of de voorspelling van 24 uur vooraf
    ophalen = weer_voorspelling_ophalen if voorspelling else weer_data_ophalen
    weer = ophalen(start, einde, timezone="UTC")

    # weer data omzetten naar 5 minuten interval
    weer = naar_5_minuten(weer)

    # weer data ook UTC tijd
    return weer.tz_localize("UTC")


def _energie_met_weer(voorspelling=False):

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

    weer = _weer_op_5_minuten(start, einde, voorspelling)

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


def laad_data():

    # energie data met het gemeten weer
    return _energie_met_weer(voorspelling=False)


def haal_modeldata_op():

    # Energie en weer data ophalen en combineren
    data = laad_data()

    # Tijdfeatures toevoegen
    data = tijd_features_aanmaken(data)

    return data


def laad_data_voorspellingen():

    # energie data met de historische weersvoorspellingen (24 uur vooraf)
    return _energie_met_weer(voorspelling=True)


def haal_modeldata_voorspelling_op():

    # Energie en voorspelde weerdata ophalen en combineren
    data = laad_data_voorspellingen()

    # Tijdfeatures toevoegen
    data = tijd_features_aanmaken(data)

    return data


def voeg_weer_toe(df, voorspelling=False):
    """Weerkolommen toevoegen aan een reeks met een kolom 'timestamp'.

    Voor notebooks die de reeks al zelf inladen, bijvoorbeeld met evaluatie.laad_reeks.
    De rijen, de volgorde en de tijdzone van de reeks blijven gelijk.
    """

    tijd_utc = pd.to_datetime(df["timestamp"], utc=True)

    # het begin en het einde van de reeks gebruiken als kaders
    start = tijd_utc.min().date()
    einde = (tijd_utc.max() + pd.Timedelta(days=1)).date()

    weer = _weer_op_5_minuten(start, einde, voorspelling)

    # per rij het weer van dat tijdstip erbij zetten
    data = df.copy()

    for kolom in weer.columns:
        data[kolom] = weer[kolom].reindex(pd.DatetimeIndex(tijd_utc)).to_numpy()

    return data
