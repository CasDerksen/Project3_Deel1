"""Gedeelde functies voor de rolling evaluatie van alle modellen.

Elk modelnotebook gebruikt dezelfde vensters (uit 06_Dataprep_v3) en dezelfde
functie om te scoren, zodat de modellen eerlijk te vergelijken zijn.

Gebruik in een notebook:

    import sys
    from pathlib import Path
    ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "prepared_data").exists())
    sys.path.append(str(ROOT / "models"))
    import evaluatie as ev

    df = ev.laad_reeks("huis", "consumption")
    origins = ev.maak_origins(df, "val")
    detail = ev.rolling_evaluatie(voorspel, df, origins)
    ev.samenvatting(detail)

`voorspel` is een functie van het notebook zelf: die krijgt een lijst origins
(rijnummers in df) en geeft per origin de voorspelling terug. Het model wordt
vooraf getraind en tijdens de evaluatie niet meer aangepast.
"""

from pathlib import Path

import numpy as np
import pandas as pd

PUNTEN_PER_DAG = 288
INPUT_LEN = 7 * PUNTEN_PER_DAG
MAX_HORIZON = 3 * PUNTEN_PER_DAG

#Stappen vooruit waarop elk model gescoord wordt (1 stap = 5 minuten)
HORIZONS = {"5 min": 1,
            "10 min": 2,
            "15 min": 3,
            "30 min": 6,
            "60 min": 12,
            "1 dag": 288,
            "2 dagen": 576,
            "3 dagen": 864}


def project_root():
    #Projectmap zoeken, werkt vanuit elke submap
    return next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "prepared_data").exists())


def laad_reeks(locatie, target_type, bestand="forecast_targets_long.parquet"):
    """Eén reeks (bijvoorbeeld huis, consumption) als volledig 5-minuten raster.

    Rijnummer = positie in de tijd, dus 288 rijen verder is altijd 1 dag later.
    """
    data = pd.read_parquet(project_root() / "prepared_data" / bestand)

    if "set" not in data.columns:
        raise ValueError(f"'{bestand}' heeft geen kolom 'set'. Draai eerst 06_Dataprep_v3")

    df = data[(data["location"] == locatie) & (data["target_type"] == target_type)].copy()
    df = df.sort_values("timestamp").reset_index(drop=True)

    #Controle: geen ontbrekende of dubbele tijdstippen, anders kloppen de rijnummers niet
    stappen = df["timestamp"].diff().dropna()
    if not (stappen == pd.Timedelta(minutes=5)).all():
        raise ValueError("De reeks is geen volledig 5-minuten raster")

    return df


def laad_vensters():
    return pd.read_parquet(project_root() / "prepared_data" / "vensters.parquet")


def maak_origins(df, set_naam, venster=None, stap=12, horizon=MAX_HORIZON):
    """Vaste voorspelmomenten binnen de vensters van één set ("val", "test" of "stresstest").

    Een origin is het rijnummer van het eerste punt dat voorspeld wordt. Alles vóór
    de origin mag het model als input gebruiken. De hele horizon ligt binnen
    hetzelfde venster. `stap=12` betekent elk uur een nieuwe voorspelling.
    """
    goed = df["set"] == set_naam
    if venster is not None:
        goed = goed & (df["venster"] == venster)

    origins = []

    for naam, deel in df[goed].groupby("venster", sort=False):
        begin, eind = deel.index[0], deel.index[-1]
        origins.append(np.arange(begin, eind - horizon + 2, stap))

    if not origins:
        return np.array([], dtype=int)

    return np.sort(np.concatenate(origins))


def train_origins(df, input_len=INPUT_LEN, horizon=MAX_HORIZON, stap=12, sets=("train",),
                  kolom="pow_imputed", volledig=True):
    """Voorspelmomenten om op te trainen.

    Een trainvoorbeeld (input + horizon) mag geen enkel punt uit een andere set
    bevatten. Met `volledig=True` vallen ook voorbeelden met een leeg punt af.
    Voor het eindmodel kan `sets=("train", "val")`.
    """
    bruikbaar = df["set"].isin(sets).to_numpy()
    if volledig:
        bruikbaar = bruikbaar & df[kolom].notna().to_numpy()

    #Aantal bruikbare punten in het stuk [origin - input_len, origin + horizon)
    opgeteld = np.concatenate([[0], np.cumsum(bruikbaar)])
    origins = np.arange(input_len, len(df) - horizon + 1, stap)
    aantal = opgeteld[origins + horizon] - opgeteld[origins - input_len]

    return origins[aantal == input_len + horizon]


def rolling_evaluatie(voorspel, df, origins, horizons=HORIZONS, input_len=INPUT_LEN, blok=256):
    """Rolling evaluatie met een vast model.

    De origins worden op volgorde van tijd afgewerkt. Per origin geeft `voorspel`
    een voorspelling en die wordt op de gekozen stappen vergeleken met de echte
    meting (`pow_w`). Punten zonder echte meting tellen niet mee: er wordt nooit
    gescoord tegen een ingevulde waarde.

    voorspel(origins) geeft een array terug met per origin één rij:
    - óf de hele voorspelling van minstens 864 stappen,
    - óf alleen de gevraagde stappen, in de volgorde van `horizons`.
    Stappen die een model niet kan voorspellen mogen NaN zijn.
    """
    origins = np.sort(np.asarray(origins, dtype=int))
    stappen = np.array(list(horizons.values()))
    echt_alles = df["pow_w"].to_numpy(dtype=float)

    #Welk deel van de 7 dagen input is echt gemeten?
    gemeten = np.concatenate([[0], np.cumsum(~np.isnan(echt_alles))])
    start_input = np.maximum(origins - input_len, 0)
    input_dekking = (gemeten[origins] - gemeten[start_input]) / input_len

    delen = []

    for i in range(0, len(origins), blok):
        deel = origins[i:i + blok]
        pred = np.asarray(voorspel(deel), dtype=float)

        if pred.ndim != 2 or pred.shape[0] != len(deel):
            raise ValueError(f"voorspel moet een array van ({len(deel)}, stappen) teruggeven, kreeg {pred.shape}")

        if pred.shape[1] == len(stappen):
            pred_stap = pred
        elif pred.shape[1] >= stappen.max():
            pred_stap = pred[:, stappen - 1]
        else:
            raise ValueError(f"voorspel gaf {pred.shape[1]} stappen terug, verwacht {len(stappen)} of minstens {stappen.max()}")

        #Stap k vooruit hoort bij rij origin + k - 1
        echt_stap = echt_alles[deel[:, None] + stappen[None, :] - 1]

        delen.append(pd.DataFrame({
            "origin": np.repeat(df["timestamp"].to_numpy()[deel], len(stappen)),
            "set": np.repeat(df["set"].to_numpy()[deel], len(stappen)),
            "venster": np.repeat(df["venster"].to_numpy()[deel], len(stappen)),
            "input_dekking": np.repeat(input_dekking[i:i + blok], len(stappen)),
            "horizon": np.tile(list(horizons), len(deel)),
            "stappen": np.tile(stappen, len(deel)),
            "echt": echt_stap.ravel(),
            "voorspeld": pred_stap.ravel()}))

    detail = pd.concat(delen, ignore_index=True)
    detail["origin"] = pd.to_datetime(detail["origin"], utc=True).dt.tz_convert(df["timestamp"].dt.tz)
    detail["fout"] = detail["voorspeld"] - detail["echt"]

    return detail


def samenvatting(detail, per_venster=False):
    """MAE en RMSE per horizon, over alle punten met een echte meting én een voorspelling."""
    geldig = detail.dropna(subset=["fout"])
    groep = ["venster", "stappen", "horizon"] if per_venster else ["stappen", "horizon"]

    tabel = geldig.groupby(groep, sort=True)["fout"].agg(
        aantal="size",
        MAE=lambda fout: fout.abs().mean(),
        RMSE=lambda fout: np.sqrt((fout ** 2).mean()))

    return tabel.reset_index("stappen", drop=True)


def bewaar(detail, model, locatie, target_type):
    """Uitkomst opslaan zodat het vergelijkingsnotebook alle modellen naast elkaar kan zetten."""
    map_resultaten = project_root() / "models" / "results" / "rolling"
    map_resultaten.mkdir(parents=True, exist_ok=True)

    resultaat = detail.assign(model=model, location=locatie, target_type=target_type)
    set_naam = "-".join(sorted(resultaat["set"].unique()))
    pad = map_resultaten / f"{model}_{locatie}_{target_type}_{set_naam}.parquet"
    resultaat.to_parquet(pad, index=False)

    return pad


def laad_resultaten():
    map_resultaten = project_root() / "models" / "results" / "rolling"
    return pd.concat([pd.read_parquet(pad) for pad in sorted(map_resultaten.glob("*.parquet"))], ignore_index=True)


def vul_input(waarden):
    """Lege punten in de input vullen met alleen data van eerder.

    Voor modellen die geen lege input aankunnen. Eerst de waarde van 1 t/m 7 dagen
    eerder op hetzelfde tijdstip, daarna de laatste bekende waarde, anders 0.
    Kijkt nooit vooruit, dus ook in validatie en test te gebruiken voor de input.
    """
    reeks = pd.Series(np.asarray(waarden, dtype=float))
    gevuld = reeks.copy()

    for dagen in range(1, 8):
        gevuld = gevuld.fillna(reeks.shift(dagen * PUNTEN_PER_DAG))

    return gevuld.ffill().fillna(0.0).to_numpy()


def seasonal_naive(df):
    """Baseline als voorbeeld van een voorspel-functie: de laatste echt gemeten dag wordt 3 keer herhaald."""
    echt = vul_input(df["pow_w"].to_numpy(dtype=float))

    def voorspel(origins):
        laatste_dag = np.stack([echt[o - PUNTEN_PER_DAG:o] for o in origins])
        return np.tile(laatste_dag, (1, 3))

    return voorspel
