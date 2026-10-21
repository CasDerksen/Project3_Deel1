"""
Notebook/bestand 15: Streamlit-app
---
Laat voor een gekozen dag (+ de twee dagen erna) de voorspelde en echte energie zien.
Modellen per locatie (gekozen in 14_Vergelijking):
- Huis consumption : ARIMA(3,0,3) baseline      (models/saved_models/baseline_arima_huis_cons.pkl, uurdata)
- Huis generation  : SARIMAX(1,0,1)(0,1,1,24)   (models/saved_models/10a_sarimax_huis_generation.pkl, uurdata)
- Carport consumption : LightGBM                (models/saved_models/11c_lightgbm_carport_cons.txt, 5 min)
- Carport generation  : LightGBM                (models/saved_models/11d_lightgbm_carport_gen.txt, 5 min)

Starten vanuit de projectmap:  streamlit run streamlit/15_streamlit.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import lightgbm as lgb
from statsmodels.tsa.arima.model import ARIMAResults
from statsmodels.tsa.statespace.sarimax import SARIMAXResults

#Projectmap zoeken, werkt vanuit elke submap
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents, Path(__file__).resolve().parent.parent]
            if (p / "prepared_data").exists())
DATA_FILE = ROOT / "prepared_data" / "forecast_targets_with_weather.parquet"
SPLIT_FILE = ROOT / "prepared_data" / "model_split.parquet"
MODEL_DIR = ROOT / "models" / "saved_models"

TZ = "Europe/Amsterdam"
STAP_PER_DAG = 24 * 12          #288 stappen van 5 minuten
DAGEN_FORECAST = 3
HORIZON = DAGEN_FORECAST * STAP_PER_DAG
INPUT_LEN = 7 * STAP_PER_DAG    #7 dagen historie (zelfde als LightGBM-notebooks)
EXOG_FEATURES = ["Temperatuur", "GlobaleStraling", "DuurNeerslag"]   #zelfde als notebook 10a


### Data
### ___________________________________________________________________________
@st.cache_data
def laad_data():
    df = pd.read_parquet(DATA_FILE)
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_convert(TZ)
    return df


@st.cache_data
def laad_split():
    return pd.read_parquet(SPLIT_FILE).iloc[0]


@st.cache_data
def reeks_5min(location, target_type):
    """5-minutenreeks van één locatie/target, zelfde filter als de LightGBM-notebooks."""
    df = laad_data()
    deel = df[(df["location"] == location) & (df["target_type"] == target_type)].copy()
    deel = deel.dropna(subset=["pow_imputed", "Windrichting", "WindSnelheid", "LuchtVochtigheid"])
    return deel.sort_values("timestamp").reset_index(drop=True)


@st.cache_data
def reeks_uur(location, target_type):
    """Uurgemiddelde van één locatie/target, zelfde als notebook 10a."""
    df = laad_data()
    deel = df[(df["location"] == location) & (df["target_type"] == target_type)].set_index("timestamp")
    return deel.resample("1h").agg(
        y_model=("pow_imputed", "mean"),
        y_echt=("pow_w", "mean"),
        Temperatuur=("Temperatuur", "mean"),
        GlobaleStraling=("GlobaleStraling", "mean"),
        DuurNeerslag=("DuurNeerslag", "mean"),
    )


### Modellen
### ___________________________________________________________________________
@st.cache_resource
def laad_arima():
    return ARIMAResults.load(MODEL_DIR / "baseline_arima_huis_cons.pkl")


@st.cache_resource
def laad_sarimax():
    return SARIMAXResults.load(MODEL_DIR / "10a_sarimax_huis_generation.pkl")


@st.cache_resource
def laad_lgbm(bestand):
    return lgb.Booster(model_file=str(MODEL_DIR / bestand))


### Voorspellen huis (uurdata, 72 uur)
### ___________________________________________________________________________
def voorspel_huis_cons(dag_start):
    """ARIMA(3,0,3): model toepassen op de historie tot de gekozen dag en 72 uur vooruit voorspellen."""
    uur = reeks_uur("huis", "consumption")
    historie = uur.loc[:dag_start - pd.Timedelta(hours=1), "y_model"]      #NaN mag, statsmodels vult dat zelf in
    fit = laad_arima().apply(historie.to_numpy())            #zelfde parameters, nieuwe data
    pred = fit.forecast(steps=72)
    index = pd.date_range(dag_start, periods=72, freq="1h")
    return pd.DataFrame({"Echt": uur["y_model"].reindex(index).to_numpy(), "Voorspelling": pred}, index=index)


def voorspel_huis_gen(dag_start):
    """SARIMAX(1,0,1)(0,1,1,24) met weer als exogene variabelen (geschaald op train, zoals notebook 10a)."""
    uur = reeks_uur("huis", "generation")
    split = laad_split()
    train = uur[(uur.index >= split["train_start"]) & (uur.index < split["train_end"])]
    exog = ((uur[EXOG_FEATURES] - train[EXOG_FEATURES].mean()) / train[EXOG_FEATURES].std()).ffill().bfill()

    historie = uur.loc[:dag_start - pd.Timedelta(hours=1)]
    index = pd.date_range(dag_start, periods=72, freq="1h")
    fit = laad_sarimax().apply(historie["y_model"].to_numpy(), exog=exog.loc[historie.index].to_numpy())
    pred = fit.get_forecast(steps=72, exog=exog.reindex(index).to_numpy()).predicted_mean
    return pd.DataFrame({"Echt": uur["y_model"].reindex(index).to_numpy(), "Voorspelling": pred}, index=index)


### Voorspellen carport (5 min, 864 stappen)
### ___________________________________________________________________________
def bouw_features(deel, o):
    """Zelfde features als notebook 11c/11d voor één forecastmoment o (index van de eerste voorspelde stap)."""
    tijd = deel["timestamp"]
    minuut_van_dag = tijd.dt.hour * 60 + tijd.dt.minute
    dag_van_week = tijd.dt.dayofweek + tijd.dt.hour / 24
    dag_van_jaar = tijd.dt.dayofyear
    tijd_features = pd.DataFrame({
        "uur_sin": np.sin(2 * np.pi * minuut_van_dag / 1440),
        "uur_cos": np.cos(2 * np.pi * minuut_van_dag / 1440),
        "week_sin": np.sin(2 * np.pi * dag_van_week / 7),
        "week_cos": np.cos(2 * np.pi * dag_van_week / 7),
        "jaar_sin": np.sin(2 * np.pi * dag_van_jaar / 365.25),
        "jaar_cos": np.cos(2 * np.pi * dag_van_jaar / 365.25),
        "dag_van_week": tijd.dt.dayofweek,
        "is_weekend": (tijd.dt.dayofweek >= 5).astype(int),
    })
    rad = np.deg2rad(deel["Windrichting"])
    weer_features = pd.DataFrame({
        "Temperatuur": deel["Temperatuur"],
        "GlobaleStraling": deel["GlobaleStraling"],
        "DuurNeerslag": deel["DuurNeerslag"],
        "SomNeerslag": deel["SomNeerslag"],
        "WindSnelheid": deel["WindSnelheid"],
        "LuchtVochtigheid": deel["LuchtVochtigheid"],
        "wind_sin": np.sin(rad),
        "wind_cos": np.cos(rad),
    })
    y = deel["pow_imputed"].to_numpy(np.float64)
    cumsum = np.concatenate([[0.0], np.cumsum(y)])
    straling_24u = weer_features["GlobaleStraling"].rolling(STAP_PER_DAG, min_periods=1).mean().to_numpy(np.float32)

    def gem_venster(eind, lengte):
        return (cumsum[eind] - cumsum[eind - lengte]) / lengte

    h = np.arange(HORIZON)
    t = o + h
    k = h // STAP_PER_DAG + 1
    X = pd.DataFrame({
        "stap": h.astype(np.float32),
        "dag_vooruit": (k - 1).astype(np.float32),
        "laatste": np.repeat(y[o - 1], HORIZON),
        "gem_1u": np.repeat(gem_venster(o, 12), HORIZON),
        "gem_24u": np.repeat(gem_venster(o, STAP_PER_DAG), HORIZON),
        "gem_7d": np.repeat(gem_venster(o, INPUT_LEN), HORIZON),
        "zelfde_tijd_recent": y[t - STAP_PER_DAG * k],
        "zelfde_tijd_gem7": np.mean([y[t - STAP_PER_DAG * (k + j)] for j in range(7)], axis=0),
        "zelfde_tijd_vorige_week": y[t - INPUT_LEN],
        "zelfde_tijd_vorige_week_gem1u": gem_venster(np.minimum(t - INPUT_LEN + 6, o), 12),
        "straling_24u": np.repeat(straling_24u[o - 1], HORIZON),
    }).astype(np.float32)
    X[list(tijd_features.columns)] = tijd_features.to_numpy(np.float32)[t]
    X[list(weer_features.columns)] = weer_features.to_numpy(np.float32)[t]     #weer op het voorspelde tijdstip
    return X, y[t], pd.DatetimeIndex(deel["timestamp"].iloc[t])


def voorspel_carport(target_type, bestand, dag_start):
    deel = reeks_5min("carport", target_type)
    model = laad_lgbm(bestand)
    o = int((deel["timestamp"] >= dag_start).idxmax())        #eerste rij van de gekozen dag
    X, echt, tijd = bouw_features(deel, o)
    pred = model.predict(X[model.feature_name()])             #alleen de features die dit model kent
    return pd.DataFrame({"Echt": echt, "Voorspelling": pred}, index=tijd)


### Hulpfuncties
### ___________________________________________________________________________
def kies_dag(label, start, eind, standaard):
    return st.date_input(label, value=standaard, min_value=start, max_value=eind, key=label)


def advies(cons, gen, duur_uren, wat):
    """Grafieken + per dag het beste aaneengesloten blok van `duur_uren` uur (hoogste gemiddelde overschot = opwek - verbruik)."""
    df = pd.DataFrame({"Verbruik": cons["Voorspelling"], "Opwek": gen["Voorspelling"]}).dropna()
    df["Overschot"] = df["Opwek"] - df["Verbruik"]
    stap = (df.index[1] - df.index[0])
    venster = max(1, int(pd.Timedelta(hours=duur_uren) / stap))         #aantal stappen in het blok

    st.subheader("Voorspelling verbruik en opwek (W)")
    st.line_chart(df[["Verbruik", "Opwek"]])
    st.subheader("Verwacht overschot (opwek - verbruik, W)")
    st.line_chart(df[["Overschot"]])

    st.subheader(f"Advies: {wat}")
    #Gemiddeld overschot per blok dat eindigt op dit tijdstip, daarna per dag het beste blok kiezen
    gem_blok = df["Overschot"].rolling(venster).mean()
    regels = []
    for dag, deel in gem_blok.dropna().groupby(gem_blok.dropna().index.normalize()):
        eind = deel.idxmax()
        begin = eind - stap * (venster - 1)
        over = deel.max()
        oordeel = ("goed moment, er is naar verwachting genoeg eigen opwek" if over > 0
                   else "geen echt overschot verwacht; dit is het minst slechte moment")
        regels.append({"Dag": f"{dag:%a %d-%m}", "Beste blok": f"{begin:%H:%M} - {eind + stap:%H:%M}",
                       "Gem. overschot (W)": round(over), "Oordeel": oordeel})
    st.table(pd.DataFrame(regels).set_index("Dag"))

    with st.expander("Echte waarden en fout per dag"):
        echt = pd.DataFrame({"Verbruik echt": cons["Echt"], "Verbruik voorspeld": cons["Voorspelling"],
                             "Opwek echt": gen["Echt"], "Opwek voorspeld": gen["Voorspelling"]})
        st.line_chart(echt)
        dagen = echt.index.normalize()
        mae = pd.DataFrame({"MAE verbruik (W)": (cons["Voorspelling"] - cons["Echt"]).abs().groupby(dagen).mean(),
                            "MAE opwek (W)": (gen["Voorspelling"] - gen["Echt"]).abs().groupby(dagen).mean()}).round(0)
        mae.index = [f"{d:%a %d-%m}" for d in mae.index]
        st.write(mae)


### App
### ___________________________________________________________________________
st.title("SERENE – energievoorspelling")
st.caption("Kies een dag; de app laat die dag en de twee dagen erna zien (72 uur vooruit vanaf 00:00).")

split = laad_split()
#Dagen die gekozen kunnen worden: in de testperiode, met 7 dagen historie ervoor en 3 dagen data erna
eerste_dag = (pd.Timestamp(split["test_start"]).tz_convert(TZ) + pd.Timedelta(days=7)).normalize().date()
laatste_dag = (pd.Timestamp(split["test_end"]).tz_convert(TZ) - pd.Timedelta(days=DAGEN_FORECAST)).normalize().date()

tab_huis, tab_carport = st.tabs(["Huis", "Carport"])

with tab_huis:
    dag = kies_dag("Dag (huis)", eerste_dag, laatste_dag, eerste_dag)
    dag_start = pd.Timestamp(dag, tz=TZ)
    duur = st.slider("Hoe lang draait het apparaat (uur)?", 1, 6, 2, key="duur_huis")
    st.write("Modellen: verbruik ARIMA(3,0,3) baseline · opwek SARIMAX(1,0,1)(0,1,1,24) · uurgemiddelden")
    with st.spinner("Voorspellen..."):
        advies(voorspel_huis_cons(dag_start), voorspel_huis_gen(dag_start), duur,
               "wanneer apparaten (wasmachine, vaatwasser, droger) aanzetten")

with tab_carport:
    dag = kies_dag("Dag (carport)", eerste_dag, laatste_dag, eerste_dag)
    dag_start = pd.Timestamp(dag, tz=TZ)
    duur = st.slider("Hoe lang moet de auto laden (uur)?", 1, 12, 4, key="duur_carport")
    st.write("Modellen: verbruik LightGBM (11c) · opwek LightGBM (11d) · 5-minutendata")
    with st.spinner("Voorspellen..."):
        advies(voorspel_carport("consumption", "11c_lightgbm_carport_cons.txt", dag_start),
               voorspel_carport("generation", "11d_lightgbm_carport_gen.txt", dag_start), duur,
               "wanneer de auto aan de lader")
