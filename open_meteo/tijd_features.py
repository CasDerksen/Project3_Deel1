import numpy as np

def tijd_features_aanmaken(data):

    # Nederlandse tijd gebruiken
    uur = data["timestamp_nl"].dt.hour

    dag_van_week = data["timestamp_nl"].dt.dayofweek

    # Tijdfeatures maken
    data["uur_sin"] = np.sin(2 * np.pi * uur / 24)
    data["uur_cos"] = np.cos(2 * np.pi * uur / 24)

    data["week_sin"] = np.sin(2 * np.pi * dag_van_week / 7)
    data["week_cos"] = np.cos(2 * np.pi * dag_van_week / 7)

    return data
