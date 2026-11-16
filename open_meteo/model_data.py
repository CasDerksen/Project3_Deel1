from exploration.open_meteo_weer_merge import laad_data
from open_meteo.tijd_features import tijd_features_aanmaken

def haal_modeldata_op():

    # Energie- en weergegevens ophalen en combineren
    data = laad_data()

    # Tijdfeatures toevoegen
    data = tijd_features_aanmaken(data)

    return data
