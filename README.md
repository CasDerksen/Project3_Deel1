# Project 3: Energie Forecasting / Management SERENE project 
# Odessa Al_Did, Cas Derksen, Rik Garrelts, Matthijn Thijert, Stijn Ydel

## Git Repository
https://github.com/OdessaA/Project3_Deel1

## Preview (aanbevolen)
ctrl+shift+v (eerst op dit bestand klikken met cursor) voor markdown preview -> Aanbevolen

## Requirements

IN dit project wordt Python gebruikt met verschillende libraries voor data understanding, data preparation, moddeling en evaluation

Benodigdheden

Python 3.10.6

De packages uit requirements.txt

## Deel 1

## Informative Markdown Table

|File                                                |Omschrijving                                         |
|----------------------------------------------------|-----------------------------------------------------|
|datasets                                            |                                                     |
|`weerdata_heino.csv`                                |Weerdata uit Heino                                   |
|`weerdata_info.txt`                                 |Informatie over de kolommen in weerdata_heino.csv    |
|`weerdata_schoon.csv`                               |De opgeschoonde weerdata uit Heino                   |
|datasets iecon                                      |                                                     |
|`_MACOSX`                                           |De dataset van de opdracht gever                     |
|`datasets iecon`                                    |De dataset van de opdracht gever                     |
|exploration                                         |                                                     |
| `00_Dataset_structuur.ipynb`                       |De structuur van de data set                         |
| `01_DataVoorbereiding.ipynb`                       |Het opsplitsen van de data set op basis van locatie  |
| `02_Huis.ipynb`                                    |Algemene controle van de huis data                   |
| `03_Middenhuis.ipynb`                              |Algemene controle van de middenhuis data             |
| `04_Carport.ipynb`                                 |Algemene controle van de carport data                |
| `05_Weerdata.ipynb`                                |Algemene controle van de weer data                   |
| `06_Dataprep.ipynb`                                |De data prep van de data                             |
| `07_Impute_tests.ipynb`                            |Het testen van de impute data                        |
| `08_Weermerge.ipynb`                               |Het samenvoegen van de hoofd data met de weer data   |
|Locatie_data                                        |                                                     |
| `Carport.parquet`                                  |Data bestand met de data uit de Carport              |
| `huis788.parquet`                                  |Data bestand met de data uit het huis                |
| `middenhuis479.parquet`                            |Data bestand met de data uit het middenhuis          |
|Models                                              |                                                     |
|&nbsp;&nbsp;baseline/                               |                                                     |
|&nbsp;&nbsp;`15_Persistence.ipynb`      |Persistence (waarde van gisteren)                                |
|&nbsp;&nbsp;`arima_grid.csv`            |De waarden van p,q,P,Q voor arima met bijbehorende AIC           |
|&nbsp;&nbsp;`Base_Impute.ipynb`         |Het imputen van de eerste data voor de baseline                  |
|&nbsp;&nbsp;`Baseline_SARIMA_vMS.ipynb` |De baseline Sarima( eerste baseline)                             |
|&nbsp;&nbsp;`basic_impute_data.gzip`    |De opslag van de geimpute data voor de baseline                  |
|&nbsp;&nbsp;`sarima_grid.csv`           |De opslag van de geimpute data voor de baseline                  |
|&nbsp;&nbsp;LightGBM/                               |                                                     |
|&nbsp;&nbsp;`11_LightGBM_huis_cons.ipynb`|LightGBM model voor de forecast van consumption huis            |
|&nbsp;&nbsp;`11b_LightGBM_huis_gen.ipynb`|LightGBM model voor de forecast van generation huis             |
|&nbsp;&nbsp;lightning_logs/              |Worden automatisch gegenereerd bij het runnen van het Patch model|
|&nbsp;&nbsp;`version_0`                              |                                                    | 
|&nbsp;&nbsp;`version_1`                              |                                                    |
|&nbsp;&nbsp;`version_2`                              |                                                    |
|&nbsp;&nbsp;`version_3`                              |                                                    |
|&nbsp;&nbsp;LSTM_en_GRU/                             |                                                    |
|&nbsp;&nbsp; carport_model/                          |                                                    |
|&nbsp;&nbsp;`gru_consumption_carport.ipynb`| GRU model voor de forecast van consumption van carport       |
|&nbsp;&nbsp;`gru_generation_carport.ipynb`|GRU model voor de forecast van generation van carport          |
|&nbsp;&nbsp;`lstm_consumption_carport.ipynb`|LSTM model voor de forecast van consumption van huis         |
|&nbsp;&nbsp;`lstm_generation_carport.ipynb`|LSTM model voor de forecast van generation van huis           | 
|&nbsp;&nbsp; huis_modellen                          |                                 |
|&nbsp;&nbsp;`gru_consumption_huis_weer.ipynb`|GRU model voor de forecast consumption met weer data voor huis|
|&nbsp;&nbsp;`gru_consumption_huis.ipynb`|GRU model voor de forecast consumption voor huis                   |
|&nbsp;&nbsp;`gru_generation_huis .ipynb`|GRU model voor de forecast generation voor huis                    |
|&nbsp;&nbsp;`gru_generation_huis_weer_mlp.ipynb`|GRU model voor de generation consumption met weer data en een mlp erachter voor huis|
|&nbsp;&nbsp;`gru_generation_huis_weer.ipynb`|GRU model voor de forecast generation met weer data voor huis  |
|&nbsp;&nbsp;`lstm_consumption_huis.ipynb`|LSTM model voor de forecast consumption voor carport              |
|&nbsp;&nbsp;`lstm_generation_huis_weer_mlp.ipynb`|LSTM model voor de forecast generation met weer data em een mlp erachter voor carport|
|&nbsp;&nbsp;`lstm_generation_huis_weer.ipynb`|LSTM model voor de forecast generation met weer data voor carport|
|&nbsp;&nbsp;`lstm_generation_huis.ipynb`|LSTM model voor de forecast generation voor carport                   |
|&nbsp;&nbsp; PatchTST/                              |                                                          |
|&nbsp;&nbsp;`12_NHITS_PatchTST_huis_cons.ipynb`                |N-HITS en Patch_TST model voor de forecast consumption van huis|
|&nbsp;&nbsp; results/                               |                                                                       |
|&nbsp;&nbsp;`10a_sarimax_huis_generation.csv`       |De resulateten van de sarimax op generation van huis                   |
|&nbsp;&nbsp;`10b_sarimax_carport_generation.csv`    |De resulateten van de sarimax op generation van carport                |
|&nbsp;&nbsp;`10c_sarimax_huis_consumption.csv`      |De resulateten van de sarimax op consumption van huis                  |
|&nbsp;&nbsp;`10d_sarimax_carport_consumption.csv`   |De resulateten van de sarimax op consumption van carport               |
|&nbsp;&nbsp;`11_lightgbm_huis_cons.csv`             |De resulateten van de LightGBM op consumption van huis                 |
|&nbsp;&nbsp;`11b_lightgbm_huis_gen.csv`             |De resulateten van de LightGBM op generation van huis                  |
|&nbsp;&nbsp;`12_nhits_patchtst_huis_cons.csv`       |De resulateten van de N_HITS_Patch model op consumption van huis       |
|&nbsp;&nbsp;`13_timesfm_huis_cons.csv`              |De resulateten van de times_fm model op consumption van huis           |
|&nbsp;&nbsp;`15_persistence_huis_cons.csv`          |De resulateten van het persistence model, de waarden van een dag eerder|
|&nbsp;&nbsp;`15_persistence_huis_gen.csv`           |De resulateten van het persistence model, de waarden van een dag eerder|
|&nbsp;&nbsp;`baseline_huis_cons.csv`                |De resulateten van de baseline op consumption van huis                 |
|&nbsp;&nbsp;`gru_consumption_huis_weer.csv`         |De resulateten van de GRU op consumption van huis met weer data        |
|&nbsp;&nbsp;`gru_consumption_huis.csv`              |De resulateten van de GRU op consumption van huis                      |
|&nbsp;&nbsp;`gru_generation_huis_weer_mlp.csv`      |De resulateten van de GRU meet een mlp op generation van huis met weer data|
|&nbsp;&nbsp;`gru_generation_huis_weer.csv`          |De resulateten van de GRU op generation van huis met weer              |
|&nbsp;&nbsp;`lstm_generation_huis_weer_mlp.csv`     |De resulateten van de LSTM met een mlp op generation van huis met weer data|
|&nbsp;&nbsp;`lstm_generation_huis_weer.csv`         |De resulateten van de LSTM op generation van huis met weer data        |
|&nbsp;&nbsp;SARIMAX                                 |                                                                       |
|&nbsp;&nbsp;`10a_Huis_Generation_SARIMAX.ipynb`     |Sarimax model op generation van huis                                   |
|&nbsp;&nbsp;`10b_Carport_Generation_SARIMAX.ipynb`  |Sarimax model op generation van carport                                |
|&nbsp;&nbsp;`10c_Huis_Consumption_SARIMAX.ipynb`    |Sarimax model op consumption van huis                                  |
|&nbsp;&nbsp;`10d_Carport_Consumption_SARIMAX.ipynb` |Sarimax model op consumption van carport                               |
|&nbsp;&nbsp;`Train-Test-Val--Split.ipynb`           |train-test split voor de Sarimax modellen                              |
|&nbsp;&nbsp; TimesFM                                |                                                                       |
|&nbsp;&nbsp;`13_TimesFM_huis_cons.ipynb`            |TimesFM model op consumption van huis                                  |
|&nbsp;&nbsp; Vergelijking                           |                                                                       |
|&nbsp;&nbsp;`14_Vergelijking.ipynb`                 |Vergelijking van alle genoemde modellen                                |
|prepared_data                                       |                                                                       |
|`data_preparation_overzicht.parquet`                |Overzicht van de data preparatie                                       |
|`forecast_targets_long.parquet`                     |De geprepareerde data                                                  |
|`forecast_targets_with_weather.parquet`             |De geprepareerde data + de weer data                                   |
|`prepared_energy_long.parquet`                      |De geprepareerde data                                                  |
|`testdata_impute_carport`                           |De geimpute data van carport                                           |
|`testdata_impute_huis`                              |De geimpute data van huis                                              |
|`weather_features.parquet`                          |De geprepareerde weer dataset                                          |
|Streamlit                                           |                                                                       |
|`15_streamlit.py`                                   |Streamlit waarin het dashboard word gemaakt                            |
## Aanbevolen gebruik
1. Installeer bovenstaande requirements.  

2. Zorg dat `datasets iecon` in je omgeving staat en run notebook `exploration\01_DataVoorbereiden.ipynb` om de drie verschillende Parquet bestanden in het mapje Locatie_data te krijgen.

3. Open en run `exploration\02_Huis.ipynb` met de verkregen `locatie_data\huis788.parquet` dataset, `exploration\03_Middenhuis.ipynb` met de verkregen `locatie_data\middenhuis479.parquet` dataset en de `exploration\04_Carport.ipynb` met de verkregen `locatie_data\carport.parquet` dataset. Voor de Data Exploration van de data.

4. Open en run `exploration\05_Weerdata.ipynb` met de weerdataset: `datasets\weerdata_heino.csv` om vervolgens `datasets\weerdata_schoon.csv` te creeëren.

5. Open en run `exploration\06_Dataprep.ipynb` met de verkregen datasets uit stap 2. Om de Geprepareerd datasets te creeëren.

6. Open en run `exploration\08_Weermerge.ipynb` met de verkregen dataset om de originele dataset en de weer data te mergen tot 1 bestand.

7. Je hebt nu alle databestanden om de modellen te kunnen runnen!!!!!!!!!!!!!!!

8. Om alle gemaakte modellen met elkaar te vergelijken run je `models\Vergelijking\14_Vergelijking.ipynb`
