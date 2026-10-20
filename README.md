# Project 3: Energie Forecasting / Management SERENE project 
# Odessa Al_Did, Cas Derksen, Rik Garrelts, Matthijn Thijert, Stijn Ydel

## Git Repository
https://github.com/OdessaA/Project3_Deel1

## Preview (aanbevolen)
ctrl+shift+v (eerst op dit bestand klikken met cursor) voor markdown preview -> Aanbevolen

## Requirements
alembic==1.20.0
asttokens==3.0.2
cloudpickle==3.1.2
colorama==0.4.6
colorlog==6.12.0
comm==0.2.3
contourpy==1.3.2
cramjam==2.11.0
cycler==0.12.1
debugpy==1.8.21
decorator==5.3.1
exceptiongroup==1.3.1
executing==2.2.1
fastparquet==2026.5.0
filelock==4.0.12
fonttools==4.65.0
formulaic==1.2.2
fsspec==2026.7.0
greenlet==3.5.6
interface_meta==2.0.1
ipykernel==7.3.0
ipython==8.39.0
jedi==0.20.0
Jinja2==3.1.6
joblib==1.6.0
jupyter_client==8.10.0
jupyter_core==5.9.1
kiwisolver==1.5.1
lightgbm==4.7.0
Mako==1.4.3
MarkupSafe==3.0.4
matplotlib==3.10.9
matplotlib-inline==0.2.2
mpmath==1.3.0
narwhals==2.26.0
nest-asyncio2==1.7.2
networkx==3.4.2
numpy==2.2.6
optuna==5.0.0
packaging==26.3
pandas==2.3.3
parso==0.8.7
patsy==1.0.3
pillow==12.3.0
platformdirs==4.11.8
prompt_toolkit==3.0.53
psutil==7.2.2
pure_eval==0.2.4
pyarrow==25.0.1
Pygments==2.21.0
pyparsing==3.3.3
python-dateutil==2.9.0.post0
pytz==2026.3.post1
PyYAML==6.0.3
pyzmq==27.2.0
scikit-learn==1.7.2
scipy==1.15.3
seaborn==0.13.2
six==1.17.0
SQLAlchemy==2.0.54
stack-data==0.6.3
statsmodels==0.15.0
sympy==1.14.0
threadpoolctl==3.7.0
tomli==2.5.0
torch==2.14.1+cu126
torchvision==0.29.1+cu126
tornado==6.5.8
tqdm==4.70.1
traitlets==5.16.1
typing_extensions==4.16.0
tzdata==2026.4
wcwidth==0.8.3
wrapt==2.5.0


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
|&nbsp;&nbsp;`14_Vergelijking.ipynb`                  Vergelijking van alle genoemde modellen                                |
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
Je hebt nu alle databestanden om de modellen te kunnen runnen!!!!!!!!!!!!!!!



7. Om alle gemaakte modellen met elkaar te vergelijken run je `models\Vergelijking\14_Vergelijking.ipynb`


Tot hier ben ik gekomen 



7. Gebruik alle 12 `2025-xx_rws_filedata.csv` datasets en run vervolgens notebook `06_File_EDA_Processing.ipynb`.

8. Open en run `07_UitschietersAnalyse.ipynb` met verkregen `train_weather.parquet`, `val_weather.parquet` en `test_weather.parquet`.

9. Gebruik `Wegentabel.csv` en run `08_Mergen_Files.ipynb` met verkregen `FileData.csv`, `train_clean.parquet`, `test_clean.prquet` en `val_clean.parquet`.

10. Gebruik `Nederland.csv.zst` en run `09_Afstandchecker.ipynb` met verkregen `train_merged.parquet`.

11. Gebruik `disruptions-2025.csv` en run `TreinStoringenEDA.ipynb`.

12. Open en run `Trein_invloed.ipynb` met verkregen `FileData.csv` en `trajecten_overijssel.csv`.

13. Open en run `01_DecisionTreeRegressor.ipynb` met verkregen `train_merged.parquet`, `test_merged.parquet` en `val_merged.parquet`.

14. Open en run `02_RandomForestRegressor.ipynb` met verkregen `train_merged.parquet`, `test_merged.parquet` en `val_merged.parquet`.

15. Open eindmodel! `03_XGBoost.ipynb` en gebruik vervolgens de verkregen `test_merged.parquet`, `train_merged.parquet` en `val_merged.parquet` om voorspellingen te doen.

16. Voor de evaluatie in notebook `04_Evaluatie.ipynb`, gebruik verkregen `xgb_eindresultaten.csv`, `xgb_evaluatie_test.parquet` en `xgb_feature_importance.csv`.

Bronnenlijst
Filedata - https://downloads.rijkswaterstaatdata.nl/filedata/
Treindata - https://www.rijdendetreinen.nl/open-data/treinstoringen
Wijkdata - https://www.cbs.nl/nl-nl/dossier/nederland-regionaal/geografische-data/wijk-en-buurtkaart-2025
Weerdata - https://www.daggegevens.knmi.nl/klimatologie/uurgegevens
