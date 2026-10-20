# Project 3: Energie Forecasting / Management SERENE project 
Data Project 2 modaliteit  - Oscar Herder, Rik Garrelts, Cas Derksen, Omar Dewidar

## Git Repository


## Preview (aanbevolen)
ctrl+shift+v (eerst op dit bestand klikken met cursor) voor markdown preview -> Aanbevolen

## Requirements
- Python 3.10.6
- Jupyter Notebook
- Packages:
  - pandas 2.3.3
  - matplotlib 3.10.8
  - seaborn 0.13.2
  - numpy 2.2.6
  - geopandas 1.1.3
  - duckdb 1.5.1
  - scipy 1.15.3
  - scikit-learn 1.7.2
  - xgboost 3.2.0
  - pyarrow
  - fastparquet
  - zstandard
  - openpyxl
  - xml
  - gzip
  - re 
  - warnings

## Deel 1

## Informative Markdown Table

|File                                                |Omschrijving                                         |
|----------------------------------------------------|-----------------------------------------------------|
|RittenData                                          |                                                     |
|`ruwe_data.parquet`                                 |Dataset ruwe data                                    |
|`datatrain.parquet`                                 |Dataset traindata 76%                                |
|`test.parquet`                                      |Dataset testdata 5%                                  |
|`validate.parquet`                                  |Dataset validatiedata 19%                            |
|TreinData                                           |                                                     |
|`disruptions-2025.csv`                              |Treinstoringdata uit 2025                            |
|`trajecten_overijssel.csv`                          |Treinstoringdata uit 2025 (gefilterd op Overijssel)  |
|FileData                                            |                                                     |
| `2025-xx_rws_filedata.csv`                         |FileData uit 2025 (xx = 01-12 datasets)              |
| `FileData.csv`                                     |Samengevoegde FileDatasets uit 2025                  |
| `files_ov.csv`                                     |FileData uit 2025 (gefilterd op Overijssel)          |
|NederlanseAdressen                                  |                                                     |
| `Nederland.csv.zst`                                |Adresdata uit heel Nederland                         |
| `'Provincie'.csv.zst`                              |Adresdata uit verschillende provincies               |
|Weerdata                                            |                                                     |
| `heino.csv`                                        |Weerdata uit Heino                                   |
| `hoogeveen.csv`                                    |Weerdata uit Hoogeveen                               |
| `marknesse.csv`                                    |Weerdata uit Marknesse                               |
| `twenthe.csv`                                      |Weerdata uit Twente                                  |
| `info_weer.txt`                                    |Tekstinfobestand weerdata                            |
| `info_overijssel_weerdata.txt`                     |Tekstinfobestand weerdata (Overijssel)               |
|WijkBuurtData                                       |                                                     |
| `wijkenbuurten_2025_v1.gpkg`                       |Wijkdataset uit 2025                                 |
|Exploration Notebooks                               |                                                     |
| `01_EDA.ipynb`                                     |Exploratory Data Analysis notebook                   |
| `02_Data_Preperation.ipynb`                        |Data Preperation notebook                            |
| `03_eda_overijssel.ipynb`                          |EDA notebook (Overijssel)                            |
| `04_Wijkcodes.ipynb`                               |Wijkcodes notebook                                   | 
| `05_WeerData_Processing.ipynb`                     |Weerdata notebook                                    |
| `06_File_EDA_Processing.ipynb`                     |File notebook (EDA)                                  |
| `07_UitschietersAnalyse.ipynb`                     |Uitschieters notebook                                |
| `08_Mergen_Files.ipynb`                            |File notebook (merge)                                |
| `09_Afstandchecker.ipynb`                          |Afstandcheck notebook                                |
| `TreinStoringenEDA.ipynb`                          |Treinstoringen notebook (EDA)                        |
| `Trein_invloed.ipynb`                              |Treinstoringen notebook (invloed)                    |
|Processed! Data                                     |                                                     |
| `clean_weerdata.parquet`                           |Weerdata clean                                       |
| `train_processed.parquet`                          |Traindata processed                                  |
| `test_processed.parquet`                           |Testdata processed                                   |
| `val_processed.parquet`                            |valdata processed                                    |
| `test_clean.parquet`                               |Testdata clean                                       |
| `test_geo.parquet`                                 |Testdata geo                                         |
| `test_merged.parquet`                              |Testdata merged                                      |
| `test_overijssel.parquet`                          |Testdata Overijssel                                  |
| `test_weather.parquet`                             |Testdata weer                                        |
| `train_clean.parquet`                              |Traindata clean                                      |
| `train_geo.parquet`                                |Traindata geo                                        |
| `train_merged.parquet`                             |Traindata merged                                     |
| `train_overijssel.parquet`                         |Traindata Overijssel                                 |
| `train_weather.parquet`                            |Traindata weer                                       |
| `val_clean.parquet`                                |Validatiedata clean                                  |
| `val_geo.parquet`                                  |Validatiedata geo                                    |
| `val_merged.parquet`                               |Validatiedata merged                                 |
| `val_overijssel.parquet`                           |Validatiedata Overijssel                             |
| `val_weather.parquet`                              |Validatiedata weer                                   |
| `Wegentabel.csv`                                   |Origin-Destination wegentabel                        |
|Models                                              |                                                     |
| `oud_DecisionTreeRegressor.ipynb`                  |DTR model (oud)                                      |
| `oud_RandomForestRegressor.ipynb`                  |RFR model (oud)                                      |
| `01_DecisionTreeRegressor.ipynb`                   |DTR model                                            |
| `02_RandomForestRegressor.ipynb`                   |RFR model                                            |
| `03_XGBoost.ipynb`                                 |XGB model                                            |
|Evaluatie                                           |                                                     |
| `04_Evaluatie.ipynb`                               |Evaluatie eindmodel (XGB)                            |
| `xgb_eindresultaten.csv`                           |Eindresultaten xgb                                   |
| `xgb_evaluatie_test.parquet`                       |Evaluatie test xgb                                   |
| `xgb_feature_importance.csv`                       |Feature importance xgb                               |
|Informatie overig                                   |                                                     |
| `README.md`                                        |Readme file                                          |


## Aanbevolen gebruik
1. Installeer bovenstaande requirements.  

2. Gebruik `ruwe_data.parquet` en run notebook `01_EDA.ipynb` om vervolgens de train-test-val split aan te maken van de ruwe dataset.

3. Open en run `02_Data_Preperation.ipynb` met verkregen `datatrain.parquet`, `test.parquet` en `validate.parquet` datasets.

4. Open en run `03_eda_overijssel.ipynb` met verkregen `train_processed.parquet`, `test_processed.parquet` en `val_processed.parquet` datasets.

5. Gebruik `wijkenbuurten_2025_v1.gpkg` en run vervolgens notebook `04_Wijkcodes.ipynb` met verkregen `train_overijssel.parquet`, `test_overijssel.parquet` en `val_overijssel.parquet` datasets.

6. Gebruik `wijkenbuurten_2025_v1.gpkg`, `heino.csv`, `hoogeveen.csv`, `marknesse.csv` en `twenthe.csv` en run vervolgens notebook `05_WeerData_Processing.ipynb` met verkregen `train_geo.parquet`, `test_geo.parquet` en `val_geo.parquet`.

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
