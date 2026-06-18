# ASSUMPTIONS

Pressupostos sobre dados e modelo (pedido pela skill). **TODO: preencher/validar com o EDA.**

## Dados
- **Raw:** `data/01_raw/portugal_listings.csv` — 135.536 linhas, 25 colunas.
- **Target:** `Price` em **EUR** (regressão). Tem ~0.2% nulos -> linhas sem target são
  **descartadas na ingestion** (o target tem de existir). TODO: confirmar moeda/escala no EDA.
- **`PublishDate`:** **~78% nula** (só ~21% têm data, quase todas de out/2024). Por isso:
  - **NÃO** se usa um split temporal para `ref_data`/`ana_data` — usa-se um **split aleatório
    reprodutível** (`seed`, `ingestion.reference_fraction=0.8`).
  - Como `event_time` na feature store fica reservado para a Fase 5 (com a ressalva dos nulos).
- **`ref_data` / `ana_data`:** gerados por `ingestion.split_reference_analysis` (ref = baseline
  de treino; ana = "batch novo" para drift + inferência). TODO (Fase 3): injetar drift
  artificial em `ana_data` para demonstrar deteção.
- **Outliers conhecidos:** existe pelo menos um preço ~1.38 bilião € e áreas negativas/impossíveis
  (ver EDA) — tratados em `preprocessing_train.clean_data`. TODO: documentar limites aplicados.
- **Missing:** nulos MUITO elevados em várias colunas (ConservationStatus 86%, BuiltArea 80%,
  GrossArea 80%, Floor 79%, LotSize 71%, NumberOfBedrooms 65%...). Estratégia de imputação —
  ou drop de colunas demasiado vazias — em `parameters_preprocessing.yml`. TODO: justificar.
- **Skew do target:** EDA mostra skew forte -> testar `log1p(Price)` (`use_log_target`).
  TODO: registar comparação com/sem log.

## Features
- Conjunto de colunas esperado em `parameters.yml:raw_columns`.
- Partilha numérico/categórico em `parameters.yml` **confirmada contra os dtypes do raw**
  (numéricas = float64; categóricas = object). 
- **Alta cardinalidade:** `Town`/`City` têm muitos valores distintos -> one-hot direto explode.
  TODO (Fase 1): agrupar/target-encode ou usar só `District`.
- **Booleanos como `object`:** `HasParking`/`Garage`/`Elevator`/`ElectricCarsCharging` vêm como
  texto (True/False/NaN). TODO (Fase 1): normalizar para bool.
- Features disponíveis em **inferência** = mesmas do treino (sem o target). TODO: confirmar
  que o batch novo traz todas as colunas necessárias.

## Modelo
- Candidatos: RandomForest / GradientBoosting (+ XGBoost/LightGBM opcional).
- Métrica de seleção: **RMSE** (menor = melhor). Baseline = média do `Price`.
- Champion/challenger gerido via **MLflow Model Registry**.

## Limitações / riscos (ver também secção "risks & mitigations" do relatório)
- **Pandas-only:** não escala para dados muito maiores -> Spark/Polars como mitigação.
- **Hopsworks externo:** depende de API key + serviço cloud; `to_feature_store=false` permite
  correr a partir dos CSVs locais.
- **Optuna `n_trials` limitado por tempo:** trade-off qualidade vs duração.
- **Drift artificial** usado para demonstrar deteção; em produção seria contínuo.
