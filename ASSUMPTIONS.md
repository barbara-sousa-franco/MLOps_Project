# ASSUMPTIONS

Pressupostos sobre dados e modelo (pedido pela skill). **TODO: preencher/validar com o EDA.**

## Dados
- **Target:** `Price` em **EUR** (regressão). TODO: confirmar moeda/escala no EDA.
- **`PublishDate`** usado como `event_time` na feature store. TODO: confirmar formato/coerência.
- **Outliers conhecidos:** existe pelo menos um preço ~1.38 bilião € e áreas negativas/impossíveis
  (ver EDA) — tratados em `preprocessing_train.clean_data`. TODO: documentar limites aplicados.
- **Missing:** muitas colunas têm nulls (ex: ConstructionYear, Floor, áreas). Estratégia de
  imputação em `parameters_preprocessing.yml`. TODO: justificar escolhas.
- **Skew do target:** EDA mostra skew forte -> testar `log1p(Price)` (`use_log_target`).
  TODO: registar comparação com/sem log.

## Features
- Conjunto de colunas esperado em `parameters.yml:raw_columns`.
- Partilha numérico/categórico em `parameters.yml`. TODO: validar com o EDA.
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
