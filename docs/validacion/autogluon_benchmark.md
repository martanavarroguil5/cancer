# Benchmark AutoGluon y techo practico del dataset

## Objetivo

Se ejecuto AutoGluon como benchmark externo para comprobar si un AutoML tabular
moderno encontraba senal predictiva no capturada por el pipeline propio. Tras
recuperar `CASOCANCER_04_ECONOMICOS.csv`, el benchmark separa tres escenarios:

- `base`: modelo limpio principal, sin variables economicas ni proxies debiles.
- `safe_all`: variables prediagnostico ampliadas, incluyendo `tipo_seguro`, pero
  excluyendo costes/uso hospitalario.
- `economic_sensitivity`: incluye costes/uso hospitalario solo para cuantificar
  fuga temporal; no es un candidato operativo.

## Protocolo

- Mismo dataset unido por `paciente_id`: 50.001 filas y 38 columnas desde 6 CSV.
- Misma semilla `42`.
- Mismo split limpio: `train_inner` 32.000 filas, validacion 8.000 filas y test
  sellado 10.001 filas.
- `vive`, `paciente_id`, `cancer` y `alcohol` excluidos en todas las vistas.
- `coste_total`, `coste_farmaco`, `num_ingresos` y `dias_hospital` excluidos en
  `base` y `safe_all`; incluidos solo en `economic_sensitivity`.
- AutoGluon uso `TabularPredictor(label="cancer", eval_metric="f1")`,
  `presets="best_v150"`, bagging con `use_bag_holdout=True` y calibracion de
  umbral en validacion.
- La metrica final reportada usa el mismo barrido de umbral del proyecto sobre
  validacion (`0.10` a `0.90`, paso `0.01`) y aplica ese umbral una sola vez en
  test.

Las salidas completas se generaron en `outputs/autogluon/`, directorio ignorado
por git por tamano y reproducibilidad.

## Reproduccion

AutoGluon se instala en un entorno separado para no alterar el entorno principal
TensorFlow/CUDA:

```bash
python3 -m venv .venv-autogluon
source .venv-autogluon/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-autogluon.txt
```

Comandos ejecutados:

```bash
.venv-autogluon/bin/python scripts/validacion/run_autogluon_benchmark.py \
  --feature-view base \
  --preset best_v150 \
  --time-limit 900 \
  --overwrite

.venv-autogluon/bin/python scripts/validacion/run_autogluon_benchmark.py \
  --feature-view safe_all \
  --preset best_v150 \
  --time-limit 600 \
  --overwrite

.venv-autogluon/bin/python scripts/validacion/run_autogluon_benchmark.py \
  --feature-view economic_sensitivity \
  --preset best_v150 \
  --time-limit 300 \
  --overwrite
```

## Resultados

| Modelo | Vista | Umbral validacion | Precision cancer=1 | Recall cancer=1 | F1 cancer=1 | AUC-ROC | AUC-PR |
|---|---|---:|---:|---:|---:|---:|---:|
| HistGradientBoosting propio | base | 0.60 | 0.499 | 0.660 | **0.568** | 0.830 | 0.579 |
| AutoGluon `best_v150` | base | 0.48 | 0.546 | 0.555 | 0.550 | 0.830 | 0.582 |
| AutoGluon `best_v150` | safe_all | 0.48 | 0.571 | 0.558 | 0.564 | 0.844 | 0.614 |
| AutoGluon `best_v150` | economic_sensitivity | 0.40 | 0.993 | 0.967 | 0.980 | 0.998 | 0.997 |

Detalle de AutoGluon:

- En `base`, el mejor modelo fue `WeightedEnsemble_L2`, con peso completo en
  `NeuralNetTorch_r37_BAG_L1`.
- En `safe_all`, el mejor modelo fue `WeightedEnsemble_L2`, combinando
  `NeuralNetTorch_r37_BAG_L1` y `NeuralNetTorch_r144_BAG_L1`.
- En `economic_sensitivity`, el mejor ensemble combino `CatBoost_c1_BAG_L1` y
  `LightGBMPrep_r41_BAG_L1`, alcanzando metricas casi perfectas.
- Las variantes `NeuralNetFastAI` saltaron por `ImportError` en los workers de
  AutoGluon en este entorno; CatBoost, LightGBM, XGBoost y NeuralNetTorch si se
  entrenaron.

## Lectura

AutoGluon no encontro una familia de modelos ni un ensemble limpio que superase
claramente al baseline propio en aquel benchmark historico con F1. Tras cambiar
el objetivo operativo a F1/recall, esta comparativa queda como evidencia
auxiliar: `safe_all` mejora AUC-ROC y AUC-PR, y el modelo recomendado actual se
elige por sensibilidad validada sin incluir variables de fuga temporal.

La sensibilidad economica es la senal mas importante del nuevo CSV: pasar de F1
`0.564` en `safe_all` a F1 `0.980` al permitir costes/uso hospitalario es
compatible con fuga temporal o variables posteriores al diagnostico/tratamiento.
Por tanto, el CSV 04 se audita y se considera en el pipeline, pero sus variables
de coste y utilizacion no deben alimentar el modelo operativo de cribado sin una
definicion temporal externa.
