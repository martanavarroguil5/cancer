# Super hypertraining ML + MLP

## Objetivo

Este experimento es una bateria computacionalmente costosa para intentar
exprimir el mejor modelo tabular limpio y la mejor MLP usando la RTX 3090. No
sustituye al pipeline oficial: sirve para investigacion controlada y deja una
traza auditable de candidatos, umbrales y ranking final.

Reglas del experimento:

- usa `train_inner`, `valid` y `test` del split oficial;
- selecciona hiperparametros y umbrales por validacion;
- usa F1 de `cancer=1` como criterio operativo;
- reporta test sin usarlo para decidir;
- usa vistas limpias por defecto: `metadata_core`, `safe_all`, `engineered_selected`;
- bloquea `economic_sensitivity` salvo que se pase `--include-leakage-view`.

## Reproduccion

Primero activar CUDA para que TensorFlow/Keras vea la RTX 3090:

```bash
source scripts/activate_cuda.sh
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

Prueba corta validada:

```bash
python scripts/validacion/run_super_hypertraining.py \
  --profile smoke \
  --mode quick \
  --views base \
  --ml-limit 2 \
  --mlp-suite smoke \
  --mlp-limit 1 \
  --epochs 3 \
  --patience 2 \
  --reduce-patience 1 \
  --output-dir outputs/metrics/super_hypertraining_smoke
```

Pasada recomendada fuerte pero razonable:

```bash
python scripts/validacion/run_super_hypertraining.py \
  --profile rtx3090 \
  --mode full \
  --views base safe_all engineered_selected \
  --require-gpu \
  --gpu-only \
  --mlp-suite rtx3090 \
  --epochs 160 \
  --patience 18 \
  --reduce-patience 7 \
  --ensemble-top-k 8 \
  --ensemble-max-members 4 \
  --output-dir outputs/metrics/super_hypertraining_rtx3090_gpu_f1_full_v2
```

Por defecto los mejores artefactos se guardan en
`<output-dir>/models/` para que una smoke no pise modelos de otra corrida. Si se
quiere exportar explicitamente a `models/`, anadir:

```bash
--models-dir models
```

Pasada intermedia si se quiere controlar mejor el tiempo:

```bash
python scripts/validacion/run_super_hypertraining.py \
  --profile balanced \
  --mode full \
  --views base safe_all \
  --mlp-suite refine \
  --epochs 90 \
  --patience 10 \
  --reduce-patience 5 \
  --output-dir outputs/metrics/super_hypertraining_balanced
```

## Tamano de busqueda

Con seed 42:

| Perfil | Candidatos ML por vista | Candidatos MLP por vista | Uso previsto |
|---|---:|---:|---|
| `smoke` | 3 | 2 | Sanity check rapido. |
| `balanced` | 79 | 9 | Busqueda seria de duracion moderada. |
| `rtx3090` | 337 | 21 | Busqueda pesada para ejecucion larga. |

Se pueden acotar con `--ml-limit` y `--mlp-limit`.

La busqueda tambien calcula ensembles soft-voting limpios. Para ML clasico se
limita por defecto a los 12 mejores candidatos por vista (`--ensemble-top-k 12`)
para evitar una explosion combinatoria; para ahorrar tiempo se puede usar
`--skip-ensembles`.

## Salidas

El directorio elegido en `--output-dir` contiene:

- `ml_results.csv`: ranking parcial/final de modelos ML.
- `ml_ensembles.csv`: ensembles simples de los mejores modelos ML por vista.
- `mlp_results.csv`: ranking de MLPs.
- `mlp_ensembles.csv`: ensembles simples de MLPs cuando hay al menos dos
  candidatas validas.
- `final_ranking.csv`: comparacion unificada ML, MLP y ensembles.
- `summary.json` y `summary.md`: resumen auditable.
- `best_ml_marker.json` y `best_mlp_marker.json`: mejor candidato por
  validacion.

Los mejores modelos locales se guardan por defecto en `<output-dir>/models/`:

- `super_best_ml.joblib`
- `super_best_mlp.keras`
- `super_best_mlp_preprocessor.joblib`

Si se pasa `--models-dir models`, se escriben en el directorio global `models/`.

## Validacion realizada

La pasada fuerte `rtx3090` se ejecuto correctamente con TensorFlow creando el
dispositivo `GPU:0` en la NVIDIA GeForce RTX 3090 y con backends GPU para
XGBoost/LightGBM/CatBoost. Se evaluaron 672 candidatos ML, 63 MLPs y ensembles
limpios. El ranking queda gobernado por `safe_all` y F1:

| Fuente | Modelo | Vista | Umbral | F1 valid | Recall valid | Precision valid | F1 test | Recall test | Precision test |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| ML | XGBoostF1Balanced | `safe_all` | 0.38 | **0.6769** | 0.8574 | 0.3674 | 0.6794 | 0.8564 | 0.3720 |
| ML | CatBoost GPU | `safe_all` | 0.40 | 0.6758 | 0.8406 | 0.3788 | 0.6799 | 0.8429 | 0.3833 |
| MLPEnsemble | SELU+baseline+deep | `safe_all` | 0.42 | 0.6728 | 0.8354 | 0.3783 | 0.6724 | 0.8336 | 0.3792 |
| MLP | `selu_256_128_64_no_bn` | `safe_all` | 0.38 | 0.6721 | 0.8451 | 0.3695 | 0.6738 | 0.8440 | 0.3730 |
| ML | LightGBM GPU | `safe_all` | 0.35 | 0.6690 | 0.8620 | 0.3530 | 0.6708 | 0.8611 | 0.3561 |

Conclusion historica aplicada al pipeline: `safe_all` produjo el mejor ranking
limpio y motivo anadir el candidato `XGBoostF1Balanced` con los hiperparametros
ganadores. La entrega final revalidada usa `safe_all` como vista operativa y
selecciona umbrales por F1 en validacion.
