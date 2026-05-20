# Decisiones de modelado

## Objetivo

La metrica principal es F1 de la clase positiva `cancer=1`. Se calcula precision,
recall, F1, AUC-ROC y accuracy en test, pero el ranking y el umbral operativo se
seleccionan por F1 en validacion interna.

## Datos y split

- Se cargan seis CSV desde `data/raw/`.
- La union se realiza por `paciente_id` con validacion `one_to_one`.
- Split estratificado 80/20 para test.
- Del 80% de train se separa una validacion interna del 20%.
- El preprocesamiento se ajusta solo en entrenamiento mediante `ColumnTransformer`.

## Politica de features

Vista final: `safe_all`.

Se excluyen siempre:

- `paciente_id`
- `cancer`
- `alcohol`
- `vive`
- `coste_total`
- `coste_farmaco`
- `num_ingresos`
- `dias_hospital`

La vista `safe_all` conserva variables prediagnostico disponibles, incluidas
comorbilidades y variables sociodemograficas. Estas variables se interpretan como
predictores disponibles, no como causa clinica. La vista `economic_sensitivity`
queda bloqueada por defecto en `scripts/run_pipeline.py` y solo se permite con
`--allow-leakage-view` para auditoria.

## Modelos

Modelos ML complejos:

- RandomForest
- ExtraTrees
- HistGradientBoosting
- HistGradientBoostingRegularized
- GradientBoosting
- XGBoostF1Balanced
- XGBoost
- XGBoostAUC

Baseline:

- LogisticRegression con `class_weight="balanced"`.

MLP:

- Dense(256) -> Dense(128) -> Dense(64)
- Activacion SELU
- Dropout 0.08 / 0.06 / 0.04
- L2 ligera
- AdamW
- `class_weight`
- EarlyStopping y ReduceLROnPlateau monitorizando `val_f1_best`

## Resultado final

Ejecucion:

```bash
source scripts/activate_cuda.sh
python scripts/run_pipeline.py --mode full --seed 42 --feature-view safe_all
```

Ranking final en test:

| Modelo | Umbral | Precision | Recall | F1 | AUC-ROC | AUC-PR |
|---|---:|---:|---:|---:|---:|---:|
| XGBoostAUC_cuda | 0.66 | 0.578 | 0.591 | **0.584** | 0.848 | 0.624 |
| ValidationSoftVoting | 0.52 | 0.549 | 0.625 | 0.584 | 0.847 | 0.622 |
| XGBoostF1Balanced_cuda | 0.67 | 0.578 | 0.586 | 0.582 | 0.848 | 0.623 |
| GradientBoosting | 0.28 | 0.539 | 0.628 | 0.580 | 0.846 | 0.622 |
| HistGradientBoostingRegularized | 0.63 | 0.532 | 0.637 | 0.580 | 0.843 | 0.610 |
| MLP | 0.64 | 0.520 | 0.620 | 0.565 | 0.840 | 0.607 |

Modelo tecnico recomendado: `XGBoostAUC_cuda`.

Opcion clinicamente interesante para cribado: `ValidationSoftVoting`, porque
mantiene practicamente el mismo F1 (`0.584`) y aumenta el recall de `0.591` a
`0.625`, recuperando 65 positivos adicionales en test a costa de mas falsos
positivos.

## Interpretacion

Los modelos de boosting son los mas adecuados para estos datos tabulares. La MLP
cumple el enunciado y aprende senal util, pero queda por debajo del mejor ML en
F1 de test.

En contexto medico no se recomienda una lectura binaria rigida. La presentacion
final propone dos zonas de uso:

- `>= 0.66`: alto riesgo con buena precision.
- `0.43 - 0.66`: zona gris para revision clinica o prueba complementaria.

La viabilidad tecnica es positiva para un prototipo de ranking de riesgo, no para
una implantacion clinica autonoma sin validacion externa.

## Limitaciones

- Dataset sintetico.
- Falta validacion temporal y externa.
- Riesgo de sesgo por variables sociodemograficas y comorbilidades.
- Las variables de coste, uso hospitalario y estado vital se excluyen por fuga.
- Accuracy no debe guiar la decision por el desbalance de clases.
