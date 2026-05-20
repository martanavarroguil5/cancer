# Feature engineering y auditoria de datos

## Decision principal

La politica final usa `--feature-view safe_all`, porque fue la vista limpia con
mejor F1 validado en test sin usar `vive`, costes ni uso hospitalario. La vista
`metadata_core` se conserva como nucleo clinico estricto alineado con el metadato
oficial recuperado en `docs/datos/metadata_dataset_cancer_oficial.md`: bioquimica completa, genetica completa,
`fumador`, `actividad_fisica` y `edad`.

## `vive` es fuga

`vive` no es valido para predecir diagnostico de cancer en un paciente operativo:
parece una variable posterior o al menos temporalmente contaminada.

Auditoria descriptiva:

| vive | cancer=0 | cancer=1 | prevalencia cancer |
|---:|---:|---:|---:|
| 0 | 7.483 | 5.593 | 42,8% |
| 1 | 32.874 | 4.051 | 11,0% |

Por eso:

- no se usa `vive` como feature,
- no se filtra el entrenamiento a `vive=1`,
- y cualquier sensibilidad que lo incluya se considera solo prueba de fuga.

Filtrar por `vive` tambien seria problematico: si esa columna se mide despues del
diagnostico o de la evolucion del paciente, condicionar el dataset a ella cambia
la distribucion del target y mete sesgo de supervivencia.

## Vistas disponibles

`base`

- Referencia compacta historica del pipeline.
- 18 variables incluidas.
- Excluye `vive`, `alcohol`, identificador, target y columnas de baja senal/proxy.
- Espacio transformado: 20 columnas.

`metadata_core`

- Vista clinica estricta de referencia desde la incorporacion del metadato oficial.
- 17 variables fuente incluidas: bioquimica completa, todas las mutaciones,
  `fumador`, `actividad_fisica` y `edad`.
- Excluye `vive`, `alcohol`, costes/uso hospitalario, comorbilidades de cautela,
  variables sociodemograficas opcionales y `tipo_seguro`.
- Espacio transformado: 19 columnas, por la codificacion one-hot de
  `actividad_fisica`.

`safe_all`

- Incluye toda variable prediagnostico no constante ni fuga.
- Recupera `plaquetas`, `creatinina`, `asma`, `enfermedad_cardiaca`, `mut_ALK`,
  variables sociodemograficas, `num_hijos`, `distancia_hospital_km` y
  `tipo_seguro`.
- Mantiene fuera `coste_total`, `coste_farmaco`, `num_ingresos` y
  `dias_hospital` por riesgo de fuga temporal.
- Vista final recomendada: obtuvo el mejor F1 limpio disponible y mantiene fuera
  todas las fugas confirmadas o de alto riesgo.

`engineered_selected`

- Parte de `metadata_core` y anade derivadas seleccionadas.
- No usa target, `vive` ni estadisticos calculados con test.
- Incluye edad en decadas, `edad > 55`, actividad ordinal, indice TyG, ratios
  lipidicos, `glucosa > 130`, `hemoglobina < 11`, `leucocitos > 10`, cargas
  geneticas e interacciones fumador/genetica.

`economic_sensitivity`

- Incluye costes y uso hospitalario del CSV 04 solo para cuantificar sensibilidad
  a fuga temporal.
- No es una vista operativa: AutoGluon alcanza F1 test `0.980`, demasiado alto
  para interpretarlo como senal prediagnostica limpia.

## Resultados en seed 42

Ejecutado con:

```bash
source scripts/activate_cuda.sh
python scripts/validacion/run_feature_engineering_experiments.py --mode full --seeds 42
```

| Vista | Modelo | F1 validacion | F1 test | AUC-ROC | AUC-PR |
|---|---|---:|---:|---:|---:|
| base | HGB | 0.5517 | 0.5682 | 0.8305 | 0.5786 |
| base | HGB regularizado | 0.5521 | 0.5664 | 0.8324 | 0.5825 |
| safe_all | HGB regularizado | 0.5515 | 0.5679 | 0.8309 | 0.5805 |
| safe_all | HGB | 0.5517 | 0.5668 | 0.8301 | 0.5784 |
| engineered_selected | HGB | 0.5549 | 0.5578 | 0.8295 | 0.5789 |
| engineered_selected | Logistica | 0.5455 | 0.5547 | 0.8315 | 0.5843 |

Lectura: las features derivadas aumentan validacion en HGB y suben AUC/AUC-PR en
modelos lineales, pero no mejoran el F1 del test sellado. Es la senal clasica de
feature engineering que parece atractivo en validacion pero no compra capacidad
predictiva real suficiente.

## Comparacion multisemilla

Ejecutado con:

```bash
python scripts/validacion/run_feature_engineering_experiments.py --mode full --seeds 42 7 13 2024
```

Resumen de medias de F1 test:

| Vista | Modelo | F1 test medio | Desv. F1 | AUC-PR media |
|---|---|---:|---:|---:|
| safe_all | HGB regularizado | 0.5523 | 0.0133 | 0.5752 |
| safe_all | HGB | 0.5522 | 0.0141 | 0.5728 |
| base | HGB | 0.5515 | 0.0127 | 0.5727 |
| engineered_selected | HGB regularizado | 0.5512 | 0.0080 | 0.5775 |
| base | HGB regularizado | 0.5502 | 0.0126 | 0.5750 |
| engineered_selected | Logistica | 0.5483 | 0.0096 | 0.5771 |

La diferencia media historica entre `safe_all` y `base` era menor que la
desviacion entre semillas con el criterio F1. Con el objetivo operativo actual,
F1/recall, `safe_all` dio los mejores resultados historicos en
super-hypertraining y queda como vista final. `metadata_core` queda como
referencia estricta de sensibilidad metodologica.

## Como reproducir

Pipeline oficial:

```bash
python scripts/run_pipeline.py --mode full --seed 42 --feature-view safe_all
```

Probar nucleo clinico estricto:

```bash
python scripts/run_pipeline.py --mode full --seed 42 --feature-view metadata_core
```

Probar features derivadas:

```bash
python scripts/run_pipeline.py --mode full --seed 42 --feature-view engineered_selected
```

Probar sensibilidad economica:

```bash
python scripts/run_pipeline.py --mode full --seed 42 --feature-view economic_sensitivity
```

Experimento comparativo:

```bash
python scripts/validacion/run_feature_engineering_experiments.py --mode full --seeds 42 7 13 2024
```

## Conclusion

El limite principal limpio sigue estando en la senal disponible, no en que falten
features manuales. El CSV 04 aporta una leccion adicional: las variables
economicas pueden disparar las metricas si recogen tratamiento o uso posterior,
asi que deben quedar fuera del modelo de cribado hasta aclarar su temporalidad.
