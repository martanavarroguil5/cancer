# Prediccion de Diagnostico de Cancer

Pipeline profesional y reproducible para evaluar si los datos multimodales del
caso permiten anticipar `cancer=1`. La metrica principal del proyecto es F1 de la
clase positiva; los umbrales se eligen solo en validacion y el test queda sellado
para la evaluacion final.

## Estructura

- `data/raw/`: seis CSV originales del caso, unidos por `paciente_id`.
- `docs/`: enunciado, metadata, decisiones, entorno y validaciones auxiliares.
- `src/cancer_ml/`: paquete reutilizable de carga, features, modelos, metricas,
  graficas y evaluacion.
- `scripts/run_pipeline.py`: ejecucion principal F1.
- `scripts/generate_five_slide_presentation.py`: genera la presentacion final.
- `scripts/smoke_test.py`: comprobacion rapida del proyecto.
- `scripts/validacion/`: experimentos auxiliares, incluido AutoGluon.
- `tests/`: pruebas de contrato contra leakage, split y seleccion de umbral.
- `outputs/`: metricas, figuras y PDF final regenerados localmente.
- `models/`: modelos entrenados localmente.

`outputs/`, `models/`, `tmp/` y entornos virtuales estan ignorados por git.

## Instalacion

Para desplegar el dashboard en Streamlit Cloud, usa las dependencias minimas de
la app:

```bash
pip install -r requirements.txt
```

Para regenerar el pipeline completo y entrenar modelos localmente con CUDA, usa
el entorno de entrenamiento:

```bash
source scripts/activate_cuda.sh
pip install -r requirements-training.txt
```

## Validacion

```bash
source scripts/activate_cuda.sh
python -m unittest discover -s tests -v
python scripts/smoke_test.py
```

## Ejecucion Final

El mejor resultado limpio con los datos disponibles se obtiene con `safe_all`,
que excluye `vive`, costes, uso hospitalario, identificadores, target y
constantes. Las variables de cautela se usan solo como predictores disponibles,
no como evidencia causal clinica.

```bash
source scripts/activate_cuda.sh
python scripts/run_pipeline.py --mode full --seed 42 --feature-view safe_all
```

La vista `economic_sensitivity` esta bloqueada por defecto porque incluye
variables con riesgo de fuga temporal. Solo puede ejecutarse como auditoria:

```bash
python scripts/run_pipeline.py --mode quick --feature-view economic_sensitivity --allow-leakage-view
```

## Resultado Final Validado

Split estratificado con semilla 42:

- Train: 40.000 pacientes.
- Validacion interna: 8.000 pacientes.
- Test: 10.001 pacientes.
- Prevalencia `cancer=1`: 19,29%.
- CSV cargados: 6/6.
- Vista final: `safe_all`.
- Metrica primaria: F1 de `cancer=1`.

Ranking final en test:

| Modelo | Umbral validacion | Precision | Recall | F1 | AUC-ROC | AUC-PR |
|---|---:|---:|---:|---:|---:|---:|
| XGBoostAUC_cuda | 0.66 | 0.578 | 0.591 | **0.584** | 0.848 | 0.624 |
| ValidationSoftVoting | 0.52 | 0.549 | 0.625 | 0.584 | 0.847 | 0.622 |
| XGBoostF1Balanced_cuda | 0.67 | 0.578 | 0.586 | 0.582 | 0.848 | 0.623 |
| GradientBoosting | 0.28 | 0.539 | 0.628 | 0.580 | 0.846 | 0.622 |
| HistGradientBoostingRegularized | 0.63 | 0.532 | 0.637 | 0.580 | 0.843 | 0.610 |
| MLP | 0.64 | 0.520 | 0.620 | 0.565 | 0.840 | 0.607 |

Modelo recomendado: `XGBoostAUC_cuda`. La MLP cumple el enunciado con tres capas
ocultas, Dropout, Early Stopping, ReduceLROnPlateau y `class_weight`, pero en este
dataset tabular los boosting quedan por delante.

## Artefactos

- PDF de 5 diapositivas:
  `outputs/reports/presentacion_final_cancer_5_diapositivas.pdf`
- Metricas finales:
  `outputs/metrics/model_metrics.csv`
- Resumen de ejecucion:
  `outputs/metrics/run_summary.json`
- Model card:
  `outputs/metrics/model_card.json`
- Figuras:
  `outputs/figures/`
- Modelo recomendado:
  `models/modelo_recomendado_xgboost_auc.joblib`

## Limpieza y entrega

AutoGluon se conserva solo como
validacion opcional documentada en `docs/validacion/autogluon_benchmark.md`; la
entrega principal se basa en el pipeline reproducible de `scripts/run_pipeline.py`.

## Decision

Los datos son viables para una prueba tecnica de cribado, pero no bastan para
implantacion clinica real sin validacion externa. La decision recomendada es usar
`XGBoostAUC_cuda` como baseline operativo limpio, mantener la MLP como comparador
y solicitar historico longitudinal prediagnostico, imagen, antecedentes
familiares, sintomas, tratamientos, estadio tumoral y validacion por hospital.
