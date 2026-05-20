# Documentacion del proyecto

Este directorio queda dividido entre documentacion de entrega y material de
validacion. La fuente de verdad para presentar el trabajo es el README raiz, la
presentacion final y las metricas generadas en `outputs/metrics/`.

## Entrega

- `enunciado_cancer.pdf`: enunciado original del caso.
- `datos/metadata_dataset_cancer_oficial.md`: metadato oficial recuperado.
- `datos/metadata_operativa_dataset_cancer.md`: metadata adaptada a los CSV
  reales del repositorio.
- `modelado/decisiones_modelado.md`: decisiones finales de variables, modelos,
  metricas, umbrales y limitaciones.

## Validacion auxiliar

- `validacion/leakage_stress_test.md`: auditoria de fuga temporal.
- `validacion/mlp_experiments.md`: busqueda de arquitecturas MLP.
- `validacion/super_hypertraining.md`: experimentos extensivos de modelos.
- `validacion/autogluon_benchmark.md`: comparativa AutoGluon opcional.

## Entorno

- `entorno/entorno_cuda_rtx3090.md`: notas reproducibles del entorno CUDA local.

## Artefactos finales

- Presentacion: `outputs/reports/presentacion_final_cancer_5_diapositivas.pdf`
- Metricas: `outputs/metrics/model_metrics.csv`
- Resumen: `outputs/metrics/run_summary.json`
- Modelo recomendado: `models/modelo_recomendado_xgboost_auc.joblib`
