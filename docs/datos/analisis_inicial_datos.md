# Analisis inicial del caso cancer

## Requisitos del PDF

El caso pide evaluar la viabilidad de anticipar diagnostico de cancer con datos
multimodales de pacientes. El entregable final son exactamente 5 diapositivas y
el codigo fuente ejecutable.

Contenido tecnico obligatorio:

- Cargar y unir los seis CSV por `paciente_id`.
- Seleccionar features justificandolas con `metadata_dataset_cancer.md`.
- Preprocesar categoricas, aplicar `StandardScaler` y hacer split estratificado 80/20.
- Entrenar al menos 3 modelos ML complejos y 1 MLP.
- Evaluar precision, recall, F1 de la clase `cancer = 1`, AUC-ROC y accuracy.
- La MLP debe tener al menos tres capas ocultas, Dropout, EarlyStopping y `class_weight`.
- El umbral de la MLP debe ajustarse en validacion, no en test.

## Inventario local

CSV presentes tras la extraccion desde BD:

- `CASOCANCER_01_BIOQUIMICOS.csv`
- `CASOCANCER_02_CLINICOS.csv`
- `CASOCANCER_03_GENETICOS.csv`
- `CASOCANCER_04_ECONOMICOS.csv`
- `CASOCANCER_05_GENERALES.csv`
- `CASOCANCER_06_SOCIODEMOGRAFICOS.csv`

El PDF del enunciado indica que hay seis colecciones. La primera version del
proyecto se construyo con cinco CSV, pero la conexion directa a la BD permitio
recuperar `CASOCANCER_04_ECONOMICOS.csv` y reconstruir
`docs/datos/metadata_operativa_dataset_cancer.md`.

## Calidad inicial de datos

Los seis CSV disponibles tienen:

- 50.001 filas cada uno.
- Sin `paciente_id` duplicados.
- Mismos `paciente_id` entre colecciones.
- Separador coma y BOM UTF-8.

Variable objetivo:

- `cancer = 1`: 9.644 pacientes.
- `cancer = 0`: 40.357 pacientes.
- Prevalencia positiva: 19,29%.
- Desbalance negativo/positivo: 4,18:1.

## Observaciones importantes

- `alcohol` es constante en los datos disponibles; conviene excluirlo.
- `vive` debe revisarse: puede ser una variable posterior al diagnostico y generar
  fuga temporal si se usa como predictor.
- `coste_total`, `coste_farmaco`, `num_ingresos` y `dias_hospital` del CSV 04
  tienen alto riesgo de reflejar tratamiento o uso hospitalario posterior al
  diagnostico; se excluyen del modelo limpio y se reservan para sensibilidad.
- `cancer` solo debe usarse como target.
- El metadata operativo esta en `docs/datos/metadata_operativa_dataset_cancer.md`.
