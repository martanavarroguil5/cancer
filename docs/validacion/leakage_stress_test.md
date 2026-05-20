# Stress test de fuga por variable

## Objetivo

Se anadio un experimento de hiper-entrenamiento controlado para comprobar si
alguna variable predice el target de forma anormal, si una variable excluida
dispara el modelo base, o si algun grupo de columnas introduce fuga temporal.

El experimento no sustituye al pipeline oficial. Sirve como auditoria:

- `single`: entrena una variable individual.
- `group`: entrena grupos tematicos completos.
- `base_plus`: anade una variable excluida al modelo base limpio.
- `base_minus`: ablation de variables del modelo base, usado en la pasada quick.

## Reproduccion

Pasada rapida completa, incluyendo ablations:

```bash
python scripts/validacion/run_leakage_stress_test.py --mode quick --seed 42
```

Pasada fuerte usada para confirmar sospechosos principales:

```bash
python scripts/validacion/run_leakage_stress_test.py \
  --mode full \
  --seed 42 \
  --skip-base-minus \
  --output-dir outputs/metrics/leakage_stress_test_full
```

Salidas:

- `outputs/metrics/leakage_stress_test/leakage_stress_results.csv`
- `outputs/metrics/leakage_stress_test/leakage_stress_summary.md`
- `outputs/metrics/leakage_stress_test_full/leakage_stress_results.csv`
- `outputs/metrics/leakage_stress_test_full/leakage_stress_summary.md`

## Resultado principal

Pasada `full`, 67 escenarios, seed 42:

| Escenario | Tipo | F1 test | AUC-ROC | AUC-PR | Delta F1 vs base | Lectura |
|---|---|---:|---:|---:|---:|---|
| `base_clean` | group | 0.566 | 0.830 | 0.578 | 0.000 | Referencia limpia. |
| `safe_all_clean` | group | 0.578 | 0.841 | 0.602 | 0.012 | Mejora ligera, sin firma de fuga fuerte. |
| `economic_sensitivity` | group | 0.980 | 0.998 | 0.996 | 0.414 | Fuga critica. |
| `economic_cost_use_only` | group | 0.978 | 0.997 | 0.995 | 0.413 | Fuga critica aun sin otras variables. |
| `known_risk_only` | group | 0.981 | 0.998 | 0.996 | 0.415 | Fuga critica. |
| `all_raw_including_risks` | group | 0.981 | 0.999 | 0.997 | 0.416 | Todas las variables quedan contaminadas por costes/uso. |

## Variables individuales

| Variable | F1 test sola | AUC-ROC sola | AUC-PR sola | Lectura |
|---|---:|---:|---:|---|
| `coste_total` | 0.974 | 0.992 | 0.989 | Fuga critica, no usar. |
| `dias_hospital` | 0.960 | 0.991 | 0.984 | Fuga critica, no usar. |
| `coste_farmaco` | 0.948 | 0.990 | 0.981 | Fuga critica, no usar. |
| `num_ingresos` | 0.688 | 0.882 | 0.694 | Riesgo alto; refleja utilizacion sanitaria. |
| `vive` | 0.480 | 0.688 | 0.320 | Fuga temporal conocida; no usar. |
| `tipo_seguro` | 0.366 | 0.620 | 0.260 | No muestra fuga fuerte; se excluye del modelo base por proxy/senal marginal. |

## Base mas variable excluida

| Variable anadida a base | F1 test | AUC-ROC | AUC-PR | Delta F1 vs base | Lectura |
|---|---:|---:|---:|---:|---|
| `coste_total` | 0.974 | 0.997 | 0.993 | +0.408 | Fuga critica. |
| `dias_hospital` | 0.964 | 0.995 | 0.988 | +0.398 | Fuga critica. |
| `coste_farmaco` | 0.953 | 0.994 | 0.987 | +0.387 | Fuga critica. |
| `num_ingresos` | 0.758 | 0.940 | 0.845 | +0.192 | Fuga/riesgo asistencial alto. |
| `vive` | 0.609 | 0.863 | 0.652 | +0.044 | Fuga temporal confirmada por mejora artificial. |
| `tipo_seguro` | 0.578 | 0.843 | 0.605 | +0.012 | Mejora pequena; no parece fuga critica. |

## Decision

El experimento confirma que la politica actual es prudente:

- mantener fuera `coste_total`, `coste_farmaco`, `num_ingresos` y
  `dias_hospital` del modelo operativo;
- mantener fuera `vive` por temporalidad/post-diagnostico;
- usar `safe_all` como vista final, ya que mejora ligeramente el F1 limpio sin
  la firma de fuga critica de costes/uso;
- mantener `metadata_core` como referencia clinica estricta, por separar el
  nucleo recomendado de comorbilidades de cautela y variables opcionales.

Este stress test no demuestra causalidad ni temporalidad por si solo, pero si
detecta una firma cuantitativa muy fuerte de fuga: variables que solas alcanzan
F1 cercano a `0.95-0.97` no son coherentes con un modelo prediagnostico limpio.
