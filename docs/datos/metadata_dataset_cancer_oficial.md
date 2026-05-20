# Metadata del Dataset — Caso de Uso: Predicción de Diagnóstico de Cáncer

> **Universidad Alfonso X el Sabio**  
> Asignatura: Bases de Datos e Inteligencia Artificial  
> Curso: 2025-2026  
> Clave de unión entre colecciones: `paciente_id`  
> Total de registros por colección: **50 001 pacientes**

---

## Resumen del dataset

| Colección | Fichero | Nº variables | Tipo de información |
|---|---|:---:|---|
| Bioquímica | `MONGO01_bioquimicos.csv` | 8 | Analítica sanguínea |
| Clínica | `MONGO02_clinicos.csv` | 8 | Diagnósticos y comorbilidades |
| Genética | `MONGO03_geneticos.csv` | 8 | Mutaciones oncogénicas |
| Económica | `MONGO04_economicos.csv` | 6 | Costes y uso de recursos sanitarios |
| Hábitos | `MONGO05_generales.csv` | 5 | Estilo de vida |
| Sociodemografía | `MONGO06_sociodemograficos.csv` | 8 | Perfil social y demográfico |

**Variable objetivo:** `cancer` (en `MONGO02_clinicos.csv`) — Clasificación binaria, prevalencia ≈ **19 %**

---

## MONGO01\_bioquimicos.csv — Analítica sanguínea

| Campo | Tipo | Mín | Máx | Media | Descripción clínica |
|---|---|---:|---:|---:|---|
| `paciente_id` | `string` | — | — | — | Identificador único del paciente |
| `glucosa` | `float` | 55.0 | 168.9 | 102.6 | Glucemia en ayunas (mg/dL). >126 indica diabetes |
| `colesterol` | `float` | 120.0 | 307.4 | 194.1 | Colesterol total sérico (mg/dL) |
| `trigliceridos` | `float` | 50.0 | 321.7 | 156.3 | Triglicéridos séricos (mg/dL). Elevados en obesidad |
| `hemoglobina` | `float` | 8.0 | 18.0 | 13.9 | Hemoglobina (g/dL). <11 indica anemia; correlaciona con cáncer |
| `leucocitos` | `float` | 2.0 | 14.6 | 7.1 | Leucocitos (×10³/µL). >10 sugiere inflamación crónica |
| `plaquetas` | `float` | 100.0 | 434.3 | 254.8 | Recuento plaquetario (×10³/µL) |
| `creatinina` | `float` | 0.35 | 1.86 | 1.01 | Creatinina sérica (mg/dL). Marcador de función renal |

> **Nota:** glucosa, hemoglobina y leucocitos presentan correlación con `cancer` por diseño del modelo generativo.

---

## MONGO02\_clinicos.csv — Historia clínica

| Campo | Tipo | Valores | Prevalencia | Descripción clínica |
|---|---|---|:---:|---|
| `paciente_id` | `string` | P1000000 … P1050000 | — | Clave de unión |
| `diabetes` | `int (0/1)` | 0 = No, 1 = Sí | 35 % | Diabetes mellitus diagnosticada |
| `hipertension` | `int (0/1)` | 0 = No, 1 = Sí | 45 % | Hipertensión arterial |
| `obesidad` | `int (0/1)` | 0 = No, 1 = Sí | 36 % | IMC ≥ 30 |
| **`cancer`** ⭐ | **`int (0/1)`** | **0 = No, 1 = Sí** | **19 %** | **Variable objetivo** |
| `enfermedad_cardiaca` | `int (0/1)` | 0 = No, 1 = Sí | 16 % | Cardiopatía diagnosticada |
| `asma` | `int (0/1)` | 0 = No, 1 = Sí | 8 % | Asma bronquial |
| `epoc` | `int (0/1)` | 0 = No, 1 = Sí | 10 % | Enfermedad pulmonar obstructiva crónica |

> ⚠️ **Data leakage potencial:** `diabetes`, `hipertension` y `obesidad` correlacionan con `cancer` por diseño. Su inclusión como features debe justificarse.

---

## MONGO03\_geneticos.csv — Mutaciones oncogénicas

| Campo | Tipo | Valores | Prevalencia | Gen y tipo de cáncer asociado |
|---|---|---|:---:|---|
| `paciente_id` | `string` | P1000000 … | — | Clave de unión |
| `mut_BRCA1` | `int (0/1)` | 0 = No portador, 1 = Portador | 8 % | Supresor tumoral. Cáncer de mama y ovario hereditario |
| `mut_TP53` | `int (0/1)` | 0 = No portador, 1 = Portador | 12 % | Guardián del genoma. Múltiples tipos de cáncer |
| `mut_EGFR` | `int (0/1)` | 0 = No portador, 1 = Portador | 9 % | Receptor de crecimiento. Adenocarcinoma pulmonar |
| `mut_KRAS` | `int (0/1)` | 0 = No portador, 1 = Portador | 13 % | Proto-oncogén. Cáncer pancreático y colorrectal |
| `mut_PIK3CA` | `int (0/1)` | 0 = No portador, 1 = Portador | 9 % | Vía PI3K/AKT/mTOR. Cáncer de mama, colon |
| `mut_ALK` | `int (0/1)` | 0 = No portador, 1 = Portador | 5 % | Reordenamiento. Cáncer de pulmón no microcítico |
| `mut_BRAF` | `int (0/1)` | 0 = No portador, 1 = Portador | 7 % | Quinasa MAP. Melanoma y cáncer colorrectal |

> **Nota:** estas son las variables con mayor peso predictivo en el modelo. Son predictores causales directos y **sí deben incluirse** como features.

---

## MONGO04\_economicos.csv — Recursos sanitarios

| Campo | Tipo | Valores / Rango | Media | Descripción |
|---|---|---|:---:|---|
| `paciente_id` | `string` | P1000000 … | — | Clave de unión |
| `tipo_seguro` | `string` | Público / Privado / Mixto | — | Modalidad de cobertura sanitaria |
| `coste_total` | `float` | 500 – 95 572 € | 15 209 € | Coste total del episodio asistencial |
| `coste_farmaco` | `float` | 101 – 41 932 € | 4 967 € | Coste de medicación |
| `num_ingresos` | `int` | 0 – 9 | 0.87 | Número de ingresos hospitalarios |
| `dias_hospital` | `int` | 0 – 168 días | 24.9 | Días totales de hospitalización |

> ⛔ **Estas variables NO deben usarse como features.** Son consecuencia del diagnóstico de cáncer, no causas. Su uso introduce data leakage severo.

---

## MONGO05\_generales.csv — Hábitos de vida

| Campo | Tipo | Valores | Distribución | Descripción |
|---|---|---|:---:|---|
| `paciente_id` | `string` | P1000000 … | — | Clave de unión |
| `fumador` | `int (0/1)` | 0 = No, 1 = Sí | 38 % fumadores | Tabaquismo activo. Factor de riesgo de múltiples cánceres |
| `alcohol` | `int` | 1 (constante) | 100 % = 1 | Variable constante — **no informativa, excluir** |
| `actividad_fisica` | `string` | Alta / Moderada / Baja | 20 / 35 / 45 % | Nivel de actividad física habitual. Factor protector |
| `vive` | `int (0/1)` | 0 = Fallecido, 1 = Vivo | 74 % vive | Supervivencia al cierre del seguimiento |

> ⛔ `vive` es consecuencia del diagnóstico y **no debe usarse como feature**. `alcohol` es constante y no aporta información.

---

## MONGO06\_sociodemograficos.csv — Perfil sociodemográfico

| Campo | Tipo | Valores / Rango | Media | Descripción |
|---|---|---|:---:|---|
| `paciente_id` | `string` | P1000000 … | — | Clave de unión |
| `edad` | `int` | 20 – 90 años | 54.5 | Edad en años. El riesgo oncológico acumulado crece con la edad |
| `nivel_educativo` | `string` | Sin estudios / Primaria / Secundaria / Universitario | — | Nivel formativo más alto alcanzado |
| `nivel_ingresos` | `string` | Muy bajo / Bajo / Medio / Alto | — | Nivel socioeconómico del hogar |
| `zona` | `string` | Urbana / Semiurbana / Rural | 55 / 25 / 20 % | Zona de residencia |
| `estado_civil` | `string` | Soltero / Casado / Divorciado / Viudo | — | Estado civil |
| `num_hijos` | `int` | 0 – 8 | 1.51 | Número de hijos |
| `distancia_hospital_km` | `float` | 0.5 – 232.7 km | 25.6 | km al hospital de referencia |

---

## Modelo generativo de la variable objetivo `cancer`

La etiqueta se ha asignado mediante un modelo logístico con factores médicamente validados:

```
P(cancer = 1) = sigmoid(β₀ + Σ wₖ·xₖ + ε)

    β₀ = −4.0   (intercepto, calibrado para prevalencia ≈ 19 %)
    ε  ~ N(0, 0.8)   (ruido biológico residual)
```

| Factor de riesgo | Peso (wₖ) | Efecto |
|---|:---:|---|
| Mutación BRCA1 | +2.0 | Riesgo fuerte |
| Mutación TP53 | +1.8 | Riesgo fuerte |
| Fumador | +1.5 | Riesgo fuerte |
| Mutación KRAS | +1.4 | Riesgo moderado-alto |
| Glucosa > 130 mg/dL | +1.2 | Riesgo moderado |
| Obesidad | +1.1 | Riesgo moderado |
| Mutación EGFR | +1.0 | Riesgo moderado |
| Hemoglobina < 11 g/dL | +0.9 | Marcador de riesgo |
| Mutación PIK3CA | +0.8 | Riesgo leve |
| Leucocitos > 10 ×10³/µL | +0.7 | Inflamación crónica |
| Mutación BRAF | +0.6 | Riesgo leve |
| Hipertensión | +0.5 | Comorbilidad asociada |
| Edad > 55 años | +0.4 | Riesgo acumulado |
| **Actividad física ALTA** | **−1.2** | **Factor protector** |
| **Actividad física MODERADA** | **−0.6** | **Protección parcial** |

---

## Guía de selección de features para el pipeline

| Variable | ¿Usar? | Motivo |
|---|:---:|---|
| glucosa, colesterol, trigliceridos, hemoglobina, leucocitos, plaquetas, creatinina | ✅ Sí | Predictores bioquímicos causales |
| mut_BRCA1, mut_TP53, mut_EGFR, mut_KRAS, mut_PIK3CA, mut_ALK, mut_BRAF | ✅ Sí | Predictores genéticos con mayor peso |
| fumador | ✅ Sí | Factor de riesgo causal |
| actividad_fisica (codificar: Baja=0, Moderada=1, Alta=2) | ✅ Sí | Factor protector causal |
| edad | ✅ Sí | Proxy de riesgo acumulado |
| alcohol | ❌ No | Constante, sin varianza |
| vive | ❌ No | Consecuencia del diagnóstico (leakage) |
| coste_total, coste_farmaco, dias_hospital, num_ingresos | ❌ No | Consecuencias del diagnóstico (leakage) |
| diabetes, hipertension, obesidad, enfermedad_cardiaca, asma, epoc | ⚠️ Valorar | Comorbilidades: correlacionan con cancer por diseño; riesgo de leakage indirecto |
| nivel_educativo, nivel_ingresos, zona, estado_civil, num_hijos, distancia_hospital_km | ⚠️ Opcional | Variables sociodemográficas de bajo peso predictivo |