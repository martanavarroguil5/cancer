# Metadata del dataset de cancer

Reconstruido a partir del enunciado `docs/enunciado_cancer.pdf`, la base de datos Azure SQL
`usecases`, el esquema `CASOCANCER` y el metadato oficial recuperado en
`docs/datos/metadata_dataset_cancer_oficial.md`.

Fecha de reconstruccion inicial: 2026-04-28.  
Actualizacion con metadato oficial: 2026-05-06.

## Actualizacion oficial

El archivo `docs/datos/metadata_dataset_cancer_oficial.md` confirma la guia de uso que debe gobernar el pipeline:

- Usar como predictores principales: bioquimica completa, mutaciones geneticas,
  `fumador`, `actividad_fisica` codificada como `Baja=0`, `Moderada=1`,
  `Alta=2`, y `edad`.
- Excluir siempre: `paciente_id`, `cancer` como feature, `alcohol`, `vive`,
  `coste_total`, `coste_farmaco`, `num_ingresos` y `dias_hospital`.
- Tratar como variables de cautela: `diabetes`, `hipertension`, `obesidad`,
  `enfermedad_cardiaca`, `asma` y `epoc`, porque correlacionan con `cancer` por
  diseno y pueden introducir leakage indirecto si se usan sin justificacion.
- Tratar como opcionales/de bajo peso: `nivel_educativo`, `nivel_ingresos`,
  `zona`, `estado_civil`, `num_hijos` y `distancia_hospital_km`.
- El modelo generativo oficial usa factores como `glucosa > 130`,
  `hemoglobina < 11`, `leucocitos > 10`, `edad > 55`, tabaquismo, actividad
  fisica protectora y mutaciones oncogenicas.

Los nombres oficiales del metadato usan `MONGO01_*` ... `MONGO06_*`; en este
repositorio los CSV equivalentes estan exportados como `CASOCANCER_01_*` ...
`CASOCANCER_06_*`.

## Resumen general

El dataset simula seis colecciones/documentos de pacientes. Todas comparten la
clave `paciente_id` y deben unirse mediante una relacion 1:1.

| Elemento | Valor |
|---|---:|
| Esquema SQL | `CASOCANCER` |
| Numero de tablas/CSV | 6 |
| Registros por tabla | 50.001 |
| `paciente_id` unicos por tabla | 50.001 |
| Interseccion de `paciente_id` entre tablas | 50.001 |
| Duplicados observados en `paciente_id` | 0 |
| Valores nulos observados en CSV | 0 |
| Variable objetivo | `cancer` |
| Prevalencia de `cancer = 1` | 19,29% |
| Registros `cancer = 1` | 9.644 |
| Registros `cancer = 0` | 40.357 |
| Razon de desbalance `0:1` | 4,18:1 |

Nota tecnica: en SQL Server todas las columnas aparecen como `IS_NULLABLE = YES`,
pero en los CSV exportados no se observan valores nulos. Aun asi, conviene incluir
comprobaciones de nulos en el pipeline.

## Tablas disponibles

| Tabla/CSV | Filas | Descripcion |
|---|---:|---|
| `CASOCANCER_01_BIOQUIMICOS` | 50.001 | Variables bioquimicas continuas de analitica. |
| `CASOCANCER_02_CLINICOS` | 50.001 | Comorbilidades clinicas binarias y variable objetivo `cancer`. |
| `CASOCANCER_03_GENETICOS` | 50.001 | Indicadores binarios de mutaciones geneticas. |
| `CASOCANCER_04_ECONOMICOS` | 50.001 | Seguro y variables de coste/uso hospitalario. |
| `CASOCANCER_05_GENERALES` | 50.001 | Habitos, actividad fisica y estado vital. |
| `CASOCANCER_06_SOCIODEMOGRAFICOS` | 50.001 | Edad y variables sociodemograficas. |

## Clave, objetivo y reglas de uso

| Campo | Tipo SQL | Tipo ML | Valores | Uso recomendado | Advertencia |
|---|---|---|---|---|---|
| `paciente_id` | `varchar(max)` | Identificador | `P1000000` a `P1050000` | Usar solo para unir tablas. | Excluir siempre del entrenamiento. |
| `cancer` | `bigint` | Objetivo binario | 0/1 | Variable `y` a predecir. | No usar como feature. Aplicar division estratificada por esta variable. |

## `CASOCANCER_01_BIOQUIMICOS`

Todas las variables de esta tabla son numericas continuas (`float` en SQL Server).
Las unidades clinicas no estan documentadas en la BD; los nombres sugieren
biomarcadores habituales, pero deben tratarse como variables sinteticas.

| Campo | Tipo SQL | Rango observado | Media | Descripcion | Advertencia de uso |
|---|---|---:|---:|---|---|
| `paciente_id` | `varchar(max)` | 50.001 ids unicos | NA | Clave de paciente. | Usar solo para union. |
| `glucosa` | `float` | 55,00 - 179,23 | 102,19 | Medida bioquimica de glucosa. | Estandarizar antes de modelos sensibles a escala. |
| `colesterol` | `float` | 120,00 - 320,00 | 193,66 | Medida bioquimica de colesterol. | Estandarizar. |
| `trigliceridos` | `float` | 50,00 - 321,68 | 156,23 | Medida bioquimica de trigliceridos. | Estandarizar. |
| `hemoglobina` | `float` | 8,00 - 18,00 | 13,93 | Medida bioquimica de hemoglobina. | Estandarizar. |
| `leucocitos` | `float` | 2,00 - 15,08 | 7,15 | Recuento de leucocitos. | Estandarizar. |
| `plaquetas` | `float` | 100,00 - 489,79 | 254,97 | Recuento de plaquetas. | Estandarizar. |
| `creatinina` | `float` | 0,35 - 2,10 | 1,00 | Medida bioquimica de creatinina. | Estandarizar. |

## `CASOCANCER_02_CLINICOS`

Variables clinicas binarias. En todas ellas `1` indica presencia/positivo y `0`
ausencia/negativo.

| Campo | Tipo SQL | Valores | Prevalencia de 1 | Descripcion | Advertencia de uso |
|---|---|---|---:|---|---|
| `paciente_id` | `varchar(max)` | 50.001 ids unicos | NA | Clave de paciente. | Usar solo para union. |
| `diabetes` | `bigint` | 0/1 | 34,47% | Indicador de diabetes. | Feature clinica valida si se conoce antes del diagnostico. |
| `hipertension` | `bigint` | 0/1 | 44,32% | Indicador de hipertension. | Feature clinica valida si se conoce antes del diagnostico. |
| `obesidad` | `bigint` | 0/1 | 35,42% | Indicador de obesidad. | Feature clinica valida si se conoce antes del diagnostico. |
| `cancer` | `bigint` | 0/1 | 19,29% | Diagnostico de cancer. | Variable objetivo. No incluir en `X`. |
| `enfermedad_cardiaca` | `bigint` | 0/1 | 16,63% | Indicador de enfermedad cardiaca. | Feature clinica valida si se conoce antes del diagnostico. |
| `asma` | `bigint` | 0/1 | 8,16% | Indicador de asma. | Feature clinica valida si se conoce antes del diagnostico. |
| `epoc` | `bigint` | 0/1 | 9,18% | Indicador de EPOC. | Feature clinica valida si se conoce antes del diagnostico. |

## `CASOCANCER_03_GENETICOS`

Variables geneticas binarias. En todas ellas `1` indica mutacion detectada y `0`
mutacion no detectada.

| Campo | Tipo SQL | Valores | Prevalencia de 1 | Descripcion | Advertencia de uso |
|---|---|---|---:|---|---|
| `paciente_id` | `varchar(max)` | 50.001 ids unicos | NA | Clave de paciente. | Usar solo para union. |
| `mut_BRCA1` | `bigint` | 0/1 | 8,08% | Indicador de mutacion `BRCA1`. | Puede ser muy informativa; evitar interpretacion causal simplista. |
| `mut_TP53` | `bigint` | 0/1 | 12,03% | Indicador de mutacion `TP53`. | Puede ser muy informativa; evitar interpretacion causal simplista. |
| `mut_EGFR` | `bigint` | 0/1 | 9,82% | Indicador de mutacion `EGFR`. | Puede ser muy informativa; evitar interpretacion causal simplista. |
| `mut_KRAS` | `bigint` | 0/1 | 12,97% | Indicador de mutacion `KRAS`. | Puede ser muy informativa; evitar interpretacion causal simplista. |
| `mut_PIK3CA` | `bigint` | 0/1 | 8,98% | Indicador de mutacion `PIK3CA`. | Puede ser muy informativa; evitar interpretacion causal simplista. |
| `mut_ALK` | `bigint` | 0/1 | 4,92% | Indicador de mutacion `ALK`. | Baja prevalencia; vigilar estabilidad en validacion. |
| `mut_BRAF` | `bigint` | 0/1 | 6,91% | Indicador de mutacion `BRAF`. | Baja prevalencia; vigilar estabilidad en validacion. |

## `CASOCANCER_04_ECONOMICOS`

Variables de seguro, coste y uso hospitalario. Estas variables pueden estar
temporalmente despues del diagnostico o muy correlacionadas con el tratamiento.
Para un modelo de cribado previo al diagnostico, se recomienda probar resultados
con y sin esta tabla y justificar la decision.

| Campo | Tipo SQL | Rango/valores observados | Media/prevalencia | Descripcion | Advertencia de uso |
|---|---|---|---:|---|---|
| `paciente_id` | `varchar(max)` | 50.001 ids unicos | NA | Clave de paciente. | Usar solo para union. |
| `tipo_seguro` | `varchar(max)` | Publico 56,2%; Privado 23,9%; Mixto 19,9% | NA | Tipo de cobertura sanitaria. | Codificar como categorica; puede introducir sesgo socioeconomico. |
| `coste_total` | `float` | 500,00 - 102.256,29 | 15.170,91 | Coste sanitario total asociado al paciente. | Alto riesgo de fuga temporal si el coste se genera tras diagnostico/tratamiento. |
| `coste_farmaco` | `float` | 100,07 - 41.932,14 | 4.930,03 | Coste de medicacion/farmacos. | Alto riesgo de fuga temporal si refleja tratamiento oncologico. |
| `num_ingresos` | `bigint` | 0 - 9 | 0,88 | Numero de ingresos hospitalarios. | Posible proxy de gravedad o diagnostico; justificar inclusion. |
| `dias_hospital` | `bigint` | 0 - 181 | 24,91 | Dias acumulados de hospitalizacion. | Posible fuga si se calcula despues del diagnostico. |

## `CASOCANCER_05_GENERALES`

Variables generales de habitos y estado vital.

| Campo | Tipo SQL | Rango/valores observados | Media/prevalencia | Descripcion | Advertencia de uso |
|---|---|---|---:|---|---|
| `paciente_id` | `varchar(max)` | 50.001 ids unicos | NA | Clave de paciente. | Usar solo para union. |
| `fumador` | `bigint` | 0/1 | 38,02% | Indicador de tabaquismo. | Feature valida si se conoce antes del diagnostico. |
| `alcohol` | `bigint` | 1 en el 100,00% | 100,00% | Indicador de consumo de alcohol. | Variable constante; eliminar del entrenamiento porque no aporta varianza. |
| `actividad_fisica` | `varchar(max)` | Baja 45,1%; Moderada 34,8%; Alta 20,1% | NA | Nivel de actividad fisica. | Codificar como categorica ordinal o nominal; justificar criterio. |
| `vive` | `bigint` | 0/1 | 73,85% | Estado vital del paciente. | Riesgo muy alto de fuga y de variable posterior al diagnostico; excluir en modelos de cribado. |

## `CASOCANCER_06_SOCIODEMOGRAFICOS`

Variables demograficas y socioeconomicas. Son utiles para modelado predictivo, pero
requieren cautela por posibles sesgos de equidad.

| Campo | Tipo SQL | Rango/valores observados | Media/prevalencia | Descripcion | Advertencia de uso |
|---|---|---|---:|---|---|
| `paciente_id` | `varchar(max)` | 50.001 ids unicos | NA | Clave de paciente. | Usar solo para union. |
| `edad` | `bigint` | 20 - 90 | 54,43 | Edad del paciente. | Feature importante; revisar distribucion por clase. |
| `nivel_educativo` | `varchar(max)` | Secundaria 39,9%; Primaria 25,2%; Universitario 24,8%; Sin estudios 10,2% | NA | Nivel educativo maximo. | Posible proxy socioeconomico; evaluar sesgo. |
| `nivel_ingresos` | `varchar(max)` | Medio 39,9%; Bajo 28,4%; Alto 19,8%; Muy bajo 11,9% | NA | Nivel de ingresos. | Posible sesgo socioeconomico; codificar con cuidado. |
| `zona` | `varchar(max)` | Urbana 54,8%; Semiurbana 25,3%; Rural 19,9% | NA | Entorno de residencia. | Puede capturar accesibilidad sanitaria; evaluar sesgo. |
| `estado_civil` | `varchar(max)` | Casado 47,9%; Soltero 24,9%; Divorciado 15,2%; Viudo 12,1% | NA | Estado civil. | Codificar como categorica nominal. |
| `num_hijos` | `bigint` | 0 - 8 | 1,50 | Numero de hijos. | Puede actuar como proxy demografico; revisar importancia. |
| `distancia_hospital_km` | `float` | 0,50 - 250,00 | 24,86 | Distancia al hospital en kilometros. | Estandarizar; posible proxy de acceso sanitario. |

## Recomendaciones para seleccion de features

Para un modelo cuyo objetivo sea anticipar diagnostico de cancer en un sistema de
cribado, la seleccion alineada con el metadato oficial es:

Incluir inicialmente:

- Variables bioquimicas: `glucosa`, `colesterol`, `trigliceridos`, `hemoglobina`,
  `leucocitos`, `plaquetas`, `creatinina`.
- Variables geneticas: todas las columnas `mut_*`.
- Habitos y demografia principal: `fumador`, `actividad_fisica`, `edad`.

Excluir por defecto:

- `paciente_id`: identificador, no predictor.
- `cancer`: variable objetivo.
- `alcohol`: constante en el dataset.
- `vive`: probable variable posterior al diagnostico y posible fuga de datos.
- `coste_total`, `coste_farmaco`, `num_ingresos`, `dias_hospital`: consecuencias
  del diagnostico/tratamiento; incluir solo para sensibilidad de fuga.
- `tipo_seguro`: no aparece como predictor recomendado en la guia oficial y puede
  actuar como proxy socioeconomico.

Valorar solo con justificacion:

- Comorbilidades clinicas: `diabetes`, `hipertension`, `obesidad`,
  `enfermedad_cardiaca`, `asma`, `epoc`.
- Variables sociodemograficas opcionales: `nivel_educativo`, `nivel_ingresos`,
  `zona`, `estado_civil`, `num_hijos`, `distancia_hospital_km`.

## Preprocesamiento recomendado

- Unir los seis CSV por `paciente_id` con inner join y verificar que quedan
  50.001 registros.
- Separar `X` e `y`, usando `cancer` como objetivo.
- Usar division estratificada 80%/20% por `cancer`.
- Estandarizar variables numericas continuas con `StandardScaler`.
- Codificar categoricas con `OneHotEncoder` o codificacion ordinal justificada.
- Gestionar el desbalance con `class_weight`, metricas centradas en la clase
  positiva y ajuste de umbral en validacion.
- No optimizar hiperparametros ni umbral sobre el conjunto de test.

## Variables categoricas detectadas

| Campo | Valores |
|---|---|
| `tipo_seguro` | `Publico`, `Privado`, `Mixto` |
| `actividad_fisica` | `Baja`, `Moderada`, `Alta` |
| `nivel_educativo` | `Sin estudios`, `Primaria`, `Secundaria`, `Universitario` |
| `nivel_ingresos` | `Muy bajo`, `Bajo`, `Medio`, `Alto` |
| `zona` | `Urbana`, `Semiurbana`, `Rural` |
| `estado_civil` | `Casado`, `Soltero`, `Divorciado`, `Viudo` |

## Fuente de reconstruccion

- Enunciado: `docs/enunciado_cancer.pdf`, que indica la existencia esperada de
  `metadata_dataset_cancer.md`.
- BD consultada: `usecases`, esquema `CASOCANCER`.
- Tipos SQL obtenidos desde `INFORMATION_SCHEMA.COLUMNS`.
- Rangos, medias, prevalencias y categorias calculados sobre los CSV exportados
  desde las tablas `CASOCANCER_*`.
