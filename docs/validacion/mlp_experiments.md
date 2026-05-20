# Experimentos opcionales de MLP

Este documento separa explicitamente la investigacion de MLPs del pipeline
principal. El pipeline oficial sigue siendo:

```bash
source scripts/activate_cuda.sh
python scripts/run_pipeline.py --mode full --seed 42
```

La utilidad experimental vive en:

```bash
source scripts/activate_cuda.sh
python scripts/validacion/run_mlp_experiments.py --suite broad --epochs 70 --patience 9 --reduce-patience 4
```

## Que hace

- Entrena variantes MLP con el mismo split limpio de train interno, validacion y
  test.
- Selecciona arquitecturas y umbrales solo con validacion.
- Guarda ranking, historicos por candidato y ensembles opcionales en
  `outputs/metrics/`, que esta ignorado por git.
- Puede filtrar candidatos con `--names` y repetir semillas con `--seed`.

## Que no hace

- No cambia los modelos clasicos del pipeline principal.
- No elige miembros del `ValidationSoftVoting` final.
- No debe usarse para sobrescribir conclusiones mirando el test candidato a
  candidato.

## Resultado incorporado al pipeline

La investigacion justifico cambiar la MLP principal de `128-64-32 + Adam` a:

- `256-128-64` neuronas.
- `BatchNormalization`.
- Dropout decreciente `0.18`, `0.16`, `0.12`.
- L2 ligera `1e-5`.
- `AdamW` con learning rate `7e-4`, `weight_decay=1e-4` y `clipnorm=1.0`.
- Early stopping y reduccion de learning rate monitorizando el mejor F1 de
  validacion (`val_f1_best`).

En el split principal `seed=42`, la MLP paso aproximadamente de F1 `0.557` a
`0.561` en test, con umbral seleccionado en validacion `0.66`.

## Resultado no incorporado

Se probaron redes mas grandes, variantes residuales y activaciones `swish`,
`gelu` y `selu`. Algunas mejoraron una carrera concreta, pero no fueron lo
suficientemente estables para justificar meterlas en el pipeline principal.

Tambien se probo un ensemble solo de MLPs. En `seed=42`, el mejor ensemble por
validacion fue `swish_512_256_128_64+dense_256_128_64_low_dropout`, con F1 test
aproximado `0.560`. Se deja fuera del pipeline principal porque:

- duplica coste de entrenamiento,
- no supera de forma clara a la MLP compacta incorporada,
- no mejora al mejor modelo global (`HistGradientBoosting`, F1 `0.568`),
- y anadir otro ensemble haria el entregable menos sencillo sin ganancia clara.

## Lectura metodologica

El resultado encaja con datos tabulares: mas parametros no compran necesariamente
mas senal. La mejora util vino de regularizacion, optimizador, criterio de
parada alineado con F1 y ajuste de umbral limpio en validacion.
