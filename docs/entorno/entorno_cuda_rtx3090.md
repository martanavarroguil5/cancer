# Entorno CUDA para RTX 3090

Comprobado el 2026-04-28 en esta maquina:

- GPU: NVIDIA GeForce RTX 3090.
- Driver NVIDIA: 580.126.09.
- CUDA reportado por `nvidia-smi`: 13.0.
- `nvcc` del sistema: 12.0.140.
- Python del entorno: 3.12.

## Decision de instalacion

El enunciado pide una MLP con `Dense`, `BatchNormalization`, `Dropout`,
`EarlyStopping`, `ReduceLROnPlateau` y `class_weight`, asi que el entorno queda
preparado para TensorFlow/Keras con GPU:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
deactivate
source scripts/activate_cuda.sh
```

No se ha instalado PyTorch en el mismo entorno para evitar mezclar dos pilas CUDA
pesadas si no hace falta para el trabajo.

## Uso

```bash
source scripts/activate_cuda.sh
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
jupyter lab
```

El script `scripts/activate_cuda.sh` activa `.venv` y ajusta `LD_LIBRARY_PATH`
para priorizar las librerias
CUDA/cuDNN instaladas por `tensorflow[and-cuda]`. En esta maquina era necesario
porque la libreria `libnvJitLink.so.12` del sistema es de CUDA 12.0 y TensorFlow
2.21 necesita la version instalada dentro del entorno.

Verificacion realizada:

- `pip check`: sin dependencias rotas.
- TensorFlow 2.21 detecta `/physical_device:GPU:0`.
- Compute capability detectada: `(8, 6)`, correspondiente a RTX 3090.
- Una multiplicacion de matrices de TensorFlow se ejecuto en `/GPU:0`.
- XGBoost 3.2.0 entreno correctamente con `device="cuda"`.
