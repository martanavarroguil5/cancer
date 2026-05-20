"""Red neuronal MLP con validacion limpia y class_weight."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight

from cancer_ml.config import MODELS_DIR, ModeConfig, set_global_seed
from cancer_ml.evaluation import threshold_search

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class MLPConfig:
    """Configuracion reproducible de una arquitectura MLP candidata."""

    name: str
    hidden_units: tuple[int, ...]
    block_type: str = "dense"
    activation: str = "relu"
    dropout: float | tuple[float, ...] = 0.20
    batch_norm: bool = True
    l2: float = 0.0
    optimizer: str = "adamw"
    learning_rate: float = 7e-4
    weight_decay: float = 1e-4
    batch_size: int | None = None
    label_smoothing: float = 0.0
    monitor: str = "val_f1_best"
    warmup_epochs: int = 4
    clipnorm: float | None = 1.0

    def to_row(self) -> dict[str, object]:
        row = asdict(self)
        row["hidden_units"] = "x".join(str(unit) for unit in self.hidden_units)
        if isinstance(self.dropout, tuple):
            row["dropout"] = ",".join(str(value) for value in self.dropout)
        return row


DEFAULT_MLP_CONFIG = MLPConfig(
    name="selu_256_128_64_no_bn",
    hidden_units=(256, 128, 64),
    activation="selu",
    dropout=(0.08, 0.06, 0.04),
    batch_norm=False,
    l2=1e-6,
    optimizer="adamw",
    learning_rate=6e-4,
    weight_decay=1e-5,
    batch_size=512,
)


@dataclass
class MLPArtifacts:
    model: object
    preprocessor: object
    history: pd.DataFrame
    validation_proba: np.ndarray
    validation_target: np.ndarray
    test_proba: np.ndarray
    config: MLPConfig | None = None
    validation_f1: float | None = None
    validation_threshold: float | None = None


def train_mlp(
    preprocessor,
    X_train_inner: pd.DataFrame,
    y_train_inner: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    X_test: pd.DataFrame,
    mode_config: ModeConfig,
    seed: int,
    models_dir: Path = MODELS_DIR,
    mlp_config: MLPConfig | None = None,
) -> MLPArtifacts:
    """Entrena la MLP; si CUDA falla, reintenta colocando el entrenamiento en CPU."""

    config = mlp_config or DEFAULT_MLP_CONFIG
    transformer = clone(preprocessor)
    X_train_t = _as_float32(transformer.fit_transform(X_train_inner))
    X_valid_t = _as_float32(transformer.transform(X_valid))
    X_test_t = _as_float32(transformer.transform(X_test))

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1]),
        y=np.asarray(y_train_inner).astype(int),
    )
    class_weight = {0: float(class_weights[0]), 1: float(class_weights[1])}
    LOGGER.info("Class_weight MLP: %s.", class_weight)

    try:
        return _fit_mlp_on_device(
            X_train_t,
            np.asarray(y_train_inner).astype(int),
            X_valid_t,
            np.asarray(y_valid).astype(int),
            X_test_t,
            transformer,
            class_weight,
            mode_config,
            config,
            seed,
            models_dir,
            device=None,
        )
    except Exception as exc:
        LOGGER.warning("MLP con dispositivo por defecto fallo: %s", exc)
        return _fit_mlp_on_device(
            X_train_t,
            np.asarray(y_train_inner).astype(int),
            X_valid_t,
            np.asarray(y_valid).astype(int),
            X_test_t,
            transformer,
            class_weight,
            mode_config,
            config,
            seed,
            models_dir,
            device="/CPU:0",
        )


def _fit_mlp_on_device(
    X_train_t: np.ndarray,
    y_train: np.ndarray,
    X_valid_t: np.ndarray,
    y_valid: np.ndarray,
    X_test_t: np.ndarray,
    transformer,
    class_weight: dict[int, float],
    mode_config: ModeConfig,
    mlp_config: MLPConfig,
    seed: int,
    models_dir: Path,
    device: str | None,
) -> MLPArtifacts:
    import tensorflow as tf

    set_global_seed(seed)
    tf.keras.backend.clear_session()
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass

    context = tf.device(device) if device else _nullcontext()
    with context:
        model = build_mlp(X_train_t.shape[1], seed, mlp_config)
        batch_size = mlp_config.batch_size or mode_config.batch_size
        monitor_mode = "max" if _monitor_should_maximize(mlp_config.monitor) else "min"
        validation_metrics = ValidationThresholdMetrics(
            X_valid_t,
            y_valid,
            batch_size=batch_size,
        )
        callbacks = [
            validation_metrics,
            tf.keras.callbacks.EarlyStopping(
                monitor=mlp_config.monitor,
                mode=monitor_mode,
                patience=mode_config.mlp_patience,
                min_delta=1e-4,
                restore_best_weights=True,
                start_from_epoch=mlp_config.warmup_epochs,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor=mlp_config.monitor,
                mode=monitor_mode,
                factor=0.5,
                patience=mode_config.mlp_reduce_patience,
                min_lr=1e-5,
            ),
        ]
        LOGGER.info(
            "Entrenando MLP %s%s.",
            mlp_config.name,
            f" en {device}" if device else "",
        )
        history = model.fit(
            X_train_t,
            y_train,
            validation_data=(X_valid_t, y_valid),
            epochs=mode_config.mlp_epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=2,
        )
        validation_proba = model.predict(X_valid_t, batch_size=batch_size, verbose=0).ravel()
        test_proba = model.predict(X_test_t, batch_size=batch_size, verbose=0).ravel()

    history_df = pd.DataFrame(history.history)
    validation_threshold, _ = threshold_search(y_valid, validation_proba, None)
    validation_f1 = f1_score(
        y_valid,
        (validation_proba >= validation_threshold).astype(int),
        pos_label=1,
        zero_division=0,
    )
    models_dir.mkdir(parents=True, exist_ok=True)
    model.save(models_dir / "MLP.keras")
    return MLPArtifacts(
        model=model,
        preprocessor=transformer,
        history=history_df,
        validation_proba=validation_proba,
        validation_target=y_valid,
        test_proba=test_proba,
        config=mlp_config,
        validation_f1=float(validation_f1),
        validation_threshold=float(validation_threshold),
    )


def build_mlp(input_dim: int, seed: int, mlp_config: MLPConfig | None = None):
    import tensorflow as tf

    config = mlp_config or DEFAULT_MLP_CONFIG
    tf.keras.utils.set_random_seed(seed)
    inputs = tf.keras.Input(shape=(input_dim,), name="features")
    if config.block_type == "residual":
        x = _build_residual_stack(inputs, config, tf)
    elif config.block_type == "dense":
        x = _build_dense_stack(inputs, config, tf)
    else:
        raise ValueError(f"block_type no soportado para MLP: {config.block_type}")
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="cancer_probability")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=f"cancer_mlp_{config.name}")
    model.compile(
        optimizer=_build_optimizer(config, tf),
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=config.label_smoothing),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.AUC(name="auc_pr", curve="PR"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


class ValidationThresholdMetrics:
    """Callback que calcula el mejor F1/umbral en validacion al final de cada epoch."""

    def __init__(
        self,
        X_valid: np.ndarray,
        y_valid: np.ndarray,
        batch_size: int,
    ) -> None:
        self.X_valid = X_valid
        self.y_valid = np.asarray(y_valid).astype(int)
        self.batch_size = batch_size

    def set_model(self, model) -> None:
        self.model = model

    def set_params(self, params) -> None:
        self.params = params

    def __getattr__(self, name: str):
        if name.startswith("on_"):
            return self._noop
        raise AttributeError(name)

    def _noop(self, *args, **kwargs) -> None:
        return None

    def on_epoch_end(self, epoch: int, logs: dict[str, float] | None = None) -> None:
        logs = logs if logs is not None else {}
        validation_proba = self.model.predict(self.X_valid, batch_size=self.batch_size, verbose=0).ravel()
        threshold, table = threshold_search(self.y_valid, validation_proba, None)
        best = table.sort_values(
            ["f1_positive", "precision_positive", "recall_positive"],
            ascending=[False, False, False],
        ).iloc[0]
        logs["val_f1_best"] = float(best["f1_positive"])
        logs["val_precision_best"] = float(best["precision_positive"])
        logs["val_recall_best"] = float(best["recall_positive"])
        logs["val_threshold_best"] = float(threshold)


def _build_dense_stack(inputs, config: MLPConfig, tf):
    x = inputs
    for index, units in enumerate(config.hidden_units):
        x = _dense_norm_activation(x, units, config, tf, name=f"dense_{index + 1}")
        dropout = _dropout_for_layer(config.dropout, index)
        if dropout > 0:
            x = tf.keras.layers.Dropout(dropout, name=f"dropout_{index + 1}")(x)
    return x


def _build_residual_stack(inputs, config: MLPConfig, tf):
    x = inputs
    for index, units in enumerate(config.hidden_units):
        shortcut = x
        y = _dense_norm_activation(x, units, config, tf, name=f"residual_{index + 1}_dense_1")
        dropout = _dropout_for_layer(config.dropout, index)
        if dropout > 0:
            y = tf.keras.layers.Dropout(dropout, name=f"residual_{index + 1}_dropout_1")(y)
        y = tf.keras.layers.Dense(
            units,
            use_bias=not config.batch_norm,
            kernel_initializer=_kernel_initializer(config.activation),
            kernel_regularizer=_kernel_regularizer(config, tf),
            name=f"residual_{index + 1}_dense_2",
        )(y)
        if config.batch_norm:
            y = tf.keras.layers.BatchNormalization(name=f"residual_{index + 1}_bn_2")(y)
        if shortcut.shape[-1] != units:
            shortcut = tf.keras.layers.Dense(
                units,
                use_bias=False,
                kernel_initializer=_kernel_initializer(config.activation),
                kernel_regularizer=_kernel_regularizer(config, tf),
                name=f"residual_{index + 1}_projection",
            )(shortcut)
        x = tf.keras.layers.Add(name=f"residual_{index + 1}_add")([shortcut, y])
        x = tf.keras.layers.Activation(config.activation, name=f"residual_{index + 1}_activation")(x)
        if dropout > 0:
            x = tf.keras.layers.Dropout(dropout, name=f"residual_{index + 1}_dropout_2")(x)
    return x


def _dense_norm_activation(x, units: int, config: MLPConfig, tf, name: str):
    x = tf.keras.layers.Dense(
        units,
        use_bias=not config.batch_norm,
        kernel_initializer=_kernel_initializer(config.activation),
        kernel_regularizer=_kernel_regularizer(config, tf),
        name=name,
    )(x)
    if config.batch_norm:
        x = tf.keras.layers.BatchNormalization(name=f"{name}_bn")(x)
    return tf.keras.layers.Activation(config.activation, name=f"{name}_activation")(x)


def _build_optimizer(config: MLPConfig, tf):
    optimizer_name = config.optimizer.lower()
    kwargs = {"learning_rate": config.learning_rate}
    if config.clipnorm is not None:
        kwargs["clipnorm"] = config.clipnorm
    if optimizer_name == "adamw":
        return tf.keras.optimizers.AdamW(weight_decay=config.weight_decay, **kwargs)
    if optimizer_name == "adam":
        return tf.keras.optimizers.Adam(**kwargs)
    raise ValueError(f"Optimizador no soportado para MLP: {config.optimizer}")


def _kernel_regularizer(config: MLPConfig, tf):
    return tf.keras.regularizers.L2(config.l2) if config.l2 else None


def _kernel_initializer(activation: str) -> str:
    if activation.lower() == "selu":
        return "lecun_normal"
    return "he_normal"


def _dropout_for_layer(dropout: float | tuple[float, ...], index: int) -> float:
    if isinstance(dropout, tuple):
        return float(dropout[min(index, len(dropout) - 1)])
    return float(dropout)


def _monitor_should_maximize(monitor: str) -> bool:
    metric = monitor.lower()
    return any(token in metric for token in ("auc", "f1", "precision", "recall", "accuracy"))


def _as_float32(array) -> np.ndarray:
    if hasattr(array, "toarray"):
        array = array.toarray()
    return np.asarray(array, dtype=np.float32)


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False
