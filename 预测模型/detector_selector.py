# -*- coding: utf-8 -*-
"""
探测器选择器模块
修改说明：
1. 训练前对数据进行 shuffle，避免时序偏移导致划分不均
2. 添加验证集 R² 打印，便于监控选择器质量
3. build_model 命名避免与其他模块冲突（函数名加前缀）
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from sklearn.metrics import r2_score


def _build_selector_net(n_detectors, cfg):
    """构建探测器选择器网络"""
    inp = Input(shape=(n_detectors,))
    x = Dense(64, activation='relu')(inp)
    x = Dense(32, activation='relu')(x)
    x = Dense(16, activation='relu')(x)
    out = Dense(1, activation='linear')(x)
    model = Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg["learning_rate"]),
        loss='mse'
    )
    return model


def train_detector_selector(data, cfg):
    """
    训练探测器选择器，返回最重要的 top-3 探测器列名

    优化点：
    - 训练前 shuffle 数据，避免时序偏移
    - 打印验证集 R² 供调试
    """
    detector_cols = [col for col in data.columns if col.startswith('Device')]
    if not detector_cols:
        raise ValueError("数据中未找到以 'Device' 开头的探测器列。")

    X_det = data[detector_cols].values
    y = data['fire height'].values

    # 优化：先 shuffle 再划分，避免时序偏移
    rng = np.random.RandomState(cfg.get("seed", 42))
    indices = rng.permutation(len(X_det))
    X_det = X_det[indices]
    y = y[indices]

    split_idx = int(0.8 * len(X_det))
    X_train, X_val = X_det[:split_idx], X_det[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    selector_model = _build_selector_net(len(detector_cols), cfg)
    selector_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100, batch_size=16, verbose=0
    )

    # 评估选择器质量
    y_val_pred = selector_model.predict(X_val, verbose=0).flatten()
    val_r2 = r2_score(y_val, y_val_pred)
    print(f"[探测器选择器] 验证集 R² = {val_r2:.4f}")

    # 按第一层权重绝对值排序
    first_layer_weights = np.abs(selector_model.layers[1].get_weights()[0]).flatten()
    detector_importance = sorted(
        zip(detector_cols, first_layer_weights),
        key=lambda x: x[1], reverse=True
    )

    top_n = min(3, len(detector_cols))
    selected = [det[0] for det in detector_importance[:top_n]]
    print(f"[探测器选择器] 选定探测器: {selected}")
    return selected
