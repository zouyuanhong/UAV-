# -*- coding: utf-8 -*-
"""
深度学习模型模块
修改说明：
1. 移除未被调用的 train_deep_learning_model() 函数，消除冗余
2. build_model 中用 LayerNormalization 替代 BatchNormalization（小batch更稳定）
3. 删除重复的 read_csv_auto()，统一使用 main_app.py 中的版本
"""
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, Dropout, LayerNormalization
)
from tensorflow.keras.models import Model


def build_model(input_dim, cfg):
    """
    构建深度学习模型
    优化：使用 LayerNormalization 替代 BatchNormalization
          LayerNorm 不依赖 batch 统计量，小 batch_size 下更稳定
    """
    x_in = Input(shape=(input_dim,))
    x = x_in

    for i in range(cfg["n_layers"]):
        # 逐层减半神经元，最小16
        units = max(16, int(cfg["neurons"] / (2 ** i)))
        x = Dense(units, activation="relu")(x)
        x = LayerNormalization()(x)          # 优化：替换 BatchNorm
        x = Dropout(cfg["dropout"])(x)

    x_out = Dense(1, activation="linear")(x)
    model = Model(x_in, x_out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg["learning_rate"]),
        loss="mse",
        metrics=["mae"]
    )
    return model
