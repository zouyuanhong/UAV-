# -*- coding: utf-8 -*-
"""
动态融合模型模块（解耦架构 + 多因子动态权重版）

核心改动：
1. 融合 MLP 输入扩展为 [dl_pred, formula_pred, time_norm, hrr_norm, detector_temps...]
2. DL模型、经验公式、融合MLP 三个组件独立保存、独立加载
3. 预测时在 Python 中完成融合，不依赖 Keras 图序列化
4. predict_one_df 输出附加列：DL预测值、经验公式预测值、近似动态权重
5. DL模型使用全部列作为输入特征
6. 阶段2允许下降但限制降幅，最终输出做平滑处理
7. DL预测值整体下移，使其在60s时从零开始增长
"""
import os
import json
import time
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from config import DEFAULTS, ARTIFACT_ROOT
from empirical_formula_model import predict_multi_var_formula

warnings.filterwarnings("ignore")
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False


# ===================== DL 预测值下移 =====================
def _shift_dl_to_zero_at_grow(time_seq, dl_pred, cfg):
    """
    将 DL 预测值整体下移，使其在 START_GROW (60s) 时刻从零开始。

    做法：找到时间序列中最接近 60s 的样本点，取该点的 dl_pred 值作为偏移量，
    整体减去该偏移量，再将 60s 之前的值截断为 0（避免出现负值）。
    """
    grow_time = cfg["START_GROW"]  # 60.0

    # 找到最接近 60s 的样本索引
    idx_at_grow = np.argmin(np.abs(time_seq - grow_time))
    offset = dl_pred[idx_at_grow]

    shifted = dl_pred - offset

    # 60s 之前不应有预测高度，强制归零
    shifted[time_seq < grow_time] = 0.0

    # 下移后不应出现负值
    shifted = np.maximum(shifted, 0.0)

    return shifted


# ===================== 物理规则修正 =====================
def correct_flame_height(time_seq, pred_seq, cfg):
    """
    基于物理规则修正火焰高度预测。

    阶段划分：
      - t <= 20s          → 恒定初始高度 0.6m
      - 20s < t < 60s     → 过渡区间，高度为 0
      - 60s <= t < 100s   → 增长段，允许下降但降幅受限
      - t >= 100s         → 稳定期，不允许下降

    阶段2下降规则：
      降幅上限 = min(当前高度 × 0.2, 0.6m)

    最终平滑处理：
      消除台阶和陡增，确保从 60s 起缓慢上升
    """
    cfg = {k: float(v) for k, v in cfg.items()}
    corrected = pred_seq.copy().astype(float)

    # ── 阶段1：t <= 20s → 恒定初始高度 ──
    corrected[time_seq <= cfg["STAGE1_END"]] = cfg["INIT_HEIGHT"]

    # ── 过渡段：20s < t < 60s → 最小高度 ──
    mask_mid = (time_seq > cfg["STAGE1_END"]) & (time_seq < cfg["STAGE2_END"])
    corrected[mask_mid] = cfg["MIN_HEIGHT"]

    # ── 阶段2：t >= 60s → 约束上限，允许下降但限制降幅 ──
    idx = np.where(time_seq >= cfg["START_GROW"])[0]
    for i in idx:
        corrected[i] = min(corrected[i], cfg["MAX_HEIGHT"])

        if i > 0 and time_seq[i - 1] >= cfg["START_GROW"]:
            prev = corrected[i - 1]
            max_drop = min(0.2 * prev, 0.6)
            if corrected[i] < prev - max_drop:
                corrected[i] = prev - max_drop

        if time_seq[i] >= cfg["START_STABLE"] and i > 0:
            if corrected[i] < corrected[i - 1]:
                corrected[i] = corrected[i - 1]

    corrected = np.clip(corrected, cfg["MIN_HEIGHT"], cfg["MAX_HEIGHT"])

    # ── 最终平滑处理 ──
    corrected = _smooth_prediction(time_seq, corrected, cfg)

    return corrected


def _smooth_prediction(time_seq, pred_seq, cfg):
    """
    最终平滑处理：
      1. 从 60s 起高度从 0 缓慢上升，而非突然跳变
      2. 消除曲线前中段的"台阶"和陡增
      3. 末尾接近 MAX_HEIGHT (53.4m) 的段落不做平滑
    """
    smoothed = pred_seq.copy().astype(float)

    grow_mask = time_seq >= cfg["START_GROW"]
    if not np.any(grow_mask):
        return smoothed

    grow_indices = np.where(grow_mask)[0]

   # ① 增长起始段 smoothstep 渐变
    fade_duration = 200.0
    fade_start = cfg["START_GROW"]
    fade_end = fade_start + fade_duration

    for i in grow_indices:
        t = time_seq[i]
        if t < fade_end:
            x = np.clip((t - fade_start) / fade_duration, 0.0, 1.0)
            factor = x * x * (3.0 - 2.0 * x)
            smoothed[i] *= factor

    # ② EMA 平滑，仅在远离上限时生效
    ema_alpha = 0.3
    smooth_threshold = 0.95 * cfg["MAX_HEIGHT"]

    for j in range(1, len(grow_indices)):
        i_curr = grow_indices[j]
        i_prev = grow_indices[j - 1]
        if smoothed[i_prev] < smooth_threshold:
            smoothed[i_curr] = ema_alpha * smoothed[i_curr] + (1.0 - ema_alpha) * smoothed[i_prev]

    # ③ 稳定期二次保护
    for j in range(1, len(grow_indices)):
        i_curr = grow_indices[j]
        i_prev = grow_indices[j - 1]
        if time_seq[i_curr] >= cfg["START_STABLE"]:
            if smoothed[i_curr] < smoothed[i_prev]:
                smoothed[i_curr] = smoothed[i_prev]

    smoothed = np.clip(smoothed, cfg["MIN_HEIGHT"], cfg["MAX_HEIGHT"])
    return smoothed


# ===================== 融合输入构建 =====================
def _build_fusion_inputs(dl_pred, formula_pred, data_subset,
                         selected_detectors):
    """
    组装融合 MLP 的输入特征。
    输入维度：2 + 1 + 1 + N = 4 + N
    """
    time_norm = data_subset["Time"].values / 200.0
    hrr_norm = data_subset["HRR"].values / 1000.0
    detector_temps = data_subset[selected_detectors].values

    parts = [
        dl_pred.reshape(-1, 1),
        formula_pred.reshape(-1, 1),
        time_norm.reshape(-1, 1),
        hrr_norm.reshape(-1, 1),
        detector_temps
    ]
    return np.concatenate(parts, axis=1).astype(np.float32)


def _get_fusion_input_dim(selected_detectors):
    """计算融合 MLP 的输入维度"""
    return 4 + len(selected_detectors)


# ===================== 融合 MLP 构建 =====================
def build_fusion_mlp(input_dim, cfg):
    """
    构建融合权重 MLP。
    输入: [dl_pred, formula_pred, time_norm, hrr_norm, detector_temps...]  (4+N 维)
    输出: 融合后的火焰高度预测                                              (1维)
    """
    inp = Input(shape=(input_dim,))
    x = Dense(32, activation='relu')(inp)
    x = Dense(16, activation='relu')(x)
    x = Dense(8, activation='relu')(x)
    out = Dense(1, activation='linear')(x)
    model = Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg["learning_rate"]),
        loss='mse',
        metrics=['mae']
    )
    return model


# ===================== 辅助函数 =====================
def _resolve_target_col(df):
    for col in ['fire height', '火焰高度']:
        if col in df.columns:
            return col
    raise ValueError("数据中未找到 'fire height' 或 '火焰高度' 列")


def top_feature_importance_by_corr(df, features, target, topk=15):
    vals = []
    y = df[target].values
    for f in features:
        x = df[f].values
        if np.std(x) < 1e-12:
            continue
        corr = np.corrcoef(x, y)[0, 1]
        vals.append((f, abs(corr) if not np.isnan(corr) else 0.0))
    return sorted(vals, key=lambda z: z[1], reverse=True)[:topk]


def make_report_figure(y_train, pred_train, y_test, pred_test, feat_importance):
    train_res = pred_train - y_train
    test_res = pred_test - y_test

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    ax1, ax2, ax3, ax4 = axes.flatten()

    ax1.scatter(y_train, pred_train, s=35, alpha=0.8, color="#4C9AC5")
    m1 = min(y_train.min(), pred_train.min())
    m2 = max(y_train.max(), pred_train.max())
    ax1.plot([m1, m2], [m1, m2], "r--", lw=2, label="理想预测线")
    ax1.set_title(f"训练集性能\nR²={r2_score(y_train, pred_train):.4f}")
    ax1.set_xlabel("实际火焰高度 (m)")
    ax1.set_ylabel("预测火焰高度 (m)")
    ax1.grid(alpha=0.3)
    ax1.legend()

    ax2.scatter(y_test, pred_test, s=45, alpha=0.8, color="#C05A8A")
    m1 = min(y_test.min(), pred_test.min())
    m2 = max(y_test.max(), pred_test.max())
    ax2.plot([m1, m2], [m1, m2], "r--", lw=2, label="理想预测线")
    ax2.set_title(f"测试集性能\nR²={r2_score(y_test, pred_test):.4f}")
    ax2.set_xlabel("实际火焰高度 (m)")
    ax2.set_ylabel("预测火焰高度 (m)")
    ax2.grid(alpha=0.3)
    ax2.legend()

    ax3.hist(train_res, bins=20, alpha=0.7, color="#E79B24", label="训练集误差")
    ax3.hist(test_res, bins=20, alpha=0.7, color="#D45A3A", label="测试集误差")
    ax3.axvline(0, color="k", linestyle="--", lw=1.8, label="零误差线")
    ax3.set_title("预测误差分布")
    ax3.set_xlabel("预测误差 (m)")
    ax3.set_ylabel("频次")
    ax3.grid(alpha=0.3)
    ax3.legend()

    if feat_importance:
        names = [x[0] for x in feat_importance][::-1]
        scores = [x[1] for x in feat_importance][::-1]
        ax4.barh(names, scores, color="#7EA46B")
    ax4.set_title("前15个重要特征（相关性近似）")
    ax4.set_xlabel("特征重要性")
    ax4.grid(axis="x", alpha=0.3)

    fig.suptitle("火焰高度预测模型分析报告", fontsize=18)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


# ===================== 模型训练入口 =====================
def train_and_save(train_dfs, cfg, bundle_name,
                   train_detector_selector,
                   fit_multi_var_model,
                   build_model):
    """
    三层训练流水线：
      第一层  → 探测器选择
      第二层A → 经验公式拟合
      第二层B → DL模型训练（全部列作为输入）
      第三层  → 融合MLP训练
    """
    np.random.seed(cfg["seed"])
    tf.random.set_seed(cfg["seed"])

    data = pd.concat(train_dfs, ignore_index=True)

    target_col = _resolve_target_col(data)
    if target_col != "fire height":
        data = data.rename(columns={target_col: "fire height"})
        print(f"[训练] 目标列 '{target_col}' 已统一重命名为 'fire height'")

    req = ["Time", "fire height", "HRR"]
    miss = [c for c in req if c not in data.columns]
    if miss:
        raise ValueError(f"训练数据缺少必要列: {miss}")
    data = data.dropna(subset=req).copy()

    # ══════════ 第一层：探测器选择 ══════════
    selected_detectors = train_detector_selector(data, cfg)
    print(f"[第一层完成] 选定探测器: {selected_detectors}")

    # ══════════ 第二层A：经验公式拟合 ══════════
    formula_params = fit_multi_var_model(data, cfg, selected_detectors)
    print(f"[第二层A完成] 经验公式参数: "
          f"a={formula_params[0]:.4f}, b={formula_params[1]:.4f}, c={formula_params[2]:.4f}")

    # ══════════ 第二层B：DL模型训练 ══════════
    target = "fire height"
    features = [c for c in data.columns if c != target]
    print(f"[第二层B] DL 输入特征数: {len(features)}")
    if not features:
        raise ValueError("没有可用训练特征，请检查数据列。")

    X = data[features].values
    y = data[target].values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    indices = np.arange(len(Xs))
    train_idx, test_idx = train_test_split(
        indices, test_size=cfg["test_size"], random_state=cfg["seed"]
    )

    X_train, X_test = Xs[train_idx], Xs[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    data_train = data.iloc[train_idx].reset_index(drop=True)
    data_test = data.iloc[test_idx].reset_index(drop=True)

    dl_model = build_model(X_train.shape[1], cfg)
    dl_callbacks = [
        EarlyStopping(monitor="val_loss", patience=cfg["early_stop_patience"],
                       restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                           patience=max(3, cfg["early_stop_patience"] // 2),
                           min_lr=1e-6)
    ]
    dl_model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        callbacks=dl_callbacks,
        verbose=0
    )
    print("[第二层B完成] DL 模型训练完成")

    # ══════════ 第三层：融合MLP训练 ══════════
    # 获取 DL 预测值，并整体下移使其在 60s 时从零开始
    dl_train_pred_raw = dl_model.predict(X_train, verbose=0).flatten()
    dl_test_pred_raw = dl_model.predict(X_test, verbose=0).flatten()

    dl_train_pred = _shift_dl_to_zero_at_grow(
        data_train["Time"].values, dl_train_pred_raw, cfg
    )
    dl_test_pred = _shift_dl_to_zero_at_grow(
        data_test["Time"].values, dl_test_pred_raw, cfg
    )
    print("[第二层B完成] DL 预测值已下移，60s 时从零开始")

    formula_train_pred = predict_multi_var_formula(
        data_train, formula_params, cfg, selected_detectors
    )
    formula_test_pred = predict_multi_var_formula(
        data_test, formula_params, cfg, selected_detectors
    )

    fusion_X_train = _build_fusion_inputs(
        dl_train_pred, formula_train_pred, data_train, selected_detectors
    )
    fusion_X_test = _build_fusion_inputs(
        dl_test_pred, formula_test_pred, data_test, selected_detectors
    )

    fusion_input_dim = _get_fusion_input_dim(selected_detectors)
    print(f"[第三层] 融合 MLP 输入维度: {fusion_input_dim}")

    fusion_mlp = build_fusion_mlp(fusion_input_dim, cfg)
    fusion_callbacks = [
        EarlyStopping(monitor="val_loss", patience=cfg.get("fusion_patience", 15),
                       restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=8, min_lr=1e-6)
    ]
    fusion_mlp.fit(
        fusion_X_train, y_train,
        validation_data=(fusion_X_test, y_test),
        epochs=cfg.get("fusion_epochs", 80),
        batch_size=cfg["batch_size"],
        callbacks=fusion_callbacks,
        verbose=0
    )
    print("[第三层完成] 融合 MLP 训练完成")

    # ── 评估 ──
    pred_train = fusion_mlp.predict(fusion_X_train, verbose=0).flatten()
    pred_test = fusion_mlp.predict(fusion_X_test, verbose=0).flatten()

    metrics = {
        "train_r2": float(r2_score(y_train, pred_train)),
        "test_r2": float(r2_score(y_test, pred_test)),
        "train_rmse": float(np.sqrt(mean_squared_error(y_train, pred_train))),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, pred_test))),
        "train_mae": float(mean_absolute_error(y_train, pred_train)),
        "test_mae": float(mean_absolute_error(y_test, pred_test)),
    }
    print(f"[评估] 测试集 R²={metrics['test_r2']:.4f}, "
          f"RMSE={metrics['test_rmse']:.4f}m, MAE={metrics['test_mae']:.4f}m")

    feat_imp = top_feature_importance_by_corr(data, features, target, topk=15)
    report_fig = make_report_figure(y_train, pred_train, y_test, pred_test, feat_imp)

    ts = time.strftime("%Y%m%d_%H%M%S")
    bundle_dir = os.path.join(ARTIFACT_ROOT, f"{bundle_name}_{ts}")
    os.makedirs(bundle_dir, exist_ok=True)

    dl_model.save(os.path.join(bundle_dir, "dl_model.keras"))
    fusion_mlp.save(os.path.join(bundle_dir, "fusion_mlp.keras"))
    joblib.dump(scaler, os.path.join(bundle_dir, "scaler.joblib"))

    for name, obj in [
        ("features.json", features),
        ("formula_params.json",
         {"a": formula_params[0], "b": formula_params[1], "c": formula_params[2]}),
        ("config.json", cfg),
        ("selected_detectors.json", selected_detectors),
        ("metrics.json", metrics),
    ]:
        with open(os.path.join(bundle_dir, name), "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)

    report_fig.savefig(os.path.join(bundle_dir, "训练分析报告.png"),
                        dpi=300, bbox_inches="tight")
    print(f"[保存] 模型已保存到: {bundle_dir}")

    return bundle_dir, metrics, report_fig


# ===================== 模型加载与预测 =====================
def load_bundle(bundle_dir):
    required = [
        "dl_model.keras", "fusion_mlp.keras", "scaler.joblib",
        "features.json", "formula_params.json", "config.json",
        "selected_detectors.json"
    ]
    missing = [f for f in required
               if not os.path.exists(os.path.join(bundle_dir, f))]
    if missing:
        raise FileNotFoundError(
            f"模型包缺少文件: {missing}\n"
            f"请使用新代码重新训练模型（旧模型包格式不兼容）。"
        )

    dl_model = tf.keras.models.load_model(
        os.path.join(bundle_dir, "dl_model.keras"), compile=False
    )
    fusion_mlp = tf.keras.models.load_model(
        os.path.join(bundle_dir, "fusion_mlp.keras"), compile=False
    )
    scaler = joblib.load(os.path.join(bundle_dir, "scaler.joblib"))

    def _j(name):
        with open(os.path.join(bundle_dir, name), "r", encoding="utf-8") as f:
            return json.load(f)

    features = _j("features.json")
    fp = _j("formula_params.json")
    formula_params = [fp["a"], fp["b"], fp["c"]]
    cfg = _j("config.json")
    selected_detectors = _j("selected_detectors.json")

    return dl_model, fusion_mlp, scaler, features, formula_params, cfg, selected_detectors


def _estimate_dynamic_weights(dl_pred, formula_pred, fused_pred):
    diff = dl_pred - formula_pred
    alpha = np.where(
        np.abs(diff) > 1e-6,
        (fused_pred - formula_pred) / diff,
        0.5
    )
    return np.clip(alpha, 0.0, 1.0)


def predict_one_df(df, dl_model, fusion_mlp, scaler, features,
                    formula_params, cfg, selected_detectors):
    """
    对单个 DataFrame 执行预测。

    流程：
      DL预测 → 下移至60s从零开始 → 经验公式预测 → 融合MLP → 物理修正 + 平滑

    输出 DataFrame 附加列：
      - DL_Pred_Raw       — DL 原始预测值（下移前）
      - DL_Pred           — DL 预测值（下移后，60s时为零）
      - Formula_Pred      — 经验公式预测值
      - DL_Weight         — DL 近似动态权重
      - Formula_Weight    — 经验公式近似动态权重
      - Final_Fusion_Pred — 最终融合预测值
    """
    for c in ["Time", "HRR"] + selected_detectors:
        if c not in df.columns:
            raise ValueError(f"预测文件缺少必要列: {c}")

    lack = [f for f in features if f not in df.columns]
    if lack:
        raise ValueError(f"预测文件缺少训练特征列（前10个）: {lack[:10]}")

    # ① DL 模型原始预测
    Xs = scaler.transform(df[features].values)
    dl_pred_raw = dl_model.predict(Xs, verbose=0).flatten()

    # ② DL 预测值下移，使其在 60s 时从零开始
    dl_pred = _shift_dl_to_zero_at_grow(
        df["Time"].values, dl_pred_raw, cfg
    )

    # ③ 经验公式预测
    formula_pred = predict_multi_var_formula(
        df, formula_params, cfg, selected_detectors
    )

    # ④ 融合 MLP 预测
    fusion_input = _build_fusion_inputs(
        dl_pred, formula_pred, df, selected_detectors
    )
    fused_pred = fusion_mlp.predict(fusion_input, verbose=0).flatten()

    # ⑤ 近似动态权重估计
    dl_weight = _estimate_dynamic_weights(dl_pred, formula_pred, fused_pred)
    formula_weight = 1.0 - dl_weight

    # ⑥ 物理规则修正 + 平滑
    final_pred = correct_flame_height(df["Time"].values, fused_pred, cfg)

    out = df.copy()
    out["DL_Pred_Raw"] = dl_pred_raw
    out["DL_Pred"] = dl_pred
    out["Formula_Pred"] = formula_pred
    out["DL_Weight"] = dl_weight
    out["Formula_Weight"] = formula_weight
    out["Final_Fusion_Pred"] = final_pred
    return out
