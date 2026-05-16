# -*- coding: utf-8 -*-
"""
经验公式模型模块
修改说明：
1. fit_multi_var_model 添加拟合质量打印（残差中位数）
2. predict_multi_var_formula 后段强制一阶导数单增（加速生长），触墙后贴住
"""
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


def exponential_saturation_func_vec(t, T, HRR, a, b, c, cfg):
    """指数饱和函数：h = H_max * (1 - exp(-a * growth^b / c))"""
    growth = np.clip(t - cfg["START_GROW"], 0, None) * (T / 200.0) * (HRR / 1000.0)
    h = cfg["MAX_HEIGHT"] * (1 - np.exp(
        -a * np.power(np.maximum(growth, 0), b) / (c + 1e-6)
    ))
    h = np.where(t < cfg["START_GROW"], 0.0, h)
    return np.minimum(h, cfg["MAX_HEIGHT"])


def fit_multi_var_model(data, cfg, selected_detectors):
    """
    拟合多变量经验公式，返回参数 [a, b, c]

    优化：打印拟合残差中位数，便于调试
    """
    need = ["Time", "fire height", "HRR"] + selected_detectors
    for c in need:
        if c not in data.columns:
            raise ValueError(f"训练数据缺少必要列：{c}")

    df_valid = data[
        (data["Time"] >= cfg["START_GROW"]) & (data["fire height"] > 0.1)
    ].copy()

    if len(df_valid) < 10:
        print("[经验公式] 有效样本不足(<10)，使用默认参数 [0.02, 1.2, 0.8]")
        return [0.02, 1.2, 0.8]

    t = df_valid["Time"].values
    T = df_valid[selected_detectors].mean(axis=1).values
    H = df_valid["HRR"].values
    y = df_valid["fire height"].values

    def f(x, a, b, c):
        return exponential_saturation_func_vec(x[0], x[1], x[2], a, b, c, cfg)

    try:
        popt, _ = curve_fit(
            f, (t, T, H), y,
            bounds=([0.001, 0.5, 0.3], [0.1, 3.0, 3.0]),
            maxfev=10000
        )
        residuals = np.abs(y - f((t, T, H), *popt))
        print(f"[经验公式] 拟合完成: a={popt[0]:.4f}, b={popt[1]:.4f}, c={popt[2]:.4f}")
        print(f"[经验公式] 残差中位数: {np.median(residuals):.3f}m, "
              f"90%分位: {np.percentile(residuals, 90):.3f}m")
        return popt.tolist()
    except Exception as e:
        print(f"[经验公式] 拟合失败({e})，使用默认参数")
        return [0.02, 1.2, 0.8]


def predict_multi_var_formula(df, params, cfg, selected_detectors):
    """
    使用经验公式进行预测。

    后段（t >= 60s）强制规则：
      ① 一阶导数（斜率）单调递增：火焰高度加速上升，不会减速
      ② 上限为 MAX_HEIGHT (53.4m)，触及后贴墙

    数学含义：d²h/dt² >= 0（凸函数），曲线下凸，越来越陡，
    直到撞到 53.4m 天花板后变为水平。
    """
    a, b, c = params
    cfg = {k: float(v) for k, v in cfg.items()}

    t = df["Time"].values
    T = df[selected_detectors].mean(axis=1).values
    H = df["HRR"].values

    pred = np.zeros(len(df), dtype=float)

    # 阶段1：t <= 20s → 恒定初始高度
    pred[t <= cfg["STAGE1_END"]] = cfg["INIT_HEIGHT"]

    # 过渡段：20s < t < 60s → 最小高度
    mask_mid = (t > cfg["STAGE1_END"]) & (t < cfg["STAGE2_END"])
    pred[mask_mid] = cfg["MIN_HEIGHT"]

    # 阶段2：t >= 60s → 经验公式原始计算
    mask_grow = t >= cfg["START_GROW"]
    if np.any(mask_grow):
        raw_grow = exponential_saturation_func_vec(
            t[mask_grow], T[mask_grow], H[mask_grow], a, b, c, cfg
        )
        pred[mask_grow] = raw_grow

    # ── 后段强制一阶导数单增 ──
    idx = np.where(mask_grow)[0]
    if len(idx) > 1:
        n = len(idx)

        # ① 计算离散斜率（一阶差分）
        slopes = np.array([
            pred[idx[j + 1]] - pred[idx[j]] for j in range(n - 1)
        ])

        # ② 斜率不得为负（高度只增不减）
        slopes = np.maximum(slopes, 0.0)

        # ③ 强制斜率单调递增：若 slope[j] < slope[j-1]，则向上修正
        for j in range(1, len(slopes)):
            if slopes[j] < slopes[j - 1]:
                slopes[j] = slopes[j - 1]

        # ④ 从第一个增长点开始，用修正后的斜率重建高度序列
        for j in range(n - 1):
            pred[idx[j + 1]] = pred[idx[j]] + slopes[j]

        # ⑤ 上限截断 + 触墙后贴住
        hit_wall = False
        for i in idx:
            if hit_wall:
                pred[i] = cfg["MAX_HEIGHT"]
            elif pred[i] >= cfg["MAX_HEIGHT"]:
                pred[i] = cfg["MAX_HEIGHT"]
                hit_wall = True

    return pred
