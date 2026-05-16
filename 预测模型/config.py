# -*- coding: utf-8 -*-
"""
统一配置中心：所有常量、阶段阈值、默认超参数集中管理
消除各文件中重复硬编码的问题
"""

DEFAULTS = {
    # === 模型超参数 ===
    "seed": 42,
    "test_size": 0.2,
    "neurons": 256,
    "n_layers": 3,
    "dropout": 0.10,
    "learning_rate": 1e-3,
    "batch_size": 32,          # 优化：从8提升到32，配合BatchNorm更稳定
    "epochs": 400,
    "early_stop_patience": 25,

    # === 物理阶段规则 ===
    "MAX_HEIGHT": 53.4,
    "MIN_HEIGHT": 0.0,
    "INIT_HEIGHT": 0.6,
    "STAGE1_END": 20.0,        # t <= 20s：恒定初始高度
    "STAGE2_END": 60.0,        # 20s < t < 60s：过渡区间
    "START_GROW": 60.0,        # t >= 60s：开始增长
    "DROP_LIMIT": -0.2,        # 单步最大降幅
    "START_STABLE": 100.0,     # t >= 100s：进入稳定期（不允许下降）

    # === 融合层参数 ===
    "fusion_epochs": 80,       # 优化：从50提升到80
    "fusion_patience": 15,     # 优化：从10提升到15
}

ARTIFACT_ROOT = "artifacts"
