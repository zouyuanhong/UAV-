# -*- coding: utf-8 -*-
"""
Streamlit 主界面
适配解耦架构 + 多因子动态权重：predict_one_df 现在输出附加列
新增：批量下载全部预测结果为 ZIP
"""
import io
import os
import zipfile
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

from detector_selector import train_detector_selector
from deep_learning_model import build_model
from empirical_formula_model import fit_multi_var_model
from dynamic_fusion_model import (
    DEFAULTS, ARTIFACT_ROOT, train_and_save,
    load_bundle, predict_one_df
)


def read_csv_auto(uploaded_file):
    """自动兼容 header=1 或 header=0 的 CSV"""
    raw = uploaded_file.getvalue()
    for hdr in [1, 0]:
        try:
            df = pd.read_csv(io.BytesIO(raw), header=hdr)
            if "Time" in df.columns:
                return df
        except Exception:
            continue
    raise ValueError(f"文件 {uploaded_file.name} 无法识别，需包含 Time 列。")


# ===================== Streamlit UI =====================
st.set_page_config(page_title="火焰高度预测可视化系统", layout="wide")
st.title("🔥 火焰高度预测可视化系统（训练 / 预测分离）")

tab_train, tab_pred = st.tabs(["1) 深度学习训练界面", "2) 预测界面（批量）"])

# ────────── 训练界面 ──────────
with tab_train:
    st.subheader("上传训练数据（可多文件）")
    train_files = st.file_uploader(
        "选择训练CSV", type=["csv"], accept_multiple_files=True, key="train_upload"
    )

    st.markdown("### 参数设置")
    col1, col2, col3 = st.columns(3)
    with col1:
        neurons = st.slider("神经元数量（首层）", 32, 1024, DEFAULTS["neurons"], step=32)
        n_layers = st.slider("网络层数", 1, 6, DEFAULTS["n_layers"])
        dropout = st.slider("Dropout比例", 0.0, 0.6, float(DEFAULTS["dropout"]), step=0.01)
    with col2:
        lr = st.number_input(
            "学习率", min_value=1e-5, max_value=1e-1,
            value=float(DEFAULTS["learning_rate"]), format="%.5f"
        )
        batch_size = st.selectbox("批次大小", [4, 8, 16, 32, 64, 128], index=3)
        epochs = st.slider("训练轮数(Epochs)", 20, 1000, DEFAULTS["epochs"], step=10)
    with col3:
        patience = st.slider("早停耐心值", 3, 200, DEFAULTS["early_stop_patience"])
        bundle_name = st.text_input("模型保存名称", value="fire_height_model")

    current_cfg = DEFAULTS.copy()
    current_cfg.update({
        "neurons": neurons, "n_layers": n_layers, "dropout": dropout,
        "learning_rate": lr, "batch_size": batch_size,
        "epochs": epochs, "early_stop_patience": patience
    })

    # 参数对比表
    compare_rows = []
    for k in ["neurons", "n_layers", "dropout", "learning_rate",
              "batch_size", "epochs", "early_stop_patience"]:
        compare_rows.append({
            "参数": k,
            "默认值": DEFAULTS[k],
            "当前值": current_cfg[k],
            "是否调整": "是" if DEFAULTS[k] != current_cfg[k] else "否"
        })
    st.dataframe(pd.DataFrame(compare_rows), use_container_width=True)

    if st.button("开始训练并保存模型", type="primary"):
        if not train_files:
            st.error("请先上传训练数据。")
        else:
            try:
                dfs = [read_csv_auto(f) for f in train_files]
                with st.spinner("训练中，请稍候..."):
                    bundle_dir, metrics, fig = train_and_save(
                        dfs, current_cfg, bundle_name,
                        train_detector_selector,
                        fit_multi_var_model,
                        build_model
                    )
                st.success(f"训练完成，已保存到：{bundle_dir}")

                c1, c2, c3 = st.columns(3)
                c1.metric("测试集 R²", f"{metrics['test_r2']:.4f}")
                c2.metric("测试集 RMSE", f"{metrics['test_rmse']:.4f}")
                c3.metric("测试集 MAE", f"{metrics['test_mae']:.4f}")

                st.pyplot(fig)
            except Exception as e:
                st.error(f"训练失败：{e}")

# ────────── 预测界面 ──────────
with tab_pred:
    st.subheader("加载已训练模型并进行批量预测")

    bundle_dirs = sorted(
        [os.path.join(ARTIFACT_ROOT, d) for d in os.listdir(ARTIFACT_ROOT)
         if os.path.isdir(os.path.join(ARTIFACT_ROOT, d))],
        reverse=True
    )

    if not bundle_dirs:
        st.info("暂无已保存模型，请先在训练界面streamlit run main_app.py训练一次。")
    else:
        selected_bundle = st.selectbox("选择本地模型目录", bundle_dirs)

        pred_files = st.file_uploader(
            "选择预测CSV（可批量）", type=["csv"],
            accept_multiple_files=True, key="pred_upload"
        )
        if st.button("开始批量预测", type="primary"):
            if not pred_files:
                st.error("请先上传预测数据。")
            else:
                try:
                    (dl_model, fusion_mlp, scaler, features,
                     formula_params, cfg_loaded,
                     selected_detectors) = load_bundle(selected_bundle)
                    st.success("模型加载成功。")

                    # ── 收集全部预测结果，用于批量下载 ──
                    all_results = {}  # {文件名: DataFrame}

                    progress = st.progress(0)
                    for idx, f in enumerate(pred_files):
                        df = read_csv_auto(f)
                        out = predict_one_df(
                            df, dl_model, fusion_mlp, scaler,
                            features, formula_params,
                            cfg_loaded, selected_detectors
                        )

                        # 保存到汇总字典
                        result_name = f"预测结果_{os.path.splitext(f.name)[0]}.csv"
                        all_results[result_name] = out

                        # ── 图1：预测曲线 ──
                        fig2, ax = plt.subplots(figsize=(12, 5))
                        ax.plot(out["Time"], out["Final_Fusion_Pred"],
                                "r-", lw=2.8, label="最终融合预测")
                        ax.plot(out["Time"], out["DL_Pred"],
                                "b--", lw=1.2, alpha=0.6, label="DL 模型预测")
                        ax.plot(out["Time"], out["Formula_Pred"],
                                "g--", lw=1.2, alpha=0.6, label="经验公式预测")
                        ax.axhline(cfg_loaded["MAX_HEIGHT"], color="orange",
                                   ls=":", lw=1.5,
                                   label=f"{cfg_loaded['MAX_HEIGHT']}m上限")
                        ax.axvline(cfg_loaded["START_GROW"], color="gray",
                                   ls=":", lw=1.5,
                                   label=f"{cfg_loaded['START_GROW']}s开始增长")
                        ax.set_title(f"预测曲线：{f.name}")
                        ax.set_xlabel("时间 (s)")
                        ax.set_ylabel("火焰高度 (m)")
                        ax.grid(alpha=0.3)
                        ax.legend()
                        st.pyplot(fig2)

                        # ── 图2：动态权重变化曲线 ──
                        fig3, ax2 = plt.subplots(figsize=(12, 3.5))
                        ax2.plot(out["Time"], out["DL_Weight"],
                                 "b-", lw=2, label="DL 模型权重 α")
                        ax2.plot(out["Time"], out["Formula_Weight"],
                                 "g-", lw=2, label="经验公式权重 1-α")
                        ax2.fill_between(out["Time"], 0, out["DL_Weight"],
                                         alpha=0.15, color="blue")
                        ax2.fill_between(out["Time"], 0, out["Formula_Weight"],
                                         alpha=0.15, color="green")
                        ax2.set_ylim(-0.05, 1.05)
                        ax2.set_title(f"动态融合权重：{f.name}")
                        ax2.set_xlabel("时间 (s)")
                        ax2.set_ylabel("权重")
                        ax2.grid(alpha=0.3)
                        ax2.legend()
                        st.pyplot(fig3)

                        # ── 数据预览（含全部中间值） ──
                        preview_cols = [
                            "Time", "DL_Pred", "Formula_Pred",
                            "DL_Weight", "Formula_Weight", "Final_Fusion_Pred"
                        ]
                        available_cols = [c for c in preview_cols if c in out.columns]
                        st.markdown(f"**{f.name}** 预测结果预览（前20行）")
                        st.dataframe(out[available_cols].head(20),
                                     use_container_width=True)

                        progress.progress((idx + 1) / len(pred_files))

                    # ══════════════════════════════════════════════
                    #  全部预测完成后：批量下载功能
                    # ══════════════════════════════════════════════
                    st.markdown("---")
                    st.subheader("📦 批量下载全部预测结果")

                    # 显示汇总信息
                    st.markdown(f"共完成 **{len(all_results)}** 个文件的预测：")
                    summary_rows = []
                    for name, df_out in all_results.items():
                        summary_rows.append({
                            "文件名": name,
                            "数据行数": len(df_out),
                            "列数": len(df_out.columns),
                            "Final_Fusion_Pred 最大值": f"{df_out['Final_Fusion_Pred'].max():.2f} m",
                        })
                    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

                    # ── 批量下载按钮：打包为 ZIP ──
                    zip_buffer = io.BytesIO()
                    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                        for name, df_out in all_results.items():
                            csv_bytes = df_out.to_csv(index=False).encode("utf-8-sig")
                            zf.writestr(name, csv_bytes)
                    zip_buffer.seek(0)

                    st.download_button(
                        label=f"⬇️ 一键下载全部 ({len(all_results)} 个文件, ZIP)",
                        data=zip_buffer,
                        file_name="火焰高度预测结果_批量下载.zip",
                        mime="application/zip",
                        type="primary",
                    )

                    # ── 逐个下载按钮（可选展开） ──
                    with st.expander("展开查看逐个文件下载", expanded=False):
                        for name, df_out in all_results.items():
                            csv_bytes = df_out.to_csv(index=False).encode("utf-8-sig")
                            st.download_button(
                                label=f"下载 {name}",
                                data=csv_bytes,
                                file_name=name,
                                mime="text/csv",
                                key=f"dl_single_{name}",
                            )

                except Exception as e:
                    st.error(f"预测失败：{e}")
