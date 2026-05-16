import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
from pathlib import Path

# -------------------------- 基础配置 --------------------------
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 200

# 颜色库（沿用原程序）
colors = [
    '#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00',
    '#a65628', '#f781bf', '#66c2a5', '#fc8d62', '#1b9e77'
]

# 参数名称与单位（根据实际实验修改）
PARAM_NAMES = ["下洗气流流速", "无人机高度", "无人机距离"]
PARAM_UNITS = ["m_s", "m", "m"]

# 基准工况 0_0_0_0 的固定替换值
BASELINE_OVERRIDE_VALUE = 691.02

# 异常值关键词列表
ABNORMAL_KEYWORDS = [
    "未找到", "未达到", "未检测", "无数据", "异常",
    "null", "none", "nan", "n/a", "na", "#n/a", "#ref!",
    "-", "--", "---", "—", "——"
]

# ============== 【修改①】统一字号常量（便于全局调整） ==============
FONT_TITLE = 14
FONT_LABEL = 13
FONT_TICK = 11
FONT_LEGEND = 11
FONT_BAR_TEXT = 9


# 数据清洗函数
def clean_value(raw_str):
    s = str(raw_str).strip().lower()
    if s == '' or s == 'nan':
        return np.nan
    for kw in ABNORMAL_KEYWORDS:
        if kw in s:
            return np.nan
    try:
        return float(s)
    except (ValueError, TypeError):
        return np.nan


# -------------------------- 核心解析函数 --------------------------
def parse_condition_name(name):
    match = re.match(r'预测结果_(.+?)_pre(?:\.csv)?$', str(name).strip())
    if match:
        return match.group(1).split('_')
    return None


def is_baseline(parts):
    return all(p.strip() == '0' for p in parts)


def make_label(parts, diff_flags):
    if is_baseline(parts):
        return "不设无人机"
    segs = []
    n = len(parts)
    for i in range(n):
        if diff_flags[i]:
            name = PARAM_NAMES[i] if i < len(PARAM_NAMES) else f"参数{i + 1}"
            unit = PARAM_UNITS[i] if i < len(PARAM_UNITS) else ""
            segs.append(f"{name}{parts[i]}{unit}")
    return " | ".join(segs) if segs else "_".join(parts)


# -------------------------- 单文件绘图函数 --------------------------
def plot_bar_from_csv(file_path, output_dir):
    try:
        # ========== 1. 数据读取 ==========
        df = pd.read_csv(file_path, header=None, dtype=str)
        raw_names = df.iloc[0, :].astype(str).values

        raw_cleaned = []
        abnormal_cols = []
        for col_idx in range(df.shape[1]):
            raw_str = df.iloc[1, col_idx] if col_idx < df.shape[1] else ''
            cleaned = clean_value(raw_str)
            if np.isnan(cleaned):
                abnormal_cols.append((col_idx, str(raw_str).strip()))
            raw_cleaned.append(cleaned)

        raw_values = np.array(raw_cleaned, dtype=float)

        if abnormal_cols:
            abnormal_info = ", ".join(
                [f"第{idx}列('{val}')" for idx, val in abnormal_cols]
            )
            print(f"  [清洗] 文件 {Path(file_path).name} 中发现异常值: {abnormal_info}，已跳过")

        # ========== 2. 解析各工况参数 + 强制替换基准值 + 过滤异常值 ==========
        parsed = []
        for i, (name, val) in enumerate(zip(raw_names, raw_values)):
            if np.isnan(val):
                continue

            parts = parse_condition_name(name)
            if parts is None:
                print(f"  跳过（命名不符）: {name}")
                continue

            if is_baseline(parts):
                val = BASELINE_OVERRIDE_VALUE

            parsed.append({'name': name, 'parts': parts, 'value': val})

        if not parsed:
            return False, f"文件 {Path(file_path).name} 中未找到符合规律的工况名称"

        clean_summary = ""
        if abnormal_cols:
            abnormal_names = [raw_names[idx] if idx < len(raw_names) else f"列{idx}"
                              for idx, _ in abnormal_cols]
            clean_summary = f"（已清洗{len(abnormal_cols)}个异常值: {', '.join(abnormal_names)}）"

        n_parts = len(parsed[0]['parts'])

        # ========== 3. 分离基准行与实验行 ==========
        baseline = [e for e in parsed if is_baseline(e['parts'])]
        experiments = [e for e in parsed if not is_baseline(e['parts'])]

        # ========== 4. 分析参数变化 ==========
        if experiments:
            exp_part_arrays = [[e['parts'][i] for e in experiments] for i in range(n_parts)]
            diff_flags = [len(set(pa)) > 1 for pa in exp_part_arrays]
        else:
            diff_flags = [False] * n_parts

        # ========== 5. 排序 ==========
        sort_key_idx = next((i for i in range(n_parts) if diff_flags[i]), 0)
        experiments.sort(
            key=lambda e: int(e['parts'][sort_key_idx])
            if e['parts'][sort_key_idx].lstrip('-').isdigit() else 0
        )

        ordered = baseline + experiments

        # ========== 6. 生成精简图例 ==========
        labels = [make_label(e['parts'], diff_flags) for e in ordered]
        values = [e['value'] for e in ordered]

        # ========== 7. 自动生成图题 ==========
        if experiments:
            ref = experiments[0]['parts']
        elif baseline:
            ref = baseline[0]['parts']
        else:
            ref = [''] * n_parts

        fixed_info = []
        varying_names = []
        for i in range(n_parts):
            name = PARAM_NAMES[i] if i < len(PARAM_NAMES) else f"参数{i + 1}"
            unit = PARAM_UNITS[i] if i < len(PARAM_UNITS) else ""
            if diff_flags[i]:
                varying_names.append(name)
            else:
                fixed_info.append(f"{name}{ref[i]}{unit}")

        if fixed_info and varying_names:
            title = "、".join(fixed_info)# + f"，不同{'与'.join(varying_names)}预测结果柱状图"
        elif varying_names:
            title = f"不同{'与'.join(varying_names)}预测结果柱状图"
        else:
            title = "预测结果柱状图"

        # ========== 8. 绘制柱状图 ==========
        x = np.arange(len(labels))
        bar_colors = [colors[i % len(colors)] for i in range(len(values))]

        # ============== 【修改②】固定图片比例 14:6 ==============
        fig, ax = plt.subplots(figsize=(14, 6))
        # ========================================================

        bars = ax.bar(
            x, values,
            width=0.5,
            color=bar_colors,
            edgecolor='white',
            linewidth=0.8
        )

        # 基准横虚线
        ax.axhline(
            y=BASELINE_OVERRIDE_VALUE,
            color='#CC0000',
            linestyle='--',
            linewidth=0.9,
            alpha=0.7,
            label=f'基准工况值 ({BASELINE_OVERRIDE_VALUE})'
        )

        # 柱顶数值标注
        y_max = max(max(values), BASELINE_OVERRIDE_VALUE) if values else BASELINE_OVERRIDE_VALUE
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + y_max * 0.012,
                f"{val:.4f}",
                ha='center', va='bottom',
                fontsize=FONT_BAR_TEXT,
                fontweight='bold'
            )

        # 基准虚线右侧标注
        ax.text(
            x[-1] + 0.35,
            BASELINE_OVERRIDE_VALUE + y_max * 0.01,
            f'{BASELINE_OVERRIDE_VALUE:.2f}',
            ha='left', va='bottom',
            fontsize=FONT_BAR_TEXT,
            color='#CC0000',
            fontweight='bold'
        )

        # ============== 【修改③】所有字体加大 ==============
        #ax.set_xlabel("实验条件", fontsize=FONT_LABEL, fontweight="bold")
        ax.set_ylabel("火焰到达顶端所用时间", fontsize=FONT_LABEL, fontweight="bold")
        ax.set_title(title, fontsize=FONT_TITLE, fontweight="bold", pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=FONT_TICK, rotation=25, ha='right')
        # ==================================================

        # ============== 【修改④】Y轴从0开始 ==============
        y_margin_top = y_max * 0.18
        ax.set_ylim(0, y_max + y_margin_top)
        # ==================================================

        ax.tick_params(direction='in', length=4, width=0.8, labelsize=FONT_TICK)
        ax.yaxis.grid(True, linestyle='--', alpha=0.5)
        ax.set_axisbelow(True)
        ax.legend(loc="upper right", fontsize=FONT_LEGEND, frameon=True)

        # ========== 9. 保存图片 ==========
        file_name = Path(file_path).stem
        save_path = os.path.join(output_dir, f"{title}_预测结果柱状图.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=500, bbox_inches="tight")
        plt.close()

        msg = f"成功生成：{save_path}"
        if clean_summary:
            msg += f"\n    {clean_summary}"
        return True, msg

    except Exception as e:
        return False, f"处理文件 {Path(file_path).name} 失败：{str(e)}"


# -------------------------- 可视化界面（无改动） --------------------------
class BatchPlotApp:
    def __init__(self, root):
        self.root = root
        self.root.title("预测结果柱状图批量绘图工具")
        self.root.geometry("700x550")
        self.root.resizable(False, False)

        self.selected_files = []
        self.output_dir = tk.StringVar()
        self._create_widgets()

    def _create_widgets(self):
        frame_file = ttk.LabelFrame(
            self.root,
            text="选择CSV文件（每个文件第1行为工况名称，第2行为数据，一个文件绘制一张图）",
            padding=10
        )
        frame_file.pack(fill="x", padx=20, pady=10)

        ttk.Button(frame_file, text="批量选择文件", command=self.select_files).pack(side="left")
        self.lbl_file_count = ttk.Label(frame_file, text="已选文件：0个")
        self.lbl_file_count.pack(side="left", padx=10)

        frame_output = ttk.LabelFrame(self.root, text="输出目录", padding=10)
        frame_output.pack(fill="x", padx=20, pady=10)

        ttk.Entry(frame_output, textvariable=self.output_dir, state="readonly", width=40).pack(side="left")
        ttk.Button(frame_output, text="选择目录", command=self.select_output_dir).pack(side="left", padx=10)

        frame_progress = ttk.LabelFrame(self.root, text="处理进度", padding=10)
        frame_progress.pack(fill="both", expand=True, padx=20, pady=10)

        self.progress_bar = ttk.Progressbar(frame_progress, orient="horizontal", length=400, mode="determinate")
        self.progress_bar.pack(pady=5)

        self.txt_log = tk.Text(frame_progress, height=8, width=55, font=("Consolas", 9))
        self.txt_log.pack(pady=5)

        ttk.Button(self.root, text="开始批量绘图", command=self.start_batch_plot, style="Accent.TButton").pack(pady=10)

    def select_files(self):
        files = filedialog.askopenfilenames(
            title="选择CSV文件",
            filetypes=[("CSV文件", "*.csv"), ("所有文件", "*.*")]
        )
        if files:
            self.selected_files = list(files)
            self.lbl_file_count.config(text=f"已选文件：{len(self.selected_files)}个")
            self._log(f"已选择{len(self.selected_files)}个CSV文件")

    def select_output_dir(self):
        dir_path = filedialog.askdirectory(title="选择图片输出目录")
        if dir_path:
            self.output_dir.set(dir_path)
            self._log(f"输出目录已设置为：{dir_path}")

    def start_batch_plot(self):
        if not self.selected_files:
            messagebox.showwarning("警告", "请先选择要处理的CSV文件！")
            return
        if not self.output_dir.get():
            messagebox.showwarning("警告", "请先选择图片输出目录！")
            return

        self.txt_log.delete(1.0, tk.END)

        total_files = len(self.selected_files)
        self.progress_bar["maximum"] = total_files
        self.progress_bar["value"] = 0

        success_count = 0
        for idx, file in enumerate(self.selected_files):
            self.progress_bar["value"] = idx + 1
            self.root.update_idletasks()

            success, msg = plot_bar_from_csv(file, self.output_dir.get())
            self._log(msg)
            if success:
                success_count += 1

        self._log(f"\n批量处理完成！成功：{success_count}个，失败：{total_files - success_count}个")
        messagebox.showinfo(
            "完成",
            f"批量绘图完成！\n成功生成{success_count}张图片\n失败{total_files - success_count}个文件"
        )

    def _log(self, msg):
        self.txt_log.insert(tk.END, f"{msg}\n")
        self.txt_log.see(tk.END)


# -------------------------- 程序入口 --------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = BatchPlotApp(root)
    root.mainloop()
