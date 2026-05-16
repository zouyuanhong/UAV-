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

# 颜色库
colors = [
    "#000000","#51a16a","#a15188","#5335c4","#c45335","#35a6c4","#1b5362","#CC0000"
]

# 多样式标记 + 线型
markers = ['o', '^', 's', 'D', 'v', '<', '>', 'p','*']
linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':','--']

# -------------------------- 核心绘图函数 --------------------------
def plot_temperature_curve(file_path, output_dir):
    """
    单个CSV文件绘图函数
    :param file_path: CSV文件路径
    :param output_dir: 图片输出目录
    """
    try:
        # 数据读取
        df = pd.read_csv(file_path)
        #print(df)
        name_col = df.iloc[:, 0]
        data_cols = df.iloc[:, 38:55]
        new_data = pd.concat([name_col, data_cols], axis=1)
        #print(new_data)

        raw_labels = df.iloc[1:, 0].astype(str).values
        #print(raw_labels)
        #z_raw_labels=df.iloc[0, 0].astype(str).values
        z_raw_labels=["不设无人机"]
        #print(z_raw_labels)
        x = np.arange(5, 54, 3)

        # 拆分标签段
        all_parts = [name.split('_') for name in raw_labels]
        part1 = [p[0] for p in all_parts]
        part2 = [p[1] for p in all_parts]
        part3 = [p[2] for p in all_parts]

        p1_diff = len(set(part1)) > 1
        p2_diff = len(set(part2)) > 1
        p3_diff = len(set(part3)) > 1

        # 取固定段数值
        fixed_v1 = part1[0] if part1 else ""
        fixed_v2 = part2[0] if part2 else ""
        fixed_v3 = part3[0] if part3 else ""

        # 生成精简图例
        def get_simplified_labels():
            new_names = []
            for p in all_parts:
                v1, v2, v3 = p[0], p[1], p[2]
                label_parts = []
                if p1_diff:
                    label_parts.append(f"下洗气流流速{v1}m/s")
                if p2_diff:
                    label_parts.append(f"无人机高度{v2}m")
                if p3_diff:
                    label_parts.append(f"无人机距离{v3}m")
                new_names.append(" | ".join(label_parts))
            # 按数字排序
            new_names = sorted(new_names, key=lambda x: int(re.findall(r"\d+", x)[0]) if re.findall(r"\d+", x) else 0)
            return new_names

        new_labels = get_simplified_labels()

        new_labels = [z_raw_labels] + new_labels
        #print(new_labels)

        '''all_parts = [name.split('_') for name in raw_labels]
        int(part1 = [p[0] for p in all_parts])
        int(part2 = [p[1] for p in all_parts])
        int(part3 = [p[2] for p in all_parts])
        print(part1)
        print(part2)
        print(part3)'''

        # 自动生成图题
        if p1_diff:
            title = f"{fixed_v2}m高度、{fixed_v3}m距离，不同下洗气流流速温升曲线"
            new_data = new_data.sort_values(by=new_data.columns[0], key=lambda x: x.str.split('_').str[0].astype(int))
            #print(new_data)
        elif p2_diff:
            title = f"{fixed_v1}m/s流速、{fixed_v3}m距离，不同无人机高度温升曲线"
            new_data = new_data.sort_values(by=new_data.columns[0], key=lambda x: x.str.split('_').str[1].astype(int))
            #print(new_data)
        elif p3_diff:
            title = f"{fixed_v1}m/s流速、{fixed_v2}m高度，不同无人机距离温升曲线"
            new_data = new_data.sort_values(by=new_data.columns[0], key=lambda x: x.str.split('_').str[2].astype(int))
            #print(new_data)
        else:
            title = "温升曲线对比"

        data_cols = new_data.iloc[:, 1:]
        print(data_cols)

        # 绘图
        fig, ax = plt.subplots(figsize=(4, 3))

        for idx, (label, y_series) in enumerate(zip(new_labels, data_cols.values)):
            # 处理"未达到"为NaN
            y = np.array([np.nan if str(val) == "未达到" else float(val) for val in y_series])
            c = colors[idx % len(colors)]
            m = markers[idx % len(markers)]
            ls = linestyles[idx % len(linestyles)]
            
            ax.plot(
                x, y,
                color=c,
                linewidth=0.7,
                linestyle=ls,
                marker=m,
                markersize=4,
                markerfacecolor=c,
                markeredgecolor='white',
                markeredgewidth=0.8,
                label=label
            )

        # 图表样式
        ax.set_xlabel("探测器高度（m）", fontsize=7, fontweight="bold")
        ax.set_ylabel("探测器温度升至300℃所用时间（s）", fontsize=7, fontweight="bold")
        ax.set_title(title, fontsize=7, fontweight="bold", pad=7)
        ax.set_xticks(x)

        # 刻度样式
        ax.tick_params(
            direction='in',
            length=3,
            width=0.7,
            labelsize=7
        )

        # 图例
        ax.legend(loc="lower right", fontsize=6, frameon=True)

        # 保存图片（基于原CSV文件名生成）
        file_name = Path(file_path).stem
        save_path = os.path.join(output_dir, f"{file_name}_温升曲线.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()  # 关闭画布释放内存
        return True, f"成功生成：{save_path}"
    except Exception as e:
        return False, f"处理文件{file_path}失败：{str(e)}"

# -------------------------- 可视化界面 --------------------------
class BatchPlotApp:
    def __init__(self, root):
        self.root = root
        self.root.title("温升曲线批量绘图工具")
        self.root.geometry("700x550")
        self.root.resizable(False, False)

        # 选中的文件列表
        self.selected_files = []
        # 输出目录
        self.output_dir = tk.StringVar()

        # 界面布局
        self._create_widgets()

    def _create_widgets(self):
        # 1. 文件选择区域
        frame_file = ttk.LabelFrame(self.root, text="选择CSV文件", padding=10)
        frame_file.pack(fill="x", padx=20, pady=10)

        ttk.Button(frame_file, text="批量选择文件", command=self.select_files).pack(side="left")
        self.lbl_file_count = ttk.Label(frame_file, text="已选文件：0个")
        self.lbl_file_count.pack(side="left", padx=10)

        # 2. 输出目录选择区域
        frame_output = ttk.LabelFrame(self.root, text="输出目录", padding=10)
        frame_output.pack(fill="x", padx=20, pady=10)

        ttk.Entry(frame_output, textvariable=self.output_dir, state="readonly", width=40).pack(side="left")
        ttk.Button(frame_output, text="选择目录", command=self.select_output_dir).pack(side="left", padx=10)

        # 3. 进度显示区域
        frame_progress = ttk.LabelFrame(self.root, text="处理进度", padding=10)
        frame_progress.pack(fill="both", expand=True, padx=20, pady=10)

        self.progress_bar = ttk.Progressbar(frame_progress, orient="horizontal", length=400, mode="determinate")
        self.progress_bar.pack(pady=5)

        self.txt_log = tk.Text(frame_progress, height=8, width=55, font=("Consolas", 9))
        self.txt_log.pack(pady=5)

        # 4. 开始处理按钮
        ttk.Button(self.root, text="开始批量绘图", command=self.start_batch_plot, style="Accent.TButton").pack(pady=10)

    def select_files(self):
        """选择多个CSV文件"""
        files = filedialog.askopenfilenames(
            title="选择CSV文件",
            filetypes=[("CSV文件", "*.csv"), ("所有文件", "*.*")]
        )
        if files:
            self.selected_files = list(files)
            self.lbl_file_count.config(text=f"已选文件：{len(self.selected_files)}个")
            self._log(f"已选择{len(self.selected_files)}个CSV文件")

    def select_output_dir(self):
        """选择输出目录"""
        dir_path = filedialog.askdirectory(title="选择图片输出目录")
        if dir_path:
            self.output_dir.set(dir_path)
            self._log(f"输出目录已设置为：{dir_path}")

    def start_batch_plot(self):
        """开始批量绘图"""
        if not self.selected_files:
            messagebox.warning("警告", "请先选择要处理的CSV文件！")
            return
        if not self.output_dir.get():
            messagebox.warning("警告", "请先选择图片输出目录！")
            return

        # 清空日志
        self.txt_log.delete(1.0, tk.END)
        # 设置进度条
        total_files = len(self.selected_files)
        self.progress_bar["maximum"] = total_files
        self.progress_bar["value"] = 0

        # 批量处理每个文件
        success_count = 0
        for idx, file in enumerate(self.selected_files):
            self.progress_bar["value"] = idx + 1
            self.root.update_idletasks()  # 更新界面

            # 处理单个文件
            success, msg = plot_temperature_curve(file, self.output_dir.get())
            self._log(msg)
            if success:
                success_count += 1

        # 处理完成提示
        self._log(f"\n批量处理完成！成功：{success_count}个，失败：{total_files - success_count}个")
        messagebox.showinfo("完成", f"批量绘图完成！\n成功生成{success_count}张图片\n失败{total_files - success_count}个文件")

    def _log(self, msg):
        """日志输出"""
        self.txt_log.insert(tk.END, f"{msg}\n")
        self.txt_log.see(tk.END)  # 自动滚动到底部

# -------------------------- 程序入口 --------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = BatchPlotApp(root)
    root.mainloop()