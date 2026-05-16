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
#markers = ['o', '^', 's', 'D', 'v', '<', '>', 'p','*']
linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':','--']

# -------------------------- 核心绘图函数 --------------------------
def plot_heat_release_rate(files, output_dir):
    """
    多CSV文件合并绘图函数（热释放速率）
    :param files: CSV文件路径列表
    :param output_dir: 图片输出目录
    """
    try:
        # 存储所有文件的数据
        all_data = []
        file_names = []
        time_data = None
        
        # 第一步：读取所有文件数据，确定时间截取点
        max_time_threshold = 800  # 截至800秒
        cut_length = None
        
        # 1. 读取所有文件，获取时间列和数据列
        for file in files:
            df = pd.read_csv(file)
            # 第一列是时间，第二列是热释放速率
            time_series = df.iloc[1:, 0].astype(float)
            data_series = df.iloc[1:, 1]
            
            # 处理非数值数据
            data_series = pd.to_numeric(data_series, errors='coerce')
            
            # 找到时间刚好超过800秒的位置
            over_800_idx = np.where(time_series > max_time_threshold)[0]
            if len(over_800_idx) > 0:
                valid_length = over_800_idx[0] + 1  # 包含超过800秒的第一个点
            else:
                valid_length = len(time_series)  # 所有数据都≤800秒
            
            # 记录有效长度，取所有文件中的最小有效长度（保证时间轴对齐）
            if cut_length is None or valid_length < cut_length:
                cut_length = valid_length
            
            # 存储文件数据和名称
            all_data.append((time_series, data_series))
            file_names.append(Path(file).stem)

        raw_labels = file_names[1:]
        print(raw_labels)
        z_raw_labels=["不设无人机"]
        
        # 2. 统一截取数据（按最小有效长度）
        cut_time = None
        plot_data = []
        for idx, (time_series, data_series) in enumerate(all_data):
            # 截取数据
            cut_time_series = time_series[:cut_length]
            cut_data_series = data_series[:cut_length]
            
            # 仅用第一个文件的时间轴作为统一横轴
            if idx == 0:
                cut_time = cut_time_series
            
            plot_data.append(cut_data_series)

        #print(plot_data)
        #print(np.array(plot_data).shape)

            

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
        
        # 3. 生成图例（沿用原命名规则）
        def get_simplified_labels():
            new_names = []
            for p in all_parts:
                v1, v2, v3 = p[0], p[1], p[2]
                label_parts = []
                if p1_diff:
                    label_parts.append(f"下洗气流流速{v1}m/s")
                    figure_name = f"{v2}m高度、{v3}m距离"
                if p2_diff:
                    label_parts.append(f"无人机高度{v2}m")
                    figure_name = f"{v1}m/s流速、{v3}m距离"
                if p3_diff:
                    label_parts.append(f"无人机距离{v3}m")
                    figure_name = f"{v1}m/s流速、{v2}m高度"
                label_parts.append(f"{figure_name}")
                new_names.append(" | ".join(label_parts))
            # 按数字排序
            #new_names = sorted(new_names, key=lambda x: int(re.findall(r"\d+", x)[0]) if re.findall(r"\d+", x) else 0)
            return new_names
        
        new_labels = z_raw_labels + get_simplified_labels()
        # 按数字排序
        new_labels = sorted(new_labels, key=lambda x: int(re.findall(r"\d+", x)[0]) if re.findall(r"\d+", x) else 0)
        
        '''# 4. 生成图题（沿用原规则）
        first_name_parts = raw_labels[0].split('_') if '_' in raw_labels[0] else []
        if len(first_name_parts)>=3:
            p1, p2, p3 = first_name_parts[0], first_name_parts[1], first_name_parts[2]
            title = f"{p2}m高度、{p3}m距离，不同下洗气流流速热释放速率曲线"
        else:
            title = "热释放速率曲线对比"'''

        # 把原始数据转成 DataFrame
        df = pd.DataFrame(plot_data)  # plot_data 是你的 (6,334) 数组

        # 在第一列插入名称列
        df.insert(0, "filename", file_names)

        # ===================== 排序逻辑（完全保留你的逻辑，只修复语法） =====================
        if p1_diff:
            title = f"{fixed_v2}m高度、{fixed_v3}m距离，不同下洗气流流速温升曲线"
            # 按第 1 段数字排序（0,15,15...）
            df = df.sort_values(by="filename", key=lambda x: x.str.split('_').str[0].astype(int))

        elif p2_diff:
            title = f"{fixed_v1}m/s流速、{fixed_v3}m距离，不同无人机高度温升曲线"
            # 按第 2 段数字排序
            df = df.sort_values(by="filename", key=lambda x: x.str.split('_').str[1].astype(int))

        elif p3_diff:
            title = f"{fixed_v1}m/s流速、{fixed_v2}m高度，不同无人机距离温升曲线"
            # 按第 3 段数字排序
            df = df.sort_values(by="filename", key=lambda x: x.str.split('_').str[2].astype(int))

        # 最终排序后的数据
        new_data = df.reset_index(drop=True)  # 重置索引
        print("最终数据形状：", new_data.shape)  # (6, 335)
        print(new_data)

        plot_data = new_data.iloc[:, 1:]
        print(plot_data)
        
        # 5. 绘图
        fig, ax = plt.subplots(figsize=(4, 3))

        # ============== 修复核心：遍历DataFrame的【行数据】，而不是列 ==============
        # 排序后，从df中提取排序后的文件名，重新生成对应标签
        sorted_filenames = new_data["filename"].tolist()
        # 生成排序后的标签（和数据一一对应）
        def get_sorted_labels(filenames):
            labels = []
            for name in filenames:
                if name == '0_0_0_0_hrr':
                    labels.append("不设无人机")
                    continue
                p = name.split('_')
                v1, v2, v3 = p[0], p[1], p[2]
                label_parts = []
                if p1_diff:
                    label_parts.append(f"下洗气流流速{v1}m/s")
                if p2_diff:
                    label_parts.append(f"无人机高度{v2}m")
                if p3_diff:
                    label_parts.append(f"无人机距离{v3}m")
                labels.append(" | ".join(label_parts))
            return labels

        # 获取和排序后数据匹配的标签
        sorted_labels = get_sorted_labels(sorted_filenames)

        # 遍历【每一行数据】（每条曲线）+ 匹配标签
        for idx, (label, (_, row_data)) in enumerate(zip(sorted_labels, new_data.iterrows())):
            y_series = row_data.iloc[1:]  # 提取数值部分（334个点）
            c = colors[idx % len(colors)]
            ls = linestyles[idx % len(linestyles)]
            
            ax.plot(
                cut_time.values, y_series.values,  # 统一转numpy数组，保证维度匹配
                color=c,
                linewidth=0.7,
                linestyle=ls,
                markersize=4,
                markerfacecolor=c,
                markeredgecolor='white',
                markeredgewidth=0.8,
                label=label
            )

        # 图表样式
        ax.set_xlabel("时间（s）", fontsize=9, fontweight="bold")
        ax.set_ylabel("热释放速率", fontsize=9, fontweight="bold")
        ax.set_title(title, fontsize=10, fontweight="bold", pad=10)
        ax.set_xlim(0, max(cut_time))

        # 刻度样式
        ax.tick_params(direction='in', length=3, width=0.7, labelsize=8)
        ax.legend(loc="best", fontsize=7, frameon=True)


        '''for p in all_parts:
            v1, v2, v3 = p[0], p[1], p[2]
            if p1_diff:
                figure_name = f"{v2}m高度、{v3}m距离"
            if p2_diff:
                figure_name = f"{v1}m|s流速、{v3}m距离"
            if p3_diff:
                figure_name = f"{v1}m|s流速、{v2}m高度"'''

        # 保存图片
        main_file_name = Path(files[0]).stem
        #main_file_name = figure_name
        save_path = os.path.join(output_dir, f"{main_file_name}_热释放速率曲线.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()
        
        # 记录有效数据长度
        log_msg = f"有效数据长度（截至{max_time_threshold}秒）：{cut_length}个数据点"
        return True, f"成功生成：{save_path}\n{log_msg}"
    
    except Exception as e:
        return False, f"处理文件失败：{str(e)}"

# -------------------------- 可视化界面 --------------------------
class BatchPlotApp:
    def __init__(self, root):
        self.root = root
        self.root.title("热释放速率曲线批量绘图工具")
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
        ttk.Button(self.root, text="开始绘图（多文件同图）", command=self.start_batch_plot, style="Accent.TButton").pack(pady=10)

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
        """开始绘图（多文件合并为一张图）"""
        if not self.selected_files:
            messagebox.warning("警告", "请先选择要处理的CSV文件！")
            return
        if not self.output_dir.get():
            messagebox.warning("警告", "请先选择图片输出目录！")
            return

        # 清空日志
        self.txt_log.delete(1.0, tk.END)
        # 设置进度条
        self.progress_bar["maximum"] = 1
        self.progress_bar["value"] = 0

        # 处理多文件合并绘图
        self.progress_bar["value"] = 1
        self.root.update_idletasks()
        
        success, msg = plot_heat_release_rate(self.selected_files, self.output_dir.get())
        self._log(msg)

        # 处理完成提示
        if success:
            messagebox.showinfo("完成", "绘图完成！\n已将所有选中文件的数据绘制在同一张图中")
        else:
            messagebox.showerror("失败", f"绘图失败：{msg}")

    def _log(self, msg):
        """日志输出"""
        self.txt_log.insert(tk.END, f"{msg}\n")
        self.txt_log.see(tk.END)  # 自动滚动到底部

# -------------------------- 程序入口 --------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = BatchPlotApp(root)
    root.mainloop()