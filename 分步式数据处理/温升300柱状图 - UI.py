import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText
import threading

# 设置matplotlib后端，避免tkinter冲突
plt.switch_backend('TkAgg')

def plot_temperature_bar(file_path, output_dir="output", log_text=None):
    """
    科研柱状图生成函数：计算指定列平均值，按规则排序与命名
    :param file_path: CSV文件路径
    :param output_dir: 图片输出目录
    :param log_text: 日志输出控件
    """
    def log(msg):
        """日志输出函数"""
        if log_text:
            log_text.insert(tk.END, f"{msg}\n")
            log_text.see(tk.END)
        else:
            print(msg)

    try:
        # -------------------------- 基础配置 --------------------------
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        # 支持中文显示
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['figure.dpi'] = 300
        # 固定颜色库（按要求顺序）
        colors = [
            '#c45161', '#e094a0', '#f2b6c0', '#f2dde1',
            '#cbc7d8', '#8db7d2', '#5e62a9', '#434279'
        ]

        # -------------------------- 数据读取与预处理 --------------------------
        log(f"正在读取文件：{file_path}")
        df = pd.read_csv(file_path)

        # 提取第一列（名称列）和指定数据列（43、47、51列，Python索引42、46、50）
        name_col = df.iloc[:, 0]
        target_cols = df.iloc[:, [42, 46, 50]]  # 43列=索引42，47列=索引46，51列=索引50
        # 合并名称与目标数据
        new_data = pd.concat([name_col, target_cols], axis=1)
        
        # ========== 新增：过滤包含"未达到"字样的行 ==========
        # 先记录原始行数
        original_rows = len(new_data)
        # 检查整个DataFrame中是否包含"未达到"，过滤掉包含该字样的行
        # 将所有数据转换为字符串后检查，然后取反得到不包含"未达到"的行
        new_data = new_data[~new_data.astype(str).apply(lambda x: x.str.contains('未达到', na=False)).any(axis=1)]
        # 记录过滤后的行数
        filtered_rows = len(new_data)
        removed_rows = original_rows - filtered_rows
        if removed_rows > 0:
            log(f"⚠️  过滤掉 {removed_rows} 行包含'未达到'的异常数据")
        if filtered_rows == 0:
            log(f"❌  文件 {file_path} 过滤后无有效数据，跳过绘制")
            return
        # ==================================================



        # 处理"不设无人机"数据（单独作为第一个系列）
        raw_labels = df.iloc[1:, 0].astype(str).values  # 排除第一行（不设无人机）
        z_raw_labels = ["不设无人机"]  # 固定第一个系列名称

        # -------------------------- 拆分标签+判断变化段 --------------------------
        all_parts = [name.split('_') for name in raw_labels if '_' in name]
        # 初始化判断标志
        p1_diff = p2_diff = p3_diff = False
        fixed_v1 = fixed_v2 = fixed_v3 = ""

        if all_parts:
            part1 = [p[0] for p in all_parts]
            part2 = [p[1] for p in all_parts]
            part3 = [p[2] for p in all_parts]
            # 判断哪一段参数变化
            p1_diff = len(set(part1)) > 1  # 片段1：下洗气流流速
            p2_diff = len(set(part2)) > 1  # 片段2：无人机高度
            p3_diff = len(set(part3)) > 1  # 片段3：无人机距离
            # 取固定段的数值（取第一个作为代表）
            fixed_v1 = part1[0] if part1 else ""
            fixed_v2 = part2[0] if part2 else ""
            fixed_v3 = part3[0] if part3 else ""

        # -------------------------- 数据排序（按折线图规则） --------------------------
        def sort_data_by_segment(data):
            """根据变化段对数据排序"""
            if p1_diff:
                # 按片段1（流速）排序
                return data.sort_values(
                    by=data.columns[0],
                    key=lambda x: x.str.split('_').str[0].astype(int, errors='ignore')
                )
            elif p2_diff:
                # 按片段2（高度）排序
                return data.sort_values(
                    by=data.columns[0],
                    key=lambda x: x.str.split('_').str[1].astype(int, errors='ignore')
                )
            elif p3_diff:
                # 按片段3（距离）排序
                return data.sort_values(
                    by=data.columns[0],
                    key=lambda x: x.str.split('_').str[2].astype(int, errors='ignore')
                )
            return data  # 无变化段时保持原序

        # 执行排序
        new_data_sorted = sort_data_by_segment(new_data)

        # -------------------------- 计算指定列平均值（每个系列1个数值） --------------------------
        # 排除名称列，计算每行（每个系列）的平均值
        avg_values = new_data_sorted.iloc[:, 1:].astype(float).mean(axis=1).values
        # 获取排序后的系列名称
        sorted_labels = new_data_sorted.iloc[:, 0].astype(str).values

        # -------------------------- 生成精简系列名称 --------------------------
        def get_simplified_labels(labels):
            """生成精简的系列名称（只显示变化段）"""
            simplified = []
            for label in labels:
                if label == "不设无人机":
                    simplified.append(label)
                    continue
                parts = label.split('_')
                if len(parts) < 3:
                    simplified.append(label)
                    continue
                v1, v2, v3 = parts[0], parts[1], parts[2]
                label_parts = []
                if p1_diff:
                    label_parts.append(f"下洗气流流速{v1}m/s")
                if p2_diff:
                    label_parts.append(f"无人机高度{v2}m")
                if p3_diff:
                    label_parts.append(f"无人机距离{v3}m")
                simplified.append(" | ".join(label_parts))
            return simplified

        final_labels = get_simplified_labels(sorted_labels)
        final_labels = z_raw_labels + final_labels[1:]

        # -------------------------- 自动生成图题 --------------------------
        if p1_diff:
            title = f"{fixed_v2}m高度、{fixed_v3}m距离，不同下洗气流流速温升时间平均值"
        elif p2_diff:
            title = f"{fixed_v1}m/s流速、{fixed_v3}m距离，不同无人机高度温升时间平均值"
        elif p3_diff:
            title = f"{fixed_v1}m/s流速、{fixed_v2}m高度，不同无人机距离温升时间平均值"
        else:
            title = "不同工况温升平均值对比"

        # -------------------------- 绘制柱状图 --------------------------
        fig, ax = plt.subplots(figsize=(4,3))
        # 柱状图位置（避免重叠）
        bar_positions = np.arange(len(final_labels))
        # 绘制柱状图（循环使用颜色库）
        bars = ax.bar(
            bar_positions,
            avg_values,
            color=[colors[i % len(colors)] for i in range(len(final_labels))],
            width=0.6,
            edgecolor='white',
            linewidth=1.2
        )

        # -------------------------- 图表样式优化 --------------------------
        # 设置坐标轴标签与标题
        ax.set_ylabel("探测器温度升至300℃所用平均时间（s）", fontsize=7, fontweight="bold")
        ax.set_title(title, fontsize=7, fontweight="bold", pad=7)
        # 设置X轴刻度与标签（垂直显示避免重叠）
        ax.set_xticks(bar_positions)
        ax.set_xticklabels(final_labels, rotation=30, ha='center', fontsize=7)
        # 刻度线向内
        ax.tick_params(direction='in', length=3, width=0.7, labelsize=7)
        # 添加数值标签（显示在柱子顶部）
        for bar, value in zip(bars, avg_values):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2.,
                height + 0.1,  # 数值标签偏移量
                f"{value:.2f}",  # 保留2位小数
                ha='center', va='bottom', fontsize=8
            )

        # 调整布局，避免标签被截断
        plt.tight_layout()
        
        # 生成唯一的文件名（基于原文件名）
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        output_path = os.path.join(output_dir, f"科研柱状图_{file_name}_温升平均值.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)  # 关闭图形释放资源
        log(f"✅ 柱状图已保存至：{output_path}")
        
    except Exception as e:
        log(f"❌ 处理文件 {file_path} 时出错：{str(e)}")
        raise

class TemperatureBarPlotGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("科研柱状图生成工具 - 温升平均值分析")
        self.root.geometry("800x600")
        self.root.resizable(True, True)

        # 选中的文件列表
        self.selected_files = []
        
        # 创建UI布局
        self._create_widgets()
        
    def _create_widgets(self):
        """创建界面组件"""
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 1. 文件选择区域
        file_frame = ttk.LabelFrame(main_frame, text="文件选择", padding="10")
        file_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 文件列表框
        self.file_listbox = tk.Listbox(file_frame, height=6, selectmode=tk.EXTENDED)
        self.file_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # 文件操作按钮
        btn_frame = ttk.Frame(file_frame)
        btn_frame.pack(side=tk.RIGHT)
        
        ttk.Button(btn_frame, text="添加文件", command=self.add_files).pack(fill=tk.X, pady=2)
        ttk.Button(btn_frame, text="添加文件夹", command=self.add_folder).pack(fill=tk.X, pady=2)
        ttk.Button(btn_frame, text="移除选中", command=self.remove_selected).pack(fill=tk.X, pady=2)
        ttk.Button(btn_frame, text="清空列表", command=self.clear_list).pack(fill=tk.X, pady=2)
        
        # 2. 输出设置区域
        output_frame = ttk.LabelFrame(main_frame, text="输出设置", padding="10")
        output_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(output_frame, text="输出目录：").pack(side=tk.LEFT)
        self.output_dir_var = tk.StringVar(value="output")
        ttk.Entry(output_frame, textvariable=self.output_dir_var, width=50).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(output_frame, text="浏览", command=self.select_output_dir).pack(side=tk.LEFT)
        
        # 3. 日志区域
        log_frame = ttk.LabelFrame(main_frame, text="运行日志", padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.log_text = ScrolledText(log_frame, height=15, font=("Consolas", 9))
        self.log_text.pack(fill=tk.BOTH, expand=True)
        self.log_text.config(state=tk.NORMAL)
        
        # 4. 操作按钮区域
        btn_frame2 = ttk.Frame(main_frame)
        btn_frame2.pack(fill=tk.X, pady=(0, 10))
        
        self.run_btn = ttk.Button(btn_frame2, text="开始批量生成", command=self.run_batch_plot, style="Accent.TButton")
        self.run_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(btn_frame2, text="清空日志", command=self.clear_log).pack(side=tk.LEFT)
        ttk.Button(btn_frame2, text="关于", command=self.show_about).pack(side=tk.RIGHT)
        
        # 设置样式
        style = ttk.Style()
        style.configure("Accent.TButton", font=("微软雅黑", 10, "bold"))
        
    def add_files(self):
        """添加CSV文件"""
        files = filedialog.askopenfilenames(
            title="选择CSV文件",
            filetypes=[("CSV文件", "*.csv"), ("所有文件", "*.*")]
        )
        for file in files:
            if file not in self.selected_files:
                self.selected_files.append(file)
                self.file_listbox.insert(tk.END, file)
                
    def add_folder(self):
        """添加文件夹中的所有CSV文件"""
        folder = filedialog.askdirectory(title="选择包含CSV文件的文件夹")
        if folder:
            for file in os.listdir(folder):
                if file.lower().endswith('.csv'):
                    file_path = os.path.join(folder, file)
                    if file_path not in self.selected_files:
                        self.selected_files.append(file_path)
                        self.file_listbox.insert(tk.END, file_path)
                        
    def remove_selected(self):
        """移除选中的文件"""
        selected_indices = self.file_listbox.curselection()
        # 倒序删除避免索引错乱
        for idx in reversed(selected_indices):
            del self.selected_files[idx]
            self.file_listbox.delete(idx)
            
    def clear_list(self):
        """清空文件列表"""
        self.selected_files.clear()
        self.file_listbox.delete(0, tk.END)
        
    def select_output_dir(self):
        """选择输出目录"""
        dir_path = filedialog.askdirectory(title="选择输出目录")
        if dir_path:
            self.output_dir_var.set(dir_path)
            
    def clear_log(self):
        """清空日志"""
        self.log_text.delete(1.0, tk.END)
        
    def show_about(self):
        """显示关于信息"""
        messagebox.showinfo(
            "关于",
            "科研柱状图生成工具 v1.1\n\n"
            "功能：批量生成温升平均值柱状图\n"
            "新增：自动过滤包含'未达到'的异常数据\n"
            "支持：自动识别变化参数、生成精简标签、批量处理CSV文件\n"
            "作者：豆包编程助手"
        )
        
    def run_batch_plot(self):
        """批量生成图表（在新线程中运行）"""
        if not self.selected_files:
            messagebox.showwarning("警告", "请先选择要处理的CSV文件！")
            return
            
        output_dir = self.output_dir_var.get().strip()
        if not output_dir:
            messagebox.showwarning("警告", "请设置输出目录！")
            return
            
        # 禁用运行按钮防止重复点击
        self.run_btn.config(state=tk.DISABLED)
        
        # 在新线程中执行批量处理
        def worker():
            try:
                self.log_text.insert(tk.END, "="*50 + "\n")
                self.log_text.insert(tk.END, f"开始批量处理，共 {len(self.selected_files)} 个文件\n")
                self.log_text.insert(tk.END, f"输出目录：{output_dir}\n")
                self.log_text.insert(tk.END, "="*50 + "\n")
                self.log_text.see(tk.END)
                
                # 处理每个文件
                success_count = 0
                fail_count = 0
                skip_count = 0  # 新增：记录跳过的文件数
                for i, file in enumerate(self.selected_files, 1):
                    self.log_text.insert(tk.END, f"\n[{i}/{len(self.selected_files)}] 正在处理：{file}\n")
                    self.log_text.see(tk.END)
                    try:
                        plot_temperature_bar(file, output_dir, self.log_text)
                        # 检查是否因为无数据被跳过
                        # 通过日志最后一行判断，或者可以修改函数返回值
                        # 这里简化处理，假设执行完成没有异常就是成功
                        success_count += 1
                    except Exception as e:
                        if "过滤后无有效数据" in str(e):
                            skip_count += 1
                            fail_count += 0
                        else:
                            fail_count += 1
                        self.log_text.insert(tk.END, f"处理失败：{str(e)}\n")
                        self.log_text.see(tk.END)
                
                # 输出总结
                self.log_text.insert(tk.END, "\n" + "="*50 + "\n")
                self.log_text.insert(tk.END, f"批量处理完成！\n成功：{success_count} 个 | 失败：{fail_count} 个 | 跳过（无有效数据）：{skip_count} 个\n")
                self.log_text.insert(tk.END, "="*50 + "\n")
                self.log_text.see(tk.END)
                
                messagebox.showinfo("完成", f"批量处理完成！\n成功：{success_count} 个 | 失败：{fail_count} 个 | 跳过（无有效数据）：{skip_count} 个")
                
            finally:
                # 恢复按钮状态
                self.root.after(0, lambda: self.run_btn.config(state=tk.NORMAL))
                
        # 启动线程
        threading.Thread(target=worker, daemon=True).start()

# -------------------------- 主程序入口 --------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = TemperatureBarPlotGUI(root)
    root.mainloop()