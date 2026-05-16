import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import threading
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# -------------------------- 全局绘图配置（中文、字体、样式）--------------------------
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 120

# -------------------------- 核心绘图函数（批量处理单个文件）--------------------------
def plot_single_file(file_path, save_dir):
    """
    处理单个文件并绘图
    :param file_path: 文件路径（csv/xlsx）
    :param save_dir: 图片保存目录
    """
    try:
        # 1. 读取文件（自动兼容CSV/Excel）
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path)
        else:
            return False, "不支持的文件格式！"

        # 2. 校验数据列数（必须3列：时间、连续值、离散值）
        if df.shape[1] < 3:
            return False, "数据列数不足，必须包含3列！"

        # 3. 按位置取列（不依赖列名，通用性更强）
        time_col = df.iloc[:, 0]    # 第1列：时间（横轴）
        pred_col = df.iloc[:, 1]    # 第2列：连续曲线
        real_col = df.iloc[:, 2]    # 第3列：离散实测点

        # 4. 筛选有效实测数据（去除空值）
        valid_mask = real_col.notna()
        x_real = time_col[valid_mask]
        y_real = real_col[valid_mask]
        y_pred_real = pred_col[valid_mask]

        if len(x_real) == 0:
            return False, "文件中无有效实测离散数据！"

        # 5. 计算误差（绝对误差+百分偏差）
        abs_err = y_real.values - y_pred_real.values
        per_err = []
        for yp in y_pred_real.values:
            if abs(yp) > 1e-3:
                per_err.append(abs_err[len(per_err)] / yp * 100)
            else:
                per_err.append(np.nan)  # 预测值为0，不计算百分比

        # 6. 创建画布
        plt.figure(figsize=(6, 4))

        # 绘制连续预测曲线
        plt.plot(time_col, pred_col, color='#1f77b4', linewidth=2, label='预测火焰高度')
        # 绘制离散实测散点
        plt.scatter(x_real, y_real, color='#ff7f0e', s=50, zorder=5, label='实际火焰高度')

        # 绘制垂直误差线 + 标注差值
        for t, yp, yr, ae, pe in zip(x_real, y_pred_real, y_real, abs_err, per_err):
            # 垂直误差线
            plt.plot([t, t], [yp, yr], color='red', linewidth=1.2, alpha=0.7)
            # 标注文字（正负差值+百分偏差）
            if np.isnan(pe):
                txt = f'{ae:+.2f}'
            else:
                txt = f'{ae:+.2f}\n({pe:+.1f}%)'
            plt.text(t, yr + 0.4, txt, fontsize=7, ha='center')

        # 图表美化
        plt.xlabel('时间 (s)', fontsize=12)
        plt.ylabel('火焰高度（m）', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)
        plt.tight_layout()

        # 7. 保存图片（以原文件名命名）
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        save_path = os.path.join(save_dir, f'{file_name}_对比图.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()  # 关闭画布，释放内存

        return True, f"成功：{file_name}"

    except Exception as e:
        return False, f"失败：{str(e)}"

# -------------------------- 批量处理函数（后台运行，防界面卡顿）--------------------------
def batch_process(files, save_dir, text_widget):
    """批量处理所有选中的文件"""
    if not files or not save_dir:
        messagebox.showwarning("提示", "请先选择文件和输出目录！")
        return

    text_widget.insert(tk.END, "========== 开始批量绘图 ==========\n")
    text_widget.see(tk.END)

    for file in files:
        success, msg = plot_single_file(file, save_dir)
        text_widget.insert(tk.END, f"{msg}\n")
        text_widget.see(tk.END)  # 自动滚动到最新日志

    text_widget.insert(tk.END, "========== 全部处理完成 ==========\n\n")
    text_widget.see(tk.END)
    messagebox.showinfo("完成", "所有文件已批量绘图并保存！")

# -------------------------- GUI可视化界面 --------------------------
class PlotGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("三列数据批量绘图工具 | 时间+连续曲线+离散实测")
        self.root.geometry("650x450")  # 窗口大小

        # 变量存储
        self.selected_files = []
        self.save_directory = ""

        # 界面组件
        # 1. 选择文件按钮
        tk.Button(root, text="选择多个CSV/Excel文件", command=self.choose_files,
                  width=25, height=2).pack(pady=10)

        # 2. 选择输出文件夹按钮
        tk.Button(root, text="选择图片保存目录", command=self.choose_save_dir,
                  width=25, height=2).pack(pady=5)

        # 3. 开始绘图按钮
        tk.Button(root, text="开始批量绘图", command=self.start_plot,
                  bg="#2196F3", fg="white", width=25, height=2).pack(pady=10)

        # 4. 日志显示框
        tk.Label(root, text="处理日志：").pack(anchor='w', padx=10)
        self.log_text = scrolledtext.ScrolledText(root, width=80, height=15)
        self.log_text.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)

    def choose_files(self):
        """选择多个数据文件"""
        files = filedialog.askopenfilenames(
            title="选择数据文件",
            filetypes=[("所有支持文件", "*.csv *.xlsx *.xls"),
                       ("CSV文件", "*.csv"),
                       ("Excel文件", "*.xlsx *.xls")]
        )
        if files:
            self.selected_files = files
            self.log_text.insert(tk.END, f"已选择 {len(files)} 个文件\n")
            self.log_text.see(tk.END)

    def choose_save_dir(self):
        """选择保存目录"""
        save_dir = filedialog.askdirectory(title="选择图片保存文件夹")
        if save_dir:
            self.save_directory = save_dir
            self.log_text.insert(tk.END, f"保存目录：{save_dir}\n")
            self.log_text.see(tk.END)

    def start_plot(self):
        """启动批量绘图（子线程运行，防界面卡死）"""
        if not self.selected_files:
            messagebox.showwarning("提示", "请先选择数据文件！")
            return
        if not self.save_directory:
            messagebox.showwarning("提示", "请先选择图片保存目录！")
            return

        # 子线程运行批量任务
        thread = threading.Thread(
            target=batch_process,
            args=(self.selected_files, self.save_directory, self.log_text)
        )
        thread.daemon = True
        thread.start()

# -------------------------- 启动程序 --------------------------
if __name__ == "__main__":
    main_root = tk.Tk()
    app = PlotGUI(main_root)
    main_root.mainloop()