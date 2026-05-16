import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
import threading
import pandas as pd
import os
import re

# -------------------------- 通用核心拼接函数 --------------------------
def merge_devc_hrr(devc_path, hrr_path, work_id, save_dir):
    """
    通用：拼接单个devc和hrr文件
    :return: (成功/失败, 信息)
    """
    try:
        # 读取文件（兼容编码）
        try:
            df_devc = pd.read_csv(devc_path, encoding='utf-8')
        except:
            df_devc = pd.read_csv(devc_path, encoding='gbk')
        try:
            df_hrr = pd.read_csv(hrr_path, encoding='utf-8')
        except:
            df_hrr = pd.read_csv(hrr_path, encoding='gbk')

        # 校验行数
        if len(df_devc) != len(df_hrr):
            return False, f"失败：{work_id} 文件行数不匹配"

        # 核心拼接：hrr第二列及以后拼接到devc末尾
        df_hrr_merge = df_hrr.iloc[:, 1:]
        df_result = pd.concat([df_devc, df_hrr_merge], axis=1)

        # 保存文件
        save_name = f"{work_id}_pre.csv"
        save_path = os.path.join(save_dir, save_name)
        df_result.to_csv(save_path, index=False, encoding='utf-8-sig')
        return True, f"成功：{work_id} → {save_name}"

    except Exception as e:
        return False, f"失败：{work_id} 错误：{str(e)}"

# -------------------------- 模式1：子文件夹工况处理 --------------------------
def process_mode1(root_dir, save_dir, log_widget):
    """模式1：根目录下每个子文件夹为一个工况，内含devc+hrr"""
    work_dirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    if not work_dirs:
        log_widget.insert(tk.END, "❌ 未找到任何工况子文件夹！\n")
        return

    log_widget.insert(tk.END, f"========== 模式1 - 共找到 {len(work_dirs)} 个工况 ==========\n")
    success, fail = 0, 0
    for work_dir in work_dirs:
        devc, hrr, work_id = None, None, None
        for f in os.listdir(work_dir):
            if f.endswith('_devc.csv'):
                devc = os.path.join(work_dir, f)
                work_id = re.sub(r'_devc\.csv$', '', f)
            elif f.endswith('_hrr.csv'):
                hrr = os.path.join(work_dir, f)
        if not devc or not hrr:
            log_widget.insert(tk.END, f"⏭️ 跳过：{os.path.basename(work_dir)} 缺少文件\n")
            fail +=1
            continue
        res, msg = merge_devc_hrr(devc, hrr, work_id, save_dir)
        log_widget.insert(tk.END, msg + "\n")
        success += res
        fail += not res
    log_widget.insert(tk.END, f"✅ 模式1处理完成 | 成功：{success} | 失败：{fail}\n\n")

# -------------------------- 模式2：分离文件夹匹配处理（新增） --------------------------
def process_mode2(devc_root, hrr_root, save_dir, log_widget):
    """模式2：独立DEVC文件夹 + 独立HRR文件夹，自动按工况编号匹配"""
    # 1. 扫描两个文件夹的所有文件
    devc_files = {}
    hrr_files = {}

    # 提取DEVC文件与工况编号
    for f in os.listdir(devc_root):
        if f.endswith('_devc.csv'):
            wid = re.sub(r'_devc\.csv$', '', f)
            devc_files[wid] = os.path.join(devc_root, f)

    # 提取HRR文件与工况编号
    for f in os.listdir(hrr_root):
        if f.endswith('_hrr.csv'):
            wid = re.sub(r'_hrr\.csv$', '', f)
            hrr_files[wid] = os.path.join(hrr_root, f)

    # 取交集（同时存在的工况）
    common_ids = set(devc_files.keys()) & set(hrr_files.keys())
    if not common_ids:
        log_widget.insert(tk.END, "❌ 未匹配到任何相同工况编号的文件！\n")
        return

    log_widget.insert(tk.END, f"========== 模式2 - 匹配到 {len(common_ids)} 个工况 ==========\n")
    success, fail, skip = 0, 0, 0

    # 处理匹配成功的工况
    for wid in common_ids:
        res, msg = merge_devc_hrr(devc_files[wid], hrr_files[wid], wid, save_dir)
        log_widget.insert(tk.END, msg + "\n")
        success += res
        fail += not res

    # 提示未匹配的文件
    devc_only = set(devc_files.keys()) - common_ids
    hrr_only = set(hrr_files.keys()) - common_ids
    if devc_only:
        log_widget.insert(tk.END, f"⚠️ 仅DEVC存在：{', '.join(devc_only)}\n")
    if hrr_only:
        log_widget.insert(tk.END, f"⚠️ 仅HRR存在：{', '.join(hrr_only)}\n")

    log_widget.insert(tk.END, f"✅ 模式2处理完成 | 成功：{success} | 失败：{fail}\n\n")

# -------------------------- 批量任务入口 --------------------------
def start_task(mode, root_dir, devc_dir, hrr_dir, save_dir, log_widget):
    log_widget.delete(1.0, tk.END)
    try:
        if mode == 1:
            if not root_dir or not save_dir:
                messagebox.showwarning("提示", "请选择根文件夹和保存目录！")
                return
            process_mode1(root_dir, save_dir, log_widget)
        else:
            if not devc_dir or not hrr_dir or not save_dir:
                messagebox.showwarning("提示", "请选择DEVC/HRR文件夹和保存目录！")
                return
            process_mode2(devc_dir, hrr_dir, save_dir, log_widget)
        messagebox.showinfo("完成", "批量拼接任务全部执行完毕！")
    except Exception as e:
        messagebox.showerror("错误", f"程序异常：{str(e)}")

# -------------------------- 可视化GUI界面 --------------------------
class MergeGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("双模式工况数据拼接工具 | devc+hrr 全自动匹配")
        self.root.geometry("750x550")

        # 变量
        self.mode = tk.IntVar(value=1)
        self.root_dir = tk.StringVar()
        self.devc_dir = tk.StringVar()
        self.hrr_dir = tk.StringVar()
        self.save_dir = tk.StringVar()

        # 顶部：模式选择
        tk.Label(root, text="选择处理模式：", font=("微软雅黑", 11)).pack(pady=5)
        frame_mode = tk.Frame(root)
        frame_mode.pack(pady=5)
        ttk.Radiobutton(frame_mode, text="模式1：工况子文件夹（内含devc+hrr）", variable=self.mode, value=1, command=self.switch_mode).grid(row=0, column=0, padx=10)
        ttk.Radiobutton(frame_mode, text="模式2：分离文件夹（总DEVC + 总HRR）", variable=self.mode, value=2, command=self.switch_mode).grid(row=0, column=1, padx=10)

        # 路径选择区域
        self.frame_path = tk.Frame(root)
        self.frame_path.pack(pady=10, fill=tk.X, padx=20)
        self.switch_mode()  # 初始化界面

        # 按钮区域
        tk.Button(root, text="开始批量拼接", command=self.run_thread, bg="#2196F3", fg="white", width=25, height=2).pack(pady=10)

        # 日志区域
        tk.Label(root, text="处理日志：", font=("微软雅黑", 10)).pack(anchor='w', padx=20)
        self.log_text = scrolledtext.ScrolledText(root, width=90, height=20)
        self.log_text.pack(padx=20, pady=5, fill=tk.BOTH, expand=True)

    def switch_mode(self):
        """切换模式，刷新路径界面"""
        for widget in self.frame_path.winfo_children():
            widget.destroy()
        mode = self.mode.get()

        if mode == 1:
            # 模式1：根文件夹
            tk.Label(self.frame_path, text="工况根文件夹：").grid(row=0, column=0, sticky='w')
            tk.Entry(self.frame_path, textvariable=self.root_dir, width=50).grid(row=0, column=1, padx=5)
            tk.Button(self.frame_path, text="选择", command=self.choose_root).grid(row=0, column=2)
        else:
            # 模式2：DEVC + HRR 两个文件夹
            tk.Label(self.frame_path, text="DEVC总文件夹：").grid(row=0, column=0, sticky='w')
            tk.Entry(self.frame_path, textvariable=self.devc_dir, width=50).grid(row=0, column=1, padx=5)
            tk.Button(self.frame_path, text="选择", command=self.choose_devc).grid(row=0, column=2)

            tk.Label(self.frame_path, text="HRR总文件夹：").grid(row=1, column=0, sticky='w', pady=5)
            tk.Entry(self.frame_path, textvariable=self.hrr_dir, width=50).grid(row=1, column=1, padx=5)
            tk.Button(self.frame_path, text="选择", command=self.choose_hrr).grid(row=1, column=2)

        # 统一保存目录
        tk.Label(self.frame_path, text="结果保存目录：").grid(row=2, column=0, sticky='w', pady=5)
        tk.Entry(self.frame_path, textvariable=self.save_dir, width=50).grid(row=2, column=1, padx=5)
        tk.Button(self.frame_path, text="选择", command=self.choose_save).grid(row=2, column=2)

    # 路径选择函数
    def choose_root(self): self.root_dir.set(filedialog.askdirectory(title="选择工况根文件夹"))
    def choose_devc(self): self.devc_dir.set(filedialog.askdirectory(title="选择DEVC总文件夹"))
    def choose_hrr(self): self.hrr_dir.set(filedialog.askdirectory(title="选择HRR总文件夹"))
    def choose_save(self): self.save_dir.set(filedialog.askdirectory(title="选择结果保存文件夹"))

    def run_thread(self):
        """子线程运行，防卡顿"""
        thread = threading.Thread(target=start_task, args=(
            self.mode.get(),
            self.root_dir.get(),
            self.devc_dir.get(),
            self.hrr_dir.get(),
            self.save_dir.get(),
            self.log_text
        ))
        thread.daemon = True
        thread.start()

# -------------------------- 启动 --------------------------
if __name__ == "__main__":
    main_root = tk.Tk()
    app = MergeGUI(main_root)
    main_root.mainloop()