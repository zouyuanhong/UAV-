"""
CSV温度数据处理器 - 检测300℃首次到达时间
功能：
1. 批量处理文件夹及子文件夹中的CSV文件
2. 可筛选包含特定关键词的CSV文件
3. 检测每个探测器首次达到300℃的时间
4. 探测器按【6竖列分组 + 每列高度Z从低到高】排序
5. 列顺序：Y=-1.4 → Y≈0 → Y=3 → Y=-3 → Y=6 → Y=-6
6. 保存处理结果到新的CSV文件
"""
import pandas as pd
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, scrolledtext
import glob
from datetime import datetime
import threading

class CSVDataProcessor:
    def __init__(self, root):
        self.root = root
        self.root.title("CSV温度数据处理器 - 6列高度排序版")
        self.root.geometry("900x700")
        
        # 变量初始化
        self.folder_path = ""
        self.keyword = ""
        self.processing = False
        self.result_data = []
        self.detector_names = []
        
        # 🔥 固定6列分组 + 高度从低到高（你提供的探测器清单）
        self.FIXED_SORTED_DETECTORS = [
            # 列1：Y=-1.4（高度Z从低到高）
            "Device270","Device","Device272","Device271","Device274","Device273",
            "Device276","Device275","Device278","Device277","Device280","Device279",
            "Device282","Device281","Device284","Device283","Device286","Device285",
            "Device288","Device287","Device290","Device289","Device292","Device291",
            "Device294","Device293","Device296","Device295","Device298","Device297",
            "Device300","Device299","Device302","Device301","Device304","Device303",
            
            # 列2：Y≈0
            "Device305","Device306","Device307","Device308","Device309","Device310",
            "Device315","Device311","Device312","Device313","Device316","Device314",
            "Device319","Device318","Device321","Device320","Device317","Device322",
            
            # 列3：Y=3.0
            "Device359","Device360","Device361","Device362","Device363","Device364",
            "Device369","Device365","Device366","Device367","Device370","Device368",
            "Device372","Device371","Device373","Device375","Device374","Device376",
            
            # 列4：Y=-3.0
            "Device377","Device378","Device379","Device380","Device381","Device382",
            "Device387","Device383","Device384","Device385","Device388","Device386",
            "Device390","Device389","Device391","Device393","Device392","Device394",
            
            # 列5：Y=6.0
            "Device395","Device396","Device397","Device398","Device399","Device400",
            "Device405","Device401","Device402","Device403","Device406","Device404",
            "Device408","Device407","Device409","Device410","Device411","Device412",
            
            # 列6：Y=-6.0
            "Device413","Device414","Device415","Device418","Device416","Device417",
            "Device423","Device419","Device421","Device420","Device424","Device422",
            "Device429","Device428","Device430","Device427","Device426","Device425"
        ]
        
        # 创建界面
        self.create_widgets()

    def create_widgets(self):
        folder_frame = ttk.LabelFrame(self.root, text="1. 文件夹设置")
        folder_frame.pack(fill=tk.X, padx=10, pady=5, ipady=5)
        ttk.Label(folder_frame, text="目标文件夹:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.folder_entry = ttk.Entry(folder_frame, width=60)
        self.folder_entry.grid(row=0, column=1, padx=5, pady=5, columnspan=2)
        ttk.Button(folder_frame, text="浏览", command=self.select_folder).grid(row=0, column=3, padx=5, pady=5)
        ttk.Label(folder_frame, text="文件名关键词:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.keyword_entry = ttk.Entry(folder_frame, width=60)
        self.keyword_entry.grid(row=1, column=1, padx=5, pady=5, columnspan=2)
        ttk.Label(folder_frame, text="(留空处理全部CSV)").grid(row=1, column=3, padx=5, pady=5)
        ttk.Label(folder_frame, text="✅ 排序规则：6列分组 + 每列高度Z从低到高", foreground="green").grid(row=2, column=0, columnspan=4, padx=5, pady=2)

        control_frame = ttk.LabelFrame(self.root, text="2. 处理控制")
        control_frame.pack(fill=tk.X, padx=10, pady=5, ipady=5)
        self.process_btn = ttk.Button(control_frame, text="开始处理", command=self.start_process)
        self.process_btn.pack(side=tk.LEFT, padx=10, pady=5)
        self.save_btn = ttk.Button(control_frame, text="保存结果", command=self.save_results, state=tk.DISABLED)
        self.save_btn.pack(side=tk.LEFT, padx=10, pady=5)
        self.clear_btn = ttk.Button(control_frame, text="清空日志", command=self.clear_log)
        self.clear_btn.pack(side=tk.LEFT, padx=10, pady=5)

        progress_frame = ttk.LabelFrame(self.root, text="3. 处理进度")
        progress_frame.pack(fill=tk.X, padx=10, pady=5, ipady=5)
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, padx=10, pady=5)
        self.status_label = ttk.Label(progress_frame, text="就绪")
        self.status_label.pack(padx=10, pady=2)

        log_frame = ttk.LabelFrame(self.root, text="4. 处理日志")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.log_text = scrolledtext.ScrolledText(log_frame, height=20, state=tk.DISABLED)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        result_frame = ttk.LabelFrame(self.root, text="5. 结果预览（已按6列+高度排序）")
        result_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.result_text = scrolledtext.ScrolledText(result_frame, height=10, state=tk.DISABLED)
        self.result_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def select_folder(self):
        folder = filedialog.askdirectory(title="选择CSV文件夹")
        if folder:
            self.folder_path = folder
            self.folder_entry.delete(0, tk.END)
            self.folder_entry.insert(0, folder)
            self.log(f"已选文件夹：{folder}")

    def log(self, msg):
        self.log_text.config(state=tk.NORMAL)
        t = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{t}] {msg}\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)
        self.root.update_idletasks()

    def clear_log(self):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state=tk.DISABLED)
        self.log("日志已清空")

    def update_progress(self, p, s):
        self.progress_var.set(p)
        self.status_label.config(text=s)
        self.root.update_idletasks()

    def start_process(self):
        if self.processing:
            messagebox.showwarning("提示", "正在处理中")
            return
        self.folder_path = self.folder_entry.get().strip()
        self.keyword = self.keyword_entry.get().strip()
        if not self.folder_path or not os.path.exists(self.folder_path):
            messagebox.showerror("错误", "请选择有效文件夹")
            return
        self.processing = True
        self.process_btn.config(state=tk.DISABLED)
        self.save_btn.config(state=tk.DISABLED)
        self.result_data = []
        threading.Thread(target=self.process_csv).start()

    def process_csv(self):
        try:
            pattern = os.path.join(self.folder_path, "**", f"*{self.keyword}*.csv" if self.keyword else "*.csv")
            files = glob.glob(pattern, recursive=True)
            if not files:
                self.log("未找到CSV")
                self.update_progress(0, "完成")
                self.processing = False
                self.process_btn.config(state=tk.NORMAL)
                return
            total = len(files)
            self.log(f"找到 {total} 个CSV")
            self.update_progress(0, f"共{total}个文件")

            for i, path in enumerate(files, 1):
                name = os.path.basename(path)
                self.log(f"→ 处理 {i}/{total}：{name}")
                try:
                    df = pd.read_csv(path)
                    if df.shape[1] < 2:
                        self.log(f"⚠ {name} 格式无效，跳过")
                        continue
                    if not self.detector_names:
                        self.detector_names = df.iloc[0, 1:].tolist()
                        self.log(f"读取探测器总数：{len(self.detector_names)}")

                    time_data = pd.to_numeric(df.iloc[1:, 0], errors="coerce").dropna()
                    #name=str.replace("_.csv","")
                    row = [name]
                    for d in self.FIXED_SORTED_DETECTORS:
                        if d not in self.detector_names:
                            row.append("无此探测器")
                            continue
                        col = self.detector_names.index(d) + 1
                        temp = pd.to_numeric(df.iloc[1:, col], errors="coerce").dropna()
                        if len(temp) == 0 or len(time_data) == 0:
                            row.append("无数据")
                            continue
                        m = min(len(temp), len(time_data))
                        t_series = time_data.iloc[:m].reset_index(drop=True)
                        tp_series = temp.iloc[:m].reset_index(drop=True)
                        mask = tp_series >= 300
                        if mask.any():
                            idx = mask.idxmax()
                            row.append(f"{t_series.iloc[idx]:.2f}")
                        else:
                            row.append("未达到")
                    self.result_data.append(row)
                    self.update_progress(i/total*100, f"{i}/{total}")
                except Exception as e:
                    self.log(f"❌ 处理失败：{str(e)}")

            self.log("✅ 全部处理完成")
            self.update_progress(100, "完成")
            self.show_preview()
            self.save_btn.config(state=tk.NORMAL)
        except Exception as e:
            self.log(f"💥 异常：{str(e)}")
        finally:
            self.processing = False
            self.process_btn.config(state=tk.NORMAL)

    def show_preview(self):
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete(1.0, tk.END)
        head = ["文件名"] + self.FIXED_SORTED_DETECTORS[:10]
        self.result_text.insert(tk.END, "\t".join(str(x) for x in head) + "\t...\n")
        self.result_text.insert(tk.END, "-"*120 + "\n")
        for r in self.result_data[:10]:
            self.result_text.insert(tk.END, "\t".join(str(x) for x in r[:11]) + "\t...\n")
        self.result_text.config(state=tk.DISABLED)

    def save_results(self):
        if not self.result_data:
            messagebox.showwarning("提示", "无结果")
            return
        save_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
            initialfile=f"温升300.csv"
        )
        if not save_path:
            return
        df_out = pd.DataFrame(
            self.result_data,
            columns=["文件名"] + self.FIXED_SORTED_DETECTORS
        )
        df_out.to_csv(save_path, index=False, encoding="utf-8-sig")
        self.log(f"✅ 已保存：{save_path}")
        messagebox.showinfo("完成", f"已保存：\n{save_path}")

def main():
    root = tk.Tk()
    app = CSVDataProcessor(root)
    root.protocol("WM_DELETE_WINDOW", lambda: root.destroy() if not app.processing else messagebox.askokcancel("退出", "仍在处理，确定退出？") and root.destroy())
    root.mainloop()

if __name__ == "__main__":
    main()