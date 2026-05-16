import os
import csv
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

def process_csv_files(folder_path, output_path):
    """批量处理CSV文件核心逻辑（UTF-8编码）"""
    result_dict = {}
    
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            extracted_value = None
            
            try:
                # 读取：强制 UTF-8
                with open(file_path, "r", encoding="utf-8") as csv_file:
                    reader = csv.reader(csv_file)
                    for row in reader:
                        if not row:
                            continue
                        
                        last_col = row[-1].strip()
                        try:
                            last_col_num = float(last_col)
                        except ValueError:
                            continue
                        
                        # 匹配 53.4
                        if abs(last_col_num - 53.4) < 0.4:
                            first_col = row[0].strip()
                            extracted_value = first_col
                            break
                
                result_dict[filename] = extracted_value if extracted_value is not None else "未找到"
            
            except Exception as e:
                result_dict[filename] = f"读取失败：{str(e)}"
    
    # 写入：强制 UTF-8 + 支持Excel正常识别
    with open(output_path, "w", encoding="utf-8-sig", newline="") as out_csv:
        writer = csv.writer(out_csv)
        writer.writerow(result_dict.keys())
        writer.writerow(result_dict.values())
    
    return len(result_dict)

def select_folder():
    path = filedialog.askdirectory(title="请选择存放CSV文件的文件夹")
    if path:
        folder_entry.delete(0, tk.END)
        folder_entry.insert(0, path)

def save_result():
    path = filedialog.asksaveasfilename(
        defaultextension=".csv",
        filetypes=[("CSV文件", "*.csv")],
        title="保存结果文件",
        initialfile="提取结果.csv"
    )
    if path:
        save_entry.delete(0, tk.END)
        save_entry.insert(0, path)

def start_process():
    folder = folder_entry.get().strip()
    save_path = save_entry.get().strip()
    
    if not folder or not save_path:
        messagebox.showerror("错误", "请先选择文件夹和保存路径！")
        return
    
    if not os.path.isdir(folder):
        messagebox.showerror("错误", "文件夹路径无效！")
        return
    
    try:
        count = process_csv_files(folder, save_path)
        messagebox.showinfo("完成", f"处理成功！\n共处理 {count} 个CSV文件\n结果已保存至：\n{save_path}")
    except Exception as e:
        messagebox.showerror("处理失败", f"错误信息：{str(e)}")

# ==================== GUI 界面 ====================
root = tk.Tk()
root.title("CSV批量提取工具 - 查找最后一列第一个53.4")
root.geometry("650x280")
root.resizable(False, False)

# 文件夹选择
tk.Label(root, text="1. 选择CSV文件夹：", font=("微软雅黑", 11, "bold")).place(x=20, y=20)
folder_entry = ttk.Entry(root, width=50)
folder_entry.place(x=20, y=50)
ttk.Button(root, text="浏览", command=select_folder).place(x=470, y=48)

# 保存路径
tk.Label(root, text="2. 结果保存路径：", font=("微软雅黑", 11, "bold")).place(x=20, y=100)
save_entry = ttk.Entry(root, width=50)
save_entry.place(x=20, y=130)
ttk.Button(root, text="选择", command=save_result).place(x=470, y=128)

# 开始按钮
start_btn = ttk.Button(root, text="▶ 开始处理", command=start_process)
start_btn.place(x=220, y=180, width=200, height=50)

# 说明
tk.Label(root, text="编码：UTF-8  | 功能：提取最后一列第一个=53.4 对应的第一列数值", fg="gray").place(x=20, y=240)

root.mainloop()