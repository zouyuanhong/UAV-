import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

# -------------------------- 基础配置 --------------------------
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 200

# 颜色库
colors = [
    "#51a16a","#a15188","#5335c4","#c45335","#35a6c4","#1b5362","#000000"
]

# 多样式标记 + 线型
markers = ['o', '^', 's', 'D', 'v', '<', '>', 'p']
linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':']

# -------------------------- 数据读取 --------------------------
file_path = "E:/self.out/study/fifth/毕业论文/data/CSV/DEVC/不同距离/距离3/温升300.csv"
df = pd.read_csv(file_path)

data_cols = df.iloc[:, 38:55]
raw_labels = df.iloc[:, 0].astype(str).values
x = np.arange(6, 55, 3)

# -------------------------- 先拆分所有段，用于判断与取值 --------------------------
all_parts = [name.split('_') for name in raw_labels]
part1 = [p[0] for p in all_parts]
part2 = [p[1] for p in all_parts]
part3 = [p[2] for p in all_parts]

p1_diff = len(set(part1)) > 1
p2_diff = len(set(part2)) > 1
p3_diff = len(set(part3)) > 1

# 取固定段的数值（不变的那段就是固定值x）
fixed_v1 = part1[0]
fixed_v2 = part2[0]
fixed_v3 = part3[0]

# -------------------------- 生成精简图例 --------------------------
def get_simplified_labels(raw_names):
    new_names = []
    names = []
    for p in all_parts:
        v1, v2, v3 = p[0], p[1], p[2]
        label_parts = []
        if p1_diff:
            label_parts.append(f"下洗气流流速{v1}m/s")
        if p2_diff:
            label_parts.append(f"无人机高度{v2}m")
        if p3_diff:
            label_parts.append(f"无人机距离{v3}m")
        names.append(" | ".join(label_parts))
    
    new_names = sorted(names, key=lambda x: int(re.findall(r"\d+", x)[0]))

    return new_names

new_labels = get_simplified_labels(raw_labels)

# -------------------------- 自动生成图题 --------------------------
if p1_diff:
    # 片段1（流速）不同 → 图题：高度，距离，不同气流流速温升曲线
    title = f"{fixed_v2}m高度、{fixed_v3}m距离，不同下洗气流流速温升曲线"
elif p2_diff:
    # 片段2（高度）不同 → 图题：流速，距离，不同高度温升曲线
    title = f"{fixed_v1}m/s流速、{fixed_v3}m距离，不同无人机高度温升曲线"
elif p3_diff:
    # 片段3（距离）不同 → 图题：流速，高度，不同距离温升曲线
    title = f"{fixed_v1}m/s流速、{fixed_v2}m高度，不同无人机距离温升曲线"
else:
    title = "温升曲线对比"

# -------------------------- 绘图 --------------------------
fig, ax = plt.subplots(figsize=(4, 3))

for idx, (label, y_series) in enumerate(zip(new_labels, data_cols.values)):
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

# -------------------------- 图表样式 --------------------------
ax.set_xlabel("探测器高度（m）", fontsize=7, fontweight="bold")
ax.set_ylabel("探测器温度升至300℃所用时间（s）", fontsize=7, fontweight="bold")
ax.set_title(title, fontsize=7, fontweight="bold", pad=7)  # 动态图题
ax.set_xticks(x)

# 刻度向内 + 数字字号减小
ax.tick_params(
    direction='in',
    length=3,
    width=0.7,
    labelsize=7
)

# 图例右下角
ax.legend(loc="lower right", fontsize=6, frameon=True)

plt.tight_layout()
plt.savefig("科研折线图_最终版.png", dpi=200, bbox_inches="tight")
plt.show()