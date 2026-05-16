import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ---------------------- 数据 ----------------------
data = {
    '文件': [
        '10m高度，3m距离', '10m高度，4m距离', '12m高度，3m距离', '14m高度，6m距离',
        '16m高度，6m距离', '16m高度，7m距离', '4m高度，4m距离', '4m高度，7m距离',
        '6m高度，6m距离', '6m高度，7m距离', '8m高度，4m距离', '8m高度，6m距离'
    ],
    '预测': [604.80194, 679.20284, 576.00398, 643.20349, 626.40162, 631.20294,
             840.00377, 792.00368, 686.40014, 679.20121, 712.80040, 708.00404],
    '实际': [609.9, 683.6, 579.6, 643.5, 621.3, 633.7,
             838.2, 794.5, 683.3, 673.4, 711.1, 708.1]
}

df = pd.DataFrame(data)
df['偏差'] = df['预测'] - df['实际']
df['偏差百分比(%)'] = (df['偏差'] / df['实际']) * 100

# ---------------------- 绘图设置 ----------------------
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

x = np.arange(len(df))
width = 0.35

fig, ax1 = plt.subplots(figsize=(16, 7))

# ===================== 高级配色 =====================
color_pred = '#2E86AB'      # 预测：沉稳蓝
color_real = "#9E3BA2"      # 实际：玫瑰紫
color_dev  = '#F18F01'      # 偏差：暖橙
color_pct  = "#C78E1D"      # 百分比：醒目红

# 柱状图
bars1 = ax1.bar(x - width/2, df['预测'], width, label='模型预测值', color=color_pred)
bars2 = ax1.bar(x + width/2, df['实际'], width, label='实际模拟值', color=color_real)

ax1.set_ylabel('火焰到达顶端所需时间时间（s）', fontsize=15)
ax1.set_xticks(x)
ax1.set_xticklabels(df['文件'], rotation=25, ha='right')
ax1.legend(loc='upper left')
ax1.grid(alpha=0.2)

# ===================== 标签位置精细调整 =====================
# 预测值标签 → 偏左上
for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/5+0.3, height + 3,
             f'{height:.2f}', ha='right', va='bottom', fontsize=12, color=color_pred, fontweight='bold')

# 实际值标签 → 偏右上
for bar in bars2:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/5-0.05 , height + 3,
             f'{height:.2f}', ha='left', va='bottom', fontsize=12, color=color_real, fontweight='bold')

# 双轴：偏差折线
ax2 = ax1.twinx()
ax2.plot(x, df['偏差'], color=color_dev, marker='o', linewidth=2.5, label='预测时间与实际时间偏差')
ax2.set_ylabel('预测结果与实际结果偏差时间（s）', color=color_dev, fontsize=15)
ax2.tick_params(axis='y', direction='in', labelcolor=color_dev)


ax1.set_ylim(0, 900)
ax1.set_yticks(np.arange(0, 901, 100))
ax2.set_ylim(-6, 14)
ax2.set_yticks(np.arange(-6, 15, 2))


# ===================== 百分比标签大幅上移 =====================
for i, pct in enumerate(df['偏差百分比(%)']):
    ax1.text(i, max(df['预测'][i], df['实际'][i]) + 35,
             f'偏差百分比\n{pct:.2f}%', ha='center', color=color_pct, fontweight='bold', fontsize=12)

# 图例合并
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

plt.tight_layout()
plt.savefig('预测值_实际值_偏差_百分比.png', dpi=300)
plt.show()