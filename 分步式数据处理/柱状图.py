import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import os

def plot_temperature_bar(file_path, output_dir="output"):
    """
    科研柱状图生成函数：计算指定列平均值，按规则排序与命名
    :param file_path: CSV文件路径
    :param output_dir: 图片输出目录
    """
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
    df = pd.read_csv(file_path)
    # 提取第一列（名称列）和指定数据列（43、47、51列，Python索引42、46、50）
    name_col = df.iloc[:, 0]
    target_cols = df.iloc[:, [42, 46, 50]]  # 43列=索引42，47列=索引46，51列=索引50
    # 合并名称与目标数据
    new_data = pd.concat([name_col, target_cols], axis=1)

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
    #print(final_labels)

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
    #ax.set_xlabel("实验工况", fontsize=7, fontweight="bold")
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
    # 保存图片
    output_path = os.path.join(output_dir, "科研柱状图_温升平均值.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"柱状图已保存至：{output_path}")

# -------------------------- 函数调用（直接运行） --------------------------
if __name__ == "__main__":
    # 传入新上传的CSV文件路径
    csv_file = "E:/self.out/study/fifth/毕业论文/data/CSV/DEVC/不同高度/高度4温升300.csv"
    # 生成柱状图
    plot_temperature_bar(csv_file)