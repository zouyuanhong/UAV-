#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lateral Temperature Gradient Analyzer v3.4
- All font sizes increased
"""

import os
import re
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)

try:
    from scipy.interpolate import griddata
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ================================================================
# GLOBAL FONT CONFIGURATION
# ================================================================

FONT = {
    'title':      19,
    'suptitle':   22,
    'axis_label': 22,
    'tick':       19,
    'legend':     25,
    'annotate':   15,
    'bar_label':  15,
    'info_text':  17,
    'colorbar':   19,
}

matplotlib.rcParams.update({
    'font.size':          FONT['tick'],
    'axes.titlesize':     FONT['title'],
    'axes.labelsize':     FONT['axis_label'],
    'xtick.labelsize':    FONT['tick'],
    'ytick.labelsize':    FONT['tick'],
    'legend.fontsize':    FONT['legend'],
    'figure.titlesize':   FONT['suptitle'],
})


# ================================================================
# CONSTANTS
# ================================================================

Z_LEVELS = [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35,
            38, 41, 44, 47, 50, 53]

LAYER_DEVICES = {
    'L1': {
        'y': -1.4, 'label': 'Y=-1.4m (fire source)',
        'devices': [
            'Device', 'Device271', 'Device273', 'Device275',
            'Device277', 'Device279', 'Device281', 'Device283',
            'Device285', 'Device287', 'Device289', 'Device291',
            'Device293', 'Device295', 'Device297', 'Device299',
            'Device301', 'Device303']
    },
    'L2': {
        'y': 0.0, 'label': 'Y=0m (wall surface)',
        'devices': [
            'Device305', 'Device306', 'Device307', 'Device308',
            'Device309', 'Device310', 'Device315', 'Device311',
            'Device312', 'Device313', 'Device316', 'Device314',
            'Device319', 'Device318', 'Device321', 'Device320',
            'Device317', 'Device322']
    },

    'L3': {
        'y': 3.0, 'label': 'Y=3m (outer mid)',
        'devices': [
            'Device359', 'Device360', 'Device361', 'Device362',
            'Device363', 'Device364', 'Device369', 'Device365',
            'Device366', 'Device367', 'Device370', 'Device368',
            'Device372', 'Device371', 'Device373', 'Device375',
            'Device374', 'Device376']
    },
    'L4': {
        'y': -3.0, 'label': 'Y=-3m (inner near)',
        'devices': [
            'Device377', 'Device378', 'Device379', 'Device380',
            'Device381', 'Device382', 'Device387', 'Device383',
            'Device384', 'Device385', 'Device388', 'Device386',
            'Device390', 'Device389', 'Device391', 'Device393',
            'Device392', 'Device394']
    },
    'L5': {
        'y': -6.0, 'label': 'Y=-6m (inner far)',
        'devices': [
            'Device413', 'Device414', 'Device415', 'Device418',
            'Device416', 'Device417', 'Device423', 'Device419',
            'Device421', 'Device420', 'Device424', 'Device422',
            'Device429', 'Device428', 'Device430', 'Device427',
            'Device426', 'Device425']
    },
    'L6': {
        'y': 6.0, 'label': 'Y=6m (outer far)',
        'devices': [
            'Device395', 'Device396', 'Device397', 'Device398',
            'Device399', 'Device400', 'Device405', 'Device401',
            'Device402', 'Device403', 'Device406', 'Device404',
            'Device408', 'Device407', 'Device409', 'Device410',
            'Device411', 'Device412']
    },
}

EXCLUDED_LAYERS = {'L1'}
ALL_SORTED = sorted(LAYER_DEVICES, key=lambda k: LAYER_DEVICES[k]['y'])
ANALYSIS_LAYERS = [ln for ln in ALL_SORTED if ln not in EXCLUDED_LAYERS]
ANALYSIS_Y = [LAYER_DEVICES[ln]['y'] for ln in ANALYSIS_LAYERS]

LAYER_COLORS = {
    'L2': '#2ca02c', 'L3': '#9467bd',
    'L4': '#ff7f0e', 'L5': '#1f77b4', 'L6': '#8c564b'}

COND_COLORS = [
    '#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00',
    '#a65628', '#f781bf', '#66c2a5', '#fc8d62', '#1b9e77']
COND_STYLES = ['-', '--', '-.', ':'] * 3
COND_MARKERS = ['o', 's', '^', 'D', 'v', 'p', 'h', '*', 'X', 'P']

SEG_UNITS = {0: 'm/s流速', 1: 'm高度', 2: 'm距离'}

SINGLE_TYPES = [
    ('profile',     '横向温度剖面'),
    ('evolution',   '剖面时间演化'),
    ('grad_ts',     '梯度时序曲线'),
    ('grad_heat',   '梯度热力图'),
    ('asymmetry',   '不对称指数分析'),
    ('asym_heat',   '不对称指数热力图'),
    ('yz_contour',  'Y-Z 温度等值线图'),
    ('yz_compare',  'Y-Z 多时刻温度对比'),
]

CMP_TYPES = [
    ('cmp_profile',          '横向剖面对比'),
    ('cmp_grad_ts',          '梯度时序对比'),
    ('cmp_asymmetry',        '不对称指数对比'),
    ('cmp_peak_temp',        '峰值温度对比'),
    ('cmp_reach_time',       '升温所需时间'),
    ('cmp_outer_reach',      '外层到达时间对比'),
    ('cmp_outer_first_ht',   'L6首达高度柱状图'),
]

ALL_TYPES = SINGLE_TYPES + CMP_TYPES

PARAM_NEEDS = {
    'profile':          ['height', 'time'],
    'evolution':        ['height', 'count'],
    'grad_ts':          ['height'],
    'grad_heat':        [],
    'asymmetry':        [],
    'asym_heat':        [],
    'yz_contour':       ['time'],
    'yz_compare':       ['count'],
    'cmp_profile':      ['height', 'time'],
    'cmp_grad_ts':      ['height'],
    'cmp_asymmetry':    ['height'],
    'cmp_peak_temp':    ['layer'],
    'cmp_reach_time':   ['layer', 'threshold'],
    'cmp_outer_reach':  ['threshold'],
    'cmp_outer_first_ht': ['threshold'],
}

HINTS = {
    'profile':     '选择高度和时间',
    'evolution':   '选择高度，叠加多个时间步',
    'grad_ts':     '选择高度，显示层间梯度随时间变化',
    'grad_heat':   '总梯度(L5→L6)，时间×高度热力图',
    'asymmetry':   'AI 在 6 个代表性高度处的曲线',
    'asym_heat':   'AI 时间×高度热力图',
    'yz_contour':  '选择时间，Y-Z 等值线图',
    'yz_compare':  '自动多时刻 Y-Z 等值线对比',
    'cmp_profile':     '所有工况：相同 Z 和 t 下的 T-Y 曲线',
    'cmp_grad_ts':     '所有工况：总梯度随时间变化',
    'cmp_asymmetry':   '所有工况：AI 随时间变化',
    'cmp_peak_temp':   '所有工况：峰值温度 vs 高度',
    'cmp_reach_time':  '所有工况：达到阈值时间 vs 高度',
    'cmp_outer_reach': 'L5 和 L6 首次到达时间与高度',
    'cmp_outer_first_ht': (
        '柱高 = L6层首个达到阈值的探测器高度\n'
        '误差 = 前10个探测器(按到达时间排序)高度的标准差'),
}


# ================================================================
# AUTO-RENAME
# ================================================================

def _parse_segments(name):
    parts = name.split('_')
    if len(parts) < 3:
        return None
    try:
        return [float(p) for p in parts[:4]]
    except ValueError:
        return None

def _fmt_val(v):
    return f'{v:g}'

def auto_rename_conditions(conditions, orig_map):
    if not conditions:
        return conditions, orig_map
    old_names = list(conditions.keys())
    parsed = {}
    for dn in old_names:
        on = orig_map.get(dn, dn)
        segs = _parse_segments(on)
        if segs is not None:
            parsed[dn] = segs
    if not parsed:
        return conditions, orig_map
    baselines = [dn for dn, s in parsed.items()
                 if all(v == 0 for v in s[:3])]
    others = [dn for dn, s in parsed.items()
              if not all(v == 0 for v in s[:3])]
    vary = set()
    if len(others) >= 2:
        first = parsed[others[0]]
        for dn in others[1:]:
            s = parsed[dn]
            for i in range(3):
                if s[i] != first[i]:
                    vary.add(i)
    elif len(others) == 1:
        s = parsed[others[0]]
        for i in range(3):
            if s[i] != 0:
                vary.add(i)
    rename = {}
    for dn in baselines:
        rename[dn] = '基准工况'
    for dn in others:
        s = parsed[dn]
        parts = []
        if vary:
            for i in sorted(vary):
                parts.append(f'{_fmt_val(s[i])}{SEG_UNITS[i]}')
        else:
            for i in range(3):
                if s[i] != 0:
                    parts.append(f'{_fmt_val(s[i])}{SEG_UNITS[i]}')
        rename[dn] = ' '.join(parts) if parts else dn
    seen = {}
    final = {}
    for dn, label in rename.items():
        if label in seen:
            seen[label] += 1
            final[dn] = f'{label} ({seen[label]})'
        else:
            seen[label] = 1
            final[dn] = label
    new_conditions, new_orig = {}, {}
    for dn in old_names:
        nd = final.get(dn, dn)
        new_conditions[nd] = conditions[dn]
        new_orig[nd] = orig_map.get(dn, dn)
    return new_conditions, new_orig


# ================================================================
# ENGINE
# ================================================================

class Engine:
    def __init__(self):
        self.df = None
        self.grad_df = None
        self._pairs = None

    def load(self, path):
        df = pd.read_csv(path, header=1)
        df.columns = [c.strip().strip('"').strip() for c in df.columns]
        df['Time'] = pd.to_numeric(df['Time'], errors='coerce')
        df.dropna(subset=['Time'], inplace=True)
        df.sort_values('Time', inplace=True)
        df.reset_index(drop=True, inplace=True)
        missing = [d for info in LAYER_DEVICES.values()
                   for d in info['devices'] if d not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing[:5]}")
        self.df = df
        self.grad_df = None
        self._pairs = [
            (ANALYSIS_LAYERS[i], ANALYSIS_LAYERS[i + 1],
             LAYER_DEVICES[ANALYSIS_LAYERS[i + 1]]['y']
             - LAYER_DEVICES[ANALYSIS_LAYERS[i]]['y'])
            for i in range(len(ANALYSIS_LAYERS) - 1)]
        return df

    def temps(self, row):
        return {ln: np.array([row[d] for d in info['devices']],
                             dtype=float)
                for ln, info in LAYER_DEVICES.items()}

    def tidx(self, t):
        return int((self.df['Time'] - t).abs().idxmin())

    def tval(self, t):
        return float(self.df.iloc[self.tidx(t)]['Time'])

    def trange(self):
        return float(self.df['Time'].min()), float(self.df['Time'].max())

    def sample_times(self, n=8):
        idxs = np.linspace(0, len(self.df) - 1, n, dtype=int)
        return self.df['Time'].iloc[idxs].tolist()

    def compute_gradients(self, cb=None):
        N, H = len(self.df), len(Z_LEVELS)
        T = {ln: self.df[info['devices']].values
             for ln, info in LAYER_DEVICES.items()}
        if cb: cb(30, 100)
        grads = {}
        for n1, n2, dy in self._pairs:
            grads[f'{n1}\u2192{n2}'] = (T[n2] - T[n1]) / dy
        dy_tot = LAYER_DEVICES['L6']['y'] - LAYER_DEVICES['L5']['y']
        g_total = (T['L6'] - T['L5']) / dy_tot
        t_out = (T['L3'] + T['L6']) / 2.0
        t_in = (T['L4'] + T['L5']) / 2.0
        ai = (t_out - t_in) / (t_out + t_in + 1e-10)
        if cb: cb(70, 100)
        times = self.df['Time'].values
        data = {
            'Time': np.repeat(times, H),
            'Z': np.tile(np.array(Z_LEVELS, dtype=float), N),
            'k': np.tile(np.arange(H), N)}
        for key, arr in grads.items():
            data[f'G_{key}'] = arr.flatten()
        data['G_total'] = g_total.flatten()
        data['AI'] = ai.flatten()
        for ln in LAYER_DEVICES:
            data[f'T_{ln}'] = T[ln].flatten()
        self.grad_df = pd.DataFrame(data)
        if cb: cb(100, 100)
        return self.grad_df

    def reach_time(self, threshold=300, layer='L2'):
        devs = LAYER_DEVICES[layer]['devices']
        t = self.df['Time'].values
        out = np.full(len(devs), np.nan)
        for i, d in enumerate(devs):
            mask = self.df[d].values >= threshold
            if mask.any():
                out[i] = t[mask][0]
        return out

    def peak_temp(self, layer='L2'):
        devs = LAYER_DEVICES[layer]['devices']
        return np.array([self.df[d].max() for d in devs])

    # ---- single-condition plots ----

    def plot_profile(self, time, k):
        row = self.df.iloc[self.tidx(time)]
        T = self.temps(row)
        yv, tv = ANALYSIS_Y, [T[ln][k] for ln in ANALYSIS_LAYERS]
        fig = Figure(figsize=(11, 7), dpi=100)
        ax = fig.add_subplot(111)
        ax.plot(yv, tv, 'o-', lw=2.5, ms=10, color='#d62728',
                mfc='white', mew=2.5)
        for y, v, ln in zip(yv, tv, ANALYSIS_LAYERS):
            ax.annotate(f'{v:.0f}', (y, v),
                        textcoords='offset points', xytext=(0, 14),
                        ha='center', fontsize=FONT['annotate'],
                        color=LAYER_COLORS[ln])
        ax.set_xlabel('Y (m)', fontsize=FONT['axis_label'])
        ax.set_ylabel('温度 (°C)', fontsize=FONT['axis_label'])
        ax.set_title(f'横向温度剖面  Z={Z_LEVELS[k]}m  '
                     f't={row["Time"]:.0f}s',
                     fontsize=FONT['title'])
        ax.tick_params(labelsize=FONT['tick'])
        ax.grid(True, alpha=.3, ls='--')
        fig.tight_layout()
        return fig

    def plot_evolution(self, k, times):
        fig = Figure(figsize=(13, 8), dpi=100)
        ax = fig.add_subplot(111)
        for i, tgt in enumerate(times):
            row = self.df.iloc[self.tidx(tgt)]
            T = self.temps(row)
            tv = [T[ln][k] for ln in ANALYSIS_LAYERS]
            frac = i / max(len(times) - 1, 1)
            ax.plot(ANALYSIS_Y, tv, 'o-',
                    color=cm.hot(0.15 + 0.8 * frac),
                    lw=1.8, ms=7, label=f't={row["Time"]:.0f}s')
        ax.set_xlabel('Y (m)', fontsize=FONT['axis_label'])
        ax.set_ylabel('温度 (°C)', fontsize=FONT['axis_label'])
        ax.set_title(f'剖面时间演化  Z={Z_LEVELS[k]}m',
                     fontsize=FONT['title'])
        ax.legend(fontsize=FONT['legend'], loc='best')
        ax.tick_params(labelsize=FONT['tick'])
        ax.grid(True, alpha=.3, ls='--')
        fig.tight_layout()
        return fig

    def plot_grad_ts(self, k):
        gkeys = ([f'G_{ANALYSIS_LAYERS[i]}\u2192{ANALYSIS_LAYERS[i+1]}'
                  for i in range(len(ANALYSIS_LAYERS) - 1)]
                 + ['G_total'])
        sub = self.grad_df[self.grad_df['k'] == k].sort_values('Time')
        t = sub['Time'].values
        cols = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#e377c2']
        sty = ['--', '--', '--', '--', '-']
        fig = Figure(figsize=(15, 8), dpi=100)
        ax = fig.add_subplot(111)
        for i, gk in enumerate(gkeys):
            ax.plot(t, sub[gk].values, ls=sty[i], color=cols[i],
                    lw=2, label=gk.replace('G_', ''))
        ax.axhline(0, color='k', lw=.8)
        ax.set_xlabel('时间 (s)', fontsize=FONT['axis_label'])
        ax.set_ylabel('梯度 (°C/m)', fontsize=FONT['axis_label'])
        ax.set_title(f'横向梯度时序  Z={Z_LEVELS[k]}m',
                     fontsize=FONT['title'])
        ax.legend(fontsize=FONT['legend'], loc='best')
        ax.tick_params(labelsize=FONT['tick'])
        ax.grid(True, alpha=.3, ls='--')
        fig.tight_layout()
        return fig

    def plot_grad_heatmap(self):
        piv = self.grad_df.pivot_table(
            index='Z', columns='Time', values='G_total')
        Za, Ta, Ga = piv.index.values, piv.columns.values, piv.values
        fig = Figure(figsize=(17, 9), dpi=100)
        ax = fig.add_subplot(111)
        vmax = np.nanpercentile(np.abs(Ga), 98) or 1
        im = ax.pcolormesh(Ta, Za, Ga, cmap='RdBu_r',
                           vmin=-vmax, vmax=vmax, shading='auto')
        cb = fig.colorbar(im, ax=ax, pad=.02)
        cb.set_label('梯度 (°C/m)', fontsize=FONT['colorbar'])
        cb.ax.tick_params(labelsize=FONT['tick'])
        ax.set_xlabel('时间 (s)', fontsize=FONT['axis_label'])
        ax.set_ylabel('高度 Z (m)', fontsize=FONT['axis_label'])
        ax.set_title('总横向梯度热力图 (L5→L6)',
                     fontsize=FONT['title'])
        ax.tick_params(labelsize=FONT['tick'])
        fig.tight_layout()
        return fig

    def plot_asymmetry(self, ks=None):
        if ks is None:
            ks = [0, 3, 6, 9, 12, 15]
        nc = min(3, len(ks)); nr = (len(ks) + nc - 1) // nc
        fig = Figure(figsize=(7 * nc, 5 * nr), dpi=100)
        for i, ki in enumerate(ks):
            ax = fig.add_subplot(nr, nc, i + 1)
            sub = self.grad_df[self.grad_df['k'] == ki].sort_values('Time')
            ax.plot(sub['Time'].values, sub['AI'].values,
                    lw=2, color='#d62728')
            ax.axhline(0, color='k', lw=.8, ls='--')
            ax.set_title(f'Z={Z_LEVELS[ki]}m', fontsize=FONT['title'])
            ax.set_ylabel('AI', fontsize=FONT['axis_label'])
            ax.set_xlabel('时间 (s)', fontsize=FONT['axis_label'])
            ax.tick_params(labelsize=FONT['tick'])
            ax.grid(True, alpha=.3, ls='--')
        fig.suptitle('不对称指数', fontsize=FONT['suptitle'], y=1.02)
        fig.tight_layout()
        return fig

    def plot_asym_heatmap(self):
        piv = self.grad_df.pivot_table(
            index='Z', columns='Time', values='AI')
        Za, Ta, AIa = piv.index.values, piv.columns.values, piv.values
        fig = Figure(figsize=(17, 9), dpi=100)
        ax = fig.add_subplot(111)
        vmax = np.nanpercentile(np.abs(AIa), 98) or 1
        im = ax.pcolormesh(Ta, Za, AIa, cmap='coolwarm',
                           vmin=-vmax, vmax=vmax, shading='auto')
        cb = fig.colorbar(im, ax=ax, pad=.02)
        cb.set_label('AI', fontsize=FONT['colorbar'])
        cb.ax.tick_params(labelsize=FONT['tick'])
        ax.set_xlabel('时间 (s)', fontsize=FONT['axis_label'])
        ax.set_ylabel('高度 Z (m)', fontsize=FONT['axis_label'])
        ax.set_title('不对称指数热力图', fontsize=FONT['title'])
        ax.tick_params(labelsize=FONT['tick'])
        fig.tight_layout()
        return fig

    def _yz_grid(self, T):
        ya, za, ta = [], [], []
        for ln in ANALYSIS_LAYERS:
            for k in range(len(Z_LEVELS)):
                ya.append(LAYER_DEVICES[ln]['y'])
                za.append(Z_LEVELS[k])
                ta.append(T[ln][k])
        yi, zi = np.linspace(-7, 7, 200), np.linspace(0, 55, 200)
        YI, ZI = np.meshgrid(yi, zi)
        TI = griddata((np.array(ya), np.array(za)), np.array(ta),
                       (YI, ZI), method='cubic')
        return YI, ZI, TI, ya, za, ta

    def plot_yz_contour(self, time):
        row = self.df.iloc[self.tidx(time)]
        T = self.temps(row)
        YI, ZI, TI, ya, za, ta = self._yz_grid(T)
        fig = Figure(figsize=(9, 15), dpi=100)
        ax = fig.add_subplot(111)
        levels = np.linspace(20, max(max(ta), 30), 40)
        cs = ax.contourf(YI, ZI, TI, levels=levels, cmap='hot', extend='max')
        cb = fig.colorbar(cs, ax=ax, pad=.02, shrink=.7)
        cb.set_label('温度 (°C)', fontsize=FONT['colorbar'])
        cb.ax.tick_params(labelsize=FONT['tick'])
        ax.scatter(ya, za, c='white', s=15, marker='+', alpha=.6)
        for ln in ANALYSIS_LAYERS:
            ax.axvline(LAYER_DEVICES[ln]['y'], color='white',
                       lw=.5, ls=':', alpha=.4)
        ax.set_xlabel('Y (m)', fontsize=FONT['axis_label'])
        ax.set_ylabel('Z (m)', fontsize=FONT['axis_label'])
        ax.set_title(f'Y-Z 温度场  t={row["Time"]:.0f}s',
                     fontsize=FONT['title'])
        ax.tick_params(labelsize=FONT['tick'])
        ax.set_ylim(0, 55)
        fig.tight_layout()
        return fig

    def plot_yz_compare(self, times):
        n = len(times)
        fig = Figure(figsize=(7 * n, 15), dpi=100)
        all_t = []
        for tgt in times:
            T = self.temps(self.df.iloc[self.tidx(tgt)])
            for ln in ANALYSIS_LAYERS:
                all_t.extend(T[ln].tolist())
        levels = np.linspace(20, max(max(all_t), 30), 30)
        axes, cs = [], None
        for i, tgt in enumerate(times):
            ax = fig.add_subplot(1, n, i + 1); axes.append(ax)
            row = self.df.iloc[self.tidx(tgt)]
            T = self.temps(row)
            YI, ZI, TI, ya, za, _ = self._yz_grid(T)
            cs = ax.contourf(YI, ZI, TI, levels=levels, cmap='hot',
                             extend='max')
            ax.scatter(ya, za, c='white', s=8, marker='+', alpha=.4)
            ax.set_title(f't={row["Time"]:.0f}s', fontsize=FONT['title'])
            ax.set_xlabel('Y (m)', fontsize=FONT['axis_label'])
            if i == 0:
                ax.set_ylabel('Z (m)', fontsize=FONT['axis_label'])
            ax.tick_params(labelsize=FONT['tick'])
            ax.set_ylim(0, 55)
        if cs:
            cb = fig.colorbar(cs, ax=axes, pad=.02, shrink=.6)
            cb.set_label('温度 (°C)', fontsize=FONT['colorbar'])
            cb.ax.tick_params(labelsize=FONT['tick'])
        fig.suptitle('Y-Z 温度对比',
                     fontsize=FONT['suptitle'], y=1.01)
        fig.tight_layout()
        return fig


# ================================================================
# COMPARISON PLOT FUNCTIONS
# ================================================================

def _cstyle(i):
    return (COND_COLORS[i % len(COND_COLORS)],
            COND_STYLES[i % len(COND_STYLES)],
            COND_MARKERS[i % len(COND_MARKERS)])


def plot_cmp_profile(engines, names, time, k, title_prefix=''):
    fig = Figure(figsize=(12, 8), dpi=100)
    ax = fig.add_subplot(111)
    for i, (name, eng) in enumerate(zip(names, engines)):
        row = eng.df.iloc[eng.tidx(time)]
        T = eng.temps(row)
        tv = [T[ln][k] for ln in ANALYSIS_LAYERS]
        c, ls, mk = _cstyle(i)
        ax.plot(ANALYSIS_Y, tv, color=c, ls=ls, marker=mk,
                lw=2.5, ms=9, mfc='white', mew=2, label=name)
    ax.set_xlabel('Y (m)', fontsize=FONT['axis_label'])
    ax.set_ylabel('温度 (°C)', fontsize=FONT['axis_label'])
    ax.set_title(f'{title_prefix}横向剖面对比  Z={Z_LEVELS[k]}m',
                 fontsize=FONT['title'])
    ax.legend(fontsize=FONT['legend'], loc='best', framealpha=.9)
    ax.tick_params(labelsize=FONT['tick'])
    ax.grid(True, alpha=.3, ls='--')
    fig.tight_layout()
    return fig


def plot_cmp_grad_ts(engines, names, k, title_prefix=''):
    fig = Figure(figsize=(15, 8), dpi=100)
    ax = fig.add_subplot(111)
    for i, (name, eng) in enumerate(zip(names, engines)):
        sub = eng.grad_df[eng.grad_df['k'] == k].sort_values('Time')
        c, ls, mk = _cstyle(i)
        ax.plot(sub['Time'].values, sub['G_total'].values,
                color=c, ls=ls, lw=2.5, label=name)
    ax.axhline(0, color='k', lw=.8)
    ax.set_xlabel('时间 (s)', fontsize=FONT['axis_label'])
    ax.set_ylabel('总梯度 L5→L6 (°C/m)',
                  fontsize=FONT['axis_label'])
    ax.set_title(f'{title_prefix}梯度时序对比  Z={Z_LEVELS[k]}m',
                 fontsize=FONT['title'])
    ax.legend(fontsize=FONT['legend'], loc='best', framealpha=.9)
    ax.tick_params(labelsize=FONT['tick'])
    ax.grid(True, alpha=.3, ls='--')
    fig.tight_layout()
    return fig


def plot_cmp_asymmetry(engines, names, k, title_prefix=''):
    fig = Figure(figsize=(15, 8), dpi=100)
    ax = fig.add_subplot(111)
    for i, (name, eng) in enumerate(zip(names, engines)):
        sub = eng.grad_df[eng.grad_df['k'] == k].sort_values('Time')
        c, ls, mk = _cstyle(i)
        ax.plot(sub['Time'].values, sub['AI'].values,
                color=c, ls=ls, lw=2.5, label=name)
    ax.axhline(0, color='k', lw=.8, ls='--')
    ax.set_xlabel('时间 (s)', fontsize=FONT['axis_label'])
    ax.set_ylabel('不对称指数', fontsize=FONT['axis_label'])
    ax.set_title(f'{title_prefix}不对称指数对比  Z={Z_LEVELS[k]}m',
                 fontsize=FONT['title'])
    ax.legend(fontsize=FONT['legend'], loc='best', framealpha=.9)
    ax.tick_params(labelsize=FONT['tick'])
    ax.grid(True, alpha=.3, ls='--')
    fig.tight_layout()
    return fig


def plot_cmp_peak_temp(engines, names, layer='L2', title_prefix=''):
    fig = Figure(figsize=(11, 11), dpi=100)
    ax = fig.add_subplot(111)
    for i, (name, eng) in enumerate(zip(names, engines)):
        peaks = eng.peak_temp(layer)
        c, ls, mk = _cstyle(i)
        ax.plot(peaks, Z_LEVELS, color=c, ls=ls, marker=mk,
                lw=2.5, ms=9, mfc='white', mew=2, label=name)
    ax.set_xlabel('峰值温度 (°C)',
                  fontsize=FONT['axis_label'])
    ax.set_ylabel('高度 Z (m)', fontsize=FONT['axis_label'])
    ax.set_title(f'{title_prefix}峰值温度对比\n'
                 f'({LAYER_DEVICES[layer]["label"]})',
                 fontsize=FONT['title'])
    ax.legend(fontsize=FONT['legend'], loc='best', framealpha=.9)
    ax.tick_params(labelsize=FONT['tick'])
    ax.grid(True, alpha=.3, ls='--')
    fig.tight_layout()
    return fig


def plot_cmp_reach_time(engines, names, threshold=300, layer='L2',
                        title_prefix=''):
    fig = Figure(figsize=(11, 11), dpi=100)
    ax = fig.add_subplot(111)
    za = np.array(Z_LEVELS, dtype=float)
    for i, (name, eng) in enumerate(zip(names, engines)):
        rt = eng.reach_time(threshold, layer)
        valid = ~np.isnan(rt); c, ls, mk = _cstyle(i)
        ax.plot(rt[valid], za[valid], color=c, ls=ls, marker=mk,
                lw=2.5, ms=9, mfc='white', mew=2, label=name)
        if np.any(~valid):
            ax.scatter(rt[~valid], za[~valid], color=c, marker='x',
                       s=80, lw=2.5)
    ax.set_xlabel(f'达到 {threshold}°C 所用的时间 (s)',
                  fontsize=FONT['axis_label'])
    ax.set_ylabel('探测器高度 Z (m)', fontsize=FONT['axis_label'])
    ax.set_title(f'{title_prefix}探测器升温所需时间\n'
                 f'({LAYER_DEVICES[layer]["label"]})',
                 fontsize=FONT['title'])
    ax.legend(fontsize=FONT['legend'], loc='best', framealpha=.9)
    ax.tick_params(labelsize=FONT['tick'])
    ax.grid(True, alpha=.3, ls='--')
    fig.tight_layout()
    return fig


def table_cmp_reach_time(engines, names, threshold=300, layer='L2'):
    rows = []
    for ki, z in enumerate(Z_LEVELS):
        r = {'Z(m)': z}
        for name, eng in zip(names, engines):
            rt = eng.reach_time(threshold, layer)
            r[name] = f'{rt[ki]:.1f}' if not np.isnan(rt[ki]) else '-'
        rows.append(r)
    return pd.DataFrame(rows)


def plot_cmp_outer_reach(engines, names, threshold=300,
                         title_prefix=''):
    outer = ['L5', 'L6']
    fig = Figure(figsize=(16, 9), dpi=100)
    za = np.array(Z_LEVELS, dtype=float)
    for li, layer in enumerate(outer):
        ax = fig.add_subplot(1, 2, li + 1)
        for i, (name, eng) in enumerate(zip(names, engines)):
            rt = eng.reach_time(threshold, layer)
            valid = ~np.isnan(rt); c, ls, mk = _cstyle(i)
            ax.plot(rt[valid], za[valid], color=c, ls=ls, marker=mk,
                    lw=2.5, ms=8, mfc='white', mew=2, label=name)
            if np.any(~valid):
                ax.scatter(rt[~valid], za[~valid], color=c,
                           marker='x', s=80, lw=2.5)
        ax.set_xlabel(f'达到 {threshold}°C 的时间 (s)',
                      fontsize=FONT['axis_label'])
        ax.set_ylabel('高度 Z (m)', fontsize=FONT['axis_label'])
        ax.set_title(f'{layer}: {LAYER_DEVICES[layer]["label"]}',
                     fontsize=FONT['title'])
        ax.legend(fontsize=FONT['legend'], loc='best', framealpha=.9)
        ax.tick_params(labelsize=FONT['tick'])
        ax.grid(True, alpha=.3, ls='--')
    fig.suptitle(f'{title_prefix}外层首次到达时间 ({threshold}°C)',
                 fontsize=FONT['suptitle'], y=1.02)
    fig.tight_layout()
    return fig


def table_cmp_outer_reach(engines, names, threshold=300):
    rows = []
    for ki, z in enumerate(Z_LEVELS):
        for layer in ['L5', 'L6']:
            r = {'Z(m)': z, 'Layer': layer}
            for name, eng in zip(names, engines):
                rt = eng.reach_time(threshold, layer)
                r[name] = f'{rt[ki]:.1f}' if not np.isnan(rt[ki]) else '-'
            rows.append(r)
    return pd.DataFrame(rows)


# ================================================================
# L6 FIRST-REACH HEIGHT BAR CHART
# ================================================================

def _compute_l6_first_data(engines, names, threshold=300):
    layer = 'L6'
    za = np.array(Z_LEVELS, dtype=float)
    results = []

    for name, eng in zip(names, engines):
        rt = eng.reach_time(threshold, layer)
        valid = ~np.isnan(rt)

        entry = {
            'name': name, 'first_h': np.nan, 'first_t': np.nan,
            'std_h': np.nan, 'dt_spread': np.nan,
            'n_top': 0, 'n_valid': 0,
            'top_heights': [], 'top_times': [],
        }

        if not np.any(valid):
            results.append(entry)
            continue

        valid_times   = rt[valid]
        valid_heights = za[valid]
        order = np.argsort(valid_times)
        sorted_h = valid_heights[order]
        sorted_t = valid_times[order]
        n_top = min(10, len(order))

        entry['first_h']     = float(sorted_h[0])
        entry['first_t']     = float(sorted_t[0])
        entry['n_valid']     = int(np.sum(valid))
        entry['n_top']       = n_top
        entry['top_heights'] = sorted_h[:n_top].tolist()
        entry['top_times']   = sorted_t[:n_top].tolist()

        if n_top > 1:
            entry['std_h']      = float(np.std(sorted_h[:n_top]))
            entry['dt_spread']  = float(sorted_t[n_top - 1] - sorted_t[0])
        else:
            entry['std_h']     = 0.0
            entry['dt_spread'] = 0.0

        results.append(entry)

    return results


def plot_cmp_outer_first_height(engines, names, threshold=300,
                                title_prefix=''):
    data = _compute_l6_first_data(engines, names, threshold)

    fig = Figure(figsize=(max(9, len(names) * 2 + 2), 9), dpi=100)
    ax = fig.add_subplot(111)
    x = np.arange(len(names))

    heights = [d['first_h'] for d in data]
    stds    = [d['std_h']   for d in data]
    colors  = [COND_COLORS[i % len(COND_COLORS)]
               for i in range(len(names))]

    err_lo, err_hi = [], []
    for s, h in zip(stds, heights):
        if np.isnan(h):
            err_lo.append(0); err_hi.append(0)
        else:
            err_lo.append(min(s, h))
            err_hi.append(min(s, 53 - h))

    ax.bar(x, heights,
           yerr=[err_lo, err_hi],
           color=colors, edgecolor='black',
           linewidth=1, width=0.55, alpha=0.85,
           error_kw=dict(ecolor='#333333', lw=2.5,
                         capsize=10, capthick=2))

    for i, d in enumerate(data):
        h = d['first_h']
        if np.isnan(h):
            ax.text(i, 2, 'N/A', ha='center', va='bottom',
                    fontsize=FONT['bar_label'], color='gray')
            continue
        s  = d['std_h']
        dt = d['dt_spread']
        eh = err_hi[i]
        ax.text(i, h + eh + 2,
                f'{h:.0f}m\n'
                f'\u03c3={s:.1f}m\n'
                f'\u0394t={dt:.0f}s',
                ha='center', va='bottom',
                fontsize=FONT['bar_label'], fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha='right',
                       fontsize=FONT['tick'])
    ax.set_ylabel('高度 Z (m)', fontsize=FONT['axis_label'])
    ax.set_xlabel('工况', fontsize=FONT['axis_label'])
    ax.set_title(
        f'{title_prefix}L6层(Y=6m)首个到达{threshold}°C的探测器高度\n'
        f'误差条 = 前10个探测器(按到达时间排序)高度的σ',
        fontsize=FONT['title'])
    ax.set_ylim(0, 66)
    ax.tick_params(labelsize=FONT['tick'])
    ax.grid(True, alpha=.3, ls='--', axis='y')

    lines = ['前10个探测器（按到达时间排序）：']
    for d in data:
        if d['top_heights']:
            hs = ', '.join(f'{h:.0f}' for h in d['top_heights'])
            ts = ', '.join(f'{t:.0f}' for t in d['top_times'])
            lines.append(f"  {d['name']}:")
            lines.append(f"    H(m): [{hs}]")
            lines.append(f"    T(s): [{ts}]")
        else:
            lines.append(f"  {d['name']}: 无有效探测器")
    info_text = '\n'.join(lines)

    fig.subplots_adjust(bottom=0.30)
    fig.text(0.02, 0.01, info_text,
             fontsize=FONT['info_text'],
             fontfamily='monospace', va='bottom',
             bbox=dict(boxstyle='round', facecolor='#f0f0f0',
                       alpha=0.8))
    return fig


def table_cmp_outer_first_height(engines, names, threshold=300):
    data = _compute_l6_first_data(engines, names, threshold)
    rows = []
    for d in data:
        r = {
            'Condition': d['name'],
            '1st Height (m)': (f'{d["first_h"]:.0f}'
                               if not np.isnan(d['first_h']) else '-'),
            '1st Time (s)': (f'{d["first_t"]:.1f}'
                             if not np.isnan(d['first_t']) else '-'),
            '\u03c3 Height (m)': (f'{d["std_h"]:.2f}'
                                 if not np.isnan(d['std_h']) else '-'),
            '\u0394t (s)': (f'{d["dt_spread"]:.1f}'
                           if not np.isnan(d['dt_spread']) else '-'),
            'N valid': d['n_valid'],
        }
        if d['top_heights']:
            r['Top10 H(m)'] = ', '.join(
                f'{h:.0f}' for h in d['top_heights'])
        if d['top_times']:
            r['Top10 T(s)'] = ', '.join(
                f'{t:.0f}' for t in d['top_times'])
        rows.append(r)
    return pd.DataFrame(rows)


# ================================================================
# GUI
# ================================================================

class App:
    def __init__(self, root):
        self.root = root
        self.root.title('横向温度梯度分析器 v3.4')
        self.root.geometry('1500x920')
        self.root.minsize(1200, 700)
        self.conditions = {}
        self._orig_map = {}
        self._cond_cnt = 0
        self.canvas_widget = None
        self.nav_toolbar = None
        self.param_widgets = {}
        self._setup_font()
        self._build_menu()
        self._build_ui()
        self._set_status('就绪')
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _setup_font(self):
        from matplotlib.font_manager import FontManager
        avail = {f.name for f in FontManager().ttflist}
        for f in ['SimHei', 'Microsoft YaHei', 'KaiTi',
                   'FangSong', 'WenQuanYi Micro Hei', 'STHeiti',
                   'Noto Sans CJK SC']:
            if f in avail:
                matplotlib.rcParams['font.sans-serif'] = [f]
                matplotlib.rcParams['axes.unicode_minus'] = False
                return

    def _build_menu(self):
        mb = tk.Menu(self.root)
        fm = tk.Menu(mb, tearoff=0)
        fm.add_command(label='添加  Ctrl+O',
                       command=self._add_condition)
        fm.add_separator()
        fm.add_command(label='保存  Ctrl+S', command=self._save_fig)
        fm.add_command(label='导出 CSV', command=self._export_csv)
        fm.add_separator()
        fm.add_command(label='退出', command=self._on_close)
        mb.add_cascade(label='文件', menu=fm)
        am = tk.Menu(mb, tearoff=0)
        for key, label in SINGLE_TYPES:
            am.add_command(label=label,
                           command=lambda k=key: self._sel_type(k))
        am.add_separator()
        for key, label in CMP_TYPES:
            am.add_command(label=f'[对比] {label}',
                           command=lambda k=key: self._sel_type(k))
        mb.add_cascade(label='分析', menu=am)
        self.root.config(menu=mb)
        self.root.bind('<Control-o>',
                       lambda e: self._add_condition())
        self.root.bind('<Control-s>', lambda e: self._save_fig())
        self.root.bind('<F5>', lambda e: self._run())

    def _build_ui(self):
        pw = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        pw.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.sidebar = ttk.Frame(pw, width=320)
        pw.add(self.sidebar, weight=0)
        self.right = ttk.Frame(pw)
        pw.add(self.right, weight=1)
        self._build_sidebar()
        self._build_right()

    def _build_sidebar(self):
        cf = ttk.LabelFrame(self.sidebar,
                            text='工况列表 (L1已排除)', padding=6)
        cf.pack(fill='x', padx=5, pady=5)
        self.cond_list = tk.Listbox(cf, height=7,
                                    selectmode=tk.EXTENDED,
                                    font=('Consolas', 9))
        self.cond_list.pack(fill='x', pady=(0, 4))
        br = ttk.Frame(cf); br.pack(fill='x')
        ttk.Button(br, text='+ 添加', width=8,
                   command=self._add_condition).pack(side='left', padx=1)
        ttk.Button(br, text='- 移除', width=8,
                   command=self._remove_condition).pack(side='left', padx=1)
        ttk.Button(br, text='重命名', width=8,
                   command=self._rename_condition).pack(side='left', padx=1)
        self.cond_info = ttk.Label(cf, text='尚未加载文件',
                                   foreground='gray', wraplength=290)
        self.cond_info.pack(fill='x', pady=(4, 0))

        af = ttk.LabelFrame(self.sidebar, text='分析类型', padding=6)
        af.pack(fill='x', padx=5, pady=5)
        self.aval = tk.StringVar(value='profile')
        ttk.Label(af, text='单工况分析',
                  font=('', 9, 'bold')).pack(anchor='w')
        for key, label in SINGLE_TYPES:
            ttk.Radiobutton(af, text=label, variable=self.aval,
                            value=key,
                            command=self._update_params).pack(anchor='w')
        ttk.Separator(af).pack(fill='x', pady=4)
        ttk.Label(af, text='多工况对比',
                  font=('', 9, 'bold')).pack(anchor='w')
        for key, label in CMP_TYPES:
            ttk.Radiobutton(af, text=label, variable=self.aval,
                            value=key,
                            command=self._update_params).pack(anchor='w')

        self.pf = ttk.LabelFrame(self.sidebar, text='参数设置', padding=6)
        self.pf.pack(fill='x', padx=5, pady=5)
        self._update_params()

        bf = ttk.Frame(self.sidebar, padding=6)
        bf.pack(fill='x', padx=5, pady=5)
        ttk.Button(bf, text='\u25b6 运行 (F5)',
                   command=self._run).pack(fill='x', pady=2)
        ttk.Button(bf, text='保存图片',
                   command=self._save_fig).pack(fill='x', pady=2)
        ttk.Button(bf, text='导出 CSV',
                   command=self._export_csv).pack(fill='x', pady=2)
        ttk.Button(bf, text='清除',
                   command=self._clear).pack(fill='x', pady=2)

    def _build_right(self):
        self.plot_frame = ttk.Frame(self.right)
        self.plot_frame.pack(fill='both', expand=True)
        lf = ttk.LabelFrame(self.right, text='日志', height=120)
        lf.pack(fill='x', padx=5, pady=(0, 5))
        lf.pack_propagate(False)
        self.log = tk.Text(lf, height=4, font=('Consolas', 9),
                           wrap='word')
        sb = ttk.Scrollbar(lf, command=self.log.yview)
        self.log.configure(yscrollcommand=sb.set)
        sb.pack(side='right', fill='y')
        self.log.pack(fill='both', expand=True)
        sf = ttk.Frame(self.right)
        sf.pack(fill='x', padx=5, pady=(0, 5))
        self.status = tk.StringVar(value='就绪')
        ttk.Label(sf, textvariable=self.status).pack(side='left')
        self.progress = ttk.Progressbar(sf, length=240,
                                        mode='determinate')
        self.progress.pack(side='right', padx=5)

    def _update_params(self):
        for w in self.pf.winfo_children():
            w.destroy()
        self.param_widgets.clear()
        needs = PARAM_NEEDS.get(self.aval.get(), [])
        if 'height' in needs:
            ttk.Label(self.pf, text='高度 Z:').pack(anchor='w')
            hv = tk.StringVar(value=f'Z={Z_LEVELS[9]}m')
            ttk.Combobox(self.pf, textvariable=hv,
                         values=[f'Z={z}m' for z in Z_LEVELS],
                         state='readonly').pack(fill='x', pady=(0, 4))
            self.param_widgets['height'] = hv
        if 'time' in needs:
            ttk.Label(self.pf, text='时间 t (s):').pack(anchor='w')
            tv = tk.StringVar(value='400')
            ttk.Entry(self.pf, textvariable=tv).pack(fill='x', pady=(0, 4))
            self.param_widgets['time'] = tv
        if 'count' in needs:
            ttk.Label(self.pf, text='采样数量:').pack(anchor='w')
            cv = tk.StringVar(value='8')
            ttk.Spinbox(self.pf, from_=2, to=20,
                        textvariable=cv, width=10).pack(anchor='w',
                                                        pady=(0, 4))
            self.param_widgets['count'] = cv
        if 'layer' in needs:
            ttk.Label(self.pf, text='层:').pack(anchor='w')
            ly = tk.StringVar(value='L2')
            ttk.Combobox(self.pf, textvariable=ly,
                         values=ANALYSIS_LAYERS,
                         state='readonly').pack(fill='x', pady=(0, 4))
            self.param_widgets['layer'] = ly
        if 'threshold' in needs:
            ttk.Label(self.pf,
                      text='阈值 (°C):').pack(anchor='w')
            th = tk.StringVar(value='300')
            ttk.Entry(self.pf, textvariable=th).pack(fill='x', pady=(0, 4))
            self.param_widgets['threshold'] = th
        if not needs:
            ttk.Label(self.pf, text='（无参数）',
                      foreground='gray').pack(anchor='w')
        hint = HINTS.get(self.aval.get(), '')
        if hint:
            ttk.Label(self.pf, text=hint, foreground='gray',
                      wraplength=290, font=('', 9)).pack(
                          anchor='w', pady=(8, 0))

    def _get_k(self):
        s = self.param_widgets.get(
            'height', tk.StringVar(value=f'Z={Z_LEVELS[9]}m'))
        return Z_LEVELS.index(
            int(s.get().replace('Z=', '').replace('m', '')))

    def _get_time(self):
        return float(self.param_widgets.get(
            'time', tk.StringVar(value='400')).get())

    def _get_count(self):
        return int(self.param_widgets.get(
            'count', tk.StringVar(value='8')).get())

    def _get_layer(self):
        return self.param_widgets.get(
            'layer', tk.StringVar(value='L2')).get()

    def _get_threshold(self):
        return float(self.param_widgets.get(
            'threshold', tk.StringVar(value='300')).get())

    def _get_cmp_title_prefix(self):
        """
        读取所有已加载工况的上传时原始文件名（不含扩展名），
        以下划线分割后：
          - 若所有工况的第二段(索引1)数值相同 → 加前缀 "[值]+高度"
          - 若所有工况的第三段(索引2)数值相同 → 加前缀 "[值]+距离"
        两个条件独立判断，可同时成立。
        """
        if not self._orig_map:
            return ''

        seg1_list = []
        seg2_list = []

        for orig_name in self._orig_map.values():
            segs = _parse_segments(orig_name)
            if segs is not None and all(v == 0 for v in segs[:3]):
                continue

            parts = orig_name.split('_')
            # 第二段
            if len(parts) >= 2:
                try:
                    seg1_list.append(float(parts[1]))
                except ValueError:
                    seg1_list.append(None)
            else:
                seg1_list.append(None)
            # 第三段
            if len(parts) >= 3:
                try:
                    seg2_list.append(float(parts[2]))
                except ValueError:
                    seg2_list.append(None)
            else:
                seg2_list.append(None)

        prefix_parts = []

        # 判断第二段是否全部相同（排除None，且要求所有工况均能解析）
        valid1 = [v for v in seg1_list if v is not None]
        if len(valid1) == len(seg1_list) and len(valid1) > 1:
            if all(v == valid1[0] for v in valid1):
                prefix_parts.append(f'无人机高度{_fmt_val(valid1[0])}m')

        # 判断第三段是否全部相同（排除None，且要求所有工况均能解析）
        valid2 = [v for v in seg2_list if v is not None]
        if len(valid2) == len(seg2_list) and len(valid2) > 1:
            if all(v == valid2[0] for v in valid2):
                prefix_parts.append(f'无人机距离{_fmt_val(valid2[0])}m')

        return '  '.join(prefix_parts) + '  ' if prefix_parts else ''

    def _do_auto_rename(self):
        if not self.conditions:
            return
        self.conditions, self._orig_map = auto_rename_conditions(
            self.conditions, self._orig_map)
        self.cond_list.delete(0, tk.END)
        for dn in self.conditions:
            self.cond_list.insert(tk.END, dn)
        self._update_cond_info()
        self._plog(f'已重命名: {list(self.conditions.keys())}')

    def _add_condition(self):
        paths = filedialog.askopenfilenames(
            title='选择CSV文件',
            filetypes=[('CSV', '*.csv'), ('所有文件', '*.*')])
        if not paths:
            return

        def worker(paths):
            loaded = []
            for path in paths:
                self._cond_cnt += 1
                stem = os.path.splitext(
                    os.path.basename(path))[0]
                tmp = re.sub(
                    r'[^a-zA-Z0-9_\u4e00-\u9fff-]', '_',
                    stem)[:40] or f'Cond_{self._cond_cnt}'
                orig, c = tmp, 2
                while tmp in self.conditions:
                    tmp = f'{orig}_{c}'; c += 1
                self.root.after(0, lambda n=tmp: (
                    self._plog(f'正在加载 {n}...'),
                    self._set_status(f'正在加载 {n}...')))
                try:
                    eng = Engine()
                    eng.load(path)
                    eng.compute_gradients(
                        cb=lambda cur, tot: self.root.after(
                            0, lambda: self._set_progress(
                                cur / tot * 100)))
                    self.conditions[tmp] = eng
                    self._orig_map[tmp] = stem
                    loaded.append(tmp)
                    self.root.after(0, lambda n=tmp, p=path: (
                        self.cond_list.insert(tk.END, n),
                        self._plog(f'成功: {n}'),
                        self._set_status(
                            f'已加载 {len(self.conditions)} 个工况'),
                        self._update_cond_info()))
                except Exception as e:
                    self.root.after(0, lambda n=tmp, e=str(e): (
                        messagebox.showerror('错误', f'{n}: {e}'),
                        self._plog(f'错误 {n}: {e}')))
            self.root.after(0, lambda: self._set_progress(100))
            if loaded:
                self.root.after(100, self._do_auto_rename)

        threading.Thread(target=worker, args=(paths,),
                         daemon=True).start()

    def _remove_condition(self):
        sel = self.cond_list.curselection()
        if not sel:
            return
        names = [self.cond_list.get(i) for i in sel]
        for n in names:
            self.conditions.pop(n, None)
            self._orig_map.pop(n, None)
        for i in reversed(sel):
            self.cond_list.delete(i)
        self._plog(f'已移除: {", ".join(names)}')
        self._update_cond_info()

    def _rename_condition(self):
        sel = self.cond_list.curselection()
        if len(sel) != 1:
            messagebox.showinfo('提示', '请选中一个工况')
            return
        old = self.cond_list.get(sel[0])
        new = simpledialog.askstring('重命名', f'将 "{old}" 重命名为:',
                                     initialvalue=old)
        if new and new != old:
            if new in self.conditions:
                messagebox.showwarning('已存在', f'"{new}" 已存在')
                return
            self.conditions[new] = self.conditions.pop(old)
            self._orig_map[new] = self._orig_map.pop(old, new)
            self.cond_list.delete(sel[0])
            self.cond_list.insert(sel[0], new)
            self._update_cond_info()

    def _update_cond_info(self):
        n = len(self.conditions)
        self.cond_info.config(
            text=f'{n} 个工况' if n else '尚未加载文件',
            foreground='blue' if n else 'gray')

    def _sel_cond(self):
        sel = self.cond_list.curselection()
        name = (self.cond_list.get(sel[0]) if sel
                else next(iter(self.conditions), None))
        return (name, self.conditions[name]) if name else (None, None)

    def _all_conds(self):
        names = list(self.conditions.keys())
        return names, [self.conditions[n] for n in names]

    def _display(self, fig):
        if self.canvas_widget:
            plt.close(self.canvas_widget.figure)
            self.canvas_widget.get_tk_widget().destroy()
        if self.nav_toolbar:
            self.nav_toolbar.destroy()
        self.canvas_widget = FigureCanvasTkAgg(
            fig, master=self.plot_frame)
        self.canvas_widget.draw()
        self.nav_toolbar = NavigationToolbar2Tk(
            self.canvas_widget, self.plot_frame)
        self.nav_toolbar.update()
        self.canvas_widget.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _clear(self):
        if self.canvas_widget:
            plt.close(self.canvas_widget.figure)
            self.canvas_widget.get_tk_widget().destroy()
            self.canvas_widget = None
        if self.nav_toolbar:
            self.nav_toolbar.destroy()
            self.nav_toolbar = None

    def _plog(self, m):
        self.log.insert(tk.END, m + '\n')
        self.log.see(tk.END)

    def _set_status(self, m):
        self.status.set(m)

    def _set_progress(self, v):
        self.progress['value'] = v

    def _sel_type(self, k):
        self.aval.set(k)
        self._update_params()

    def _run(self):
        atype = self.aval.get()
        is_cmp = atype.startswith('cmp_')
        if not self.conditions:
            messagebox.showwarning('警告', '请先添加文件')
            return
        self._plog(f'\n运行: {dict(ALL_TYPES).get(atype, atype)}')
        self._set_status('分析中...')
        try:
            fig = None
            if is_cmp:
                if len(self.conditions) < 2:
                    messagebox.showinfo('提示', '需要 >= 2 个工况')
                    return
                fig = self._run_cmp(atype)
            else:
                n, eng = self._sel_cond()
                if eng is None:
                    messagebox.showwarning('警告', '请选中一个工况')
                    return
                fig = self._run_single(atype, n, eng)
            if fig:
                self._display(fig)
                self._plog('完成')
                self._set_status('已完成')
        except Exception as e:
            messagebox.showerror('错误', str(e))
            self._plog(f'错误: {e}')
            self._set_status('出错')

    def _run_single(self, atype, name, eng):
        self._plog(f'  工况: {name}')
        if atype == 'profile':
            k = self._get_k(); t = self._get_time()
            self._plog(f'  Z={Z_LEVELS[k]}m t~{eng.tval(t):.0f}s')
            return eng.plot_profile(t, k)
        elif atype == 'evolution':
            k = self._get_k(); n = self._get_count()
            return eng.plot_evolution(k, eng.sample_times(n))
        elif atype == 'grad_ts':
            return eng.plot_grad_ts(self._get_k())
        elif atype == 'grad_heat':
            return eng.plot_grad_heatmap()
        elif atype == 'asymmetry':
            return eng.plot_asymmetry()
        elif atype == 'asym_heat':
            return eng.plot_asym_heatmap()
        elif atype == 'yz_contour':
            if not HAS_SCIPY:
                raise ImportError('需要安装 scipy')
            return eng.plot_yz_contour(self._get_time())
        elif atype == 'yz_compare':
            if not HAS_SCIPY:
                raise ImportError('需要安装 scipy')
            return eng.plot_yz_compare(
                eng.sample_times(self._get_count()))
        return None

    def _run_cmp(self, atype):
        names, engines = self._all_conds()
        self._plog(f'  工况: {", ".join(names)}')

        title_prefix = self._get_cmp_title_prefix()

        if atype == 'cmp_profile':
            return plot_cmp_profile(engines, names,
                                    self._get_time(), self._get_k(),
                                    title_prefix=title_prefix)
        elif atype == 'cmp_grad_ts':
            return plot_cmp_grad_ts(engines, names, self._get_k(),
                                    title_prefix=title_prefix)
        elif atype == 'cmp_asymmetry':
            return plot_cmp_asymmetry(engines, names, self._get_k(),
                                      title_prefix=title_prefix)
        elif atype == 'cmp_peak_temp':
            return plot_cmp_peak_temp(engines, names,
                                      self._get_layer(),
                                      title_prefix=title_prefix)
        elif atype == 'cmp_reach_time':
            layer = self._get_layer(); th = self._get_threshold()
            tbl = table_cmp_reach_time(engines, names, th, layer)
            self._plog('\n' + tbl.to_string(index=False))
            return plot_cmp_reach_time(engines, names, th, layer,
                                       title_prefix=title_prefix)
        elif atype == 'cmp_outer_reach':
            th = self._get_threshold()
            tbl = table_cmp_outer_reach(engines, names, th)
            self._plog('\n' + tbl.to_string(index=False))
            return plot_cmp_outer_reach(engines, names, th,
                                        title_prefix=title_prefix)
        elif atype == 'cmp_outer_first_ht':
            th = self._get_threshold()
            tbl = table_cmp_outer_first_height(engines, names, th)
            self._plog('\n' + tbl.to_string(index=False))
            return plot_cmp_outer_first_height(engines, names, th,
                                               title_prefix=title_prefix)
        return None

    def _save_fig(self):
        if not self.canvas_widget:
            messagebox.showinfo('提示', '没有图片')
            return
        path = filedialog.asksaveasfilename(
            defaultextension='.png',
            filetypes=[('PNG', '*.png'), ('PDF', '*.pdf'),
                       ('SVG', '*.svg')])
        if path:
            self.canvas_widget.figure.savefig(
                path, dpi=200, bbox_inches='tight')
            self._plog(f'已保存: {path}')

    def _export_csv(self):
        n, eng = self._sel_cond()
        if not eng or eng.grad_df is None:
            messagebox.showwarning('警告', '请选中一个工况')
            return
        path = filedialog.asksaveasfilename(
            defaultextension='.csv',
            initialfile=f'{n}_grad.csv',
            filetypes=[('CSV', '*.csv')])
        if path:
            eng.grad_df.to_csv(path, index=False, float_format='%.4f')
            self._plog(f'已导出: {path}')

    def _on_close(self):
        plt.close('all')
        self.root.destroy()


if __name__ == '__main__':
    root = tk.Tk()
    App(root)
    root.mainloop()
