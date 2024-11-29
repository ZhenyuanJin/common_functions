# region 标准库导入
import os
import sys
import json
import pickle
import random
import shutil
import time
import warnings
from math import ceil
from multiprocessing import Process
from pathlib import Path
from typing import Union, Sequence, Callable, Optional


# 数学和科学计算库
import numpy as np
import scipy
import scipy.stats as st
from scipy.stats import gaussian_kde, zscore
from scipy.fft import fft, fftfreq
from scipy.integrate import quad
from scipy.sparse import coo_matrix, csr_matrix
import scipy.sparse as sps
from sklearn.decomposition import PCA
import jax
import jax.numpy as jnp


# 数据处理和可视化库
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, ScalarFormatter
from matplotlib.colors import BoundaryNorm, Normalize
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable


# 神经网络和脑模型库
import brainpy as bp
import brainpy.math as bm
# from brainpy._src.dyn.base import SynDyn
# from brainpy._src.mixin import AlignPost, ReturnInfo
from brainpy._src.integrators.ode.generic import odeint
from brainpy._src.integrators.joint_eq import JointEq
from brainpy._src.initialize import parameter
# from brainpy._src.context import share
from brainpy.types import ArrayType

# 脑区处理库
# import networkx as nx
# import pointpats
# import shapely.geometry


# 自定义库
import common_functions as cf
# endregion


# region 定义默认参数
E_COLOR = cf.RED
I_COLOR = cf.BLUE
BP_VERSION = '2.5.0'
if bp.__version__ != BP_VERSION:
    cf.print_sep()
    cf.print_title('please check the brainpy version', char='*')
    print('current brainpy version: {}'.format(bp.__version__))
    print('default brainpy version: {}'.format(BP_VERSION))
    cf.print_sep()
# endregion


# region brainpy使用说明
def brainpy_data_structure():
    return '所有数据按照(T, N)的形式存储，其中T表示时间点的数量，N表示神经元的数量。'


def brainpy_unit(physical_quantity):
    '''
    Note that nA * MOhm = mV, thus consistent with for example \tau * dV/dt = - ( V - V_rest ) + R * I
    '''
    if physical_quantity == 'V':
        return 'mV (10^-3 V)'
    # 这两个可能不对
    # if physical_quantity == 'I':
    #     return 'nA (10^-9 A)'
    # if physical_quantity == 'R':
    #     return 'MOhm (10^6 Ohm)'
    if physical_quantity == 'tau':
        return 'ms (10^-3 s)'
# endregion


# region 获得常用神经元参数
def get_LifRef_param(paper):
    if paper == 'Joglekar_2018_Neuron':
        E_params = {'V_th': -50.0, 'V_reset': -60.0, 'V_rest':-70.0, 'tau_ref': 2.0, 'R': 50.0, 'tau': 20.0, 'V_initializer': bp.init.Uniform(-70.0, -50.0)} # Initializer is obtained from https://brainpy-examples.readthedocs.io/en/latest/large_scale_modeling/Joglekar_2018_InterAreal_Balanced_Amplification_figure5.html
        I_params = {'V_th': -50.0, 'V_reset': -60.0, 'V_rest':-70.0, 'tau_ref': 2.0, 'R': 50.0, 'tau': 10.0, 'V_initializer': bp.init.Uniform(-70.0, -50.0)}
    if paper == 'Liang_2020_Frontiers':
        E_params = {'V_th': -50.0, 'V_reset': -60.0, 'V_rest':-70.0, 'tau_ref': 2.0, 'R': 50.0, 'tau': 20.0, 'V_initializer': bp.init.Uniform(-70.0, -50.0)}
        I_params = {'V_th': -50.0, 'V_reset': -60.0, 'V_rest':-70.0, 'tau_ref': 1.0, 'R': 50.0, 'tau': 10.0, 'V_initializer': bp.init.Uniform(-70.0, -50.0)}
    if paper == 'Wang_2002_Neuron':
        E_params = {'V_th': -50.0, 'V_reset': -55.0, 'V_rest':-70.0, 'tau_ref': 2.0, 'R': 0.04, 'tau': 20.0, 'V_initializer': bp.init.OneInit(-70.)}
        I_params = {'V_th': -50.0, 'V_reset': -55.0, 'V_rest':-70.0, 'tau_ref': 1.0, 'R': 0.05, 'tau': 10.0, 'V_initializer': bp.init.OneInit(-70.)}
    return E_params, I_params

def get_synapse_and_params(mode):
    if mode == 'Liang_2020_Frontiers_Fast':
        synapse = {'E2E': NormalizedDualExponCUBA, 'I2E': NormalizedDualExponCUBA, 'E2I': NormalizedDualExponCUBA, 'I2I': NormalizedDualExponCUBA}
        synapse_params = {'E2E': {'tau_rise': 0.5, 'tau_decay': 2.0, 'delay': 0.0, 'out_label': 'E'}, 'E2I': {'tau_rise': 0.5, 'tau_decay': 2.0, 'delay': 0.0, 'out_label': 'E'}, 'I2E': {'tau_rise': 0.5, 'tau_decay': 1.2, 'delay': 0.0, 'out_label': 'I'}, 'I2I': {'tau_rise': 0.5, 'tau_decay': 1.2, 'delay': 0.0, 'out_label': 'I'}}
    if mode == 'Liang_2020_Frontiers_Critical':
        synapse = {'E2E': NormalizedDualExponCUBA, 'I2E': NormalizedDualExponCUBA, 'E2I': NormalizedDualExponCUBA, 'I2I': NormalizedDualExponCUBA}
        synapse_params = {'E2E': {'tau_rise': 0.5, 'tau_decay': 2.0, 'delay': 0.0, 'out_label': 'E'}, 'E2I': {'tau_rise': 0.5, 'tau_decay': 2.0, 'delay': 0.0, 'out_label': 'E'}, 'I2E': {'tau_rise': 0.5, 'tau_decay': 3.0, 'delay': 0.0, 'out_label': 'I'}, 'I2I': {'tau_rise': 0.5, 'tau_decay': 3.0, 'delay': 0.0, 'out_label': 'I'}}
    if mode == 'Liang_2020_Frontiers_Slow':
        synapse = {'E2E': NormalizedDualExponCUBA, 'I2E': NormalizedDualExponCUBA, 'E2I': NormalizedDualExponCUBA, 'I2I': NormalizedDualExponCUBA}
        synapse_params = {'E2E': {'tau_rise': 0.5, 'tau_decay': 2.0, 'delay': 0.0, 'out_label': 'E'}, 'E2I': {'tau_rise': 0.5, 'tau_decay': 2.0, 'delay': 0.0, 'out_label': 'E'}, 'I2E': {'tau_rise': 0.5, 'tau_decay': 4.3, 'delay': 0.0, 'out_label': 'I'}, 'I2I': {'tau_rise': 0.5, 'tau_decay': 4.3, 'delay': 0.0, 'out_label': 'I'}}
    if mode == 'Liang_2022_PCB_AS':
        synapse = {'E2E': NormalizedExponCOBA, 'I2E': NormalizedExponCOBA, 'E2I': NormalizedExponCOBA, 'I2I': NormalizedExponCOBA}
        synapse_params = {'E2E': {'tau': 4.0, 'delay': 0.0, 'E': 0.0, 'out_label': 'E'}, 'E2I': {'tau': 4.0, 'delay': 0.0, 'E': 0.0, 'out_label': 'E'}, 'I2E': {'tau': 4.0, 'delay': 0.0, 'E': -70.0, 'out_label': 'I'}, 'I2I': {'tau': 4.0, 'delay': 0.0, 'E': -70.0, 'out_label': 'I'}}
    if mode == 'Liang_2022_PCB_Cri':
        synapse = {'E2E': NormalizedExponCOBA, 'I2E': NormalizedExponCOBA, 'E2I': NormalizedExponCOBA, 'I2I': NormalizedExponCOBA}
        synapse_params = {'E2E': {'tau': 4.0, 'delay': 0.0, 'E': 0.0, 'out_label': 'E'}, 'E2I': {'tau': 4.0, 'delay': 0.0, 'E': 0.0, 'out_label': 'E'}, 'I2E': {'tau': 8.0, 'delay': 0.0, 'E': -70.0, 'out_label': 'I'}, 'I2I': {'tau': 8.0, 'delay': 0.0, 'E': -70.0, 'out_label': 'I'}}
    if mode == 'Liang_2022_PCB_SS':
        synapse = {'E2E': NormalizedExponCOBA, 'I2E': NormalizedExponCOBA, 'E2I': NormalizedExponCOBA, 'I2I': NormalizedExponCOBA}
        synapse_params = {'E2E': {'tau': 4.0, 'delay': 0.0, 'E': 0.0, 'out_label': 'E'}, 'E2I': {'tau': 4.0, 'delay': 0.0, 'E': 0.0, 'out_label': 'E'}, 'I2E': {'tau': 11.0, 'delay': 0.0, 'E': -70.0, 'out_label': 'I'}, 'I2I': {'tau': 11.0, 'delay': 0.0, 'E': -70.0, 'out_label': 'I'}}
    if mode == 'Liang_2022_PCB_P':
        synapse = {'E2E': NormalizedExponCOBA, 'I2E': NormalizedExponCOBA, 'E2I': NormalizedExponCOBA, 'I2I': NormalizedExponCOBA}
        synapse_params = {'E2E': {'tau': 4.0, 'delay': 0.0, 'E': 0.0, 'out_label': 'E'}, 'E2I': {'tau': 4.0, 'delay': 0.0, 'E': 0.0, 'out_label': 'E'}, 'I2E': {'tau': 14.0, 'delay': 0.0, 'E': -70.0, 'out_label': 'I'}, 'I2I': {'tau': 14.0, 'delay': 0.0, 'E': -70.0, 'out_label': 'I'}}
    return synapse, synapse_params


# def get_inp_param(paper):
#     if paper == 'Liang_2020_Frontiers':
#         E_inp_rate = 5.0
#         E_inp_num = 8000
#         E_inp_prob = 0.2
#         E_inp_weight = 0.45
#         I_inp_rate = 5.0
#         I_inp_num = 8000
#         I_inp_prob = 0.2
#         I_inp_weight = 0.72
#         return {'E': {'type': 'dualexpon_poisson', 'rate': E_inp_rate, 'num': E_inp_num, 'prob': E_inp_prob, 'weight': E_inp_weight, 'tau_rise': synapse_params['E2E']['tau_rise'], 'tau_decay': synapse_params['E2E']['tau_decay']}, 'I': {'type': 'dualexpon_poisson', 'rate': I_inp_rate, 'num': I_inp_num, 'prob': I_inp_prob, 'weight': I_inp_weight, 'tau_rise': synapse_params['E2E']['tau_rise'], 'tau_decay': synapse_params['E2E']['tau_decay']}}
# endregion


# region 利用idx提取数据
def neuron_idx_data(data, indices=None, keep_size=False):
    '''
    从spike或者V中提取指定索引的神经元数据。indices可以是slice对象或单个值。
    data: 二维矩阵，其中行表示时间点，列表示神经元。(与brainpy的输出一致)
    indices: 要提取的神经元索引列表或单个值。
    keep_size: 是否保持返回数据的二维形状。
    '''
    if indices is None:
        return data
    if keep_size and isinstance(indices, int):  # 单个索引时保持二维形状
        return data[:, [indices]]
    else:
        return data[:, indices]


def time_idx_data(data, indices=None, keep_size=False):
    '''
    从spike或者V中提取指定时间点的数据。indices可以是slice对象或单个值。
    data: 二维矩阵，其中行表示时间点，列表示神经元。(与brainpy的输出一致)
    indices: 要提取的时间点索引列表或单个值。
    keep_size: 是否保持返回数据的二维形状。
    '''
    if indices is None:
        return data
    if keep_size and isinstance(indices, int):  # 单个索引时保持二维形状
        return data[[indices], :]
    else:
        return data[indices, :]
# endregion


# region 神经元放电性质计算
def spike_to_fr(spike, width, dt, neuron_idx=None, **kwargs):
    '''
    修改bp.measure.firing_rate使得一维数组的spike也能够计算firing rate
    '''
    partial_spike = neuron_idx_data(spike, neuron_idx, keep_size=True)
    return bp.measure.firing_rate(partial_spike, width, dt, **kwargs)


def get_spike_acf(spike, dt, nlags, neuron_idx=None, average=True, **kwargs):
    '''
    计算spike的自相关函数
    '''
    partial_spike = neuron_idx_data(spike, neuron_idx, keep_size=True)
    float_spike = np.array(partial_spike).astype(float)
    lag_times, multi_acf = cf.get_multi_acf(float_spike.T, T=dt, nlags=nlags, **kwargs)
    if average:
        return lag_times, np.nanmean(multi_acf, axis=0)
    else:
        return lag_times, multi_acf.T


def spike_to_fr_acf(spike, width, dt, nlags, neuron_idx=None, spike_to_fr_kwargs=None, **kwargs):
    '''
    计算spike的firing rate的自相关函数,注意,计算fr的过程自动平均了neuron_idx中的神经元。
    '''
    if spike_to_fr_kwargs is None:
        spike_to_fr_kwargs = {}
    fr = spike_to_fr(spike, width, dt, neuron_idx, **spike_to_fr_kwargs)
    return cf.get_acf(fr, T=dt, nlags=nlags, **kwargs)


def get_ISI(spike, dt, neuron_idx=None, **kwargs):
    '''
    计算spike的ISI
    '''
    partial_spike = neuron_idx_data(spike, neuron_idx, keep_size=True)
    ISI = []
    for i in range(partial_spike.shape[1]):
        spike_times = np.where(partial_spike[:, i])[0] * dt
        if len(spike_times) < 2:
            continue
        ISI.append(list(np.diff(spike_times)))
    if len(ISI) == 0:
        return []
    else:
        ISI = cf.flatten_list(ISI)
    return ISI


def get_ISI_CV(spike, dt, neuron_idx=None, **kwargs):
    '''
    计算spike的ISI CV
    '''
    partial_spike = neuron_idx_data(spike, neuron_idx, keep_size=True)
    CV = []
    for i in range(partial_spike.shape[1]):
        spike_times = np.where(partial_spike[:, i])[0] * dt
        if len(spike_times) < 2:
            continue
        CV.append(cf.get_CV(np.diff(spike_times)))
    return CV


def get_spike_FF(spike, dt, timebin_list, neuron_idx=None, **kwargs):
    '''
    计算spike的FF(对每个神经元的整个spike序列计算FF,spike被方波卷积)
    '''
    partial_spike = neuron_idx_data(spike, neuron_idx, keep_size=True)

    FF = []
    for timebin in timebin_list:
        FF_partial = []
        kernel = np.ones(timebin) / timebin
        bin_spike = cf.convolve_multi_timeseries(partial_spike.T, kernel).T
        bin_spike = bin_spike / (timebin * dt)
        for i in range(bin_spike.shape[1]):
            FF_partial.append(cf.get_FF(bin_spike[:, i]))
        FF.append(FF_partial)
    return FF


def get_spike_avalanche(spike, dt, bin_size, neuron_idx=None, **kwargs):
    '''
    计算spike的avanlance
    '''
    partial_spike = neuron_idx_data(spike, neuron_idx, keep_size=True)

    # 获得所有神经元的spike相加得到的总spike
    spike_sum = np.sum(partial_spike, axis=1)

    # 利用bin_size计算bin内的spike数量
    bin_spike = cf.bin_timeseries(spike_sum, bin_size, mode='sum')

    # 获取avalanche的开始和结束
    non_zero_starts = np.where((bin_spike != 0) & (np.roll(bin_spike, 1) == 0))[0]
    non_zero_ends = np.where((bin_spike != 0) & (np.roll(bin_spike, -1) == 0))[0]

    # 记录avalanche的各种性质
    avalanche_size = []
    avalanche_duration = []
    for start, end in zip(non_zero_starts, non_zero_ends):
        avalanche_size.append(np.sum(bin_spike[start:end+1]))
        avalanche_duration.append((end - start) * bin_size * dt)

    # 计算特定duration下size的平均值
    duration_size_map = {}
    for d, s in zip(avalanche_duration, avalanche_size):
        if d in duration_size_map:
            duration_size_map[d].append(s)
        else:
            duration_size_map[d] = [s]

    duration_avg_size = {d: np.mean(sizes) for d, sizes in duration_size_map.items()}

    return avalanche_size, avalanche_duration, duration_avg_size
# endregion


# region 作图函数
def raster_plot(ax, dt, spike, color=cf.BLUE, xlabel='time (ms)', ylabel='neuron index', title='raster', label=None, xlim=None, ylim=None, scatter_kwargs=None, set_ax_kwargs=None):
    scatter_kwargs = cf.update_dict({'s': cf.MARKER_SIZE / 2}, scatter_kwargs)
    set_ax_kwargs = cf.update_dict({'adjust_tick_size': False}, set_ax_kwargs)

    if xlim is None:
        xlim = [0, spike.shape[0]*dt]
    if ylim is None:
        ylim = [0, spike.shape[1]]

    # Get the indices of the spikes
    spike_timestep, neuron_indices = np.where(spike)
    
    # Filter spikes based on xlim
    valid_indices = (spike_timestep*dt >= xlim[0]) & (spike_timestep*dt <= xlim[1])
    
    # Only plot valid spikes
    cf.plt_scatter(ax, spike_timestep[valid_indices]*dt, neuron_indices[valid_indices], color=color, label=label, clip_on=False, **scatter_kwargs)

    cf.set_ax(ax, xlabel, ylabel, title=title, xlim=xlim, ylim=ylim, **set_ax_kwargs)


def fr_scale_raster_plot(ax, dt, spike, fr, cmap=cf.DENSITY_CMAP, xlabel='time (ms)', ylabel='neuron index', title='raster', label=None, xlim=None, ylim=None, scatter_kwargs=None, set_ax_kwargs=None):
    scatter_kwargs = cf.update_dict({'s': cf.MARKER_SIZE / 2}, scatter_kwargs)
    set_ax_kwargs = cf.update_dict({'adjust_tick_size': False}, set_ax_kwargs)

    if xlim is None:
        xlim = [0, spike.shape[0]*dt]
    if ylim is None:
        ylim = [0, spike.shape[1]]

    # Get the indices of the spikes
    spike_timestep, neuron_indices = np.where(spike)
    
    # Filter spikes based on xlim
    valid_indices = (spike_timestep*dt >= xlim[0]) & (spike_timestep*dt <= xlim[1])
    
    # Get the color based on firing rate
    c = fr[spike_timestep[valid_indices]]

    # Only plot valid spikes
    cf.plt_colorful_scatter(ax, spike_timestep[valid_indices]*dt, neuron_indices[valid_indices], c, cmap=cmap, label=label, scatter_kwargs={'clip_on': False}, **scatter_kwargs)

    cf.set_ax(ax, xlabel, ylabel, title=title, xlim=xlim, ylim=ylim, **set_ax_kwargs)


def EI_raster_plot(ax, dt, E_spike, I_spike, E_color=E_COLOR, I_color=I_COLOR, xlabel='time (ms)', ylabel='', title='raster', E_label=None, I_label=None, E_xlim=None, E_ylim=None, I_xlim=None, I_ylim=None, split_ax_kwargs=None, scatter_kwargs=None, set_ax_kwargs=None):
    split_ax_kwargs = cf.update_dict({'nrows': 2, 'sharex': True, 'hspace': cf.SIDE_PAD*3, 'height_ratios': [E_spike.shape[1], I_spike.shape[1]]}, split_ax_kwargs)
    
    ax_E, ax_I = cf.split_ax(ax, **split_ax_kwargs)
    raster_plot(ax_E, dt, E_spike, color=E_color, xlabel='', ylabel='E '+ylabel, title=title, label=E_label, xlim=E_xlim, ylim=E_ylim, scatter_kwargs=scatter_kwargs, set_ax_kwargs=set_ax_kwargs)
    raster_plot(ax_I, dt, I_spike, color=I_color, xlabel=xlabel, ylabel='I '+ylabel, title='', label=I_label,  xlim=I_xlim, ylim=I_ylim, scatter_kwargs=scatter_kwargs, set_ax_kwargs=set_ax_kwargs)

    cf.rm_ax_spine(ax_E, 'bottom')
    cf.rm_ax_tick(ax_E, 'x')
    cf.rm_ax_ticklabel(ax_E, 'x')
    cf.align_label([ax_E, ax_I], 'y')
    return ax_E, ax_I


def template_line_plot(ax, x, y, color=cf.BLUE, xlabel='x', ylabel='y', title='line plot', label=None, line_kwargs=None, set_ax_kwargs=None):
    if line_kwargs is None:
        line_kwargs = {}
    if set_ax_kwargs is None:
        set_ax_kwargs = {}
    cf.plt_line(ax, x, y, color=color, label=label, **line_kwargs)
    cf.set_ax(ax, xlabel, ylabel, title=title, **set_ax_kwargs)


def template_EI_line_plot(ax, x, E_y, I_y, xlabel='x', ylabel='y', title='line plot', E_label='E', I_label='I', line_kwargs=None, set_ax_kwargs=None):
    template_line_plot(ax, x, E_y, color=E_COLOR, xlabel=xlabel, ylabel='E '+ylabel, title=title, label=E_label, line_kwargs=line_kwargs, set_ax_kwargs=set_ax_kwargs)
    template_line_plot(ax, x, I_y, color=I_COLOR, xlabel=xlabel, ylabel='I '+ylabel, title=title, label=I_label, line_kwargs=line_kwargs, set_ax_kwargs=set_ax_kwargs)


def input_current_plot(ax, current, dt, color=cf.BLUE, xlabel='time (ms)', ylabel='input current (nA)', title='input current', label=None, line_kwargs=None, set_ax_kwargs=None):
    if set_ax_kwargs is None:
        set_ax_kwargs = {}

    x = np.arange(current.shape[0]) * dt

    if 'xlim' in set_ax_kwargs.keys() and set_ax_kwargs['xlim'] is not None:
        valid_indices = (x >= set_ax_kwargs['xlim'][0]) & (x <= set_ax_kwargs['xlim'][1])
        x = x[valid_indices]
        current = current[valid_indices]

    template_line_plot(ax, x, current, color=color, xlabel=xlabel, ylabel=ylabel, title=title, label=label, line_kwargs=line_kwargs, set_ax_kwargs=set_ax_kwargs)


def seperate_ext_input_current_plot(ax, internal_current, external_current, dt, internal_color=cf.BLUE, external_color=cf.GREEN, total_color=cf.BLACK, xlabel='time (ms)', ylabel='input current (nA)', title='input current', internal_label='internal', external_label='external', total_label='total', line_kwargs=None, set_ax_kwargs=None):
    input_current_plot(ax, internal_current, dt, color=internal_color, xlabel=xlabel, ylabel=ylabel, title=title, label=internal_label, line_kwargs=line_kwargs, set_ax_kwargs=set_ax_kwargs)
    input_current_plot(ax, external_current, dt, color=external_color, xlabel=xlabel, ylabel=ylabel, title=title, label=external_label, line_kwargs=line_kwargs, set_ax_kwargs=set_ax_kwargs)
    input_current_plot(ax, internal_current + external_current, dt, color=total_color, xlabel=xlabel, ylabel=ylabel, title=title, label=total_label, line_kwargs=line_kwargs, set_ax_kwargs=set_ax_kwargs)


def seperate_EI_input_current_plot(ax, E_current, I_current, dt, E_color=E_COLOR, I_color=I_COLOR, total_color=cf.BLACK, xlabel='time (ms)', ylabel='input current (nA)', title='input current', E_label='E', I_label='I', total_label='total', set_ax_kwargs=None, line_kwargs=None):
    input_current_plot(ax, E_current, dt, color=E_color, xlabel=xlabel, ylabel=ylabel, title=title, label=E_label, line_kwargs=line_kwargs, set_ax_kwargs=set_ax_kwargs)
    input_current_plot(ax, I_current, dt, color=I_color, xlabel=xlabel, ylabel=ylabel, title=title, label=I_label, line_kwargs=line_kwargs, set_ax_kwargs=set_ax_kwargs)
    input_current_plot(ax, E_current + I_current, dt, color=total_color, xlabel=xlabel, ylabel=ylabel, title=title, label=total_label, line_kwargs=line_kwargs, set_ax_kwargs=set_ax_kwargs)


def seperate_EI_ext_input_current_plot(ax, E_current, I_current, external_current, dt, E_color=E_COLOR, I_color=I_COLOR, external_color=cf.GREEN, total_color=cf.BLACK, xlabel='time (ms)', ylabel='input current (nA)', title='input current', E_label='E', I_label='I', external_label='external', total_label='total', set_ax_kwargs=None, line_kwargs=None):
    input_current_plot(ax, E_current, dt, color=E_color, xlabel=xlabel, ylabel=ylabel, title=title, label=E_label, line_kwargs=line_kwargs, set_ax_kwargs=set_ax_kwargs)
    input_current_plot(ax, I_current, dt, color=I_color, xlabel=xlabel, ylabel=ylabel, title=title, label=I_label, line_kwargs=line_kwargs, set_ax_kwargs=set_ax_kwargs)
    input_current_plot(ax, external_current, dt, color=external_color, xlabel=xlabel, ylabel=ylabel, title=title, label=external_label, line_kwargs=line_kwargs, set_ax_kwargs=set_ax_kwargs)
    input_current_plot(ax, E_current + I_current + external_current, dt, color=total_color, xlabel=xlabel, ylabel=ylabel, title=title, label=total_label, line_kwargs=line_kwargs, set_ax_kwargs=set_ax_kwargs)


def fr_plot(ax, fr, dt, color=cf.BLUE, xlabel='time (ms)', ylabel='firing rate (Hz)', title='firing rate', label=None, line_kwargs=None, set_ax_kwargs=None):
    if set_ax_kwargs is None:
        set_ax_kwargs = {}

    x = np.arange(fr.shape[0]) * dt
    
    if 'xlim' in set_ax_kwargs.keys() and set_ax_kwargs['xlim'] is not None:
        valid_indices = (x >= set_ax_kwargs['xlim'][0]) & (x <= set_ax_kwargs['xlim'][1])
        x = x[valid_indices]
        fr = fr[valid_indices]

    template_line_plot(ax, x, fr, color=color, xlabel=xlabel, ylabel=ylabel, title=title, label=label, line_kwargs=line_kwargs, set_ax_kwargs=set_ax_kwargs)


def EI_fr_plot(ax, E_fr, I_fr, dt, E_color=E_COLOR, I_color=I_COLOR, xlabel='time (ms)', ylabel='firing rate (Hz)', title='firing rate', E_label='E', I_label='I', set_ax_kwargs=None, line_kwargs=None):
    fr_plot(ax, E_fr, dt, color=E_color, xlabel=xlabel, ylabel=ylabel, title=title, label=E_label, line_kwargs=line_kwargs, set_ax_kwargs=set_ax_kwargs)
    fr_plot(ax, I_fr, dt, color=I_color, xlabel=xlabel, ylabel=ylabel, title=title, label=I_label, line_kwargs=line_kwargs, set_ax_kwargs=set_ax_kwargs)


def voltage_plot(ax, V, dt, threshold, color=cf.BLUE, threshold_color=cf.ORANGE, xlabel='time (ms)', ylabel='membrane potential (mV)', title='membrane potential', label=None, line_kwargs=None, set_ax_kwargs=None):
    if set_ax_kwargs is None:
        set_ax_kwargs = {}

    x = np.arange(V.shape[0]) * dt
    
    if 'xlim' in set_ax_kwargs.keys() and set_ax_kwargs['xlim'] is not None:
        valid_indices = (x >= set_ax_kwargs['xlim'][0]) & (x <= set_ax_kwargs['xlim'][1])
        x = x[valid_indices]
        V = V[valid_indices]

    template_line_plot(ax, x, V, color=color, xlabel=xlabel, ylabel=ylabel, title=title, label=label, line_kwargs=line_kwargs, set_ax_kwargs=set_ax_kwargs)
    cf.add_hline(ax, threshold, color=threshold_color, linestyle='--', label='threshold')


def LFP_plot(ax, LFP, dt, color=cf.BLUE, xlabel='time (ms)', ylabel='LFP (mV)', title='LFP', label=None, line_kwargs=None, set_ax_kwargs=None):
    if set_ax_kwargs is None:
        set_ax_kwargs = {}

    x = np.arange(LFP.shape[0]) * dt
    
    if 'xlim' in set_ax_kwargs.keys() and set_ax_kwargs['xlim'] is not None:
        valid_indices = (x >= set_ax_kwargs['xlim'][0]) & (x <= set_ax_kwargs['xlim'][1])
        x = x[valid_indices]
        LFP = LFP[valid_indices]

    template_line_plot(ax, x, LFP, color=color, xlabel=xlabel, ylabel=ylabel, title=title, label=label, line_kwargs=line_kwargs, set_ax_kwargs=set_ax_kwargs)


def EI_LFP_plot(ax, E_LFP, I_LFP, dt, E_color=E_COLOR, I_color=I_COLOR, xlabel='time (ms)', ylabel='LFP (mV)', title='LFP', E_label='E', I_label='I', set_ax_kwargs=None, line_kwargs=None):
    LFP_plot(ax, E_LFP, dt, color=E_color, xlabel=xlabel, ylabel=ylabel, title=title, label=E_label, line_kwargs=line_kwargs, set_ax_kwargs=set_ax_kwargs)
    LFP_plot(ax, I_LFP, dt, color=I_color, xlabel=xlabel, ylabel=ylabel, title=title, label=I_label, line_kwargs=line_kwargs, set_ax_kwargs=set_ax_kwargs)


def single_exp(x, amp, tau):
    return amp * np.exp(-x / tau)


def single_exp_fit(lag_times, acf):
    single_popt, single_pcov, single_error = cf.get_curvefit(lag_times, acf, single_exp)
    return single_popt, single_pcov, single_error


def get_timescale_from_acf(lag_times, acf):
    single_popt, _, _ = single_exp_fit(lag_times, acf)
    return single_popt[1]


def acf_plot(ax, lag_times, acf, exp_fit=False, color=cf.BLUE, xlabel='lag (ms)', ylabel='ACF', title='ACF', label=None, line_kwargs=None, set_ax_kwargs=None, text_x=0.05, text_y=0.9, text_color=cf.BLACK, fontsize=cf.FONT_SIZE*1.6, show_fit_line=False, before_str='', after_str='', show_tau_in_text=True):
    '''
    before str: 在text前面加的字符串
    show_tau_in_text: 是否在text中显示'tau'
    '''

    if exp_fit:
        # def single_exp(x, amp, tau):
        #     return amp * np.exp(-x / tau)
        # def double_exp(x, amp1, tau1, amp2, tau2):
        #     return amp1 * np.exp(-x / tau1) - amp2 * np.exp(-x / tau2)
        
        # single_popt, single_pcov, single_error = cf.get_curvefit(lag_times, acf, single_exp)
        single_popt, single_pcov, single_error = single_exp_fit(lag_times, acf)
        # cf.add_text(ax, f'single exp fit: tau={cf.round_float(single_popt[1])} ms, error={cf.round_float(single_error)}', x=text_x, y=text_y, fontsize=cf.FONT_SIZE*1.6, color=text_color)
        if show_tau_in_text:
            cf.add_text(ax, cf.concat_str([before_str, f'tau={cf.round_float(single_popt[1])} ms', after_str]), x=text_x, y=text_y, fontsize=fontsize, color=text_color)
        else:
            cf.add_text(ax, cf.concat_str([before_str, f'{cf.round_float(single_popt[1])} ms', after_str]), x=text_x, y=text_y, fontsize=fontsize, color=text_color)
        # double_popt, double_pcov, double_error = cf.get_curvefit(lag_times, acf, double_exp, p0=[single_popt[0],single_popt[1],0.1,0], bounds=(0,np.inf), maxfev=5000)
        
        # cf.add_text(ax, f'double exp fit: tau1={cf.round_float(double_popt[1])} ms, tau2={cf.round_float(double_popt[3])} ms, error={cf.round_float(double_error)}', x=0.05, y=0.7)

        if show_fit_line:
            fit_line = single_exp(lag_times, *single_popt)
            cf.plt_line(ax, lag_times, fit_line, color=color, linestyle='--', label='exp fit')
    template_line_plot(ax, lag_times, acf, color=color, xlabel=xlabel, ylabel=ylabel, title=title, label=label, line_kwargs=line_kwargs, set_ax_kwargs=set_ax_kwargs)
    if exp_fit:
        return single_popt[1]


def EI_acf_plot(ax, E_lag_times, E_acf, I_lag_times, I_acf, E_color=E_COLOR, I_color=I_COLOR, xlabel='lag (ms)', ylabel='ACF', title='ACF', E_label='E', I_label='I', set_ax_kwargs=None, line_kwargs=None):
    acf_plot(ax, E_lag_times, E_acf, color=E_color, xlabel=xlabel, ylabel=ylabel, title=title, label=E_label, line_kwargs=line_kwargs, set_ax_kwargs=set_ax_kwargs)
    acf_plot(ax, I_lag_times, I_acf, color=I_color, xlabel=xlabel, ylabel=ylabel, title=title, label=I_label, line_kwargs=line_kwargs, set_ax_kwargs=set_ax_kwargs)


def freq_plot(ax, freqs, power, color=cf.BLUE, xlabel='frequency (Hz)', ylabel='power', title='power spectrum', label=None, line_kwargs=None, set_ax_kwargs=None):
    set_ax_kwargs = cf.update_dict({'xlim': [0, 500]}, set_ax_kwargs)
    template_line_plot(ax, freqs, power, color=color, xlabel=xlabel, ylabel=ylabel, title=title, label=label, line_kwargs=line_kwargs, set_ax_kwargs=set_ax_kwargs)


def EI_freq_plot(ax, E_freqs, E_power, I_freqs, I_power, E_color=E_COLOR, I_color=I_COLOR, xlabel='frequency (Hz)', ylabel='power', title='power spectrum', E_label='E', I_label='I', set_ax_kwargs=None, line_kwargs=None):
    freq_plot(ax, E_freqs, E_power, color=E_color, xlabel=xlabel, ylabel=ylabel, title=title, label=E_label, line_kwargs=line_kwargs, set_ax_kwargs=set_ax_kwargs)
    freq_plot(ax, I_freqs, I_power, color=I_color, xlabel=xlabel, ylabel=ylabel, title=title, label=I_label, line_kwargs=line_kwargs, set_ax_kwargs=set_ax_kwargs)


def FF_timewindow_plot(ax, timebin_list, FF, color=cf.BLUE, xlabel='time window (ms)', ylabel='FF', title='FF & timewindow', label=None, line_kwargs=None, set_ax_kwargs=None):
    template_line_plot(ax, timebin_list, FF, color=color, xlabel=xlabel, ylabel=ylabel, title=title, label=label, line_kwargs=line_kwargs, set_ax_kwargs=set_ax_kwargs)


def template_hist_plot(ax, data, color=cf.BLUE, xlabel='x', ylabel='probability', title='histogram', label=None, hist_kwargs=None, set_ax_kwargs=None):
    '''
    绘制histogram图
    '''
    if hist_kwargs is None:
        hist_kwargs = {}
    if set_ax_kwargs is None:
        set_ax_kwargs = {}
    cf.plt_hist(ax, data, color=color, label=label, **hist_kwargs)
    cf.set_ax(ax, xlabel, ylabel, title=title, **set_ax_kwargs)


def ISI_hist_plot(ax, ISI, color=cf.BLUE, xlabel='ISI (ms)', ylabel='probability', title='ISI distribution', label=None, hist_kwargs=None, set_ax_kwargs=None):
    '''
    绘制ISI分布图
    '''
    if len(ISI) > 0:
        set_ax_kwargs = cf.update_dict({'ylog': True}, set_ax_kwargs)
    else:
        set_ax_kwargs = cf.update_dict({}, set_ax_kwargs)
    template_hist_plot(ax, ISI, color=color, xlabel=xlabel, ylabel=ylabel, title=title, label=label, hist_kwargs=hist_kwargs, set_ax_kwargs=set_ax_kwargs)


def ISI_CV_hist_plot(ax, ISI_CV, color=cf.BLUE, xlabel='ISI CV', ylabel='probability', title='ISI CV distribution', label=None, hist_kwargs=None, set_ax_kwargs=None):
    '''
    绘制ISI CV分布图
    '''
    template_hist_plot(ax, ISI_CV, color=color, xlabel=xlabel, ylabel=ylabel, title=title, label=label, hist_kwargs=hist_kwargs, set_ax_kwargs=set_ax_kwargs)


def FF_hist_plot(ax, FF, color=cf.BLUE, xlabel='FF', ylabel='probability', title='FF distribution', label=None, hist_kwargs=None, set_ax_kwargs=None):
    '''
    绘制FF分布图
    '''
    template_hist_plot(ax, FF, color=color, xlabel=xlabel, ylabel=ylabel, title=title, label=label, hist_kwargs=hist_kwargs, set_ax_kwargs=set_ax_kwargs)


def corr_hist_plot(ax, corr, color=cf.BLUE, xlabel='correlation', ylabel='probability', title='correlation distribution', label=None, hist_kwargs=None, set_ax_kwargs=None):
    '''
    绘制correlation分布图
    '''
    template_hist_plot(ax, corr, color=color, xlabel=xlabel, ylabel=ylabel, title=title, label=label, hist_kwargs=hist_kwargs, set_ax_kwargs=set_ax_kwargs)


def avalanche_size_hist_plot(ax, avalanche_size, color=cf.BLUE, xlabel='avalanche size', ylabel='probability', title='avalanche size distribution', label=None, hist_kwargs=None, set_ax_kwargs=None):
    '''
    绘制avalanche size分布图
    '''
    if len(avalanche_size) > 0:
        set_ax_kwargs = cf.update_dict({'xlog': True, 'ylog': True}, set_ax_kwargs)
    else:
        set_ax_kwargs = cf.update_dict({}, set_ax_kwargs)
    template_hist_plot(ax, avalanche_size, color=color, xlabel=xlabel, ylabel=ylabel, title=title, label=label, hist_kwargs=hist_kwargs, set_ax_kwargs=set_ax_kwargs)


def avalanche_duration_hist_plot(ax, avalanche_duration, color=cf.BLUE, xlabel='avalanche duration', ylabel='probability', title='avalanche duration distribution', label=None, hist_kwargs=None, set_ax_kwargs=None):
    '''
    绘制avalanche duration分布图
    '''
    if len(avalanche_duration) > 0:
        set_ax_kwargs = cf.update_dict({'xlog': True, 'ylog': True}, set_ax_kwargs)
    else:
        set_ax_kwargs = cf.update_dict({}, set_ax_kwargs)
    template_hist_plot(ax, avalanche_duration, color=color, xlabel=xlabel, ylabel=ylabel, title=title, label=label, hist_kwargs=hist_kwargs, set_ax_kwargs=set_ax_kwargs)


def avalanche_size_duration_plot(ax, avalanche_size, avalanche_duration, scatter_color=cf.BLUE, line_color=cf.RED, xlabel='avalanche size', ylabel='avalanche duration', title='avalanche size-duration scatter', label=None, linregress_kwargs=None, set_ax_kwargs=None):
    '''
    绘制avalanche size和duration的散点图
    '''
    if linregress_kwargs is None:
        linregress_kwargs = {}
    if set_ax_kwargs is None:
        set_ax_kwargs = {}
    cf.plt_linregress(ax, avalanche_size, avalanche_duration, label=label, scatter_color=scatter_color, line_color=line_color, **linregress_kwargs)
    cf.set_ax(ax, xlabel, ylabel, title=title, **set_ax_kwargs)


def get_raster_color(i, pos, spike, faint_num, color, pos_color, pos_alpha):
    scattered_idx = []
    colors = np.zeros((pos.shape[0], 4))  # RGBA
    for previous in range(faint_num + 1):
        if i - previous >= 0:
            alpha = np.linspace(1, 0, faint_num + 1)[previous]
            current_idx = time_idx_data(spike, i - previous) > 0
            scattered_idx.extend(np.where(current_idx)[0])
            colors[current_idx, :3] = color  # Set RGB
            colors[current_idx, 3] = alpha  # Set alpha

    scattered_idx = np.array(scattered_idx)
    unscattered_idx = np.setdiff1d(np.arange(pos.shape[0]), scattered_idx)

    # Set colors for unscattered points
    colors[unscattered_idx, :3] = pos_color
    colors[unscattered_idx, 3] = pos_alpha

    return colors


def spatial_raster_plot(ax, spike, pos, i, dt, faint_num=3, label=None, color=cf.BLUE, scatter_size=(cf.MARKER_SIZE/3)**2, pos_color=cf.RANA, pos_alpha=0.1, scale_prop=1.05, legend_loc='upper left', bbox_to_anchor=(1, 1), set_ax_kwargs=None, scatter_kwargs=None):
    '''
    绘制Raster图
    '''
    if set_ax_kwargs is None:
        set_ax_kwargs = {}
    if scatter_kwargs is None:
        scatter_kwargs = {}
    colors = get_raster_color(i, pos, spike, faint_num, color, pos_color, pos_alpha)
    if pos.shape[1] == 2:
        cf.plt_scatter(ax, pos[:, 0], pos[:, 1], color=colors, s=scatter_size, **scatter_kwargs)
        cf.plt_scatter(ax, [], [], color=color, s=scatter_size, label=label, **scatter_kwargs)
    if pos.shape[1] == 3:
        cf.plt_scatter_3d(ax, pos[:, 0], pos[:, 1], pos[:, 2], color=colors, s=scatter_size, **scatter_kwargs)
        cf.plt_scatter_3d(ax, [], [], [], color=color, s=scatter_size, label=label, **scatter_kwargs)

    expand_xlim_min, expand_xlim_max = cf.scale_range(np.min(pos[:, 0]), np.max(pos[:, 0]), scale_prop)
    expand_ylim_min, expand_ylim_max = cf.scale_range(np.min(pos[:, 1]), np.max(pos[:, 1]), scale_prop)
    ax.set_xlim([expand_xlim_min, expand_xlim_max])
    ax.set_ylim([expand_ylim_min, expand_ylim_max])
    if pos.shape[1] == 3:
        expand_zlim_min, expand_zlim_max = cf.scale_range(np.min(pos[:, 2]), np.max(pos[:, 2]), scale_prop)
        ax.set_zlim([expand_zlim_min, expand_zlim_max])
        cf.set_ax_aspect_3d(ax)
    else:
        cf.set_ax_aspect(ax)
    ax.axis('off')

    title = 't={}'.format(cf.align_decimal(i*dt, dt))
    cf.set_ax(ax, title=title, legend_loc=legend_loc, bbox_to_anchor=bbox_to_anchor, **set_ax_kwargs)


def EI_spatial_raster_plot(ax, E_spike, I_spike, E_pos, I_pos, i, dt, faint_num=3, scatter_size=(cf.MARKER_SIZE/3)**2, pos_color=cf.RANA, pos_alpha=0.1, scale_prop=1.05, legend_loc='upper left', bbox_to_anchor=(1, 1), set_ax_kwargs=None, scatter_kwargs=None):
    '''
    绘制Raster图
    '''
    if set_ax_kwargs is None:
        set_ax_kwargs = {}
    if scatter_kwargs is None:
        scatter_kwargs = {}
    E_colors = get_raster_color(i, E_pos, E_spike, faint_num, E_COLOR, pos_color, pos_alpha)
    I_colors = get_raster_color(i, I_pos, I_spike, faint_num, I_COLOR, pos_color, pos_alpha)
    colors = np.concatenate([E_colors, I_colors], axis=0)
    pos = np.concatenate([E_pos, I_pos], axis=0)
    if pos.shape[1] == 2:
        cf.plt_scatter(ax, pos[:, 0], pos[:, 1], color=colors, s=scatter_size, **scatter_kwargs)
        cf.plt_scatter(ax, [], [], color=E_COLOR, s=scatter_size, label='E', **scatter_kwargs)
        cf.plt_scatter(ax, [], [], color=I_COLOR, s=scatter_size, label='I', **scatter_kwargs)
    if pos.shape[1] == 3:
        cf.plt_scatter_3d(ax, pos[:, 0], pos[:, 1], pos[:, 2], color=colors, s=scatter_size, **scatter_kwargs)
        cf.plt_scatter_3d(ax, [], [], [], color=E_COLOR, s=scatter_size, label='E', **scatter_kwargs)
        cf.plt_scatter_3d(ax, [], [], [], color=I_COLOR, s=scatter_size, label='I', **scatter_kwargs)

    expand_xlim_min, expand_xlim_max = cf.scale_range(np.min(np.concatenate([E_pos[:, 0], I_pos[:, 0]])), np.max(np.concatenate([E_pos[:, 0], I_pos[:, 0]])), scale_prop)
    expand_ylim_min, expand_ylim_max = cf.scale_range(np.min(np.concatenate([E_pos[:, 1], I_pos[:, 1]])), np.max(np.concatenate([E_pos[:, 1], I_pos[:, 1]])), scale_prop)
    ax.set_xlim([expand_xlim_min, expand_xlim_max])
    ax.set_ylim([expand_ylim_min, expand_ylim_max])
    if E_pos.shape[1] == 3:
        expand_zlim_min, expand_zlim_max = cf.scale_range(np.min(np.concatenate([E_pos[:, 2], I_pos[:, 2]])), np.max(np.concatenate([E_pos[:, 2], I_pos[:, 2]])), scale_prop)
        ax.set_zlim([expand_zlim_min, expand_zlim_max])
        cf.set_ax_aspect_3d(ax)
    else:
        cf.set_ax_aspect(ax)
    ax.axis('off')

    title = 't={}'.format(cf.align_decimal(i*dt, dt))
    cf.set_ax(ax, title=title, legend_loc=legend_loc, bbox_to_anchor=bbox_to_anchor, **set_ax_kwargs)


def spatial_V_plot(ax, V, pos, i, dt, label=None, vmin=None, vmax=None, cmap=cf.PINEAPPLE_CMAP, scatter_size=(cf.MARKER_SIZE/3)**2, scale_prop=1.05, legend_loc='upper left', bbox_to_anchor=(1, 1), set_ax_kwargs=None, scatter_kwargs=None):
    '''
    绘制V的空间分布图
    '''
    if set_ax_kwargs is None:
        set_ax_kwargs = {}
    if scatter_kwargs is None:
        scatter_kwargs = {}

    V_now = time_idx_data(V, i)
    
    if pos.shape[1] == 2:
        cf.plt_colorful_scatter(ax, pos[:, 0], pos[:, 1], c=V_now, cmap=cmap, s=scatter_size, vmin=vmin, vmax=vmax, label=label, **scatter_kwargs)
    if pos.shape[1] == 3:
        cf.plt_colorful_scatter_3d(ax, pos[:, 0], pos[:, 1], pos[:, 2], c=V_now, cmap=cmap, s=scatter_size, vmin=vmin, vmax=vmax, label=label, **scatter_kwargs)

    expand_xlim_min, expand_xlim_max = cf.scale_range(np.min(pos[:, 0]), np.max(pos[:, 0]), scale_prop)
    expand_ylim_min, expand_ylim_max = cf.scale_range(np.min(pos[:, 1]), np.max(pos[:, 1]), scale_prop)
    ax.set_xlim([expand_xlim_min, expand_xlim_max])
    ax.set_ylim([expand_ylim_min, expand_ylim_max])
    if pos.shape[1] == 3:
        expand_zlim_min, expand_zlim_max = cf.scale_range(np.min(pos[:, 2]), np.max(pos[:, 2]), scale_prop)
        ax.set_zlim([expand_zlim_min, expand_zlim_max])
        cf.set_ax_aspect_3d(ax)
    else:
        cf.set_ax_aspect(ax)
    ax.axis('off')

    title = 't={}'.format(cf.align_decimal(i*dt, dt))
    cf.set_ax(ax, title=title, legend_loc=legend_loc, bbox_to_anchor=bbox_to_anchor, **set_ax_kwargs)


def EI_spatial_V_plot(ax, E_V, I_V, E_pos, I_pos, i, dt, E_label='E', I_label='I', vmin=None, vmax=None, cmap=cf.PINEAPPLE_CMAP, scatter_size=(cf.MARKER_SIZE/3)**2, scale_prop=1.05, legend_loc='upper left', bbox_to_anchor=(1, 1), set_ax_kwargs=None, scatter_kwargs=None):
    '''
    绘制V的空间分布图
    '''
    if set_ax_kwargs is None:
        set_ax_kwargs = {}
    if scatter_kwargs is None:
        scatter_kwargs = {}
    E_ax = ax[0]
    I_ax = ax[1]
    spatial_V_plot(ax=E_ax, V=E_V, pos=E_pos, i=i, dt=dt, label=E_label, vmin=vmin, vmax=vmax, cmap=cmap, scatter_size=scatter_size, scale_prop=scale_prop, legend_loc=legend_loc, bbox_to_anchor=bbox_to_anchor, set_ax_kwargs=set_ax_kwargs, scatter_kwargs=scatter_kwargs)
    spatial_V_plot(ax=I_ax, V=I_V, pos=I_pos, i=i, dt=dt, label=I_label, vmin=vmin, vmax=vmax, cmap=cmap, scatter_size=scatter_size, scale_prop=scale_prop, legend_loc=legend_loc, bbox_to_anchor=bbox_to_anchor, set_ax_kwargs=set_ax_kwargs, scatter_kwargs=scatter_kwargs)

    E_ax.set_title(cf.concat_str([E_label, 'V']))
    I_ax.set_title(cf.concat_str([I_label, 'V']))
    fig = E_ax.get_figure()
    cf.set_fig_title(fig, 't={}'.format(cf.align_decimal(i*dt, dt)))


def causal_spatial_raster_plot(ax, E_monitored_neuron, I_monitored_neuron, E_spike, I_spike, E_pos, I_pos, E2E_connection, E2I_connection, I2E_connection, I2I_connection, delay_step, i, dt, faint_num=3, E_label='E', I_label='I', scatter_size=(cf.MARKER_SIZE/3)**2, show_pos=True, pos_color=cf.RANA, pos_alpha=0.5, scale_prop=1.05, legend_loc='upper left', bbox_to_anchor=(1, 1), set_ax_kwargs=None, scatter_kwargs=None):
    '''
    绘制Raster图,根据delay_step和连接,画出spike传递的路径
    '''
    if isinstance(E_monitored_neuron, int):
        E_monitored_neuron = [E_monitored_neuron]
    if isinstance(I_monitored_neuron, int):
        I_monitored_neuron = [I_monitored_neuron]
    if set_ax_kwargs is None:
        set_ax_kwargs = {}
    if scatter_kwargs is None:
        scatter_kwargs = {}
    if show_pos:
        if E_pos.shape[1] == 2:
            cf.plt_scatter(ax, E_pos[:, 0], E_pos[:, 1], color=pos_color, s=scatter_size, alpha=pos_alpha, zorder=0, **scatter_kwargs)
            cf.plt_scatter(ax, I_pos[:, 0], I_pos[:, 1], color=pos_color, s=scatter_size, alpha=pos_alpha, zorder=0, **scatter_kwargs)
        if E_pos.shape[1] == 3:
            cf.plt_scatter_3d(ax, E_pos[:, 0], E_pos[:, 1], E_pos[:, 2], color=pos_color, s=scatter_size, alpha=pos_alpha, zorder=0, **scatter_kwargs)
            cf.plt_scatter_3d(ax, I_pos[:, 0], I_pos[:, 1], I_pos[:, 2], color=pos_color, s=scatter_size, alpha=pos_alpha, zorder=0, **scatter_kwargs)
    for previous in range(faint_num+1):
        if i-previous >= 0:
            alpha = np.linspace(1, 0, faint_num+1)[previous]
            if previous == 0:
                local_E_label = E_label
                local_I_label = I_label
            else:
                local_E_label = None
                local_I_label = None
            if E_pos.shape[1] == 2:
                cf.plt_scatter(ax, E_pos[time_idx_data(E_spike, i-previous) > 0, 0], E_pos[time_idx_data(E_spike, i-previous) > 0, 1], color=E_COLOR, s=scatter_size, alpha=alpha, label=local_E_label, **scatter_kwargs)
                cf.plt_scatter(ax, I_pos[time_idx_data(I_spike, i-previous) > 0, 0], I_pos[time_idx_data(I_spike, i-previous) > 0, 1], color=I_COLOR, s=scatter_size, alpha=alpha, label=local_I_label, **scatter_kwargs)
            if E_pos.shape[1] == 3:
                cf.plt_scatter_3d(ax, E_pos[time_idx_data(E_spike, i-previous) > 0, 0], E_pos[time_idx_data(E_spike, i-previous) > 0, 1], E_pos[time_idx_data(E_spike, i-previous) > 0, 2], color=E_COLOR, s=scatter_size, alpha=alpha, label=local_E_label, **scatter_kwargs)
                cf.plt_scatter_3d(ax, I_pos[time_idx_data(I_spike, i-previous) > 0, 0], I_pos[time_idx_data(I_spike, i-previous) > 0, 1], I_pos[time_idx_data(I_spike, i-previous) > 0, 2], color=I_COLOR, s=scatter_size, alpha=alpha, label=local_I_label, **scatter_kwargs)
            for source in ['E', 'I']:
                for target in ['E', 'I']:
                    if source == 'E' and target == 'E':
                        connection = E2E_connection
                        source_pos = E_pos
                        target_pos = E_pos
                        monitored_neuron = E_monitored_neuron
                        color = E_COLOR
                        spike = E_spike
                    if source == 'E' and target == 'I':
                        connection = E2I_connection
                        source_pos = E_pos
                        target_pos = I_pos
                        monitored_neuron = I_monitored_neuron
                        color = E_COLOR
                        spike = E_spike
                    if source == 'I' and target == 'E':
                        connection = I2E_connection
                        source_pos = I_pos
                        target_pos = E_pos
                        monitored_neuron = E_monitored_neuron
                        color = I_COLOR
                        spike = I_spike
                    if source == 'I' and target == 'I':
                        connection = I2I_connection
                        source_pos = I_pos
                        target_pos = I_pos
                        monitored_neuron = I_monitored_neuron
                        color = I_COLOR
                        spike = I_spike
                    for j in range(connection.shape[0]):
                        if monitored_neuron is not None:
                            for k in monitored_neuron:
                                if connection[j, k] > 0 and time_idx_data(spike, i-previous-delay_step)[j] > 0:
                                    if source_pos.shape[1] == 2:
                                        cf.add_mid_arrow(ax, source_pos[j, 0], source_pos[j, 1], target_pos[k, 0], target_pos[k, 1], fc=color, ec=color, linewidth=scatter_size/2, alpha=alpha)
                                    if source_pos.shape[1] == 3:
                                        # ax.plot([source_pos[j, 0], source_pos[k, 0]], [pos[j, 1], pos[k, 1]], [pos[j, 2], pos[k, 2]], color=color, alpha=alpha)
                                        pass
    ax.axis('off')
    title = 't={}'.format(cf.align_decimal(i*dt, dt))
    cf.set_ax(ax, title=title, legend_loc=legend_loc, bbox_to_anchor=bbox_to_anchor, **set_ax_kwargs)
    expand_xlim_min, expand_xlim_max = cf.scale_range(np.min(np.concatenate([E_pos[:, 0], I_pos[:, 0]])), np.max(np.concatenate([E_pos[:, 0], I_pos[:, 0]])), scale_prop)
    expand_ylim_min, expand_ylim_max = cf.scale_range(np.min(np.concatenate([E_pos[:, 1], I_pos[:, 1]])), np.max(np.concatenate([E_pos[:, 1], I_pos[:, 1]])), scale_prop)
    ax.set_xlim([expand_xlim_min, expand_xlim_max])
    ax.set_ylim([expand_ylim_min, expand_ylim_max])
    if E_pos.shape[1] == 3:
        expand_zlim_min, expand_zlim_max = cf.scale_range(np.min(np.concatenate([E_pos[:, 2], I_pos[:, 2]])), np.max(np.concatenate([E_pos[:, 2], I_pos[:, 2]])), scale_prop)
        ax.set_zlim([expand_zlim_min, expand_zlim_max])
        cf.set_ax_aspect_3d(ax)
    else:
        cf.set_ax_aspect(ax)


def neuron_frame(dim, margin, fig_ax_kwargs, plot_func, plot_func_kwargs, elev_list, azim_list, folder, part_figname, save_fig_kwargs, i=None):
    if dim == 2:
        fig, ax = cf.get_fig_ax(margin=margin, **fig_ax_kwargs)
        plot_func(ax=ax, i=i, **plot_func_kwargs)
        figname = cf.concat_str([part_figname, 'i='+str(i)])
        filename = os.path.join(folder, figname)
        cf.save_fig(fig, filename, formats=['png'], pkl=False, dpi=100, **save_fig_kwargs)
        return filename
    if dim == 3:
        fig, ax = cf.get_fig_ax_3d(margin=margin, **fig_ax_kwargs)
        plot_func(ax=ax, i=i, **plot_func_kwargs)
        figname = cf.concat_str([part_figname, 'i='+str(i)])
        filename = os.path.join(folder, figname)
        fig_paths_dict, _ = cf.save_fig_3d(fig, filename, elev_list=elev_list, azim_list=azim_list, generate_video=False, formats=['png'], dpi=100, pkl=False, **save_fig_kwargs)
        return fig_paths_dict


def neuron_video(plot_func, plot_func_kwargs, dim, step_list, folder, video_name, elev_list=None, azim_list=None, margin=None, fig_ax_kwargs=None, set_ax_kwargs=None, part_figname='', save_fig_kwargs=None, video_kwargs=None, process_num=cf.PROCESS_NUM):
    '''
    绘制neuron视频

    参数:
    plot_func: 绘图函数
    plot_func_kwargs: 绘图函数的参数
    dim: 空间维度
    dt: 时间步长
    step_list: 时间步长列表
    folder: 保存文件夹
    video_name: 视频名称
    legend_loc: 图例位置
    bbox_to_anchor: 图例位置
    elev_list: 视角仰角列表
    azim_list: 视角方位角列表
    margin: 图边距
    fig_ax_kwargs: fig和ax的参数
    scatter_kwargs: scatter的参数
    set_ax_kwargs: set_ax的参数
    save_fig_kwargs: save_fig的参数
    video_kwargs: video的参数
    process_num: 进程数
    '''
    cf.mkdir(folder)

    if elev_list is None:
        elev_list = [cf.ELEV]
    if azim_list is None:
        azim_list = [cf.AZIM]
    if margin is None:
        margin = {'left': 0.1, 'right':0.75, 'bottom' : 0.1, 'top': 0.75}
    if fig_ax_kwargs is None:
        fig_ax_kwargs = {}
    if set_ax_kwargs is None:
        set_ax_kwargs = {}
    if save_fig_kwargs is None:
        save_fig_kwargs = {}
    if video_kwargs is None:
        video_kwargs = {}

    if dim == 2:
        local_args = (dim, margin, fig_ax_kwargs, plot_func, plot_func_kwargs, elev_list, azim_list, folder, part_figname, save_fig_kwargs)
        fig_paths = cf.multi_process_list_for(process_num, func=neuron_frame, args=local_args, for_list=step_list, for_idx_name='i')
        cf.fig_to_video(fig_paths, os.path.join(folder, video_name), **video_kwargs)
    if dim == 3:
        fig_paths = {}
        local_args = (dim, margin, fig_ax_kwargs, plot_func, plot_func_kwargs, elev_list, azim_list, folder, part_figname, save_fig_kwargs)
        fig_paths_dict_list = cf.multi_process_list_for(process_num, func=neuron_frame, args=local_args, for_list=step_list, for_idx_name='i')
        for fig_paths_dict in fig_paths_dict_list:
            for elev in elev_list:
                for azim in azim_list:
                    fig_paths[(elev, azim)] = fig_paths.get((elev, azim), []) + fig_paths_dict[(elev, azim)]
        for elev in elev_list:
            for azim in azim_list:
                cf.fig_to_video(fig_paths[(elev, azim)], os.path.join(folder, video_name+'_elev_{}_azim_{}'.format(str(int(elev)), str(int(azim)))), **video_kwargs)


def spike_video(E_spike, I_spike, E_pos, I_pos, dt, step_list, folder, video_name='spike_video', scatter_size=(cf.MARKER_SIZE/3)**2, faint_num=3, legend_loc='upper left', bbox_to_anchor=(1, 1), elev_list=None, azim_list=None, margin=None, fig_ax_kwargs=None, scatter_kwargs=None, set_ax_kwargs=None, part_figname='spike', save_fig_kwargs=None, video_kwargs=None, process_num=cf.PROCESS_NUM):
    margin = cf.update_dict({'left': 0.1, 'right':0.75, 'bottom' : 0.1, 'top': 0.75}, margin)
    EI_spatial_raster_kwargs = {'E_spike': E_spike, 'I_spike': I_spike, 'E_pos': E_pos, 'I_pos': I_pos, 'scatter_size': scatter_size, 'faint_num': faint_num, 'dt': dt, 'legend_loc': legend_loc, 'bbox_to_anchor': bbox_to_anchor, 'set_ax_kwargs': set_ax_kwargs, 'scatter_kwargs': scatter_kwargs}
    neuron_video(EI_spatial_raster_plot, EI_spatial_raster_kwargs, E_pos.shape[1], step_list, folder, video_name, elev_list=elev_list, azim_list=azim_list, margin=margin, fig_ax_kwargs=fig_ax_kwargs, set_ax_kwargs=set_ax_kwargs, part_figname=part_figname, save_fig_kwargs=save_fig_kwargs, video_kwargs=video_kwargs, process_num=process_num)


def V_video(E_V, I_V, E_pos, I_pos, dt, step_list, folder, vmin, vmax, cmap=cf.PINEAPPLE_CMAP, video_name='V_video', scatter_size=(cf.MARKER_SIZE/3)**2, legend_loc='upper left', bbox_to_anchor=(1, 1), elev_list=None, azim_list=None, margin=None, fig_ax_kwargs=None, scatter_kwargs=None, set_ax_kwargs=None, part_figname='V', save_fig_kwargs=None, video_kwargs=None, process_num=cf.PROCESS_NUM):
    margin = cf.update_dict({'left': 0.1, 'right':0.75, 'bottom' : 0.1, 'top': 0.75}, margin)
    fig_ax_kwargs = cf.update_dict({'ncols': 2}, fig_ax_kwargs)
    spatial_V_kwargs = {'E_V': E_V, 'I_V': I_V, 'E_pos': E_pos, 'I_pos': I_pos, 'scatter_size': scatter_size, 'dt': dt, 'vmin': vmin, 'vmax': vmax, 'cmap': cmap, 'legend_loc': legend_loc, 'bbox_to_anchor': bbox_to_anchor, 'set_ax_kwargs': set_ax_kwargs, 'scatter_kwargs': scatter_kwargs}
    neuron_video(EI_spatial_V_plot, spatial_V_kwargs, E_pos.shape[1], step_list, folder, video_name, elev_list=elev_list, azim_list=azim_list, margin=margin, fig_ax_kwargs=fig_ax_kwargs, set_ax_kwargs=set_ax_kwargs, part_figname=part_figname, save_fig_kwargs=save_fig_kwargs, video_kwargs=video_kwargs, process_num=process_num)


def casual_spike_video(E_monitored_neuron, I_monitored_neuron, E_spike, E_pos, I_spike, I_pos, E2E_connection, E2I_connection, I2E_connection, I2I_connection, delay_step, dt, folder, video_name='spike_video', scatter_size=(cf.MARKER_SIZE/3)**2, faint_num=3, legend_loc='upper left', bbox_to_anchor=(1, 1), elev_list=None, azim_list=None, margin=None, fig_ax_kwargs=None, scatter_kwargs=None, set_ax_kwargs=None, save_fig_kwargs=None, video_kwargs=None):
    margin = cf.update_dict({'left': 0.1, 'right':0.75, 'bottom' : 0.1, 'top': 0.75}, margin)
    causal_spatial_raster_kwargs = {'E_monitored_neuron': E_monitored_neuron, 'I_monitored_neuron': I_monitored_neuron, 'E_spike': E_spike, 'I_spike': I_spike, 'E_pos': E_pos, 'I_pos': I_pos, 'E2E_connection': E2E_connection, 'E2I_connection': E2I_connection, 'I2E_connection': I2E_connection, 'I2I_connection': I2I_connection, 'delay_step': delay_step, 'scatter_size': scatter_size, 'faint_num': faint_num, 'dt': dt, 'legend_loc': legend_loc, 'bbox_to_anchor': bbox_to_anchor, 'set_ax_kwargs': set_ax_kwargs, 'scatter_kwargs': scatter_kwargs}
    neuron_video(causal_spatial_raster_plot, causal_spatial_raster_kwargs, E_pos.shape[1], dt, E_spike.shape[0], folder, video_name, legend_loc=legend_loc, bbox_to_anchor=bbox_to_anchor, elev_list=elev_list, azim_list=azim_list, margin=margin, fig_ax_kwargs=fig_ax_kwargs, scatter_kwargs=scatter_kwargs, set_ax_kwargs=set_ax_kwargs, save_fig_kwargs=save_fig_kwargs, video_kwargs=video_kwargs)
# endregion


# region neuron
class EILifRef(bp.dyn.LifRefLTC):
    '''
    EI_inp分别考虑的LIF模型
    '''
    def derivative(self, V, t, I):
        return (-V + self.V_rest + self.R * I) / self.tau

    def update(self, x=None):
        x = 0. if x is None else x
        E_inp = self.sum_current_inputs(self.V.value, init=0, label='E')
        I_inp = self.sum_current_inputs(self.V.value, init=0, label='I')
        x = E_inp + I_inp + x
        return super().update(x)


class externalLifRef(bp.dyn.LifRefLTC):
    '''
    EI_inp分别考虑的LIF模型
    '''
    def derivative(self, V, t, I):
        return (-V + self.V_rest + self.R * I) / self.tau

    def update(self, x=None):
        x = 0. if x is None else x
        external_inp = self.sum_current_inputs(self.V.value, init=0, label='external') + x
        internal_inp = self.sum_current_inputs(self.V.value, init=0, label='internal')
        x = external_inp + internal_inp
        return super().update(x)
# endregion


# region synapse
class NormalizedExpon(bp.dyn.Expon):
    '''
    不同于brainpy的Expon(https://brainpy.readthedocs.io/en/latest/apis/generated/brainpy.dyn.Expon.html),这里的Expon是使用timescale归一化的,使得整个kernel积分为1
    '''
    def add_current(self, x):
        self.g.value += x / self.tau


class NormalizedDualExponV2(bp.dyn.DualExponV2):
    '''
    调整A的默认值(https://brainpy.readthedocs.io/en/latest/apis/generated/brainpy.dyn.DualExponV2.html),使得整个kernel积分为1

    注意,如果想要获取g的话,要使用这样的语法:
    定义syn
    self.syn = bf.NormalizedDualExponCUBA(self.pre, self.post, delay=None, comm=bp.dnn.CSRLinear(bp.conn.FixedProb(1., pre=self.pre.num, post=self.post.num), 1.), tau_rise=2., tau_decay=20.)
    拿到syn的两个g和a
    (self.syn.proj.refs['syn'].g_decay - self.syn.proj.refs['syn'].g_rise) * self.syn.proj.refs['syn'].a

    相比之下,NormailzedExponCUBA的g可以直接拿到
    '''
    def __init__(
        self,
        size: Union[int, Sequence[int]],
        keep_size: bool = False,
        sharding: Optional[Sequence[str]] = None,
        method: str = 'exp_auto',
        name: Optional[str] = None,
        mode: Optional[bm.Mode] = None,

        # synapse parameters
        tau_decay: Union[float, ArrayType, Callable] = 10.0,
        tau_rise: Union[float, ArrayType, Callable] = 1.,
        A: Optional[Union[float, ArrayType, Callable]] = None,
    ):
        super().__init__(name=name,
                            mode=mode,
                            size=size,
                            keep_size=keep_size,
                            sharding=sharding)

        def _format_dual_exp_A(self, A):
            A = parameter(A, sizes=self.varshape, allow_none=True, sharding=self.sharding)
            if A is None:
                A = 1 / (self.tau_decay - self.tau_rise)
            return A

        # parameters
        self.tau_rise = self.init_param(tau_rise)
        self.tau_decay = self.init_param(tau_decay)
        self.a = _format_dual_exp_A(self, A)

        # integrator
        self.integral = odeint(lambda g, t, tau: -g / tau, method=method)

        self.reset_state(self.mode)


class ExponCUBA(bp.Projection):
    def __init__(self, pre, post, delay, comm, tau, out_label=None):
        super().__init__()
        
        self.proj = bp.dyn.FullProjAlignPostMg(
        pre=pre, 
        delay=delay, 
        comm=comm,
        syn=bp.dyn.Expon.desc(post.num, tau=tau),
        out=bp.dyn.CUBA.desc(),
        post=post,
        out_label=out_label 
        )


class ExponCOBA(bp.Projection):
    def __init__(self, pre, post, delay, comm, tau, E, out_label=None):
        super().__init__()
        
        self.proj = bp.dyn.FullProjAlignPostMg(
        pre=pre, 
        delay=delay, 
        comm=comm,
        syn=bp.dyn.Expon.desc(post.num, tau=tau),
        out=bp.dyn.COBA.desc(E),
        post=post,
        out_label=out_label 
        )


class NormalizedExponCUBA(bp.Projection):
    def __init__(self, pre, post, delay, comm, tau, out_label=None):
        super().__init__()
        
        self.proj = bp.dyn.FullProjAlignPostMg(
        pre=pre, 
        delay=delay, 
        comm=comm,
        syn=NormalizedExpon.desc(post.num, tau=tau),
        out=bp.dyn.CUBA.desc(),
        post=post,
        out_label=out_label 
        )


class NormalizedExponCOBA(bp.Projection):
    def __init__(self, pre, post, delay, comm, tau, E, out_label=None):
        super().__init__()
        
        self.proj = bp.dyn.FullProjAlignPostMg(
        pre=pre, 
        delay=delay, 
        comm=comm,
        syn=NormalizedExpon.desc(post.num, tau=tau),
        out=bp.dyn.COBA.desc(E),
        post=post,
        out_label=out_label 
        )


class DualExponCUBA(bp.Projection):
    def __init__(self, pre, post, delay, comm, tau_rise, tau_decay, A=None, out_label=None):
        super().__init__()
        
        self.proj = bp.dyn.FullProjAlignPostMg(
        pre=pre, 
        delay=delay, 
        comm=comm,
        syn=bp.dyn.DualExponV2.desc(post.num, tau_rise=tau_rise, tau_decay=tau_decay, A=A),
        out=bp.dyn.CUBA.desc(),
        post=post,
        out_label=out_label 
        )


class NormalizedDualExponCUBA(bp.Projection):
    def __init__(self, pre, post, delay, comm, tau_rise, tau_decay, out_label=None):
        super().__init__()
        
        self.proj = bp.dyn.FullProjAlignPostMg(
        pre=pre, 
        delay=delay, 
        comm=comm,
        syn=NormalizedDualExponV2.desc(post.num, tau_rise=tau_rise, tau_decay=tau_decay),
        out=bp.dyn.CUBA.desc(),
        post=post,
        out_label=out_label
        )


class NormalizedDualExponCOBA(bp.Projection):
    def __init__(self, pre, post, delay, comm, tau_rise, tau_decay, E, out_label=None):
        super().__init__()
        
        self.proj = bp.dyn.FullProjAlignPostMg(
        pre=pre, 
        delay=delay, 
        comm=comm,
        syn=NormalizedDualExponV2.desc(post.num, tau_rise=tau_rise, tau_decay=tau_decay),
        out=bp.dyn.COBA.desc(E),
        post=post,
        out_label=out_label
        )

# 这两个要不要desc以及要不要改成FullProjAlignPreDSMg还没有测试
class NMDACUBA(bp.Projection):
    def __init__(self, pre, post, delay, comm, tau_rise, tau_decay, a, out_label=None):
        super().__init__()
        
        self.proj = bp.dyn.FullProjAlignPreDSMg(
        pre=pre, 
        delay=delay, 
        comm=comm,
        syn=bp.dyn.NMDA.desc(pre.num, tau_decay=tau_decay, tau_rise=tau_rise, a=a),
        out=bp.dyn.CUBA(),
        post=post,
        out_label=out_label
        )


class NMDACOBA(bp.Projection):
    def __init__(self, pre, post, delay, comm, tau_rise, tau_decay, a, E, out_label=None):
        super().__init__()
        
        self.proj = bp.dyn.FullProjAlignPreDSMg(
        pre=pre, 
        delay=delay, 
        comm=comm,
        syn=bp.dyn.NMDA.desc(pre.num, tau_decay=tau_decay, tau_rise=tau_rise, a=a),
        out=bp.dyn.COBA(E),
        post=post,
        out_label=out_label
        )


class NMDAMgBlock(bp.Projection):
    def __init__(self, pre, post, delay, comm, tau_rise, tau_decay, a, E, cc_Mg, alpha, beta, V_offset, out_label=None):
        super().__init__()
        
        self.proj = bp.dyn.FullProjAlignPreDSMg(
        pre=pre, 
        delay=delay, 
        comm=comm,
        syn=bp.dyn.NMDA.desc(pre.num, tau_decay=tau_decay, tau_rise=tau_rise, a=a),
        out=bp.dyn.MgBlock(E=E, cc_Mg=cc_Mg, alpha=alpha, beta=beta, V_offset=V_offset),
        post=post,
        out_label=out_label
        )
# endregion


# region 神经元连接
class IJConn(bp.connect.TwoEndConnector):
    """
        Connector built from the ``pre_ids`` and ``post_ids`` connections.
        Copid from brainpy, but adjust the int32 to uint32
    """
    def __init__(self, i, j, **kwargs):
        super(IJConn, self).__init__(**kwargs)

        assert isinstance(i, (np.ndarray, bm.Array, jnp.ndarray)) and i.ndim == 1
        assert isinstance(j, (np.ndarray, bm.Array, jnp.ndarray)) and j.ndim == 1
        assert i.size == j.size

        # initialize the class via "pre_ids" and "post_ids"
        self.pre_ids = jnp.asarray(i).astype(jnp.uint32)
        self.post_ids = jnp.asarray(j).astype(jnp.uint32)
        self.max_pre = self.pre_ids.max()
        self.max_post = self.post_ids.max()

    def __call__(self, pre_size, post_size):
        super(IJConn, self).__call__(pre_size, post_size)
        if self.max_pre >= self.pre_num:
            raise bp.errors.ConnectorError(f'pre_num ({self.pre_num}) should be greater than '
                            f'the maximum id ({self.max_pre}) of self.pre_ids.')
        if self.max_post >= self.post_num:
            raise bp.errors.ConnectorError(f'post_num ({self.post_num}) should be greater than '
                            f'the maximum id ({self.max_post}) of self.post_ids.')
        return self

    def build_coo(self):
        if self.pre_num <= self.max_pre:
            raise bp.errors.ConnectorError(f'pre_num ({self.pre_num}) should be greater than '
                            f'the maximum id ({self.max_pre}) of self.pre_ids.')
        if self.post_num <= self.max_post:
            raise bp.errors.ConnectorError(f'post_num ({self.post_num}) should be greater than '
                            f'the maximum id ({self.max_post}) of self.post_ids.')
        return self.pre_ids, self.post_ids


def ij_conn(pre, post, pre_size, post_size):
    '''
    利用brainpy的bp.conn.IJConn生成conn
    '''
    # conn = bp.conn.IJConn(i=pre, j=post)
    conn = IJConn(i=pre, j=post)
    conn = conn(pre_size=pre_size, post_size=post_size)
    return conn


def ij_comm(pre, post, pre_size, post_size, weight):
    '''
    利用brainpy的bp.conn.IJConn和bp.dnn.EventCSRLinear生成comm
    '''
    conn = ij_conn(pre, post, pre_size, post_size)
    return bp.dnn.EventCSRLinear(conn, weight)


@cf.not_recommend
def csr_to_conn(csr_mat):
    '''
    将csr矩阵转换为brainpy的连接矩阵
    '''
    return bp.connect.SparseMatConn(csr_mat=csr_mat)


@cf.not_recommend
def csr_to_comm(csr_mat, weight):
    '''
    将csr矩阵转换为brainpy的comm
    '''
    conn = csr_to_conn(csr_mat)
    return bp.dnn.EventCSRLinear(conn, weight)


@cf.not_recommend
def binary_conn(row_indices, col_indices, shape):
    '''
    将二进制连接矩阵转换为brainpy的连接矩阵
    '''
    return csr_to_conn(cf.binary_csr(row_indices, col_indices, shape))


@cf.not_recommend
def edr_connection(src_pos, tar_pos, LAM):
    '''
    利用edr生成连接
    '''
    distance = cf.get_mutual_distance(src_pos, tar_pos)
    prob = np.exp(-distance/LAM)
    random_num = np.random.rand(*prob.shape)
    return csr_matrix(prob > random_num)

@cf.not_recommend
def edr_conn(src_pos, tar_pos, LAM):
    '''
    利用edr生成连接
    '''
    distance = cf.get_mutual_distance(src_pos, tar_pos)
    prob = np.exp(-distance/LAM)
    random_num = np.random.rand(*prob.shape)
    return csr_to_conn(csr_matrix(prob > random_num))
# endregion


# region 神经元网络模型运行
# 停用，并且注意这里的poisson不太对，不是delta的形式
# def get_input_i(net, inp_kwargs, dt, i):
#     '''
#     注意:
#         rate以Hz为单位,而dt以ms为单位,所以需要rate * dt / 1000
#     '''
#     inp = {}
#     for g, g_inp_kwargs in inp_kwargs.items():
#         input_type = g_inp_kwargs['type']
#         if input_type == 'constant':
#             inp[g] = g_inp_kwargs['mean']
#         if input_type == 'wiener':
#             inp[g] = np.random.randn(*(net.neuron[g].size))*g_inp_kwargs['std'] + g_inp_kwargs['mean']
#         if input_type == 'poisson':
#             p = g_inp_kwargs['rate'] * dt / 1000
#             inp[g] = np.random.binomial(1, p, size=net.neuron[g].size) * g_inp_kwargs['amplitude']
#         if input_type == 'multiple_poisson':
#             p = g_inp_kwargs['rate'] * dt / 1000
#             inp[g] = np.random.binomial(g_inp_kwargs['num'], p, size=net.neuron[g].size) * g_inp_kwargs['amplitude']
#         if input_type == 'approximate_multiple_poisson':
#             p = g_inp_kwargs['rate'] * dt / 1000
#             inp[g] = np.random.normal(loc=g_inp_kwargs['num'] * p, scale=np.sqrt(g_inp_kwargs['num'] * p * (1 - p)), size=net.neuron[g].size) * g_inp_kwargs['amplitude']
#         if input_type == 'step':
#             inp[g] = g_inp_kwargs['amplitude'] if (i*dt >= g_inp_kwargs['start'] and i*dt < g_inp_kwargs['end']) else 0
#         if input_type == 'sin':
#             inp[g] = g_inp_kwargs['amplitude'] * np.sin(2 * np.pi * g_inp_kwargs['k'] * i * dt + g_inp_kwargs['phase'])
#     return inp


# def get_inp(net, inp_kwargs, dt, step_list):
#     inp = []
#     for i in step_list:
#         inp.append(get_input_i(net, inp_kwargs, dt, i))
#     return inp

        
class InpGenerator:
    def __init__(self, net):
        self.net = net

    def __call__(self, i):
        return np.array(self.net.jit_step_run(i))


def get_inp_generator(size, inp_kwargs, dt):
    '''
    size: dict
    '''
    inp = {}
    for g, g_inp_kwargs in inp_kwargs.items():
        input_type = g_inp_kwargs['type']
        if input_type == 'dualexpon_poisson':
            pre_neuron = bp.dyn.PoissonGroup
            pre_neuron_params = {'size': g_inp_kwargs['num'], 'freqs': g_inp_kwargs['rate']}
            post_neuron = bp.dyn.Lif
            post_neuron_params = {'size': size[g], 'V_rest': 0.0, 'V_reset': 0.0, 'V_th': 1.0, 'R': 1.0, 'tau': 1.0}
            comm = bp.dnn.EventCSRLinear(bp.conn.FixedProb(g_inp_kwargs['prob'], pre=g_inp_kwargs['num'], post=size[g]), weight=g_inp_kwargs['weight'])
            synapse = NormalizedDualExponCUBA
            synapse_params = {'tau_rise': g_inp_kwargs['tau_rise'], 'tau_decay': g_inp_kwargs['tau_decay'], 'delay': 0.0}
            net = NetForConstructInp(pre_neuron, post_neuron, pre_neuron_params, post_neuron_params, synapse, synapse_params, comm)

            inp[g] = InpGenerator(net)
        if input_type == 'expon_poisson':
            pre_neuron = bp.dyn.PoissonGroup
            pre_neuron_params = {'size': g_inp_kwargs['num'], 'freqs': g_inp_kwargs['rate']}
            post_neuron = bp.dyn.Lif
            post_neuron_params = {'size': size[g], 'V_rest': 0.0, 'V_reset': 0.0, 'V_th': 1.0, 'R': 1.0, 'tau': 1.0}
            comm = bp.dnn.EventCSRLinear(bp.conn.FixedProb(g_inp_kwargs['prob'], pre=g_inp_kwargs['num'], post=size[g]), weight=g_inp_kwargs['weight'])
            synapse = NormalizedExponCUBA
            synapse_params = {'tau': g_inp_kwargs['tau'], 'delay': 0.0}
            net = NetForConstructInp(pre_neuron, post_neuron, pre_neuron_params, post_neuron_params, synapse, synapse_params, comm)

            inp[g] = InpGenerator(net)
        if input_type == 'constant':
            class ConstantInp:
                def __init__(self, mean):
                    self.mean = mean

                def __call__(self, i):
                    return self.mean
            
            inp[g] = ConstantInp(g_inp_kwargs['mean'])
        if input_type == 'wiener':
            class WienerInp:
                def __init__(self, mean, std):
                    self.mean = mean
                    self.std = std

                def __call__(self, i):
                    return np.random.randn(*(size[g]))*self.std + self.mean
            
            inp[g] = WienerInp(g_inp_kwargs['mean'], g_inp_kwargs['std'])
    return inp
# endregion


# region 神经元网络模型
class NetForConstructInp(bp.DynSysGroup):
    def __init__(self, pre_neuron, post_neuron, pre_neuron_params, post_neuron_params, synapse, synapse_params, comm):
        super().__init__()

        self.pre_neuron_params = pre_neuron_params.copy()
        self.post_neuron_params = post_neuron_params.copy()
        self.synapse_params = synapse_params.copy()

        # neurons
        self.pre = pre_neuron(**self.pre_neuron_params)
        self.post = post_neuron(**self.post_neuron_params)

        # synapses
        self.synapse = synapse(pre=self.pre, post=self.post, comm=comm, **self.synapse_params)
        
    def update(self):
        self.pre()
        self.synapse()
        self.post()

        # monitor
        return self.post.sum_current_inputs(self.post.V)

@cf.deprecated
class EINet(bp.DynSysGroup):
    def __init__(self, E_neuron, I_neuron, E_params, I_params, E2E_synapse, E2I_synapse, I2E_synapse, I2I_synapse, E2E_synapse_params, E2I_synapse_params, I2E_synapse_params, I2I_synapse_params, E2E_comm, E2I_comm, I2E_comm, I2I_comm):
        super().__init__()

        self.E_params = E_params.copy()
        self.I_params = I_params.copy()
        self.E2E_synapse_params = E2E_synapse_params.copy()
        self.E2I_synapse_params = E2I_synapse_params.copy()
        self.I2E_synapse_params = I2E_synapse_params.copy()
        self.I2I_synapse_params = I2I_synapse_params.copy()

        # neurons
        self.E = E_neuron(**self.E_params)
        self.I = I_neuron(**self.I_params)

        # synapses
        self.E2E = E2E_synapse(pre=self.E, post=self.E, comm=E2E_comm, **self.E2E_synapse_params)
        self.E2I = E2I_synapse(pre=self.E, post=self.I, comm=E2I_comm, **self.E2I_synapse_params)
        self.I2E = I2E_synapse(pre=self.I, post=self.E, comm=I2E_comm, **self.I2E_synapse_params)
        self.I2I = I2I_synapse(pre=self.I, post=self.I, comm=I2I_comm, **self.I2I_synapse_params)
        
    def update(self, E_inp, I_inp):
        self.E2E()
        self.E2I()
        self.I2E()
        self.I2I()
        self.E(E_inp)
        self.I(I_inp)

        # monitor
        return self.E.spike, self.I.spike, self.E.V, self.I.V


class MultiGroupNet(bp.DynSysGroup):
    """
    多个gruop的神经元网络模型。比如EI网络。

    属性:
        neuron (dict): 包含每个组神经元对象的字典。
        synapse (dict): 包含连接神经元组的突触对象的字典。
    """
    def __init__(self, neuron, synapse, inp_neuron, inp_synapse, neuron_params, synapse_params, inp_neuron_params, inp_synapse_params, comm, inp_comm, update_output=None, seperate_EI_inp=True, seperate_ext_inp=False):
        """
        初始化 MultiGroupNet 实例。

        参数:
            neuron (dict): 包含每个组的神经元类型的字典。
            synapse (dict): 包含每个连接的突触类型的字典。(必须是{pre}2{post}的形式)
            inp_neuron (dict): 包含输入神经元类型的字典。
            inp_synapse (dict): 包含输入突触类型的字典。
            neuron_params (dict): 包含神经元初始化参数的字典。
            synapse_params (dict): 包含突触初始化参数的字典。
            inp_neuron_params (dict): 包含输入神经元初始化参数的字典。
            inp_synapse_params (dict): 包含输入突触初始化参数的字典。
            comm (dict): 包含组之间通信参数的字典。
            inp_comm (dict): 包含输入组和其他组通信参数的字典。
            update_output (list): 包含每个组输出的列表。默认为None。
            seperate_EI_inp (bool): 是否分开输入。默认为True。(假如True,则需要写好label为E和I,并使用EI的neuron模型比如EILifRef;假如为False则可以使用原版并无需写label)
            seperate_ext_inp (bool): 是否分开外部输入。默认为False。(假如False,则需要写好label为E,因为外界输入的label都是E;假如为True,代码尚未实现)
        """
        super().__init__()

        self.group = neuron.keys()
        self.inp_group = inp_neuron.keys()

        self.neuron = bm.NodeDict()
        for g in self.group:
            self.neuron[g] = neuron[g](**neuron_params[g])

        self.inp_neuron = bm.NodeDict()
        for g in inp_neuron.keys():
            self.inp_neuron[g] = inp_neuron[g](**inp_neuron_params[g])

        self.synapse = bm.NodeDict()
        for syn_type in synapse.keys():
            s, t = syn_type.split('2')
            self.synapse[f'{s}2{t}'] = synapse[f'{s}2{t}'](pre=self.neuron[s], post=self.neuron[t], comm=comm[f'{s}2{t}'], **synapse_params[f'{s}2{t}'])
        
        self.inp_synapse = bm.NodeDict()
        for syn_type in inp_synapse.keys():
            s, t = syn_type.split('2')
            self.inp_synapse[f'{s}2{t}'] = inp_synapse[f'{s}2{t}'](pre=self.inp_neuron[s], post=self.neuron[t], comm=inp_comm[f'{s}2{t}'], **inp_synapse_params[f'{s}2{t}'])

        # 设定update_output
        if update_output is None:
            self.update_output = ['spike', 'V']
        else:
            self.update_output = update_output.copy()
        
        self.seperate_EI_inp = seperate_EI_inp
        self.seperate_ext_inp = seperate_ext_inp

    def update(self, inp):
        """
        根据输入更新网络状态。

        此方法根据提供的输入更新网络中神经元和突触的状态。

        参数:
            inp (dict): 包含每个神经元组的输入的字典。

        返回:
            dict: 包含每个神经元组的输出的字典。

        注意:
            current为某个neuron_group的输入电流,而不是输出
        """
        for s in self.synapse.keys():
            self.synapse[s]()
        for s in self.inp_synapse.keys():
            self.inp_synapse[s]()
        for g in self.neuron.keys():
            if g in inp.keys():
                self.neuron[g](inp[g])
            else:
                self.neuron[g]()
        for g in self.inp_neuron.keys():
            self.inp_neuron[g]()

        
        result = {}
        if 'spike' in self.update_output:
            result['spike'] = {g: self.neuron[g].spike.value for g in self.neuron.keys()}
        if 'V' in self.update_output:
            result['V'] = {g: self.neuron[g].V.value for g in self.neuron.keys()}
        # if 'internal_current' in self.update_output:
        #     if self.seperate_EI_inp:
        #         result['internal_current'] = {}
        #         for s in ['E', 'I']:
        #             for t in self.group:
        #                 result['internal_current'][f'{s}2{t}'] = self.neuron[t].sum_current_inputs(self.neuron[t].V, label=s)
        #     else:
        #         result['internal_current'] = {t: self.neuron[t].sum_current_inputs(self.neuron[t].V) for t in self.neuron.keys()}
        # if 'external_current' in self.update_output:
        #     result['external_current'] = inp
        if 'current' in self.update_output:
            if self.seperate_ext_inp:
                raise NotImplementedError
            else:
                if self.seperate_EI_inp:
                    result['current'] = {}
                    for s in ['E', 'I']:
                        for t in self.group:
                            if t in inp.keys():
                                result['current'][f'{s}2{t}'] = self.neuron[t].sum_current_inputs(self.neuron[t].V, label=s) + inp[t]
                            else:
                                result['current'][f'{s}2{t}'] = self.neuron[t].sum_current_inputs(self.neuron[t].V, label=s)
                else:
                    result['current'] = {t: self.neuron[t].sum_current_inputs(self.neuron[t].V) for t in self.neuron.keys()}
        return result


class MultiNet(bp.DynSysGroup):
    def __init__(self, neuron, synapse, inp_neuron, inp_synapse, neuron_params, synapse_params, inp_neuron_params, inp_synapse_params, comm, inp_comm):
        """
        参数:
            neuron (dict): 包含每个组的神经元类型的字典。
            synapse (dict): 包含每个连接的突触类型的字典。(key必须是(s, t, name)的元组,如果不需要name,则name设置为None或者空字符串)
            inp_neuron (dict): 包含输入神经元类型的字典。
            inp_synapse (dict): 包含输入突触类型的字典。
            neuron_params (dict): 包含神经元初始化参数的字典。
            synapse_params (dict): 包含突触初始化参数的字典。
            inp_neuron_params (dict): 包含输入神经元初始化参数的字典。
            inp_synapse_params (dict): 包含输入突触初始化参数的字典。
            comm (dict): 包含组之间通信参数的字典。
            inp_comm (dict): 包含输入组和其他组通信参数的字典。
        """
        super().__init__()

        self.group = neuron.keys()
        self.inp_group = inp_neuron.keys()

        for g in self.group:
            setattr(self, g, neuron[g](**neuron_params[g]))

        for inp_g in self.inp_group:
            setattr(self, inp_g, inp_neuron[inp_g](**inp_neuron_params[inp_g]))
        
        for syn_type in synapse.keys():
            s, t, name = syn_type
            # cf.print_title(f'{s}2{t} {name} synapse')
            setattr(self, cf.concat_str([f'{s}2{t}', name]), synapse[syn_type](pre=getattr(self, s), post=getattr(self, t), comm=comm[syn_type], **synapse_params[syn_type]))

        for inp_syn_type in inp_synapse.keys():
            s, t, name = inp_syn_type
            # cf.print_title(f'{s}2{t} {name} synapse')
            setattr(self, cf.concat_str([f'{s}2{t}', name]), inp_synapse[inp_syn_type](pre=getattr(self, s), post=getattr(self, t), comm=inp_comm[inp_syn_type], **inp_synapse_params[inp_syn_type]))


@cf.deprecated
class MultiAreaEINet(EINet):
    def __init__(self, E_neuron, I_neuron, E_params, I_params, E2E_synapse, E2I_synapse, I2E_synapse, I2I_synapse, E2E_synapse_params, E2I_synapse_params, I2E_synapse_params, I2I_synapse_params, E2E_comm, E2I_comm, I2E_comm, I2I_comm, E_idx_to_area, I_idx_to_area):
        super().__init__(E_neuron, I_neuron, E_params, I_params, E2E_synapse, E2I_synapse, I2E_synapse, I2I_synapse, E2E_synapse_params, E2I_synapse_params, I2E_synapse_params, I2I_synapse_params, E2E_comm, E2I_comm, I2E_comm, I2I_comm)
    
    def area_fr(self):
        # 计算各个区域的firing rate
        pass


@cf.deprecated
class SpatialEINet(EINet):
    def __init__(self, E_neuron, I_neuron, E_params, I_params, E2E_synapse, E2I_synapse, I2E_synapse, I2I_synapse, E2E_synapse_params, E2I_synapse_params, I2E_synapse_params, I2I_synapse_params, E2E_comm, E2I_comm, I2E_comm, I2I_comm, E_pos, I_pos):
        super().__init__(E_neuron, I_neuron, E_params, I_params, E2E_synapse, E2I_synapse, I2E_synapse, I2I_synapse, E2E_synapse_params, E2I_synapse_params, I2E_synapse_params, I2I_synapse_params, E2E_comm, E2I_comm, I2E_comm, I2I_comm)

        self.E_pos = E_pos
        self.I_pos = I_pos


# class MultiAreaSpatialEINet(MultiAreaEINet, SpatialEINet):
#     def __init__(self, E_neuron, I_neuron, E_params, I_params, E2E_synapse, E2I_synapse, I2E_synapse, I2I_synapse, E2E_synapse_params, E2I_synapse_params, I2E_synapse_params, I2I_synapse_params, E2E_comm, E2I_comm, I2E_comm, I2I_comm, E_pos, I_pos, E_idx_to_area, I_idx_to_area):
#         MultiAreaEINet.__init__(self, E_neuron, I_neuron, E_params, I_params, E2E_synapse, E2I_synapse, I2E_synapse, I2I_synapse, E2E_synapse_params, E2I_synapse_params, I2E_synapse_params, I2I_synapse_params, E2E_comm, E2I_comm, I2E_comm, I2I_comm, E_idx_to_area, I_idx_to_area)
#         SpatialEINet.__init__(self, E_neuron, I_neuron, E_params, I_params, E2E_synapse, E2I_synapse, I2E_synapse, I2I_synapse, E2E_synapse_params, E2I_synapse_params, I2E_synapse_params, I2I_synapse_params, E2E_comm, E2I_comm, I2E_comm, I2I_comm, E_pos, I_pos)
# endregion


# region test
def test():
    bp.math.set_platform('cpu')
    # EI_ratio = 4
    EI_ratio = 8
    E2E_weight = 1
    E2I_weight = 1
    I2E_weight = -4
    I2I_weight = -4
    # E_size = 4*81
    E_size = 8*27
    I_size = E_size // EI_ratio

    E_params = {'size': E_size, 'V_th': 20.0, 'V_reset': -5.0, 'V_rest':0., 'tau_ref': 5.0, 'R': 1.0, 'tau': 10.0}
    I_params = {'size': I_size, 'V_th': 20.0, 'V_reset': -5.0, 'V_rest':0., 'tau_ref': 5.0, 'R': 1.0, 'tau': 10.0}
    E2E_synapse_params = {'delay': 0}
    E2I_synapse_params = {'delay': 0}
    I2E_synapse_params = {'delay': 0}
    I2I_synapse_params = {'delay': 0}

    E_inp_mean = 10
    E_inp_std = 30
    I_inp_mean = 10
    I_inp_std = 30

    # E_pos = np.meshgrid(np.linspace(0, 1, int(np.sqrt(E_size))) + np.random.rand(int(np.sqrt(E_size))) / E_size, np.linspace(0, 1, int(np.sqrt(E_size))) + np.random.rand(int(np.sqrt(E_size))) / E_size)
    # E_pos = np.stack([E_pos[0].flatten(), E_pos[1].flatten()], axis=1)
    # I_pos = np.meshgrid(np.linspace(0, 1, int(np.sqrt(I_size))), np.linspace(0, 1, int(np.sqrt(I_size))))
    # I_pos = np.stack([I_pos[0].flatten(), I_pos[1].flatten()], axis=1)
    E_pos = np.meshgrid(np.linspace(0, 1, round(E_size**(1/3))) + np.random.rand(round(E_size**(1/3))) / E_size, np.linspace(0, 1, round(E_size**(1/3))) + np.random.rand(round(E_size**(1/3))) / E_size, np.linspace(0, 1, round(E_size**(1/3))) + np.random.rand(round(E_size**(1/3))) / E_size)
    E_pos = np.stack([E_pos[0].flatten(), E_pos[1].flatten(), E_pos[2].flatten()], axis=1)
    print(E_pos.shape)
    I_pos = np.meshgrid(np.linspace(0, 1, round(I_size**(1/3))), np.linspace(0, 1, round(I_size**(1/3))), np.linspace(0, 1, round(I_size**(1/3))) + np.random.rand(round(I_size**(1/3))) / I_size)
    I_pos = np.stack([I_pos[0].flatten(), I_pos[1].flatten(), I_pos[2].flatten()], axis=1)

    dt = 1.
    bm.set_dt(dt)

    conn_num = 100
    LAM = 0.3

    # E2E_connection = np.random.rand(E_size, E_size)
    # E2E_connection = E2E_connection > 0.95
    # E2E_connection = csr_matrix(E2E_connection)
    E2E_connection = edr_connection(E_pos, E_pos, LAM)
    E2E_conn = csr_to_conn(E2E_connection)
    E2E_comm = bp.dnn.EventCSRLinear(conn=E2E_conn, weight=E2E_weight)

    # E2I_connection = np.zeros((E_size, I_size))
    # E2I_connection[1, 0] = 1
    # E2I_connection = csr_matrix(E2I_connection)
    E2I_connection = edr_connection(E_pos, I_pos, LAM)
    E2I_conn = csr_to_conn(E2I_connection)
    E2I_comm = bp.dnn.EventCSRLinear(conn=E2I_conn, weight=E2I_weight)

    # I2E_connection = np.zeros((I_size, E_size))
    # I2E_connection[1, 0] = 1
    # I2E_connection = csr_matrix(I2E_connection)
    I2E_connection = edr_connection(I_pos, E_pos, LAM)
    I2E_conn = csr_to_conn(I2E_connection)
    I2E_comm = bp.dnn.EventCSRLinear(conn=I2E_conn, weight=I2E_weight)

    # I2I_connection = np.zeros((I_size, I_size))
    # I2I_connection[1, 0] = 1
    # I2I_connection = csr_matrix(I2I_connection)
    I2I_connection = edr_connection(I_pos, I_pos, LAM)
    I2I_conn = csr_to_conn(I2I_connection)
    I2I_comm = bp.dnn.EventCSRLinear(conn=I2I_conn, weight=I2I_weight)

    EI_net = SpatialEINet(E_neuron=bp.dyn.LifRef, I_neuron=bp.dyn.LifRef, E_params=E_params, I_params=I_params, E2E_synapse=bp.dyn.FullProjDelta, E2I_synapse=bp.dyn.FullProjDelta, I2E_synapse=bp.dyn.FullProjDelta, I2I_synapse=bp.dyn.FullProjDelta, E2E_synapse_params=E2E_synapse_params, E2I_synapse_params=E2I_synapse_params, I2E_synapse_params=I2E_synapse_params, I2I_synapse_params=I2I_synapse_params, E2E_comm=E2E_comm, E2I_comm=E2I_comm, I2E_comm=I2E_comm, I2I_comm=I2I_comm, E_pos=E_pos, I_pos=I_pos)

    # E_inp_mean = np.ones(E_size)*0
    # E_inp_mean[1] = 30
    run_func = get_run_func(EI_net, {'mean': E_inp_mean, 'std': E_inp_std}, {'mean': I_inp_mean, 'std': I_inp_std}, E_size, I_size, input_type='wiener')


    indices = np.arange(100)
    ts = indices * bm.get_dt()
    E_spike, I_spike, E_V, I_V = bm.for_loop(run_func, indices, progress_bar=True)

    casual_spike_video([5], None, E_spike, E_pos, I_spike, I_pos, E2E_connection, E2I_connection, I2E_connection, I2I_connection, 0, dt, './video', 'casual_spike_video', scatter_size=(cf.MARKER_SIZE/3)**2, faint_num=3, legend_loc='upper left', bbox_to_anchor=(1, 1), elev_list=[cf.ELEV], azim_list=[cf.AZIM], margin={'left': 0.1, 'right':0.75, 'bottom' : 0.1, 'top': 0.75}, fig_ax_kwargs={}, scatter_kwargs={}, set_ax_kwargs={}, save_fig_kwargs={}, video_kwargs={'frame_rate': 5})

    spike_video(E_spike, E_pos, I_spike, I_pos, dt, './video', 'spike_video', scatter_size=(cf.MARKER_SIZE/3)**2, faint_num=3, legend_loc='upper left', bbox_to_anchor=(1, 1), elev_list=[cf.ELEV], azim_list=[cf.AZIM], margin={'left': 0.1, 'right':0.75, 'bottom' : 0.1, 'top': 0.75}, fig_ax_kwargs={}, scatter_kwargs={}, set_ax_kwargs={}, save_fig_kwargs={}, video_kwargs={'frame_rate': 5})
# endregion