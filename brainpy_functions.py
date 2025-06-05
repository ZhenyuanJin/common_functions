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
import abc
from collections import defaultdict
from tqdm import tqdm
from functools import partial


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
from brainpy._src import connect, initialize as init
from brainpy._src.integrators.ode.generic import odeint
from brainpy._src.integrators.joint_eq import JointEq
from brainpy._src.initialize import parameter
from brainpy._src.context import share
from brainpy.types import ArrayType
from brainpy._src.dynsys import DynamicalSystem, DynView
from brainpy._src.math.object_transform.base import StateLoadResult
# 脑区处理库
# import networkx as nx
# import pointpats
# import shapely.geometry


# 自定义库
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import common_functions as cf
# endregion


# region 定义默认参数
E_COLOR = cf.RED
I_COLOR = cf.BLUE
cf.print_title('brainpy version: {}'.format(bp.__version__), char='*')
# endregion


# region brainpy使用说明
def brainpy_data_structure():
    '''
    所有数据按照(T, N)的形式存储,其中T表示时间点的数量,N表示神经元的数量
    当training时,要注意数据维度变为(B, T, N),其中B表示batch size
    '''
    pass


def brainpy_time_axis():
    return 0


def brainpy_neuron_axis():
    return 1


def brainpy_unit(physical_quantity):
    '''
    Note that pA * GOhm = mV, thus consistent with for example \tau * dV/dt = - ( V - V_rest ) + R * I
    '''
    if physical_quantity == 'V':
        return 'mV (10^-3 V)'
    if physical_quantity == 'I':
        return 'pA (10^-12 A)'
    if physical_quantity == 'R':
        return 'GOhm (10^9 Ohm)'
    if physical_quantity == 'g':
        return 'nS (10^-9 S)'
    if physical_quantity == 'tau':
        return 'ms (10^-3 s)'
# endregion


# region gpu设置
def set_to_gpu(pre_allocate=True):
    """
    Set the platform to GPU and enable pre-allocation of GPU memory.
    """
    bm.set_platform('gpu')
    if pre_allocate:
        enable_gpu_memory_preallocation()


def enable_gpu_memory_preallocation(percent=0.95):
    """
    Enable pre-allocating the GPU memory.

    Adapted from https://brainpy.readthedocs.io/en/latest/_modules/brainpy/_src/math/environment.html#enable_gpu_memory_preallocation
    """
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'true'
    os.environ.pop('XLA_PYTHON_CLIENT_ALLOCATOR', None)
    gpu_memory_preallocation(percent)


def gpu_memory_preallocation(percent: float):
    """GPU memory allocation.

    If preallocation is enabled, this makes JAX preallocate ``percent`` of the total GPU memory,
    instead of the default 75%. Lowering the amount preallocated can fix OOMs that occur when the JAX program starts.
    """
    assert 0. <= percent < 1., f'GPU memory preallocation must be in [0., 1.]. But we got {percent}.'
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = str(percent)
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
    params = {'E': E_params, 'I': I_params}
    return params


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


def get_neuron_num_from_data(data):
    return data.shape[1]


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


def get_time_point_from_data(data):
    return data.shape[0]
# endregion


# region 数据类型转换
def dict_to_np_dict(d):
    '''
    将包含非np数组的dict内的元素全部转换为np数组
    '''
    np_dict = {}
    for k, v in d.items():
        np_dict[k] = np.array(v)
    return np_dict
# endregion


# region 神经元放电性质计算
def spike_to_fr(spike, width, dt, neuron_idx=None, **kwargs):
    '''
    修改bp.measure.firing_rate使得一维数组的spike也能够计算firing rate(但是使用方式是设定neuron_idx而不是直接传入一个一维的spike)

    注意:
    如果需要对一维的spike算,先np.reshape
    adjusted_spike = np.reshape(spike, (spike.shape[0], -1))
    '''
    partial_spike = neuron_idx_data(spike, neuron_idx, keep_size=True)
    return bp.measure.firing_rate(partial_spike, width, dt, **kwargs)


def get_spike_acf(spike, dt, nlags, neuron_idx=None, average=True, **kwargs):
    '''
    计算spike的自相关函数
    '''
    # partial_spike = neuron_idx_data(spike, neuron_idx, keep_size=True)
    # float_spike = np.array(partial_spike).astype(float)
    # lag_times, multi_acf = cf.get_multi_acf(float_spike.T, T=dt, nlags=nlags, **kwargs)
    # if average:
    #     return lag_times, np.nanmean(multi_acf, axis=0)
    # else:
    #     return lag_times, multi_acf.T
    raise NotImplementedError('get_spike_acf is not implemented yet, please use get_single_neuron_acf instead.')


def get_neuron_data_acf(neuron_data, dt, nlags, neuron_idx=None, process_num=1, **kwargs):
    '''
    计算单个神经元级别的自相关函数

    由于brainpy的数据格式是(T, N),所以需要转置再输入到acf函数中
    对于multi_acf,其shape是(N, nlags)
    '''
    partial_neuron_data = neuron_idx_data(neuron_data, neuron_idx, keep_size=True)
    lag_times, multi_acf = cf.get_multi_acf(partial_neuron_data.T, T=dt, nlags=nlags, process_num=process_num, **kwargs)
    return lag_times, multi_acf


def get_neuron_data_acovf(neuron_data, dt, nlags, neuron_idx=None, process_num=1, **kwargs):
    '''
    计算单个神经元级别的自协方差函数
    '''
    partial_neuron_data = neuron_idx_data(neuron_data, neuron_idx, keep_size=True)
    lag_times, multi_acovf = cf.get_multi_acovf(partial_neuron_data.T, T=dt, nlags=nlags, process_num=process_num, **kwargs)
    return lag_times, multi_acovf


def get_neuron_data_ccf(neuron_data_x, neuron_data_y, dt, nlags, neuron_idx_x=None, neuron_idx_y=None, process_num=1, **kwargs):
    '''
    计算神经元数据的互相关函数
    '''
    partial_neuron_data_x = neuron_idx_data(neuron_data_x, neuron_idx_x, keep_size=True)
    partial_neuron_data_y = neuron_idx_data(neuron_data_y, neuron_idx_y, keep_size=True)
    lag_times, multi_ccf = cf.get_multi_ccf(partial_neuron_data_x.T, partial_neuron_data_y.T, T=dt, nlags=nlags, process_num=process_num, **kwargs)
    return lag_times, multi_ccf


def get_neuron_data_ccovf(neuron_data_x, neuron_data_y, dt, nlags, neuron_idx_x=None, neuron_idx_y=None, process_num=1, **kwargs):
    '''
    计算神经元数据的互协方差函数
    '''
    partial_neuron_data_x = neuron_idx_data(neuron_data_x, neuron_idx_x, keep_size=True)
    partial_neuron_data_y = neuron_idx_data(neuron_data_y, neuron_idx_y, keep_size=True)
    lag_times, multi_ccovf = cf.get_multi_ccovf(partial_neuron_data_x.T, partial_neuron_data_y.T, T=dt, nlags=nlags, process_num=process_num, **kwargs)
    return lag_times, multi_ccovf


def spike_to_fr_acf(spike, width, dt, nlags, neuron_idx=None, spike_to_fr_kwargs=None, **kwargs):
    '''
    计算spike的firing rate的自相关函数,注意,计算fr的过程自动平均了neuron_idx中的神经元。
    '''
    if spike_to_fr_kwargs is None:
        spike_to_fr_kwargs = {}
    fr = spike_to_fr(spike, width, dt, neuron_idx, **spike_to_fr_kwargs)
    return cf.get_acf(fr, T=dt, nlags=nlags, **kwargs)


def _get_fr_each_neuron(spike, width, dt):
    '''
    对每个神经元的spike进行firing rate计算
    '''
    width1 = int(width / 2 / dt) * 2 + 1
    window = np.ones(width1) * 1000 / width
    return np.apply_along_axis(lambda m: np.convolve(m, window, mode='same'), axis=0, arr=spike)


def get_fr_each_neuron(spike, width, dt, process_num=1):
    spike_list = cf.split_array(spike, axis=1, n=process_num)
    r = cf.multi_process_list_for(process_num=process_num, func=_get_fr_each_neuron, kwargs={'width': width, 'dt': dt}, for_list=spike_list, for_idx_name='spike')
    r = np.concatenate(r, axis=1)
    return r


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


def single_exp(x, amp, tau):
    return amp * np.exp(-x / tau)


def single_exp_fit(lag_times, acf, fix_amp_value=None):
    try:
        if fix_amp_value is None:
            f = single_exp
        else:
            f = partial(single_exp, amp=fix_amp_value)
        single_popt, single_pcov, single_error = cf.get_curvefit(lag_times, acf, f)
        results = {}
        if fix_amp_value is not None:
            single_popt = [fix_amp_value, single_popt[0]]
        results['amp'] = single_popt[0]
        results['tau'] = single_popt[1]
        results['error'] = single_error
        results['cov'] = single_pcov
    except:
        results = {}
        results['amp'] = np.nan
        results['tau'] = np.nan
        results['error'] = np.nan
        results['cov'] = np.nan
    return results
# endregion


# region 神经元数据预处理
def sort_neuron_data(neuron_data, sort_measure, ascending=True):
    '''
    对神经元数据进行排序
    '''
    if ascending:
        return neuron_data[:, np.argsort(sort_measure)]
    else:
        return neuron_data[:, np.argsort(sort_measure)[::-1]]


def get_spike_sort_measure_by_corr_and_fr(spike, fr, fr_threshold=0.1, fallback_value=-10.0):
    '''
    计算用于排序神经元的 sort_measure,根据
    - 单个神经元的 spike 与 firing rate 的相关性
    - 神经元平均 firing rate 是否大于阈值
    
    参数：
    - spike: ndarray, shape (T, N),T为时间点,N为神经元数
    - fr: ndarray, shape (T,),对应时间段的整体 firing rate
    - fr_threshold: float,低于此阈值的神经元会被惩罚
    - fallback_value: float,对于无法计算相关性(nan)的神经元赋值为该值

    返回：
    - sort_measure: ndarray, shape (N,),每个神经元的排序度量值
    '''
    N = spike.shape[1]

    # 计算每个神经元与 firing rate 的相关性
    spike_fr_corr = np.array([
        np.corrcoef(spike[:, i], fr)[0, 1]
        for i in range(N)
    ])

    # 计算平均 firing rate 是否超过阈值
    fr_mask = (np.mean(spike, axis=0) > fr_threshold).astype(float)

    # 构造排序度量
    sort_measure = (spike_fr_corr + 1) * fr_mask

    # 将 NaN 替换为 fallback 值
    sort_measure[np.isnan(sort_measure)] = fallback_value

    return sort_measure
# endregion


# region 作图函数
def visualize_one_second(ax, x_start, y=None, fontsize=cf.LEGEND_SIZE, color=cf.BLACK):
    if y is None:
        y = ax.get_ylim()[1] * 0.9
    one_second_x = [x_start, x_start + 1000.]
    one_second_y = [y, y]
    cf.plt_line(ax, one_second_x, one_second_y, color=color)
    cf.add_text(ax, '1s', x=np.mean(one_second_x), y=np.mean(one_second_y), fontsize=fontsize, color=color, va='bottom', ha='center', transform=ax.transData)


def _get_indices_xy_for_spike_raster(ts, spike):
    # Get the indices of the spikes
    if spike.ndim == 2:
        spike_timestep, neuron_indices = np.where(spike)
    else:
        print(f'Error: spike should be 2D array, but got {spike.ndim}D array.')

    results = {}
    results['spike_timestep'] = spike_timestep
    results['neuron_indices'] = neuron_indices
    results['x'] = ts[spike_timestep]
    results['y'] = neuron_indices
    return results


def raster_plot(ax, ts, spike, color=cf.BLUE, xlabel='time (ms)', ylabel='neuron index', label=None, xlim=None, ylim=None, scatter_kwargs=None, set_ax_kwargs=None):
    if ylim is None:
        # spike 的 ylim 要设置的严格才会比较好看
        ylim = [0, spike.shape[1]]
    scatter_kwargs = cf.update_dict({'s': cf.MARKER_SIZE / 2}, scatter_kwargs)
    set_ax_kwargs = cf.update_dict({'adjust_tick_size': False}, set_ax_kwargs)
    
    r = _get_indices_xy_for_spike_raster(ts, spike)

    cf.plt_scatter(ax, r['x'], r['y'], color=color, label=label, clip_on=False, xlim=xlim, ylim=ylim, **scatter_kwargs)

    cf.set_ax(ax, xlabel, ylabel, xlim=xlim, ylim=ylim, **set_ax_kwargs)


def fr_scale_raster_plot(ax, ts, spike, fr, cmap=cf.DENSITY_CMAP, xlabel='time (ms)', ylabel='neuron index', label=None, xlim=None, ylim=None, scatter_kwargs=None, set_ax_kwargs=None):
    scatter_kwargs = cf.update_dict({'s': cf.MARKER_SIZE / 2}, scatter_kwargs)
    scatter_kwargs = cf.update_dict(scatter_kwargs, {'xlim': xlim, 'ylim': ylim})
    set_ax_kwargs = cf.update_dict({'adjust_tick_size': False}, set_ax_kwargs)
    
    r = _get_indices_xy_for_spike_raster(ts, spike, xlim=xlim, ylim=ylim)
    
    # Get the color based on firing rate
    c = fr[r['spike_timestep']]

    cf.plt_colorful_scatter(ax, r['x'], r['y'], c, cmap=cmap, label=label, scatter_kwargs={'clip_on': False}, **scatter_kwargs)

    cf.set_ax(ax, xlabel, ylabel, xlim=xlim, ylim=ylim, **set_ax_kwargs)


def EI_raster_plot(ax, ts, E_spike, I_spike, E_color=E_COLOR, I_color=I_COLOR, xlabel='time (ms)', ylabel='', E_label=None, I_label=None, E_xlim=None, E_ylim=None, I_xlim=None, I_ylim=None, split_ax_kwargs=None, scatter_kwargs=None, set_ax_kwargs=None):
    split_ax_kwargs = cf.update_dict({'nrows': 2, 'sharex': True, 'hspace': cf.SIDE_PAD*3, 'height_ratios': [E_spike.shape[1], I_spike.shape[1]]}, split_ax_kwargs)
    
    ax_E, ax_I = cf.split_ax(ax, **split_ax_kwargs)
    raster_plot(ax_E, ts, E_spike, color=E_color, xlabel='', ylabel='E '+ylabel, label=E_label, xlim=E_xlim, ylim=E_ylim, scatter_kwargs=scatter_kwargs, set_ax_kwargs=set_ax_kwargs)
    raster_plot(ax_I, ts, I_spike, color=I_color, xlabel=xlabel, ylabel='I '+ylabel, label=I_label,  xlim=I_xlim, ylim=I_ylim, scatter_kwargs=scatter_kwargs, set_ax_kwargs=set_ax_kwargs)

    cf.rm_ax_spine(ax_E, 'bottom')
    cf.rm_ax_tick(ax_E, 'x')
    cf.rm_ax_ticklabel(ax_E, 'x')
    cf.align_label([ax_E, ax_I], 'y')
    return ax_E, ax_I


def template_line_plot(ax, x, y, color=cf.BLUE, xlabel='x', ylabel='y', label=None, line_kwargs=None, set_ax_kwargs=None):
    if line_kwargs is None:
        line_kwargs = {}
    if set_ax_kwargs is None:
        set_ax_kwargs = {}

    cf.plt_line(ax, x, y, color=color, label=label, **line_kwargs)
    cf.set_ax(ax, xlabel, ylabel, **set_ax_kwargs)


def template_EI_line_plot(ax, x, E_y, I_y, xlabel='x', ylabel='y', E_label='E', I_label='I', line_kwargs=None, set_ax_kwargs=None):
    template_line_plot(ax, x, E_y, color=E_COLOR, xlabel=xlabel, ylabel='E '+ylabel, label=E_label, line_kwargs=line_kwargs, set_ax_kwargs=set_ax_kwargs)
    template_line_plot(ax, x, I_y, color=I_COLOR, xlabel=xlabel, ylabel='I '+ylabel, label=I_label, line_kwargs=line_kwargs, set_ax_kwargs=set_ax_kwargs)


def input_current_plot(ax, ts, current, color=cf.BLUE, xlabel='time (ms)', ylabel='input current (nA)', label=None, line_kwargs=None, set_ax_kwargs=None):
    template_line_plot(ax, ts, current, color=color, xlabel=xlabel, ylabel=ylabel, label=label, line_kwargs=line_kwargs, set_ax_kwargs=set_ax_kwargs)


def seperate_ext_input_current_plot(ax, ts, internal_current, external_current, internal_color=cf.BLUE, external_color=cf.GREEN, total_color=cf.BLACK, xlabel='time (ms)', ylabel='input current (nA)', internal_label='internal', external_label='external', total_label='total', line_kwargs=None, set_ax_kwargs=None):
    input_current_plot(ax, ts, internal_current, color=internal_color, xlabel=xlabel, ylabel=ylabel, label=internal_label, line_kwargs=line_kwargs, set_ax_kwargs=set_ax_kwargs)
    input_current_plot(ax, ts, external_current, color=external_color, xlabel=xlabel, ylabel=ylabel, label=external_label, line_kwargs=line_kwargs, set_ax_kwargs=set_ax_kwargs)
    input_current_plot(ax, ts, internal_current + external_current, color=total_color, xlabel=xlabel, ylabel=ylabel, label=total_label, line_kwargs=line_kwargs, set_ax_kwargs=set_ax_kwargs)


def seperate_EI_input_current_plot(ax, ts, E_current, I_current, E_color=E_COLOR, I_color=I_COLOR, total_color=cf.BLACK, xlabel='time (ms)', ylabel='input current (nA)', E_label='E', I_label='I', total_label='total', set_ax_kwargs=None, line_kwargs=None):
    input_current_plot(ax, ts, E_current, color=E_color, xlabel=xlabel, ylabel=ylabel, label=E_label, line_kwargs=line_kwargs, set_ax_kwargs=set_ax_kwargs)
    input_current_plot(ax, ts, I_current, color=I_color, xlabel=xlabel, ylabel=ylabel, label=I_label, line_kwargs=line_kwargs, set_ax_kwargs=set_ax_kwargs)
    input_current_plot(ax, ts, E_current + I_current, color=total_color, xlabel=xlabel, ylabel=ylabel, label=total_label, line_kwargs=line_kwargs, set_ax_kwargs=set_ax_kwargs)

@cf.deprecated
def seperate_EI_ext_input_current_plot(ax, ts, E_current, I_current, external_current, E_color=E_COLOR, I_color=I_COLOR, external_color=cf.GREEN, total_color=cf.BLACK, xlabel='time (ms)', ylabel='input current (nA)', E_label='E', I_label='I', external_label='external', total_label='total', set_ax_kwargs=None, line_kwargs=None):
    input_current_plot(ax, ts, E_current, color=E_color, xlabel=xlabel, ylabel=ylabel, label=E_label, line_kwargs=line_kwargs, set_ax_kwargs=set_ax_kwargs)
    input_current_plot(ax, ts, I_current, color=I_color, xlabel=xlabel, ylabel=ylabel, label=I_label, line_kwargs=line_kwargs, set_ax_kwargs=set_ax_kwargs)
    input_current_plot(ax, ts, external_current, color=external_color, xlabel=xlabel, ylabel=ylabel, label=external_label, line_kwargs=line_kwargs, set_ax_kwargs=set_ax_kwargs)
    input_current_plot(ax, ts, E_current + I_current + external_current, color=total_color, xlabel=xlabel, ylabel=ylabel, label=total_label, line_kwargs=line_kwargs, set_ax_kwargs=set_ax_kwargs)


def fr_plot(ax, ts, fr, color=cf.BLUE, xlabel='time (ms)', ylabel='firing rate (Hz)', label=None, line_kwargs=None, set_ax_kwargs=None):
    template_line_plot(ax, ts, fr, color=color, xlabel=xlabel, ylabel=ylabel, label=label, line_kwargs=line_kwargs, set_ax_kwargs=set_ax_kwargs)


def EI_fr_plot(ax, ts, E_fr, I_fr, E_color=E_COLOR, I_color=I_COLOR, xlabel='time (ms)', ylabel='firing rate (Hz)', E_label='E', I_label='I', set_ax_kwargs=None, line_kwargs=None):
    fr_plot(ax, ts, E_fr, color=E_color, xlabel=xlabel, ylabel=ylabel, label=E_label, line_kwargs=line_kwargs, set_ax_kwargs=set_ax_kwargs)
    fr_plot(ax, ts, I_fr, color=I_color, xlabel=xlabel, ylabel=ylabel, label=I_label, line_kwargs=line_kwargs, set_ax_kwargs=set_ax_kwargs)


def voltage_plot(ax, ts, V, threshold, color=cf.BLUE, threshold_color=cf.ORANGE, xlabel='time (ms)', ylabel='membrane potential (mV)', label=None, line_kwargs=None, set_ax_kwargs=None):
    template_line_plot(ax, ts, V, color=color, xlabel=xlabel, ylabel=ylabel, label=label, line_kwargs=line_kwargs, set_ax_kwargs=set_ax_kwargs)
    cf.add_hline(ax, threshold, color=threshold_color, linestyle='--', label='threshold')


def LFP_plot(ax, ts, LFP, color=cf.BLUE, xlabel='time (ms)', ylabel='LFP (mV)', label=None, line_kwargs=None, set_ax_kwargs=None):
    template_line_plot(ax, ts, LFP, color=color, xlabel=xlabel, ylabel=ylabel, label=label, line_kwargs=line_kwargs, set_ax_kwargs=set_ax_kwargs)


def EI_LFP_plot(ax, E_LFP, I_LFP, dt, E_color=E_COLOR, I_color=I_COLOR, xlabel='time (ms)', ylabel='LFP (mV)', E_label='E', I_label='I', set_ax_kwargs=None, line_kwargs=None):
    LFP_plot(ax, E_LFP, dt, color=E_color, xlabel=xlabel, ylabel=ylabel, label=E_label, line_kwargs=line_kwargs, set_ax_kwargs=set_ax_kwargs)
    LFP_plot(ax, I_LFP, dt, color=I_color, xlabel=xlabel, ylabel=ylabel, label=I_label, line_kwargs=line_kwargs, set_ax_kwargs=set_ax_kwargs)


def acf_plot(ax, lag_times, acf, tau=None, amp=None, color=cf.BLUE, xlabel='lag (ms)', ylabel='ACF', label=None, line_kwargs=None, set_ax_kwargs=None, text_x=0.2, text_y=0.9, text_color=cf.BLACK, fontsize=cf.FONT_SIZE*1.6, show_fit_line=False):
    if tau is not None:
        cf.add_text(ax, f'{cf.round_float(tau)} ms', x=text_x, y=text_y, fontsize=fontsize, color=text_color, va='center', ha='center')

    if show_fit_line:
        fit_line = single_exp(lag_times, amp=amp, tau=tau)
        cf.plt_line(ax, lag_times, fit_line, color=color, linestyle='--', label='exp fit')

    template_line_plot(ax, lag_times, acf, color=color, xlabel=xlabel, ylabel=ylabel, label=label, line_kwargs=line_kwargs, set_ax_kwargs=set_ax_kwargs)


class ACFPlotManager:
    '''
    用来方便的让text_y错开
    '''
    def __init__(self, ax, text_x):
        pass


def EI_acf_plot(ax, E_lag_times, E_acf, I_lag_times, I_acf, E_color=E_COLOR, I_color=I_COLOR, xlabel='lag (ms)', ylabel='ACF', E_label='E', I_label='I', set_ax_kwargs=None, line_kwargs=None):
    acf_plot(ax, E_lag_times, E_acf, color=E_color, xlabel=xlabel, ylabel=ylabel, label=E_label, line_kwargs=line_kwargs, set_ax_kwargs=set_ax_kwargs)
    acf_plot(ax, I_lag_times, I_acf, color=I_color, xlabel=xlabel, ylabel=ylabel, label=I_label, line_kwargs=line_kwargs, set_ax_kwargs=set_ax_kwargs)


def freq_plot(ax, freqs, power, color=cf.BLUE, xlabel='frequency (Hz)', ylabel='power', label=None, line_kwargs=None, set_ax_kwargs=None):
    set_ax_kwargs = cf.update_dict({'xlim': [0, 500]}, set_ax_kwargs)
    template_line_plot(ax, freqs, power, color=color, xlabel=xlabel, ylabel=ylabel, label=label, line_kwargs=line_kwargs, set_ax_kwargs=set_ax_kwargs)


def EI_freq_plot(ax, E_freqs, E_power, I_freqs, I_power, E_color=E_COLOR, I_color=I_COLOR, xlabel='frequency (Hz)', ylabel='power', E_label='E', I_label='I', set_ax_kwargs=None, line_kwargs=None):
    freq_plot(ax, E_freqs, E_power, color=E_color, xlabel=xlabel, ylabel=ylabel, label=E_label, line_kwargs=line_kwargs, set_ax_kwargs=set_ax_kwargs)
    freq_plot(ax, I_freqs, I_power, color=I_color, xlabel=xlabel, ylabel=ylabel, label=I_label, line_kwargs=line_kwargs, set_ax_kwargs=set_ax_kwargs)


def FF_timewindow_plot(ax, timebin_list, FF, color=cf.BLUE, xlabel='time window (ms)', ylabel='FF', label=None, line_kwargs=None, set_ax_kwargs=None):
    template_line_plot(ax, timebin_list, FF, color=color, xlabel=xlabel, ylabel=ylabel, label=label, line_kwargs=line_kwargs, set_ax_kwargs=set_ax_kwargs)


def template_hist_plot(ax, data, color=cf.BLUE, xlabel='x', ylabel='probability', label=None, hist_kwargs=None, set_ax_kwargs=None):
    '''
    绘制histogram图
    '''
    if hist_kwargs is None:
        hist_kwargs = {}
    if set_ax_kwargs is None:
        set_ax_kwargs = {}
    cf.plt_hist(ax, data, color=color, label=label, **hist_kwargs)
    cf.set_ax(ax, xlabel, ylabel, **set_ax_kwargs)


def ISI_hist_plot(ax, ISI, color=cf.BLUE, xlabel='ISI (ms)', ylabel='probability', label=None, hist_kwargs=None, set_ax_kwargs=None):
    '''
    绘制ISI分布图
    '''
    if len(ISI) > 0:
        set_ax_kwargs = cf.update_dict({'ylog': True}, set_ax_kwargs)
    else:
        set_ax_kwargs = cf.update_dict({}, set_ax_kwargs)
    template_hist_plot(ax, ISI, color=color, xlabel=xlabel, ylabel=ylabel, label=label, hist_kwargs=hist_kwargs, set_ax_kwargs=set_ax_kwargs)


def ISI_CV_hist_plot(ax, ISI_CV, color=cf.BLUE, xlabel='ISI CV', ylabel='probability', label=None, hist_kwargs=None, set_ax_kwargs=None):
    '''
    绘制ISI CV分布图
    '''
    template_hist_plot(ax, ISI_CV, color=color, xlabel=xlabel, ylabel=ylabel, label=label, hist_kwargs=hist_kwargs, set_ax_kwargs=set_ax_kwargs)


def FF_hist_plot(ax, FF, color=cf.BLUE, xlabel='FF', ylabel='probability', label=None, hist_kwargs=None, set_ax_kwargs=None):
    '''
    绘制FF分布图
    '''
    template_hist_plot(ax, FF, color=color, xlabel=xlabel, ylabel=ylabel, label=label, hist_kwargs=hist_kwargs, set_ax_kwargs=set_ax_kwargs)


def corr_hist_plot(ax, corr, color=cf.BLUE, xlabel='correlation', ylabel='probability', label=None, hist_kwargs=None, set_ax_kwargs=None):
    '''
    绘制correlation分布图
    '''
    template_hist_plot(ax, corr, color=color, xlabel=xlabel, ylabel=ylabel, label=label, hist_kwargs=hist_kwargs, set_ax_kwargs=set_ax_kwargs)


def avalanche_size_hist_plot(ax, avalanche_size, color=cf.BLUE, xlabel='avalanche size', ylabel='probability', label=None, hist_kwargs=None, set_ax_kwargs=None):
    '''
    绘制avalanche size分布图
    '''
    if len(avalanche_size) > 0:
        set_ax_kwargs = cf.update_dict({'xlog': True, 'ylog': True}, set_ax_kwargs)
    else:
        set_ax_kwargs = cf.update_dict({}, set_ax_kwargs)
    template_hist_plot(ax, avalanche_size, color=color, xlabel=xlabel, ylabel=ylabel, label=label, hist_kwargs=hist_kwargs, set_ax_kwargs=set_ax_kwargs)


def avalanche_duration_hist_plot(ax, avalanche_duration, color=cf.BLUE, xlabel='avalanche duration', ylabel='probability', title='avalanche duration distribution', label=None, hist_kwargs=None, set_ax_kwargs=None):
    '''
    绘制avalanche duration分布图
    '''
    if len(avalanche_duration) > 0:
        set_ax_kwargs = cf.update_dict({'xlog': True, 'ylog': True}, set_ax_kwargs)
    else:
        set_ax_kwargs = cf.update_dict({}, set_ax_kwargs)
    template_hist_plot(ax, avalanche_duration, color=color, xlabel=xlabel, ylabel=ylabel, title=title, label=label, hist_kwargs=hist_kwargs, set_ax_kwargs=set_ax_kwargs)


def avalanche_size_duration_plot(ax, avalanche_size, avalanche_duration, scatter_color=cf.BLUE, line_color=cf.RED, xlabel='avalanche size', ylabel='avalanche duration', label=None, linregress_kwargs=None, set_ax_kwargs=None):
    '''
    绘制avalanche size和duration的散点图
    '''
    if linregress_kwargs is None:
        linregress_kwargs = {}
    if set_ax_kwargs is None:
        set_ax_kwargs = {}
    cf.plt_linregress(ax, avalanche_size, avalanche_duration, label=label, scatter_color=scatter_color, line_color=line_color, **linregress_kwargs)
    cf.set_ax(ax, xlabel, ylabel, **set_ax_kwargs)


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
class PoissonGroupWithSeed(bp.dyn.PoissonGroup):
    def __init__(self, size, freqs, keep_size=False, sharding=None, spk_type=None, name=None, mode=None, seed=None):
        super().__init__(size=size, freqs=freqs, keep_size=keep_size, sharding=sharding, spk_type=spk_type, name=name, mode=mode)

        self.rng = bm.random.RandomState(seed_or_key=seed)
    
    def update(self):
        spikes = self.rng.rand_like(self.spike) <= (self.freqs * share['dt'] / 1000.)
        spikes = bm.asarray(spikes, dtype=self.spk_type)
        self.spike.value = spikes
        return spikes
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
    def __init__(self, pre, post, delay, comm, tau, E, out_label=None, name=None):
        super().__init__()
        if name is None:
            Expon_name = None
            COBA_name = None
        else:
            Expon_name = cf.cat('Expon', name)
            COBA_name = cf.cat('COBA', name)
        self.proj = bp.dyn.FullProjAlignPostMg(
        pre=pre, 
        delay=delay, 
        comm=comm,
        syn=bp.dyn.Expon.desc(post.num, tau=tau, name=Expon_name),
        out=bp.dyn.COBA.desc(E, name=COBA_name),
        post=post,
        out_label=out_label,
        name=name
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
    def __init__(self, pre, post, delay, comm, tau_rise, tau_decay, a, E, cc_Mg, alpha, beta, V_offset, out_label=None, name=None):
        super().__init__()
        if name is None:
            NMDA_name = None
            MgBlock_name = None
        else:
            NMDA_name = cf.cat('NMDA', name)
            MgBlock_name = cf.cat('MgBlock', name)
        self.proj = bp.dyn.FullProjAlignPreDSMg(
        pre=pre, 
        delay=delay, 
        comm=comm,
        syn=bp.dyn.NMDA.desc(pre.num, tau_decay=tau_decay, tau_rise=tau_rise, a=a, name=NMDA_name),
        out=bp.dyn.MgBlock(E=E, cc_Mg=cc_Mg, alpha=alpha, beta=beta, V_offset=V_offset, name=MgBlock_name),
        post=post,
        out_label=out_label,
        name=name
        )
# endregion


# region 神经元连接
def ij_conn(pre, post, pre_size, post_size):
    '''
    利用brainpy的bp.conn.IJConn生成conn
    '''
    conn = bp.conn.IJConn(i=pre, j=post)
    # conn = IJConn(i=pre, j=post)
    conn = conn(pre_size=pre_size, post_size=post_size)
    return conn


def ij_comm(pre, post, pre_size, post_size, weight, mode=None, name=None):
    '''
    利用brainpy的bp.conn.IJConn和bp.dnn.EventCSRLinear生成comm
    '''
    conn = ij_conn(pre, post, pre_size, post_size)
    return bp.dnn.EventCSRLinear(conn, weight, mode=mode, name=name)


class EventCSRLinearWithSingleVar(bp.dnn.EventCSRLinear):
    '''
    只有一个变量来控制整体weight的权重,weight此时不可以被训练
    '''
    def __init__(self, conn, weight, scale_var, sharding=None, mode=None, name=None, transpose=True):
        bp._src.dnn.base.Layer.__init__(self, name=name, mode=mode)
        self.conn = conn
        self.sharding = sharding
        self.transpose = transpose

        # connection
        self.indices, self.indptr = self.conn.require('csr')

        # weight
        weight = init.parameter(weight, (self.indices.size,))
        self.weight = weight
        self.scale_var = scale_var
    
    def update(self, x):
        if x.ndim == 1:
            return bm.event.csrmv(self.weight*self.scale_var, self.indices, self.indptr, x,
                                  shape=(self.conn.pre_num, self.conn.post_num),
                                  transpose=self.transpose)
        elif x.ndim > 1:
            shapes = x.shape[:-1]
            x = bm.flatten(x, end_dim=-2)
            y = jax.vmap(self._batch_csrmv)(x)
            return bm.reshape(y, shapes + (y.shape[-1],))
        else:
            raise ValueError

    def _batch_csrmv(self, x):
        return bm.event.csrmv(self.weight*self.scale_var, self.indices, self.indptr, x,
                              shape=(self.conn.pre_num, self.conn.post_num),
                              transpose=self.transpose)


def ij_comm_with_single_var(pre, post, pre_size, post_size, weight, scale_var, mode=None, name=None):
    '''
    利用brainpy的bp.conn.IJConn和bp.dnn.EventCSRLinearWithSingleVar生成comm
    '''
    conn = ij_conn(pre, post, pre_size, post_size)
    return EventCSRLinearWithSingleVar(conn, weight, scale_var, mode=mode, name=name)


class EventCSRLinearWithEtaAndScale(bp.dnn.EventCSRLinear):
    '''
    利用eta,scale来控制整体weight的权重,weight此时不可以被训练
    
    hier是一个shape为(indices.size,)的变量,用来控制每个连接的权重
    '''
    def __init__(self, conn, weight, eta, hier, scale_var, sharding=None, mode=None, name=None, transpose=True):
        bp._src.dnn.base.Layer.__init__(self, name=name, mode=mode)
        self.conn = conn
        self.sharding = sharding
        self.transpose = transpose

        # connection
        self.indices, self.indptr = self.conn.require('csr')

        # weight
        weight = init.parameter(weight, (self.indices.size,))
        self.weight = weight
        self.scale_var = scale_var
        self.eta = eta
        hier = init.parameter(hier, (self.indices.size,))
        self.hier = hier
    
    def update(self, x):
        if x.ndim == 1:
            return bm.event.csrmv(self.weight*self.scale_var*(1.+self.eta*self.hier), self.indices, self.indptr, x,
                                  shape=(self.conn.pre_num, self.conn.post_num),
                                  transpose=self.transpose)
        elif x.ndim > 1:
            shapes = x.shape[:-1]
            x = bm.flatten(x, end_dim=-2)
            y = jax.vmap(self._batch_csrmv)(x)
            return bm.reshape(y, shapes + (y.shape[-1],))
        else:
            raise ValueError

    def _batch_csrmv(self, x):
        return bm.event.csrmv(self.weight*self.scale_var*(1.+self.eta*self.hier), self.indices, self.indptr, x,
                              shape=(self.conn.pre_num, self.conn.post_num),
                              transpose=self.transpose)


def ij_comm_with_eta_and_scale(pre, post, pre_size, post_size, weight, eta, hier, scale_var, mode=None, name=None):
    '''
    利用brainpy的bp.conn.IJConn和bp.dnn.EventCSRLinearWithEtaAndScale生成comm
    '''
    conn = ij_conn(pre, post, pre_size, post_size)
    return EventCSRLinearWithEtaAndScale(conn, weight, eta, hier, scale_var, mode=mode, name=name)


def _build_block_ids(n, block_slice_list):
    '''
    示例:
    block_slice_list = [slice(0,3), slice(3,7), slice(7,10)]
    block_ids = _build_block_ids(10, block_slice_list)
    block_ids = [0, 0, 0, 1, 1, 1, 1, 2, 2, 2]
    '''
    block_ids = np.zeros(n, dtype=int)
    for i, sl in enumerate(block_slice_list):
        block_ids[sl] = i
    return block_ids


class OneToOneWithBlockVar(bp.dnn.OneToOne):
    '''
    让分块的scale_var可以被训练,但是weight不可以被训练

    block_slice_list: 分块的索引列表,假设共k个
    scale_var: 形状为(k,)的变量,每个分块对应一个变量
    '''
    def __init__(self, num, weight, block_slice_list, scale_var, sharding=None, mode=None, name=None):
        bp._src.dnn.base.Layer.__init__(self, mode=mode, name=name)

        self.num = num
        self.sharding = sharding

        weight = init.parameter(weight, (self.num,), sharding=sharding)
        self.weight = weight
        self.scale_var = scale_var
        self.block_ids = bm.array(_build_block_ids(num, block_slice_list))

    def update(self, pre_val):
        return pre_val * self.weight * self.scale_var[self.block_ids]
# endregion


# region monitor related
def get_neuron_group_size(net, g):
    neuron_group_size = getattr(net, g).size
    return neuron_group_size


class MonitorMixin:
    '''
    注意,对于所有monitor相关的函数,如果真的遇到性能上的问题,还是建议直接进行调用
    '''
    def set_slice_dim(self, bm_mode=bm.nonbatching_mode, batch_size=None):
        if bm_mode == bm.nonbatching_mode:
            self.slice_dim = 0
        elif bm_mode == bm.training_mode:
            self.slice_dim = 1
        else:
            raise ValueError(f'Unknown bm_mode: {bm_mode}')
        self.batch_size = batch_size

    def _process_slice(self, neuron_idx):
        if self.slice_dim == 0:
            return neuron_idx
        elif self.slice_dim == 1:
            return (np.arange(self.batch_size), neuron_idx)
    
    def _process_slice_for_current(self, neuron_idx):
        if isinstance(neuron_idx, slice):
            neuron_idx = cf.slice_to_array(neuron_idx)
        if self.slice_dim == 0:
            return neuron_idx
        elif self.slice_dim == 1:
            return np.ix_(np.arange(self.batch_size), neuron_idx)

    def get_function_for_monitor_V(self, g, neuron_idx=None):
        if neuron_idx is None:
            def f():
                return getattr(self.net, g).V
        else:
            neuron_idx = self._process_slice(neuron_idx)
            def f():
                return getattr(self.net, g).V[neuron_idx]
        return f

    def get_function_for_monitor_V_mean(self, g, neuron_idx=None):
        if neuron_idx is None:
            def f():
                return np.mean(getattr(self.net, g).V)
        else:
            neuron_idx = self._process_slice(neuron_idx)
            def f():
                return np.mean(getattr(self.net, g).V[neuron_idx])
        return f

    def get_function_for_monitor_spike(self, g, neuron_idx=None):
        if neuron_idx is None:
            def f():
                return getattr(self.net, g).spike
        else:
            neuron_idx = self._process_slice(neuron_idx)
            def f():
                return getattr(self.net, g).spike[neuron_idx]
        return f

    def get_function_for_monitor_spike_mean(self, g, neuron_idx=None):
        if neuron_idx is None:
            def f():
                return np.mean(getattr(self.net, g).spike)
        else:
            neuron_idx = self._process_slice(neuron_idx)
            def f():
                return np.mean(getattr(self.net, g).spike[neuron_idx])
        return f

    def get_function_for_monitor_current(self, g, neuron_idx=None, label=None):
        '''
        注意,这里切片是切在得到的array上,根据jax的需求,slice是不可以的
        '''
        if neuron_idx is None:
            def f():
                return getattr(self.net, g).sum_current_inputs(getattr(self.net, g).V, label=label)
        else:
            neuron_idx = self._process_slice_for_current(neuron_idx)
            def f():
                return getattr(self.net, g).sum_current_inputs(getattr(self.net, g).V, label=label)[neuron_idx]
        return f

    def get_function_for_monitor_current_mean(self, g, neuron_idx=None, label=None):
        '''
        注意,这里切片是切在得到的array上,根据jax的需求,slice是不可以的
        '''
        if neuron_idx is None:
            def f():
                return np.mean(getattr(self.net, g).sum_current_inputs(getattr(self.net, g).V, label=label))
        else:
            neuron_idx = self._process_slice_for_current(neuron_idx)
            def f():
                return np.mean(getattr(self.net, g).sum_current_inputs(getattr(self.net, g).V, label=label)[neuron_idx])
        return f
# endregion


# region 神经元网络模型运行
# 后面弃用
class SNNSimulator(cf.MetaModel):
    '''
    支持分chunk运行,节约内存和显存
    使用方式:
    simulator = SNNSimulator()
    simulator.set_save_mode('chunk')
    simulator.set_chunk_interval(1000.)
    此时simulator.run_time_interval会自动使用run_time_interval_in_chunks
    '''
    def __init__(self):
        self.params = {}
        super().__init__()
        self.extend_ignore_key_list('chunk_interval')
        self.set_bm_mode()
        self.set_save_mode()
        self.total_chunk_num = 0

    # region set
    def set_up(self, basedir, code_file_list, value_dir_key_before=None, both_dir_key_before=None, value_dir_key_after=None, both_dir_key_after=None, ignore_key_list=None, force_run=None):
        super().set_up(params=self.params, basedir=basedir, code_file_list=code_file_list, value_dir_key_before=value_dir_key_before, both_dir_key_before=both_dir_key_before, value_dir_key_after=value_dir_key_after, both_dir_key_after=both_dir_key_after, ignore_key_list=ignore_key_list, force_run=force_run)

    def set_optional_params_default(self):
        '''
        设置一些不强制需要的参数的默认值
        '''
        super().set_optional_params_default()
        self.set_chunk_interval(None)

    def set_simulation_results(self):
        self.simulation_results = defaultdict(list)
        self.simulation_results_type = 'dict'

    def set_random_seed(self, bm_seed=421):
        '''
        设置随机种子(全局,可以的话还是不要使用)
        '''
        bm.random.seed(bm_seed)
        self.params['bm_seed'] = bm_seed

    def set_gpu(self, id, pre_allocate=None):
        '''
        实测发现,这个set_gpu的一系列操作要写在整个py比较靠前的部分,不能对某个模型设置
        '''
        raise ValueError('请不要使用set_gpu,而是将这段代码复制到py文件的最前面')
        super().set_gpu(id)
        bm.set_platform('gpu')
        if pre_allocate is True:
            bm.enable_gpu_memory_preallocation()
        elif pre_allocate is False:
            bm.disable_gpu_memory_preallocation()
        else:
            pass

    def set_cpu(self):
        bm.set_platform('cpu')

    def set_bm_mode(self, bm_mode=None):
        '''
        设置模式
        '''
        if bm_mode is None:
            bm_mode = bm.nonbatching_mode
        bm.set_mode(bm_mode)
        cf.print_title('Set to brainpy {} mode'.format(bm_mode))
        self.bm_mode = bm_mode

    def set_dt(self, dt):
        '''
        设置dt
        '''
        self.dt = dt
        self.params['dt'] = dt
        bm.set_dt(dt)

    def set_total_simulation_time(self, total_simulation_time):
        '''
        设置总的仿真时间
        '''
        self.total_simulation_time = total_simulation_time
        self.params['total_simulation_time'] = total_simulation_time

    def set_chunk_interval(self, chunk_interval):
        '''
        设置分段运行的时间间隔
        '''
        self.chunk_interval = chunk_interval
        self.params['chunk_interval'] = chunk_interval
    
    def set_save_mode(self, save_mode='all'):
        '''
        设置保存模式

        save_mode: 'all', 全部跑完之后储存; 'chunk', 分段储存
        '''
        self.save_mode = save_mode
    # endregion

    # region get
    @abc.abstractmethod
    def get_net(self):
        '''
        获取网络模型(子类需要实现,并且定义为self.net)
        '''
        self.net = None

    @abc.abstractmethod
    def get_monitors(self):
        '''
        获取监测器(子类需要实现,并且定义为self.monitors)
        '''
        self.monitors = None

    @abc.abstractmethod
    def get_runner(self):
        '''
        获取runner(子类需要实现,并且定义为self.runner)
        '''
        self.runner = None
    # endregion

    # region run
    def initialize_model(self):
        self.get_net()
        self.get_monitors()
        self.get_runner()
        super().initialize_model()

    def update_simulation_results_from_runner(self):
        '''
        更新直接结果
        '''
        for k, v in self.runner.mon.items():
            self.simulation_results[k].append(v)

    def organize_simulation_results(self):
        '''
        整理直接结果
        '''
        for k in self.simulation_results.keys():
            self.simulation_results[k] = np.concatenate(self.simulation_results[k], axis=0)

    def organize_chunk_simulation_results(self):
        '''
        把结果重新读取整合好,再保存
        '''
        # 读取第一个chunk的metadata,获取所有的key
        chunk_idx = 0
        metadata = cf.load_pkl(cf.pj(self.outcomes_dir, f'chunk_{chunk_idx}_simulation_results', 'metadata'))

        # 对每个key,读取所有chunk的结果,并且拼接,保存(这种方式比直接读取所有chunk的结果节约内存)
        for dict_k, file_k in metadata.items():
            self.set_simulation_results()
            for chunk_idx in range(self.total_chunk_num):
                part_simulation_results = cf.load_dict_separate(cf.pj(self.outcomes_dir, f'chunk_{chunk_idx}_simulation_results'), key_to_load=[file_k])[dict_k]
                self.simulation_results[dict_k].append(part_simulation_results)
            self.simulation_results[dict_k] = np.concatenate(self.simulation_results[dict_k], axis=0)
            self.save_simulation_results()

        # 删除子chunk的所有文件夹
        for chunk_idx in range(self.total_chunk_num):
            cf.rmdir(cf.pj(self.outcomes_dir, f'chunk_{chunk_idx}_simulation_results'))

    def clear_runner_mon(self):
        '''
        清空runner的监测器
        '''
        if hasattr(self, 'runner'):
            self.runner.mon = None
            self.runner._monitors = None

    def finalize_run_detail(self):
        '''
        运行结束后,整理结果
        '''
        self.organize_simulation_results()
        self.clear_runner_mon()
        self.save_simulation_results()
        bm.clear_buffer_memory()

        if self.save_mode == 'chunk' and (not self.simulation_results_exist):
            self.organize_chunk_simulation_results()

    def log_during_run(self):
        '''
        运行过程中,打印日志
        '''
        pass

    def basic_run_time_interval(self, time_interval):
        '''
        运行模型,并且保存结果
        '''
        self.runner.run(time_interval)
        self.update_simulation_results_from_runner()
        self.log_during_run()

    def finalize_each_chunk(self, chunk_idx):
        if self.save_mode == 'chunk':
            self.organize_simulation_results()
            self.save_simulation_results(filename=f'chunk_{chunk_idx}_simulation_results')
            self.set_simulation_results()

    def run_time_interval_in_chunks(self, time_interval):
        '''
        分段运行模型,以防止内存溢出
        '''
        chunk_num = int(time_interval / self.chunk_interval) + 1
        remaining_time = time_interval
        for chunk_idx in range(chunk_num):
            run_this_chunk = False

            if self.chunk_interval <= remaining_time:
                self.basic_run_time_interval(self.chunk_interval)
                remaining_time -= self.chunk_interval
                self.total_chunk_num += 1
                run_this_chunk = True
            elif remaining_time > 0:
                self.basic_run_time_interval(remaining_time)
                remaining_time = 0
                self.total_chunk_num += 1
                run_this_chunk = True

            if run_this_chunk: # 有可能没有运行
                # 注意这里第一次跑完total_chunk_num=1,所以chunk_idx这边要-1
                self.finalize_each_chunk(chunk_idx=self.total_chunk_num-1)

    def run_time_interval(self, time_interval):
        '''
        运行模型,并且保存结果(自动选择分段运行还是直接运行)
        '''
        if self.chunk_interval is not None:
            self.run_time_interval_in_chunks(time_interval)
        else:
            self.basic_run_time_interval(time_interval)

    def run_detail(self):
        '''
        运行模型,并且保存结果

        注意: 
        当子类有多个阶段,需要重写此方法
        当内存紧张的时候,可以调用run_time_interval,分段运行
        '''
        self.run_time_interval(self.total_simulation_time)
    # endregion


class SNNSimulatorCompose(cf.Simulator):
    def _set_required_key_list(self):
        self.required_key_list = ['total_simulation_time']

    def _set_optional_key_value_dict(self):
        self.optional_key_value_dict = {
            'chunk_interval': None,
            'save_mode': 'all'
        }

    def _set_name(self):
        self.name = 'snn_simulator'

    def _config_data_keeper(self):
        pass

    def inject_net(self, net):
        self.net = net
    
    def inject_monitors(self, monitors):
        self.monitors = monitors
    
    def inject_runner(self, runner):
        self.runner = runner

    def update_results_from_runner(self):
        for k, v in self.runner.mon.items():
            if k not in self.simulation_results:
                self.simulation_results[k] = []
            self.simulation_results[k].append(v)

    def organize_results(self):
        for k in self.simulation_results.keys():
            self.simulation_results[k] = np.concatenate(self.simulation_results[k], axis=0)

    def organize_chunk_simulation_results(self):
        '''
        把结果重新读取整合好,再保存
        '''
        # 读取第一个chunk的metadata,获取所有的key
        chunk_idx = 0
        metadata = cf.load_pkl(cf.pj(self.outcomes_dir, f'chunk_{chunk_idx}_simulation_results', 'metadata'))

        # 对每个key,读取所有chunk的结果,并且拼接,保存(这种方式比直接读取所有chunk的结果节约内存)
        for dict_k, file_k in metadata.items():
            self.set_simulation_results()
            for chunk_idx in range(self.total_chunk_num):
                part_simulation_results = cf.load_dict_separate(cf.pj(self.outcomes_dir, f'chunk_{chunk_idx}_simulation_results'), key_to_load=[file_k])[dict_k]
                self.simulation_results[dict_k].append(part_simulation_results)
            self.simulation_results[dict_k] = np.concatenate(self.simulation_results[dict_k], axis=0)
            self.data_keeper.save()

        # 删除子chunk的所有文件夹
        for chunk_idx in range(self.total_chunk_num):
            cf.rmdir(cf.pj(self.outcomes_dir, f'chunk_{chunk_idx}_simulation_results'))

    def clear_runner_mon(self):
        '''
        清空runner的监测器
        '''
        if hasattr(self, 'runner'):
            self.runner.mon = None
            self.runner._monitors = None

    def finalize_run_detail(self):
        '''
        运行结束后,整理结果
        '''
        self.organize_results()
        self.clear_runner_mon()
        self.data_keeper.save()
        bm.clear_buffer_memory()

        if self.save_mode == 'chunk' and (not self.simulation_results_exist):
            self.organize_chunk_simulation_results()

    def log_during_run(self):
        pass

    def basic_run_time_interval(self, time_interval):
        self.runner.run(time_interval)
        self.update_results_from_runner()
        self.log_during_run()

    def finalize_each_chunk(self, chunk_idx):
        if self.save_mode == 'chunk':
            self.organize_results()
            self.data_keeper.save_data(filename=f'chunk_{chunk_idx}_results')
            self.data_keeper.release_memory() # 如果分段运行,一定是需要释放内存的

    def run_time_interval_in_chunks(self, time_interval):
        '''
        分段运行模型,以防止内存溢出
        '''
        chunk_num = int(time_interval / self.chunk_interval) + 1
        remaining_time = time_interval
        for chunk_idx in range(chunk_num):
            run_this_chunk = False

            if self.chunk_interval <= remaining_time:
                self.basic_run_time_interval(self.chunk_interval)
                remaining_time -= self.chunk_interval
                self.total_chunk_num += 1
                run_this_chunk = True
            elif remaining_time > 0:
                self.basic_run_time_interval(remaining_time)
                remaining_time = 0
                self.total_chunk_num += 1
                run_this_chunk = True

            if run_this_chunk: # 有可能没有运行
                # 注意这里第一次跑完total_chunk_num=1,所以chunk_idx这边要-1
                self.finalize_each_chunk(chunk_idx=self.total_chunk_num-1)

    def run_time_interval(self, time_interval):
        '''
        运行模型,并且保存结果(自动选择分段运行还是直接运行)
        '''
        if self.chunk_interval is not None:
            self.run_time_interval_in_chunks(time_interval)
        else:
            self.basic_run_time_interval(time_interval)

    def run_detail(self):
        '''
        运行模型,并且保存结果

        注意: 
        当子类有多个阶段,需要重写此方法
        当内存紧张的时候,可以调用run_time_interval,分段运行
        '''
        self.run_time_interval(self.total_simulation_time)


class SNNAnalyzerCompose(cf.Analyzer):
    def _set_name(self):
        self.name = 'snn_analyzer'


def custom_bp_running_cpu_parallel(func, params_list, num_process=10, mode='ordered'):
    '''
    参数:
    func: 需要并行计算的函数
    params_list: 需要传入的参数列表,例如[(a0, b0), (a1, b1), ...]
    num_process: 进程数
    mode: 运行模式,ordered表示有序运行,unordered表示无序运行

    注意:
    jupyter中使用时,func需要重新import,所以不建议在jupyter中使用
    实际使用时发现,改变import的模块,新的函数会被实时更新(比如在运行过程中修改了common_functions,那么对于还没运行的函数,会使用新的common_functions)
    '''
    bm.set_platform('cpu')
    total_num = len(params_list)
    
    # 将参数列表转换为分块结构(实测这样相比直接运行可以防止mem累积)
    for chunk_idx in range((total_num + num_process - 1) // num_process):
        # 计算当前分片的起止索引
        start_idx = chunk_idx * num_process
        end_idx = min((chunk_idx + 1) * num_process, total_num)
        local_num_process = end_idx - start_idx
        
        # 提取当前分片的参数并转换结构
        chunk_params = params_list[start_idx:end_idx]
        transposed_params = [list(param_chunk) for param_chunk in zip(*chunk_params)]
        
        # 打印调试信息
        cf.print_title(f"Processing chunk {chunk_idx}: [{start_idx}-{end_idx})")
        
        # 执行并行计算
        if mode == 'ordered':
            bp.running.cpu_ordered_parallel(func, transposed_params, num_process=local_num_process)
        else:
            bp.running.cpu_unordered_parallel(func, transposed_params, num_process=local_num_process)
# endregion


# region loss
def bm_ks(points1, points2):
    """
    20240221,Chen Xiaoyu,convert to BrainPy(JAX) style Based on https://github.com/mikesha2/kolmogorov_smirnov_torch/blob/main/ks_test.py
    
    Kolmogorov-Smirnov test for empirical similarity of probability distributions.
    
    Warning: we assume that none of the elements of points1 coincide with points2. 
    The test may gave false negatives if there are coincidences, however the effect
    is small.

    Parameters
    ----------
    points1 : (n1,) 1D array
        Batched set of samples from the first distribution
    points2 : (n2,) 1D array
        Batched set of samples from the second distribution
    """
    n1 = points1.shape[-1]
    n2 = points2.shape[-1]

    comb = bm.concatenate((points1, points2), axis=-1)
    comb_argsort = bm.argsort(comb,axis=-1)

    pdf1 = bm.where(comb_argsort <  n1, 1 / n1, 0)
    pdf2 = bm.where(comb_argsort >= n1, 1 / n2, 0)

    cdf1 = pdf1.cumsum(axis=-1)
    cdf2 = pdf2.cumsum(axis=-1)

    return (cdf1 - cdf2).abs().max(axis=-1)


def bm_spike_mean_to_fr(spike_mean, dt, width):
    '''
    spike_mean: shape=(time_steps, )
    '''
    width1 = int(width / 2 / dt) * 2 + 1
    window = bm.ones(width1) * 1000 / width
    return bm.convolve(spike_mean, window, mode='same')


def bm_spike_mean_to_fr_timescale(spike_mean, dt, width, nlags_list):
    fr = bm_spike_mean_to_fr(spike_mean, dt, width)
    # 掐头去尾,防止没有到稳态和边界效应
    fr = fr[int(fr.shape[0] / 4):int(fr.shape[0] * 3 / 4)]
    timescale = bm_timescale_by_multi_step_estimation(fr, nlags_list, dt)
    return timescale


def bm_multi_spike_mean_to_fr_timescale(multi_spike_mean, dt, width, nlags_list):
    '''
    multi_spike_mean: shape=(time_steps, num)
    '''
    _f = jax.vmap(bm_spike_mean_to_fr_timescale, in_axes=(0, None, None, None))
    result = _f(multi_spike_mean, dt, width, nlags_list)
    return result


def bm_spike_to_mean_freqs(spike, dt):
    '''
    在神经元维度和时间维度都进行平均,计算出频率
    '''
    return bm.mean(spike) * 1000 / dt


def bm_spike_mean_to_freqs(spike_mean, dt):
    '''
    输入一个neuron group的spike_mean,根据dt计算出对应的频率,关键是单位的转换

    spike_mean: shape=(time_steps, 1)
    dt: float, 时间步长,单位为ms

    return: shape=(time_steps, 1),单位为Hz
    '''
    return spike_mean / dt * 1000.  # 转换为Hz


def bm_spike_mean_to_mean_freqs(spike_mean, dt):
    return bm.mean(bm_spike_mean_to_freqs(spike_mean, dt))


def bm_spike_mean_to_median_freqs(spike_mean, dt):
    '''
    在时间维度上取中位数,可以更加稳健,有助于排除没到达稳态的影响
    '''
    return bm.median(bm_spike_mean_to_freqs(spike_mean, dt))


def bm_abs_loss(x, y):
    return bm.abs(x - y)


def bm_corr(x, y):
    shifted_x = x - bm.mean(x)
    shifted_y = y - bm.mean(y)
    return bm.sum(shifted_x * shifted_y) / bm.sqrt(bm.sum(shifted_x * shifted_x) * bm.sum(shifted_y * shifted_y))


def bm_delay_corr_single_step(series_to_delay, series_to_advance, nlags):
    """
    计算延迟相关系数

    series_to_delay: 需要被延迟的序列,shape = (n, )
    series_to_advance: 需要被提前的序列,shape = (n, )
    nlags: 延迟步数

    注意:
    series_to_delay 将被延迟 nlags 步
    series_to_advance 将被提前 nlags 步

    只返回nlags步的相关系数
    """
    n = series_to_delay.shape[0]
    delayed = series_to_delay[nlags:n]
    advanced = series_to_advance[0:n-nlags]
    return bm_corr(delayed, advanced)


def bm_delay_corr_multi_step(series_to_delay, series_to_advance, nlags):
    """
    计算多延迟步数的相关系数（向量化版本）

    series_to_delay: 需要被延迟的序列, shape = (n, )
    series_to_advance: 需要被提前的序列, shape = (n, )
    nlags: 延迟步数(整数)

    返回: 各延迟步数对应的相关系数,shape = (nlags, )
    """
    # 向量化映射：对每个延迟步数并行计算
    nlags_array = bm.arange(nlags + 1)
    results = []
    for lag in nlags_array:
        results.append(bm_delay_corr_single_step(series_to_delay, series_to_advance, lag))
    return bm.stack(results, axis=0)


def bm_delay_corr_single_step_multi_series(multi_series_to_delay, multi_series_to_advance, nlags):
    """
    输入: shape = (n, k),沿列(k维)vmap
    输出: shape = (k,)
    """
    _f = jax.vmap(bm_delay_corr_single_step, in_axes=(1, 1, None))
    return _f(multi_series_to_delay, multi_series_to_advance, nlags)


def bm_delay_corr_multi_step_multi_series(multi_series_to_delay, multi_series_to_advance, nlags):
    """
    输入: shape = (n, k)
    输出: shape = (nlags+1, k)
    """
    _f = jax.vmap(bm_delay_corr_multi_step, in_axes=(1, 1, None))
    result = _f(multi_series_to_delay, multi_series_to_advance, nlags)
    return bm.transpose(result, (1, 0))


def bm_timescale_by_area_under_acf(timeseries, nlags, dt):
    """
    计算时间序列的timescale
    timeseries: shape = (n, ) 或者 (n, k), 其中k为batch
    nlags: 延迟的范围

    由于时间只有积分到无穷的时候,acf下的面积才会是tau,所以使用此函数需要较大的nlags(至少要让nlags*dt大于3*tau)
    """
    local_timeseries = timeseries.reshape(timeseries.shape[0], -1)

    # 计算延迟相关系数
    delay_corr = bm_delay_corr_multi_step_multi_series(local_timeseries, local_timeseries, nlags)
    delay_corr = bm.mean(delay_corr, axis=0)
    
    # 计算时间尺度
    timescale = bm.mean(delay_corr) * local_timeseries.shape[0] * dt
    
    return timescale


def bm_timescale_by_single_step_estimation(timeseries, nlags, dt):
    '''
    只利用nlags处的延迟相关系数来估计时间尺度
    acf = exp(-t/tau)
    acf(nlags*dt) = exp(-nlags*dt/tau)
    tau = - nlags*dt / log(acf(nlags*dt))
    '''
    local_timeseries = timeseries.reshape(timeseries.shape[0], -1)

    # 计算延迟相关系数
    delay_corr = bm_delay_corr_single_step_multi_series(local_timeseries, local_timeseries, nlags)
    delay_corr = bm.mean(delay_corr)

    # 计算时间尺度
    timescale = - nlags * dt / bm.log(bm.abs(delay_corr)) # 加abs防止对负数取对数

    return timescale


def bm_timescale_by_multi_step_estimation(timeseries, nlags_list, dt):
    '''
    利用nlags_list处的延迟相关系数来估计时间尺度
    '''
    results = bm.zeros((len(nlags_list),))
    for i, nlags in enumerate(nlags_list):
        results[i] = bm_timescale_by_single_step_estimation(timeseries, nlags, dt)
    return bm.nanmedian(results) # log有可能导致nan
# endregion


# region 神经元网络模型训练
# 后面弃用
class SNNTrainer(SNNSimulator):
    '''
    注意事项:

    要训练的参数要设置成bm.TrainVar
    可以单独调用test_f_loss来建议f_loss_with_detail的正确性
    如果要共享某个TrainVar,最好在外面定义完之后直接输入到comm等的内部,并且要注意内部没有对其做额外操作
    TrainVar在外部四则运算后会转化为普通的Array,所有运算最好放到comm等的内部
    '''
    def __init__(self):
        super().__init__()
        self.current_epoch_bm = bm.Variable(bm.zeros((1, ), dtype=int))
        self.t0 = bm.Variable(bm.zeros((1, )))
    
    def set_simulation_results(self):
        self.simulation_results = defaultdict(dict)
        self.simulation_results_type = 'dict'

    def set_bm_mode(self, bm_mode=None):
        '''
        设置模式
        '''
        if bm_mode is None:
            bm_mode = bm.training_mode # 注意: 设置成training_mode之后,会产生一个batch的维度,比如说spike会变成(batch_size, time_steps, neuron_num)
        bm.set_mode(bm_mode)
        cf.print_title('Set to brainpy {} mode'.format(bm_mode))
        self.bm_mode = bm_mode

    def set_epoch(self, epoch):
        '''
        设置epoch
        '''
        self.epoch = epoch
        self.params['epoch'] = epoch

    def set_loss_tolerance(self, loss_tolerance):
        '''
        设置损失函数的容忍度(到达这个值就停止训练)
        '''
        self.loss_tolerance = loss_tolerance
        self.params['loss_tolerance'] = loss_tolerance

    def set_log_interval_epoch(self, log_interval_epoch=1):
        '''
        每log_interval_epoch记录一次
        '''
        self.log_interval_epoch = log_interval_epoch

    def set_optional_params_default(self):
        super().set_optional_params_default()
        self.set_log_interval_epoch(1)
        self.set_loss_tolerance(None)
    
    @property
    def train_vars(self):
        '''
        获取训练变量
        '''
        return self.net.train_vars().unique()

    def get_runner(self):
        pass

    @abc.abstractmethod
    def get_opt(self):
        '''
        获取优化器(子类需要实现,并且定义为self.opt)

        例如:
        lr = bp.optim.ExponentialDecayLR(lr=0.025, decay_steps=1, decay_rate=0.99975)
        self.opt = bp.optim.Adam(lr=lr, train_vars=self.train_vars)
        '''
        self.opt = None
    
    def get_f_grad(self):
        '''
        获取损失函数的梯度
        '''
        self.f_grad = bm.grad(self.f_loss_with_detail, grad_vars=self.train_vars, return_value=True, has_aux=True)

    def set_f_loss(self, batch_size, time_step, inputs=None):
        '''
        设置损失函数的信息,比如说f_loss要对比的inputs值,target值,batch_size,timestep等
        '''
        self.batch_size = batch_size
        self.params['batch_size'] = batch_size
        self.time_step = time_step
        self.params['time_step'] = time_step
        if inputs is None:
            self.inputs = bm.ones((self.batch_size, self.time_step, 1))
        else:
            self.inputs = inputs

    @property
    def warmup_monitors(self):
        '''
        将其设置为property,因为设置warmup_monitor_mode后,也许没有self.monitors,等到调用的时候再设置
        '''
        if self.warmup_monitor_mode == 'same':
            return self.monitors
        elif self.warmup_monitor_mode is None:
            return {}
        else:
            raise ValueError(f'Unknown warmup_monitor_mode: {self.warmup_monitor_mode}')

    def _process_warmup_step_list(self, warmup_step_list):
        '''
        可能输入的不是list,那么自动广播到epoch的长度
        '''
        if isinstance(warmup_step_list, int):
            warmup_step_list = [warmup_step_list] * self.epoch
        elif isinstance(warmup_step_list, list):
            if len(warmup_step_list) != self.epoch:
                raise ValueError(f'len(warmup_step_list)={len(warmup_step_list)} != epoch={self.epoch}')
        else:
            raise ValueError(f'Unknown warmup_step_list: {warmup_step_list}')
        return warmup_step_list

    def set_warmup(self, warmup_step_list, warmup_inputs_list=None, warmup_dt=None, warmup_monitor_mode=None):
        '''
        在训练之前跑一段时间

        warmup_monitor_mode: 
            None: 不监测
            'same': 监测和训练时一样的监测器(即self.monitors)
        '''
        self.warmup_step_list = self._process_warmup_step_list(warmup_step_list)
        self.params['warmup_step_list'] = self.warmup_step_list

        if warmup_inputs_list is None:
            self.warmup_inputs_list = []
            for warmup_step in self.warmup_step_list:
                self.warmup_inputs_list.append(bm.ones((self.batch_size, warmup_step, 1)))
        else:
            self.warmup_inputs_list = warmup_inputs_list

        if warmup_dt is None:
            self.warmup_dt = self.dt
        else:
            self.warmup_dt = warmup_dt
        self.params['warmup_dt'] = self.warmup_dt

        self.warmup_monitor_mode = warmup_monitor_mode
        self.params['warmup_monitor_mode'] = warmup_monitor_mode

    @abc.abstractmethod
    def f_loss_with_detail(self):
        '''
        输出为loss和其余详细信息(其余信息只占一个位置,如果有很多信息可以搞成元组一起输出)(如果不输出其他信息,用None占位)

        示例:训练网络到达指定的firing rate
        self.net.reset(self.batch_size)
        runner = bp.DSTrainer(self.net, progress_bar=False, numpy_mon_after_run=False, monitors=self.monitors)
        runner.predict(self.inputs, reset_state=False)
        output = runner.mon['E_spike']
        mean_fr_predict = bm.mean(output) * 1000 / bm.get_dt()
        return bm.square(mean_fr_predict - self.fr_target), mean_fr_predict
        '''
        pass

    @bm.cls_jit
    def f_train(self):
        grads, loss, detail = self.f_grad()
        self.opt.update(grads)
        return grads, loss, detail
    
    def initialize_model(self):
        self.get_net()
        self.get_monitors()
        self.get_opt()
        self.get_f_grad()
        cf.MetaModel.initialize_model(self)
        cf.bprt(self.train_vars, 'train_vars')

    def test_f_loss(self):
        '''
        单独调用f_loss来建议f_loss的正确性
        '''
        self.initialize_model()
        cf.print_title('test f_loss')
        print(self.f_loss_with_detail())

    def finalize_run_detail(self):
        bm.clear_buffer_memory()

    def _log_during_train_for_detail(self):
        self.simulation_results['detail'].append(self.current_detail)
        cf.better_print(f'detail: {self.current_detail}')
        self.logger.py_logger.info(f'detail: {self.current_detail}')

    def log_during_train(self):
        '''
        记录训练过程中的数据,可以在子类重写,并append到self.simulation_results对应的key中
        '''
        self.simulation_results[f'epoch_{self.current_epoch}']['epoch'] = self.current_epoch
        self.simulation_results[f'epoch_{self.current_epoch}']['loss'] = self.current_loss
        np_current_grads = {}
        for k, v in self.current_grads.items():
            np_current_grads[k] = np.array(v)
        self.simulation_results[f'epoch_{self.current_epoch}']['grads'] = np_current_grads
        cf.better_print(f'epoch: {self.current_epoch}')
        cf.better_print(f'loss: {self.current_loss}')
        self.logger.py_logger.info(f'epoch: {self.current_epoch}')
        self.logger.py_logger.info(f'loss: {self.current_loss}')
        self.logger.py_logger.info(f'grads: {self.current_grads}')
        self._log_during_train_for_detail()

        processed_wramup_mon = dict_to_np_dict(self.runner_warmup.mon)
        self.simulation_results[f'epoch_{self.current_epoch}']['warmup_mon'] = processed_wramup_mon

    def run_warmup(self):
        cf.print_title(f'Warmup with {self.warmup_step_list[self.current_epoch_bm.value[0]]} steps')
        self.runner_warmup = bp.DSTrainer(self.net, progress_bar=True, numpy_mon_after_run=False, monitors=self.warmup_monitors, dt=self.warmup_dt, t0=self.t0.value)
        self.runner_warmup.predict(self.warmup_inputs_list[self.current_epoch_bm.value[0]], reset_state=False)
        self.t0.value += self.warmup_step_list[self.current_epoch_bm.value[0]] * self.warmup_dt

    def before_f_train(self):
        '''
        每个epoch训练前的操作,可以在子类重写
        '''
        self.net.reset(self.batch_size)
        self.run_warmup()

    def after_f_train(self):
        '''
        每个epoch训练完成后的操作,可以在子类重写

        例子:
        训练SNN的权重,可以强制在每次train后调整权重的符号,利用bm.abs和-bm.abs来选择符号
        '''
        pass

    def run_detail(self):
        '''
        仍然叫做run_detail,但是实际上是训练,可以在子类重写
        '''
        for self.current_epoch in tqdm(range(self.epoch)):
            self.before_f_train()
            self.current_grads, self.current_loss, self.current_detail = self.f_train()
            self.after_f_train()
            if self.current_epoch % self.log_interval_epoch == 0:
                self.log_during_train()
                self.save_simulation_results(key_to_save=[f'epoch_{self.current_epoch}'], max_depth=2)
            if self.loss_tolerance is not None and self.current_loss < self.loss_tolerance:
                print(f'Loss tolerance reached: {self.current_loss} < {self.loss_tolerance}')
                break
            self.current_epoch_bm.value += 1

# 后面弃用
class SNNFitter(SNNTrainer):
    '''
    专门用来让SNN的某种性质达到目标值,不需要输入

    支持连续模式,即在每个epoch结束后,将当前epoch的结果传入下一个epoch
    '''
    def set_continuous_mode(self, continuous_mode=True):
        '''
        设置连续模式,如果设置为True,那么在每个epoch结束后,会将当前epoch的结果传入下一个epoch
        '''
        self.continuous_mode = continuous_mode
        self.params['continuous_mode'] = continuous_mode

    def set_optional_params_default(self):
        super().set_optional_params_default()
        self.set_continuous_mode(True)

    def set_f_loss(self, batch_size, time_step, single_loss_config_list, multi_loss_config_list=None):
        '''
        single_loss_config_list: list of dict,每个字典包含单个monitor的损失配置
            Required keys: 'monitor_key', 'target_value'
            Optional keys: 'loss_coef', 'loss_func', 'transform_func'
        multi_loss_config_list: list of dict,每个字典包含多个monitor联合的损失配置
            Required keys: 'monitor_key' (tuple), 'target_value', 'transform_func'
            Optional keys: 'loss_coef', 'loss_func'

        monitor_key: str, 监测器的key, 需要在self.monitors中定义
        target_value: float, 目标值
        loss_coef: float, 乘在每个term的loss上, 默认为1.0
        loss_func: function, 二元, 第一个参数是预测值, 第二个参数是目标值, 默认是bm_abs_loss
        transform_func: function, 将runner.mon[monitor_key]转换为目标值的函数, 当multi时, 要注意顺序和monitor_key的顺序匹配, 当single时, 默认是cf.identical_func
        '''
        super().set_f_loss(batch_size, time_step)

        if multi_loss_config_list is None:
            multi_loss_config_list = []

        # 处理单变量损失配置
        single_defaults = {
            'loss_coef': 1.0,
            'loss_func': bm_abs_loss,
            'transform_func': cf.identical_func
        }
        self.single_loss_terms = []
        for config in single_loss_config_list:
            validated = cf.update_dict(single_defaults, config)
            self.single_loss_terms.append(validated)
        self.single_loss_config_list = single_loss_config_list

        # 处理多变量联合损失配置
        multi_defaults = {
            'loss_coef': 1.0,
            'loss_func': bm_abs_loss
        }
        self.multi_loss_terms = []
        for config in multi_loss_config_list:
            validated = cf.update_dict(multi_defaults, config)
            self.multi_loss_terms.append(validated)
        self.multi_loss_config_list = multi_loss_config_list

    def f_loss_with_detail(self):
        runner = bp.DSTrainer(self.net, progress_bar=False, numpy_mon_after_run=False, monitors=self.monitors, t0=self.t0, dt=self.dt)
        runner.predict(self.inputs, reset_state=False)
        self.t0.value += self.time_step * self.dt
        loss = 0.
        single_loss_terms_detail = []
        multi_loss_terms_detail = []

        for term in self.single_loss_terms:
            monitor_key = term['monitor_key']
            target_value = term['target_value']
            loss_coef = term['loss_coef']
            loss_func = term['loss_func']
            transform_func = term['transform_func']

            output = runner.mon[monitor_key]
            transformed_output = transform_func(output)
            loss += loss_coef * loss_func(transformed_output, target_value)
            single_loss_terms_detail.append({
                'target_value': target_value,
                'loss_coef': loss_coef,
                'transformed_output': transformed_output
            })

        for term in self.multi_loss_terms:
            monitor_key = term['monitor_key']
            target_value = term['target_value']
            loss_coef = term['loss_coef']
            loss_func = term['loss_func']
            transform_func = term['transform_func']
            
            output = [runner.mon[k] for k in monitor_key]
            transformed_output = transform_func(*output)
            loss += loss_coef * loss_func(transformed_output, target_value)
            multi_loss_terms_detail.append({
                'target_value': target_value,
                'loss_coef': loss_coef,
                'transformed_output': transformed_output
            })

        return loss, (single_loss_terms_detail, multi_loss_terms_detail, runner.mon)
    
    def inherit_previous_state_when_continuous_mode(self):
        '''
        当连续模式下,将上一个epoch的状态传入下一个epoch
        '''
        if self.current_epoch > 0 and self.continuous_mode:
            assign_state(self.net, self.state)

    def before_f_train(self):
        '''
        每个epoch训练前的操作,可以在子类重写
        '''
        self.net.reset(self.batch_size)
        self.inherit_previous_state_when_continuous_mode()
        self.run_warmup()

    def log_state(self):
        self.state = extract_state(self.net)
        cf.save_dict(self.state, cf.pj(self.outcomes_dir, 'state', f'epoch_{self.current_epoch}'))

    def after_f_train(self):
        super().after_f_train()
        self.log_state()

    def _process_detail(self, detail_part, config_list, loss_type):
        '''
        将f_loss_with_detail的输出整理,将config_list中的monitor_key加入其中,并且将bp数组转换为np数组
        '''
        for i, (detail, config) in enumerate(zip(detail_part, config_list)):
            new_term = dict_to_np_dict(detail)
            new_term['monitor_key'] = config['monitor_key']
            self.simulation_results[f'epoch_{self.current_epoch}'][f'{loss_type}_term_{i}'] = new_term
            self.logger.py_logger.info(f'{loss_type}_term_{i}: {new_term}')
            cf.better_print(f'{loss_type}_term_{i}: {new_term}')

    def _log_during_train_for_detail(self):
        self._process_detail(self.current_detail[0], self.single_loss_config_list, 'single')
        self._process_detail(self.current_detail[1], self.multi_loss_config_list, 'multi')
        processed_mon = dict_to_np_dict(self.current_detail[2])
        self.simulation_results[f'epoch_{self.current_epoch}']['mon'] = processed_mon
# endregion


# region 神经元网络模型
class MultiNet(bp.DynSysGroup):
    def __init__(self, neuron, synapse, inp_neuron, inp_synapse, neuron_params, synapse_params, inp_neuron_params, inp_synapse_params, comm, inp_comm, print_info=True, clear_name_cache=True):
        """
        参数:
            neuron (dict): 包含每个组的神经元类型的字典
            synapse (dict): 包含每个连接的突触类型的字典(key必须是(s, t, name)的元组,如果不需要name,则name设置为None或者空字符串)
            inp_neuron (dict): 包含输入神经元类型的字典
            inp_synapse (dict): 包含输入突触类型的字典
            neuron_params (dict): 包含神经元初始化参数的字典
            synapse_params (dict): 包含突触初始化参数的字典
            inp_neuron_params (dict): 包含输入神经元初始化参数的字典
            inp_synapse_params (dict): 包含输入突触初始化参数的字典
            comm (dict): 包含组之间通信参数的字典
            inp_comm (dict): 包含输入组和其他组通信参数的字典
            print_info (bool): 是否打印信息
            
        建议:
            当需要很多group的时候,不一定要每个group一个neuron的key

            可以先考虑给不同group设定不同的参数来解决,这样更加高效
            tau_ref_1 = np.ones(ne_1) * 2
            tau_ref_2 = np.ones(ne_2) * 3
            tau_ref = np.concatenate([tau_ref_1, tau_ref_2])
            self.E = bp.dyn.LifRef(ne_1+ne_2, V_rest=-70., V_th=-50., V_reset=-60., tau=20., tau_ref=tau_ref,
                                V_initializer=bp.init.Normal(-55., 2.))
        
        注意:
            不要往self里面放多余的东西,只放neuron,synapse,inp_neuron,inp_synapse
            多放东西会直接报错,比如把comm放进去
        """
        super().__init__()
        if clear_name_cache:
            self.clear_name_cache()

        self.group = neuron.keys()
        self.inp_group = inp_neuron.keys()
        self.synapse_group = synapse.keys()
        self.inp_synapse_group = inp_synapse.keys()

        for g in self.group:
            setattr(self, g, neuron[g](**neuron_params[g]))

        for inp_g in self.inp_group:
            setattr(self, inp_g, inp_neuron[inp_g](**inp_neuron_params[inp_g]))
        
        for syn_type in synapse.keys():
            s, t, name = syn_type
            if print_info:
                cf.print_title(f'{s}2{t} {name} synapse')
            setattr(self, cf.concat_str([f'{s}2{t}', name]), synapse[syn_type](pre=getattr(self, s), post=getattr(self, t), comm=comm[syn_type], **synapse_params[syn_type]))

        for inp_syn_type in inp_synapse.keys():
            s, t, name = inp_syn_type
            if print_info:
                cf.print_title(f'{s}2{t} {name} synapse')
            setattr(self, cf.concat_str([f'{s}2{t}', name]), inp_synapse[inp_syn_type](pre=getattr(self, s), post=getattr(self, t), comm=inp_comm[inp_syn_type], **inp_synapse_params[inp_syn_type]))

    def clear_name_cache(self):
        '''
        2025_5_7,这样做了之后可以让Lif等对象的名字可以重复
        '''
        bp.math.clear_name_cache()

    def update(self, *args, **kwargs):
        '''
        2025_4_17,为了让其可以接收输入用于训练(在我的框架下,输入是虚假的,只是让bp来确定batch等,所以这里没有输入给原先的update)
        '''
        super().update()
    
    def reset_state(self, batch_size=1):
        '''
        2025_4_17,为了训练需要增添的
        '''
        for g in self.group:
            getattr(self, g).reset_state(batch_size)
        for inp_g in self.inp_group:
            getattr(self, inp_g).reset_state(batch_size)
        for syn_type in self.synapse_group:
            s, t, name = syn_type
            getattr(self, cf.concat_str([f'{s}2{t}', name])).reset_state(batch_size)
        for inp_syn_type in self.inp_synapse_group:
            s, t, name = inp_syn_type
            getattr(self, cf.concat_str([f'{s}2{t}', name])).reset_state(batch_size)

# 后面弃用
class SNNNetMixin:
    def get_neuron(self):
        pass

    def get_synapse(self):
        pass

    def get_inp_neuron(self):
        pass

    def get_inp_synapse(self):
        pass

    def get_neuron_params(self):
        pass

    def get_synapse_params(self):
        pass

    def get_inp_neuron_params(self):
        pass

    def get_inp_synapse_params(self):
        pass

    def get_comm(self):
        pass

    def get_inp_comm(self):
        pass

    def before_get_net(self):
        self.get_neuron()
        self.get_synapse()
        self.get_inp_neuron()
        self.get_inp_synapse()
        self.get_neuron_params()
        self.get_synapse_params()
        self.get_inp_neuron_params()
        self.get_inp_synapse_params()
        self.get_comm()
        self.get_inp_comm()

    def get_net(self):
        '''
        获取网络模型(子类可以改写,并且定义为self.net)

        注意:
        推荐将过程拆分到上面定义的函数中
        '''
        self.before_get_net()
        self.net = MultiNet(neuron=self.neuron, synapse=self.synapse, inp_neuron=self.inp_neuron, inp_synapse=self.inp_synapse, neuron_params=self.neuron_params, synapse_params=self.synapse_params, inp_neuron_params=self.inp_neuron_params, inp_synapse_params=self.inp_synapse_params, comm=self.comm, inp_comm=self.inp_comm)


class SNNNetGenerator(abc.ABC):
    @abc.abstractmethod
    def _get_neuron(self):
        pass

    @abc.abstractmethod
    def _get_synapse(self):
        pass

    @abc.abstractmethod
    def _get_inp_neuron(self):
        pass

    @abc.abstractmethod
    def _get_inp_synapse(self):
        pass

    @abc.abstractmethod
    def _get_neuron_params(self):
        pass

    @abc.abstractmethod
    def _get_synapse_params(self):
        pass

    @abc.abstractmethod
    def _get_inp_neuron_params(self):
        pass

    @abc.abstractmethod
    def _get_inp_synapse_params(self):
        pass

    def inject_conn(self, conn_dict):
        '''
        conn_dict是字典,key是(s, t, name)的元组,value是conn对象
        '''
        pass

    @abc.abstractmethod
    def _get_comm(self):
        pass

    @abc.abstractmethod
    def _get_inp_comm(self):
        pass

    def _before_get_net(self):
        self._get_neuron()
        self._get_synapse()
        self._get_inp_neuron()
        self._get_inp_synapse()
        self._get_neuron_params()
        self._get_synapse_params()
        self._get_inp_neuron_params()
        self._get_inp_synapse_params()
        self._get_comm()
        self._get_inp_comm()

    def get_net(self):
        '''
        获取网络模型(子类可以改写,并且定义为self.net)

        注意:
        推荐将过程拆分到上面定义的函数中
        '''
        self._before_get_net()
        self.net = MultiNet(neuron=self.neuron, synapse=self.synapse, inp_neuron=self.inp_neuron, inp_synapse=self.inp_synapse, neuron_params=self.neuron_params, synapse_params=self.synapse_params, inp_neuron_params=self.inp_neuron_params, inp_synapse_params=self.inp_synapse_params, comm=self.comm, inp_comm=self.inp_comm)
# endregion


# region 保存和加载
def bp_load_state_adapted(target, state_dict, **kwargs):
    """
    2025_5_8, adapted from brainpy, the order of missing and unexpected keys is changed since the brainpy bug

    also print things to make it easier to debug
    """
    nodes = target.nodes().subset(DynamicalSystem).not_subset(DynView).unique()
    missing_keys = []
    unexpected_keys = []
    failed_names = []
    for name, node in nodes.items():
        try:
            r = node.load_state(state_dict[name], **kwargs)
        except:
            r = None
            failed_names.append(name)
        if r is not None:
            missing, unexpected = r
            missing_keys.extend([f'{name}.{key}' for key in missing])
            unexpected_keys.extend([f'{name}.{key}' for key in unexpected])
    if bp.__version__ == '2.6.0':
        unexpected_keys, missing_keys = missing_keys, unexpected_keys
        if len(unexpected_keys) > 0 or len(missing_keys) > 0:
            # 两个都空的话也没必要打印warning
            cf.print_title('Note', char='!')
            cf.print_title('bp version is 2.6.0, so the order of missing and unexpected keys is changed since the brainpy bug', char='!')
            cf.print_sep(char='!')
    print(f'Failed names: {failed_names}')
    return StateLoadResult(missing_keys, unexpected_keys)


def extract_state(net):
    '''
    提取net的状态
    '''
    return bp.save_state(net)


def save_state_to_disk(net, filename, **kwargs):
    '''
    保存net的状态
    '''
    cf.mkdir(filename)
    bp.checkpoints.save_pytree(filename, net.state_dict(), **kwargs)


def load_state_from_disk(filename):
    '''
    读取net的状态
    '''
    return bp.checkpoints.load_pytree(filename)


def assign_state(net, state):
    '''
    将state赋值给net
    '''
    bp_load_state_adapted(net, state)


def load_and_assign_state(net, filename):
    '''
    读取net的状态并赋值
    '''
    state = load_state_from_disk(filename)
    assign_state(net, state)
# endregion