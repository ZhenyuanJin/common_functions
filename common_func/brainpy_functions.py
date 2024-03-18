# 标准库导入
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
# import jax.numpy as jnp

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

# 脑区处理库
# import networkx as nx
# import pointpats
# import shapely.geometry

# 自定义库
import common_functions as cf


# 定义默认参数
E_COLOR = cf.RED
I_COLOR = cf.BLUE


# brainpy使用说明
# 所有数据按照(T, N)的形式存储，其中T表示时间点的数量，N表示神经元的数量。


# 通用函数
def neuron_idx_data(data, indices=None, keep_size=False):
    '''
    从spikes或者V中提取指定索引的神经元数据。indices可以是slice对象或单个值。
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
    从spikes或者V中提取指定时间点的数据。indices可以是slice对象或单个值。
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


# 作图函数
def spike_to_fr(spikes, width, dt, neuron_idx=None, **kwargs):
    '''
    修改bp.measure.firing_rate使得一维数组的spikes也能够计算firing rate
    '''
    spikes = neuron_idx_data(spikes, neuron_idx, keep_size=True)
    return bp.measure.firing_rate(spikes, width, dt, **kwargs)


def get_spike_acf(spikes, dt, nlags, neuron_idx=None, average=True, **kwargs):
    '''
    计算spikes的自相关函数
    '''
    spikes = neuron_idx_data(spikes, neuron_idx, keep_size=True)
    lag_times, multi_acf = cf.get_acf(np.array(spikes).T, T=dt, nlags=nlags, **kwargs)
    if average:
        return lag_times, np.mean(multi_acf, axis=0)
    else:
        return lag_times, multi_acf.T


def spike_to_fr_acf(spikes, width, dt, nlags, neuron_idx=None, spike_to_fr_kwargs=None, **kwargs):
    '''
    计算spikes的firing rate的自相关函数,注意,计算fr的过程自动平均了neuron_idx中的神经元。
    '''
    if spike_to_fr_kwargs is None:
        spike_to_fr_kwargs = {}
    fr = spike_to_fr(spikes, width, dt, neuron_idx, **spike_to_fr_kwargs)
    return cf.cal_acf(fr, T=dt, nlags=nlags, **kwargs)


def spike_video(E_spikes, E_pos, I_spikes, I_pos, dt, folder, video_name='spike_video', scatter_size=5, faint_num=3, legend_loc='upper left', bbox_to_anchor=(1, 1), elev_list=None, azim_list=None, margin={'left': 0.1, 'right':0.75, 'bottom' : 0.1, 'top': 0.75}, fig_ax_kwargs=None, scatter_kwargs=None, set_ax_kwargs=None, save_fig_kwargs=None, video_kwargs=None):

    cf.mkdir(folder)

    if elev_list is None:
        elev_list = [cf.ELEV]
    if azim_list is None:
        azim_list = [cf.AZIM]
    if fig_ax_kwargs is None:
        fig_ax_kwargs = {}
    if scatter_kwargs is None:
        scatter_kwargs = {}
    if set_ax_kwargs is None:
        set_ax_kwargs = {}
    if save_fig_kwargs is None:
        save_fig_kwargs = {}
    if video_kwargs is None:
        video_kwargs = {}


    if E_pos.shape[1] == 2:
        fig_paths = []
        for i in range(E_spikes.shape[0]):
            fig, ax = cf.create_fig_ax(margin=margin, **fig_ax_kwargs)
            for previous in range(faint_num+1):
                if i-previous >= 0:
                    alpha = np.linspace(1, 0, faint_num+1)[previous]
                    if previous == 0:
                        E_label = 'E'
                        I_label = 'I'
                    else:
                        E_label = None
                        I_label = None
                    cf.plt_scatter(ax, E_pos[time_idx_data(E_spikes, i-previous) > 0, 0], E_pos[time_idx_data(E_spikes, i-previous) > 0, 1], color=E_COLOR, s=scatter_size, alpha=alpha, label=E_label, **scatter_kwargs)
                    cf.plt_scatter(ax, I_pos[time_idx_data(I_spikes, i-previous) > 0, 0], I_pos[time_idx_data(I_spikes, i-previous) > 0, 1], color=I_COLOR, s=scatter_size, alpha=alpha, label=I_label, **scatter_kwargs)
            ax.axis('equal')
            ax.axis('off')
            title = 't={}'.format(i*dt)
            cf.set_ax(ax, title=title, legend_loc=legend_loc, bbox_to_anchor=bbox_to_anchor, **set_ax_kwargs)
            figname = 'i={}'.format(str(i))
            filename = os.path.join(folder, figname)
            cf.save_fig(fig, filename, formats=['png'], **save_fig_kwargs)
            fig_paths.append(filename)

        cf.fig_to_video(fig_paths, os.path.join(folder, video_name), **video_kwargs)

    if E_pos.shape[1] == 3:
        fig_paths = {}
        for i in range(E_spikes.shape[0]):
            fig, ax = cf.create_fig_ax_3d(margin=margin, **fig_ax_kwargs)
            for previous in range(faint_num+1):
                if i-previous >= 0:
                    alpha = np.linspace(1, 0, faint_num+1)[previous]
                    if previous == 0:
                        E_label = 'E'
                        I_label = 'I'
                    else:
                        E_label = None
                        I_label = None
                    cf.plt_scatter_3d(ax, E_pos[time_idx_data(E_spikes, i-previous) > 0, 0], E_pos[time_idx_data(E_spikes, i-previous) > 0, 1], E_pos[time_idx_data(E_spikes, i-previous) > 0, 2], color=E_COLOR, s=scatter_size, alpha=alpha, label=E_label, **scatter_kwargs)
                    cf.plt_scatter_3d(ax, I_pos[time_idx_data(I_spikes, i-previous) > 0, 0], I_pos[time_idx_data(I_spikes, i-previous) > 0, 1], I_pos[time_idx_data(I_spikes, i-previous) > 0, 2], color=I_COLOR, s=scatter_size, alpha=alpha, label=I_label, **scatter_kwargs)
            ax.axis('equal')
            ax.axis('off')
            title = 't={}'.format(i*dt)
            cf.set_ax(ax, title=title, legend_loc=legend_loc, bbox_to_anchor=bbox_to_anchor, **set_ax_kwargs)
            figname = 'i={}'.format(str(i))
            filename = os.path.join(folder, figname)
            fig_paths_dict, _ = cf.save_fig_3d(fig, filename, elev_list=elev_list, azim_list=azim_list, generate_video=False, formats=['png'], **save_fig_kwargs)
            for elev in elev_list:
                for azim in azim_list:
                    fig_paths[(elev, azim)] = fig_paths.get((elev, azim), []) + fig_paths_dict[(elev, azim)]
        for elev in elev_list:
            for azim in azim_list:
                cf.fig_to_video(fig_paths[(elev, azim)], os.path.join(folder, video_name+'_elev_{}_azim_{}'.format(str(int(elev)), str(int(azim)))), **video_kwargs)


# 神经元网络模型
class EINet(bp.DynSysGroup):
    def __init__(self, E_neuron, I_neuron, E_params, I_params, E2E_synapse, E2I_synapse, I2E_synapse, I2I_synapse, E2E_synapse_params, E2I_synapse_params, I2E_synapse_params, I2I_synapse_params, E2E_comm, E2I_comm, I2E_comm, I2I_comm):
        super().__init__()

        self.E_params = E_params.copy()
        self.I_params = I_params.copy()
        self.E2E_synapse_params = E2E_synapse_params.copy()
        self.E2I_synapse_params = E2I_synapse_params.copy()
        self.I2E_synapse_params = I2E_synapse_params.copy()
        self.I2I_synapse_params = I2I_synapse_params.copy()

        # E_num = E_params['size']
        # I_num = I_params['size']

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


class MultiAreaEINet(EINet):
    def __init__(self, E_neuron, I_neuron, E_params, I_params, E2E_synapse, E2I_synapse, I2E_synapse, I2I_synapse, E2E_synapse_params, E2I_synapse_params, I2E_synapse_params, I2I_synapse_params, E2E_comm, E2I_comm, I2E_comm, I2I_comm, E_idx_to_area, I_idx_to_area):
        super().__init__(E_neuron, I_neuron, E_params, I_params, E2E_synapse, E2I_synapse, I2E_synapse, I2I_synapse, E2E_synapse_params, E2I_synapse_params, I2E_synapse_params, I2I_synapse_params, E2E_comm, E2I_comm, I2E_comm, I2I_comm)


class SpatialEINet(EINet):
    def __init__(self, E_neuron, I_neuron, E_params, I_params, E2E_synapse, E2I_synapse, I2E_synapse, I2I_synapse, E2E_synapse_params, E2I_synapse_params, I2E_synapse_params, I2I_synapse_params, E2E_comm, E2I_comm, I2E_comm, I2I_comm, E_pos, I_pos):
        super().__init__(E_neuron, I_neuron, E_params, I_params, E2E_synapse, E2I_synapse, I2E_synapse, I2I_synapse, E2E_synapse_params, E2I_synapse_params, I2E_synapse_params, I2I_synapse_params, E2E_comm, E2I_comm, I2E_comm, I2I_comm)

        self.E_pos = E_pos
        self.I_pos = I_pos


class RunSpatialEINet(SpatialEINet):
    def __init__(self, E_neuron, I_neuron, E_params, I_params, E2E_synapse, E2I_synapse, I2E_synapse, I2I_synapse, E2E_synapse_params, E2I_synapse_params, I2E_synapse_params, I2I_synapse_params, E2E_comm, E2I_comm, I2E_comm, I2I_comm, E_pos, I_pos):
        super().__init__(E_neuron, I_neuron, E_params, I_params, E2E_synapse, E2I_synapse, I2E_synapse, I2I_synapse, E2E_synapse_params, E2I_synapse_params, I2E_synapse_params, I2I_synapse_params, E2E_comm, E2I_comm, I2E_comm, I2I_comm, E_pos, I_pos)
        

def test():
    bp.math.set_platform('cpu')
    EI_ratio = 4
    E2E_weight = 1
    E2I_weight = 1
    I2E_weight = -4
    I2I_weight = -4
    E_size = 4*400
    I_size = E_size // EI_ratio

    E_params = {'size': E_size, 'V_th': 20.0, 'V_reset': -5.0, 'V_rest':0., 'tau_ref': 5.0, 'R': 1.0, 'tau': 10.0}
    I_params = {'size': I_size, 'V_th': 20.0, 'V_reset': -5.0, 'V_rest':0., 'tau_ref': 5.0, 'R': 1.0, 'tau': 10.0}
    E2E_synapse_params = {'delay': 0}
    E2I_synapse_params = {'delay': 0}
    I2E_synapse_params = {'delay': 0}
    I2I_synapse_params = {'delay': 0}

    E_inp = 40
    I_inp = 30

    E_pos = np.meshgrid(np.linspace(0, 1, int(np.sqrt(E_size))), np.linspace(0, 1, int(np.sqrt(E_size))))
    E_pos = np.stack([E_pos[0].flatten(), E_pos[1].flatten()], axis=1) + 0.1
    I_pos = np.meshgrid(np.linspace(0, 1, int(np.sqrt(I_size))), np.linspace(0, 1, int(np.sqrt(I_size))))
    I_pos = np.stack([I_pos[0].flatten(), I_pos[1].flatten()], axis=1)

    dt = 1.
    bm.set_dt(dt)

    def zero_one_csr(row_indices, col_indices, shape):
        return csr_matrix((np.ones_like(row_indices), (row_indices, col_indices)), shape=shape)

    def zero_one_conn(row_indices, col_indices, shape):
        return bp.connect.SparseMatConn(csr_mat=zero_one_csr(row_indices, col_indices, shape))

    conn_num = 100

    # E_conn_row_indices = np.repeat(np.arange(E_size), conn_num)
    # E_conn_col_indices = 
    E2E_conn = zero_one_conn(np.random.randint(0, E_size, conn_num), np.random.randint(0, E_size, conn_num), (E_size, E_size))
    # E2E_conn = bp.connect.GaussianProb(sigma=0.1, pre=E_size, post=E_size)
    E2E_comm = bp.dnn.EventCSRLinear(conn=E2E_conn, weight=E2E_weight)

    E2I_conn = zero_one_conn(np.random.randint(0, E_size, conn_num), np.random.randint(0, I_size, conn_num), (E_size, I_size))
    # E2I_conn = bp.connect.GaussianProb(sigma=0.1, pre=E_size, post=I_size)
    E2I_comm = bp.dnn.EventCSRLinear(conn=E2I_conn, weight=E2I_weight)

    I2E_conn = zero_one_conn(np.random.randint(0, I_size, conn_num), np.random.randint(0, E_size, conn_num), (I_size, E_size))
    # I2E_conn = bp.connect.GaussianProb(sigma=0.1, pre=I_size, post=E_size)
    I2E_comm = bp.dnn.EventCSRLinear(conn=I2E_conn, weight=I2E_weight)

    I2I_conn = zero_one_conn(np.random.randint(0, I_size, conn_num), np.random.randint(0, I_size, conn_num), (I_size, I_size))
    # I2I_conn = bp.connect.GaussianProb(sigma=0.1, pre=I_size, post=I_size)
    I2I_comm = bp.dnn.EventCSRLinear(conn=I2I_conn, weight=I2I_weight)

    EI_net = SpatialEINet(E_neuron=bp.dyn.LifRef, I_neuron=bp.dyn.LifRef, E_params=E_params, I_params=I_params, E2E_synapse=bp.dyn.FullProjDelta, E2I_synapse=bp.dyn.FullProjDelta, I2E_synapse=bp.dyn.FullProjDelta, I2I_synapse=bp.dyn.FullProjDelta, E2E_synapse_params=E2E_synapse_params, E2I_synapse_params=E2I_synapse_params, I2E_synapse_params=I2E_synapse_params, I2I_synapse_params=I2I_synapse_params, E2E_comm=E2E_comm, E2I_comm=E2I_comm, I2E_comm=I2E_comm, I2I_comm=I2I_comm, E_pos=E_pos, I_pos=I_pos)


    def run_fun_1(i):
        local_E_inp = np.ones(E_size)*E_inp
        local_I_inp = np.ones(I_size)*I_inp
        return EI_net.step_run(i, local_E_inp, local_I_inp)

    def run_fun_2(i):
        return EI_net.step_run(i, 0, 0)


    indices_1 = np.arange(100)
    ts_1 = indices_1 * bm.get_dt()
    print(len(indices_1))
    E_spikes_1, I_spikes_1, E_V_1, I_V_1 = bm.for_loop(
        run_fun_1, indices_1, progress_bar=True)

    indices_2 = np.arange(100, 150)
    ts_2 = indices_2 * bm.get_dt()
    print(len(indices_2))
    E_spikes_2, I_spikes_2, E_V_2, I_V_2 = bm.for_loop(
        run_fun_2, indices_2, progress_bar=True)

    indices_3 = np.arange(150, 200)
    ts_3 = indices_3 * bm.get_dt()
    print(len(indices_3))
    E_spikes_3, I_spikes_3, E_V_3, I_V_3 = bm.for_loop(
        run_fun_1, indices_3, progress_bar=True)


    fig, ax = cf.create_fig_ax()
    cf.plt_line(ax, ts_1, E_V_1[:, 0], label='E', color=cf.BLUE)
    cf.plt_line(ax, ts_1, I_V_1[:, 0], label='I', color=cf.ORANGE)
    cf.add_hline(ax, I_params['V_th'], label='Threshold', color=cf.RED)
    cf.add_hline(ax, I_params['V_reset'], label='Reset', color=cf.GREEN)
    cf.add_hline(ax, E2I_weight, label='E2I_weight', color=cf.BLACK)
    cf.add_hline(ax, I2I_weight, label='I2I_weight', color=cf.PURPLE)
    cf.set_ax(ax, 'Time (ms)', 'Membrane potential (mV)')

    fig, ax = cf.create_fig_ax()
    cf.plt_line(ax, np.concatenate([ts_1, ts_2, ts_3]), np.concatenate([E_V_1[:, 0], E_V_2[:, 0], E_V_3[:, 0]]), label='E', color=cf.BLUE)

    spike_video(E_spikes_1, E_pos, I_spikes_1, I_pos, dt, './spatial_EI_net/')



    fr = spike_to_fr(E_spikes_1, dt, dt, neuron_idx=slice(0, 10))


    fig, ax = cf.create_fig_ax()
    cf.plt_line(ax, ts_1, fr, label='Firing rate', color=cf.BLUE)



    spike_lag_times, spike_acf = get_spike_acf(E_spikes_1, dt, 10)
    fr_lag_times, fr_acf = spike_to_fr_acf(E_spikes_1, dt, dt, 10)


    fig, ax = cf.create_fig_ax()
    cf.plt_stem(ax, spike_lag_times, spike_acf, label='Spike ACF')

    fig, ax = cf.create_fig_ax()
    cf.plt_stem(ax, fr_lag_times, fr_acf, label='FR ACF')