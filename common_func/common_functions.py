# region 导入模块
# 标准库导入
import subprocess
import os
import re
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
import inspect
import types

# 数学和科学计算库
import numpy as np
import scipy
import scipy.stats as st
from scipy.stats import gaussian_kde, zscore
from scipy.integrate import quad
from scipy.sparse import coo_matrix, csr_matrix
import scipy.sparse as sps
from sklearn.decomposition import PCA, NMF
import statsmodels
from statsmodels.tsa.stattools import acf
from statsmodels.distributions.empirical_distribution import ECDF

# 数据处理和可视化库
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import FuncFormatter, ScalarFormatter
from matplotlib.colors import BoundaryNorm, Normalize, ListedColormap
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import cv2
from reportlab.pdfgen import canvas
from pdf2image import convert_from_path
# endregion


# region 定义颜色,为了防止一些奇怪的错误,不使用0而使用0.01
BLUE = (0.01, 0.45, 0.70)
RED = (1.00, 0.27, 0.01)
GREEN = (0.01, 0.65, 0.50)
BLACK = (0.01, 0.01, 0.01)
WHITE = (1.00, 1.00, 1.00)
GREY = (0.50, 0.50, 0.50)
ORANGE = (1.00, 0.50, 0.01)
PURPLE = (0.50, 0.01, 0.50)
# endregion


# region 用于plt设置
MARGIN = {'left': 0.15, 'right': 0.85, 'bottom': 0.15, 'top': 0.85}     # 默认图形边距
FONT_SIZE = 15     # 默认字体大小
TITLE_SIZE = FONT_SIZE*2     # 默认标题字体大小
SUP_TITLE_SIZE = FONT_SIZE*3     # 默认总标题字体大小
LABEL_SIZE = FONT_SIZE*2     # 默认标签字体大小
TICK_SIZE = FONT_SIZE*4/3     # 默认刻度标签字体大小
LEGEND_SIZE = FONT_SIZE*4/3     # 默认图例字体大小
LINE_WIDTH = 3     # 默认线宽
MARKER_SIZE = LINE_WIDTH*2     # 默认标记大小
AX_WIDTH = 5   # 默认图形宽度(单个图形)
AX_HEIGHT = 5   # 默认图形高度(单个图形)
FIG_SIZE = (AX_WIDTH / ( MARGIN['right'] - MARGIN['left'] ), AX_HEIGHT / ( MARGIN['top'] - MARGIN['bottom'] ))     # 默认图形大小
FIG_DPI = 100     # 默认图形分辨率
SAVEFIG_DPI = 300     # 默认保存图形分辨率
SAVEFIG_VECTOR_FORMAT = 'pdf'   # 默认保存图形的矢量图格式
SAVEFIG_RASTER_FORMAT = 'png'   # 默认保存图形的位图格式
SAVEFIG_FORMAT = SAVEFIG_VECTOR_FORMAT     # 默认保存图形的格式
TOP_SPINE = True     # 默认隐藏上方的轴脊柱
RIGHT_SPINE = True     # 默认隐藏右方的轴脊柱
LEFT_SPINE = True     # 默认显示左方的轴脊柱
BOTTOM_SPINE = True     # 默认显示下方的轴脊柱
LEGEND_LOC = 'upper right'      # 图例位置
PDF_FONTTYPE = 42     # 使pdf文件中的文字可编辑
PAD_INCHES = 0.2     # 默认图形边距
# endregion


# region plt全局参数设置
plt.rcParams['figure.figsize'] = FIG_SIZE   # 图形的大小
plt.rcParams['font.size'] = FONT_SIZE       # 字体大小
plt.rcParams['axes.labelsize'] = LABEL_SIZE       # 坐标轴标签的字体大小
plt.rcParams['axes.titlesize'] = TITLE_SIZE       # 标题的字体大小
plt.rcParams['xtick.labelsize'] = TICK_SIZE       # x轴刻度标签的字体大小
plt.rcParams['ytick.labelsize'] = TICK_SIZE       # y轴刻度标签的字体大小
plt.rcParams['legend.fontsize'] = LEGEND_SIZE       # 图例的字体大小
plt.rcParams['axes.spines.top'] = TOP_SPINE    # 设定上方的轴脊柱
plt.rcParams['axes.spines.right'] = RIGHT_SPINE    # 设定右方的轴脊柱
plt.rcParams['axes.spines.left'] = LEFT_SPINE    # 设定左方的轴脊柱
plt.rcParams['axes.spines.bottom'] = BOTTOM_SPINE    # 设定下方的轴脊柱
plt.rcParams['lines.linewidth'] = LINE_WIDTH    # 线宽
plt.rcParams['lines.markersize'] = MARKER_SIZE    # 标记大小
plt.rcParams['savefig.format'] = SAVEFIG_FORMAT    # 保存图形的格式
plt.rcParams['figure.dpi'] = FIG_DPI      # 图形的分辨率
plt.rcParams['savefig.dpi'] = SAVEFIG_DPI      # 保存图像的分辨率
plt.rcParams['legend.loc'] = LEGEND_LOC      # 图例位置
plt.rcParams['pdf.fonttype'] = PDF_FONTTYPE     # 使pdf文件中的文字可编辑
plt.rcParams['savefig.pad_inches'] = PAD_INCHES     # 图形边距
#endregion


# region 全局参数
# 随机种子
SEED = 421


# 数值处理参数
ROUND_DIGITS = 3
ROUND_FORMAT = 'general'
INF_POLICY = 'to_nan'
NAN_POLICY = 'drop'

# 文本处理参数
TEXT_PROCESS = {'capitalize': True, 'replace_underscore': ' ', 'ignore_equation_underscore': True}
FILENAME_PROCESS = {'replace_blank': '_'}

# 图形样式参数
BAR_WIDTH = 0.4
BIN_NUM = 20
XTICK_ROTATION = 90
YTICK_ROTATION = 0
STAR = '*'
STAR_SIZE = LINE_WIDTH*5
AUXILIARY_LINE_STYLE = '--'
PLT_CAP_SIZE = 5
SNS_CAP_SIZE = 0.1
ARROW_STYLE = '-|>'
ARROW_HEAD_WIDTH = 0.3
ARROW_HEAD_LENGTH = 0.5
FAINT_ALPHA = 0.5
TEXT_X = 0.05
TEXT_Y = 0.95
ELEV = 30
AZIM = 30
CBAR_LABEL_SIZE = TICK_SIZE
CBAR_TICK_SIZE = TICK_SIZE
MASK_COLOR = GREY
LABEL_PAD = LABEL_SIZE/3     # label间距(间距的单位是字体大小)
TITLE_PAD = TITLE_SIZE/3     # 标题间距(间距的单位是字体大小)
TEXT_VA = 'top'
TEXT_HA = 'left'
TEXT_KWARGS = {'verticalalignment': TEXT_VA, 'horizontalalignment': TEXT_HA, 'fontsize': FONT_SIZE, 'color': BLACK}

# 图形布局参数
BBOX_INCHES = None
SIDE_PAD = 0.03
CBAR_POSITION = {'position': 'right', 'size': 0.05, 'pad': SIDE_PAD}

# 颜色映射参数
CMAP = plt.get_cmap('viridis')
HEATMAP_CMAP = plt.get_cmap('jet')
DENSITY_CMAP = mcolors.LinearSegmentedColormap.from_list("density_map", ["#d0d0d0", BLUE])
CLABEL_KWARGS = {'inline': True, 'fontsize': FONT_SIZE, 'fmt': f'%.{ROUND_DIGITS}g'}

# 视频参数
SAVEVIDEO_FORMAT = 'mp4'
FRAME_RATE = 10

# 添加图片tag相关参数
TAG_CASE = 'lower'     # 默认tag大小写
TAG_PARENTHESES = False     # 默认tag是否加括号
FIG_TAG_POS = (0.05, 0.95)     # 默认图形tag位置
AX_TAG_POS = (-0.2, 1.2)     # 默认坐标轴tag位置
TAG_SIZE = SUP_TITLE_SIZE     # 默认tag字体大小
TAG_VA = TEXT_VA     # 默认tag垂直对齐方式
TAG_HA = TEXT_HA     # 默认tag水平对齐方式
# endregion


# region 设定gpu
def set_gpu(id):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = id  # specify which GPU(s) to be used


def find_least_used_gpu():
    # 执行nvidia-smi命令获取GPU状态
    smi_output = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits']).decode()

    # 初始化最小内存使用量和对应的GPU索引
    min_memory_used = float('inf')
    least_used_gpu_index = None

    # 解析输出，找到内存使用最少的GPU
    for line in smi_output.strip().split('\n'):
        gpu_index, memory_used = line.split(', ')
        memory_used = int(memory_used)
        if memory_used < min_memory_used:
            min_memory_used = memory_used
            least_used_gpu_index = gpu_index

    return least_used_gpu_index


def set_least_used_gpu():
    least_used_gpu = find_least_used_gpu()
    if least_used_gpu is not None:
        # 设置环境变量以使用内存使用最少的GPU
        set_gpu(least_used_gpu)
        print(f"Set to use the least used GPU: {least_used_gpu}")
    else:
        print("No GPU found.")
# endregion


# region 清空gpu内存
def clear_gpu_memory():
    print('to be done')
    # bm.disable_gpu_memory_preallocation()
# endregion


# region 随机种子
def set_seed(seed=SEED):
    '''设置随机种子'''
    random.seed(seed)
    np.random.seed(seed)


def local_seed(seed=SEED):
    '''设置局部随机种子'''
    return np.random.RandomState(seed)
# endregion


# region 获取时间
def get_time():
    '''获取当前时间'''
    return time.strftime('%Y-%m-%d-%H-%M-%S')


def get_start_time():
    '''获取开始时间'''
    return time.perf_counter()


def get_end_time():
    '''获取结束时间'''
    return time.perf_counter()


def get_interval_time(start, print_info=True, digits=ROUND_DIGITS, format_type=ROUND_FORMAT):
    '''获取时间间隔'''
    interval = time.perf_counter() - start
    if print_info:
        print(f'Interval time: {round_float(interval, digits, format_type)} s')
    return interval


def timer_decorator(func):
    '''
    计时器修饰器
    使用例子:
    @timer_decorator
    def some_function():
        # some code here
    '''
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        interval = end - start
        print(f'Interval time: {round_float(interval, digits=ROUND_DIGITS, format_type=ROUND_FORMAT)} s')
        return result
    return wrapper
# endregion


# region print
def print_sep(n=50, char='-'):
    '''打印分隔符'''
    print(char * n)


def print_title(title, n=50, char='-'):
    '''打印标题'''
    if title is not None:
        length_start = int((n-len(title))/2)
        length_end = n-len(title)-length_start
        print(length_start * char + title + length_end * char)
    else:
        print(char * n)


def print_array(arr, name=None, n=50, char='-'):
    '''打印数组'''
    print_title(name, n, char)
    print('shape: ', arr.shape)
    print('value: ', arr)
# endregion


# region 文件操作相关函数
def mkdir(fn):
    '''创建文件夹及中间文件夹'''
    if not os.path.isdir(fn):
        os.makedirs(fn)


def rmdir(fn):
    '''删除文件夹及其所有内容'''
    if os.path.isdir(fn):
        shutil.rmtree(fn, ignore_errors=True)


def cpdir(src, dst, dirs_exist_ok=True, overwrite=False):
    """
    复制一个文件夹下面的所有内容(不是大文件夹本身)到另一个文件夹里面

    参数:
    - src: 源文件夹
    - dst: 目标文件夹
    - dirs_exist_ok: 是否允许目标文件夹存在,默认为True,允许目标文件夹存在
    - overwrite: 是否允许覆盖目标文件夹,默认为False,不允许覆盖目标文件夹
    """
    if not os.path.isdir(src):
        raise ValueError("源路径不是一个目录")

    if not os.path.exists(dst):
        mkdir(dst)
    elif not dirs_exist_ok:
        raise FileExistsError("目标目录已存在")

    for root, dirs, files in os.walk(src):
        # 构造当前遍历的目录在目标路径中的对应路径
        rel_path = os.path.relpath(root, src)
        dst_path = os.path.join(dst, rel_path)

        if not os.path.exists(dst_path):
            mkdir(dst_path)

        for file in files:
            src_file_path = os.path.join(root, file)
            dst_file_path = os.path.join(dst_path, file)

            # 如果不允许覆盖，且目标文件已存在，则跳过
            if not overwrite and os.path.exists(dst_file_path):
                continue

            shutil.copy2(src_file_path, dst_file_path)


def mvdir(src, dst, overwrite=False):
    '''
    移动一个目录的最后一个层级到另一个目录里
    例子:
    mvdir('folder/subfolder', 'newfolder/')  # 将subfolder移动到newfolder下

    参数:
    - src: 源目录
    - dst: 目标目录
    - overwrite: 是否允许覆盖目标目录,默认为False,不允许覆盖目标目录
    '''
    # 如果src最后一个字符是'/',则去掉
    if src.endswith('/'):
        local_src = src[:-1]
    else:
        local_src = src

    # 使用os.path.basename直接获取目录名,无需手动查找最后一个'/'的位置
    dir_name = os.path.basename(local_src)

    # 构建最终目标路径,假设要保持原始目录名称
    final_dst = os.path.join(dst, dir_name)

    if overwrite:
        shutil.move(local_src, final_dst)
    else:
        # 检查目标路径是否存在
        if os.path.exists(final_dst):
            # 如果目标路径已存在,打印提示信息并终止操作
            print(
                f"The target folder '{final_dst}' already exists. Operation aborted.")
        else:
            # 如果目标路径不存在,执行移动操作
            shutil.move(local_src, final_dst)


def mvdir_cp_like(src, dst, dirs_exist_ok=True, overwrite=False, rm_src=True):
    """
    将一个文件夹下面的所有内容(不包括大文件夹本身)移动到另一个文件夹里面(与cpdir类似,但是是移动而不是复制)

    参数:
    - src: 源文件夹
    - dst: 目标文件夹
    - dirs_exist_ok: 是否允许目标文件夹存在,默认为True,允许目标文件夹存在
    - overwrite: 是否允许覆盖目标文件夹中的文件,默认为False,不允许覆盖目标文件夹中的文件
    """
    if not os.path.isdir(src):
        raise ValueError("源路径不是一个目录")

    if not os.path.exists(dst):
        os.mkdir(dst)
    elif not dirs_exist_ok:
        raise FileExistsError("目标目录已存在")

    for root, dirs, files in os.walk(src, topdown=False):
        # 构造当前遍历的目录在目标路径中的对应路径
        rel_path = os.path.relpath(root, src)
        dst_path = os.path.join(dst, rel_path)

        if not os.path.exists(dst_path):
            os.mkdir(dst_path)

        for file in files:
            src_file_path = os.path.join(root, file)
            dst_file_path = os.path.join(dst_path, file)

            # 如果不允许覆盖，并且目标文件已存在，则跳过
            if not overwrite and os.path.exists(dst_file_path):
                continue

            shutil.move(src_file_path, dst_file_path)

    # 删除源目录中已经为空的文件夹
    if rm_src:
        for root, dirs, files in os.walk(src, topdown=False):
            if not files and not dirs:
                os.rmdir(root)


def insert_dir(fn, insert_str):
    '''将文件夹下所有文件移动到insert_str文件夹下,insert_str文件夹放在fn文件夹下'''
    # 获取fn文件夹下的所有文件
    files = os.listdir(fn)
    # 创建insert_str文件夹
    mkdir(os.path.join(fn, insert_str))
    # 移动所有文件到insert_str文件夹下
    for file in files:
        shutil.move(os.path.join(fn, file), os.path.join(fn, insert_str, file))


def get_subdir(dir, full=True):
    '''找到文件夹下的一级子文件夹,full为True则返回全路径,否则只返回文件夹名'''
    if full:
        return [os.path.join(dir, d) for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    else:
        return [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]


def find_fig(filename, order=['.png', '.pdf', '.eps']):
    '''查找文件夹下的图片文件'''
    for ext in order:
        # Check both the original filename and the filename with each extension.
        if os.path.exists(filename) and filename.endswith(ext):
            return filename, True
        elif os.path.exists(filename+ext):
            return filename+ext, True
        elif os.path.exists(filename[-4:]+ext):
            return filename[-4:]+ext, True
    return '', False
# endregion


# region 保存和加载相关函数
def save_dict(params, filename, format_list=None):
    '''保存字典到txt和pickle文件'''
    if format_list is None:
        format_list = ['txt', 'pkl']

    # 创建文件夹
    mkdir(os.path.dirname(filename))

    # 假如filename有后缀,则添加到format_list中
    if filename.endswith('.txt') or filename.endswith('.pkl'):
        format_list.append(filename.split('.')[-1])
        filename = os.path.splitext(filename)[0]

    # 保存到txt
    if 'txt' in format_list:
        with open(filename+'.txt', 'w') as txt_file:
            for key, value in params.items():
                txt_file.write(f'{key}: {value}\n')

    # 保存到pickle
    if 'pkl' in format_list:
        save_pkl(params, filename+'.pkl')


def save_dir_dict(params, dir_key, basedir, dict_name, format_list=None):
    '''
    把一部分参数保存到路径名里,另一部分参数保存到文件里
    '''
    local_params = params.copy()
    for key in dir_key:
        basedir = os.path.join(basedir, concat_str([key, str(params[key])]))
        local_params.pop(key)
    save_dict(local_params, os.path.join(basedir, dict_name), format_list)


def save_timed_dir_dict(params, dir_key, basedir, dict_name, format_list=None):
    '''
    把一部分参数保存到路径名里,另一部分参数保存到文件里
    '''
    local_params = params.copy()
    for key in dir_key:
        basedir = os.path.join(basedir, concat_str([key, str(params[key])]))
        local_params.pop(key)
    basedir = os.path.join(basedir, get_time())
    save_dict(local_params, os.path.join(basedir, dict_name), format_list)


def save_pkl(obj, filename):
    '''保存对象到pickle文件'''
    # 创建文件夹
    mkdir(os.path.dirname(filename))

    # 保存到pickle
    if not filename.endswith('.pkl'):
        filename += '.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)


def load_pkl(filename):
    '''从pickle文件加载对象'''
    if not filename.endswith('.pkl'):
        filename += '.pkl'
    with open(filename, 'rb') as f:
        return pickle.load(f)


def param_exist(params, dir_key, basedir, pkl_name):
    '''
    比较参数,如果参数不同,则返回False
    '''
    local_params = params.copy()
    for key in dir_key:
        basedir = os.path.join(basedir, concat_str([key, str(params[key])]))
        local_params.pop(key)
    pkl_dir = os.path.join(basedir, pkl_name + '.pkl')
    if not os.path.exists(pkl_dir):
        return False
    else:
        exist_params = load_pkl(pkl_dir)
        return compare_dict(local_params, exist_params)


def timed_param_exist(params, dir_key, basedir, pkl_name):
    '''
    比较参数,如果参数不同,则返回False,这个函数会遍历时间文件夹
    '''
    local_params = params.copy()
    for key in dir_key:
        basedir = os.path.join(basedir, concat_str([key, str(params[key])]))
        local_params.pop(key)
    for time_dir in get_subdir(basedir):
        pkl_dir = os.path.join(time_dir, pkl_name + '.pkl')
        if os.path.exists(pkl_dir):
            exist_params = load_pkl(pkl_dir)
            return compare_dict(local_params, exist_params)
    return False


def compare_dict(dict1, dict2):
    '''比较两个字典的不同'''
    return dict1 == dict2


def save_df(df, filename, index=True, format_list=None):
    """
    Save a DataFrame to a CSV, Excel file, or Pickle, depending on the file extension or provided formats, with the option to include the index.

    Parameters:
    - df: DataFrame to save.
    - filename: Name of the base file to save the DataFrame to (without extension).
    - index: Boolean indicating whether to write the index to the file.
    - format_list: List of formats to save the DataFrame in.
    """
    if format_list is None:
        format_list = ['xlsx', 'pkl']

    # 创建文件夹
    mkdir(os.path.dirname(filename))

    # 将文件名和格式分开,并且将文件名的后缀添加到format_list中
    if filename.endswith('.csv') or filename.endswith('.xlsx') or filename.endswith('.pkl'):
        format_list.append(filename.split('.')[-1])
        base_filename = os.path.splitext(filename)[0]
    else:
        base_filename = filename
    
    # 分别保存到不同的文件
    for format in format_list:
        full_filename = f"{base_filename}.{format}"
        if format == 'csv':
            df.to_csv(full_filename, index=index)
        elif format == 'xlsx':
            df.to_excel(full_filename, index=index, engine='openpyxl')
        elif format == 'pkl':
            df.to_pickle(full_filename)
        else:
            raise ValueError(
                "Unsupported file type. Please use csv, xlsx, or pkl")


def load_df(filename, index_col=0, index_dtype=str, col_dtype=str):
    """
    Load a DataFrame from a CSV, Excel file, or Pickle file, depending on the file's extension.

    Parameters:
    - filename: Name of the file to load the DataFrame from (with extension).
    - index_col: Column(s) to set as index. If None, defaults to pandas' behavior.
    - index_dtype: Data type for the index column, defaults to str.
    """
    # Check if the file exists with the provided extension
    if os.path.exists(filename):
        if filename.endswith('.pkl'):
            df = pd.read_pickle(filename)
        elif filename.endswith('.xlsx'):
            df = pd.read_excel(
                filename, index_col=index_col, engine='openpyxl')
        elif filename.endswith('.csv'):
            df = pd.read_csv(filename, index_col=index_col)
        else:
            for ext in ['.pkl', '.xlsx', '.csv']:
                if os.path.exists(filename+ext):
                    return load_df(filename+ext, index_col, index_dtype, col_dtype)
            raise ValueError(
                "Unsupported file extension. Please ensure the file is .csv, .xlsx, or .pkl")

        df.index = df.index.astype(index_dtype)
        df.columns = df.columns.astype(col_dtype)

        # Convert all numerical columns to float
        df = df.apply(lambda x: x.astype(float)
                      if x.dtype.kind in 'iufc' else x)

        return df
    else:
        raise ValueError(
            "File not found. Please check the file path and extension.")
# endregion


# region 并行处理相关函数
def a():
    pass
# endregion


# region 生成字母序列
def get_tag(n, case=TAG_CASE, parentheses=TAG_PARENTHESES):
    if case == 'lower':
        # 生成小写字母标签
        tags = [chr(i) for i in range(ord('a'), ord('a') + n)]
    elif case == 'upper':
        # 生成大写字母标签
        tags = [chr(i) for i in range(ord('A'), ord('A') + n)]
    else:
        raise ValueError('Unknown case: ' + case)

    if parentheses:
        # 在每个标签前后添加括号
        tags = ['(' + tag + ')' for tag in tags]

    return tags
# endregion


# region 字符串处理相关函数
def format_text(string, text_process=None):
    '''格式化文本'''
    text_process = update_dict(TEXT_PROCESS, text_process)

    capitalize = text_process['capitalize']
    replace_underscore = text_process['replace_underscore']
    ignore_equation_underscore = text_process['ignore_equation_underscore']

    if string is None:
        return None
    else:
        # 分割字符串，保留$$之间的内容
        parts = []
        if ignore_equation_underscore:
            # 使用标志位记录是否在$$内
            in_math = False
            temp_str = ""
            for char in string:
                if char == "$" and not in_math:  # 开始$$
                    in_math = True
                    if temp_str:  # 保存$$之前的内容
                        parts.append(temp_str)
                        temp_str = ""
                    parts.append(char)
                elif char == "$" and in_math:  # 结束$$
                    in_math = False
                    parts.append(temp_str + char)
                    temp_str = ""
                elif in_math:  # $$内的内容直接添加
                    temp_str += char
                else:  # $$外的内容
                    if char == "_":
                        temp_str += replace_underscore if replace_underscore else "_"
                    else:
                        temp_str += char
            if temp_str:  # 添加最后一部分
                parts.append(temp_str)
        else:
            parts = [string.replace('_', replace_underscore)
                     if replace_underscore else string]

        # 处理大写转换
        if capitalize:
            processed_parts = [part[0].capitalize(
            ) + part[1:] if part else "" for part in parts]
        else:
            processed_parts = parts

        return "".join(processed_parts)


def round_float(number, digits=ROUND_DIGITS, format_type=ROUND_FORMAT):
    '''
    Rounds a float to a given number of digits and returns it in specified format.

    Parameters:
    - number: float, the number to be rounded.
    - digits: int, the number of digits to round to.
    - format_type: str, the format type for the output ('standard', 'scientific', 'percent', 'general'). 'general' means choose standard or scientific based on the length of the string representation.

    Returns:
    - str, the rounded number as a string in the specified format.
    '''
    if format_type == 'general':
        return f"{number:.{digits}g}"
    elif format_type == 'standard':
        return f"{number:.{digits}f}"
    elif format_type == 'scientific':
        return f"{number:.{digits}e}"
    elif format_type == 'percent':
        return f"{number:.{digits}%}"


def format_filename(string, file_process=None):
    '''格式化文件名'''
    file_process = update_dict(FILENAME_PROCESS, file_process)
    if string is None:
        return None
    else:
        return string.replace(' ', file_process['replace_blank'])


def concat_str(strs, sep='_', auto_rm_double_sep=True):
    '''连接字符串列表,并使用指定的分隔符连接'''
    if auto_rm_double_sep:
        return sep.join(strs).replace(sep*2, sep)
    else:
        return sep.join(strs)
# endregion


# region 字典处理相关函数
def create_dict(keys, values):
    '''创建字典'''
    return dict(zip(keys, values))


def union_dict(*dicts):
    '''合并多个字典'''
    # 假如有重复的key,则raise ValueError
    if len(set([k for d in dicts for k in d.keys()])) != sum([len(d) for d in dicts]):
        raise ValueError('Duplicated keys found in the input dictionaries.')
    else:
        return {k: v for d in dicts for k, v in d.items()}


def filter_dict(d, keys):
    '''过滤字典'''
    return {k: d[k] for k in keys if k in d}


def update_dict(original_dict, new_dict):
    '''更新字典'''
    if new_dict is None:
        return original_dict.copy()
    else:
        return {**original_dict, **new_dict}
# endregion


# region list处理相关函数
def flatten_list(input_list):
    flattened = []  # 初始化一个空列表，用于存放扁平化后的结果
    for item in input_list:
        if isinstance(item, list):  # 如果当前项是列表，递归调用flatten_list
            flattened.extend(flatten_list(item))
        else:
            flattened.append(item)  # 如果当前项不是列表，直接添加到结果中
    return flattened


def union_list(*lists):
    """
    获取多个列表的并集。
    
    参数:
    - lists: 一个或多个列表的可变参数。
    
    返回:
    - 一个列表，包含所有输入列表的并集。
    """
    union_set = set()
    for lst in lists:
        union_set = union_set.union(set(lst))
    return list(union_set)


def intersect_list(*lists):
    """
    获取多个列表的交集。
    
    参数:
    - lists: 一个或多个列表的可变参数。
    
    返回:
    - 一个列表，包含所有输入列表的交集。
    """
    intersection_set = set(lists[0])
    for lst in lists[1:]:
        intersection_set = intersection_set.intersection(set(lst))
    return list(intersection_set)


def rebuild_list_with_index(original, flattened, index=0):
    '''
    以original为模板，根据flattened中的元素重新构建一个列表。其括号结构与original相同，但元素顺序与flattened相同。这个函数必须要输出index才能不断调用自己，一般而言建议使用rebuild_list作为外部的接口。
    '''
    result = []
    for item in original:
        if isinstance(item, list):
            sub_list, index = rebuild_list_with_index(item, flattened, index)
            result.append(sub_list)
        else:
            result.append(flattened[index])
            index += 1
    return result, index


def rebuild_list(original, flattened):
    '''以original为模板，根据flattened中的元素重新构建一个列表。其括号结构与original相同，但元素顺序与flattened相同。'''
    return rebuild_list_with_index(original, flattened)[0]


def pure_list(l):
    '''遍历嵌套列表，将所有数组转换为列表，但保持嵌套结构不变。'''
    if isinstance(l, list):
        # 如果是列表，递归地对每个元素调用pure_list
        return [pure_list(item) for item in l]
    elif isinstance(l, np.ndarray):
        # 如果是numpy数组，先将其转换为列表，然后再对列表的每个元素递归调用pure_list
        return [pure_list(item) for item in l.tolist()]
    else:
        # 对于其他类型的元素，直接返回
        return l


def list_shape(lst):
    '''
    假定list是和numpy的array类似的嵌套列表，返回list的形状。(不能处理不规则的嵌套列表)
    '''
    if not isinstance(lst, list):
        return ()
    return (len(lst),) + list_shape(lst[0])
# endregion


# region DataFrame处理相关函数
def row_sum_df(df):
    '''计算DataFrame每行的和'''
    return df.sum(axis=1)


def row_mean_df(df):
    '''计算DataFrame每行的均值'''
    return df.mean(axis=1)


def row_std_df(df):
    '''计算DataFrame每行的标准差'''
    return df.std(axis=1)


def row_median_df(df):
    '''计算DataFrame每行的中位数'''
    return df.median(axis=1)


def col_sum_df(df):
    '''计算DataFrame每列的和'''
    return df.sum(axis=0)


def col_mean_df(df):
    '''计算DataFrame每列的均值'''
    return df.mean(axis=0)


def col_std_df(df):
    '''计算DataFrame每列的标准差'''
    return df.std(axis=0)


def col_median_df(df):
    '''计算DataFrame每列的中位数'''
    return df.median(axis=0)


def double_sum_df(df, sum_type='intersect'):
    if sum_type == 'intersect':
        sum_list = intersect_list(df.columns, df.index)
    elif sum_type == 'union':
        sum_list = union_list(df.columns, df.index)
    
    sum_series = pd.Series(np.zeros(len(sum_list)), index=sum_list)

    for sum_name in sum_list:
        for row in df.index:
            for col in df.columns:
                if row == sum_name or col == sum_name:
                    sum_series[sum_name] += df.loc[row, col]
    return sum_series
# endregion


# region array处理相关函数
def flatten_array(arr):
    '''扁平化数组'''
    return arr.flatten()
# endregion


# region 排序相关函数
def sort_dict(d, sort_by='value', reverse=False):
    '''
    Sort a dictionary by its keys or values.

    Parameters:
    - d: The dictionary to sort.
    - sort_by: Criteria for sorting ('value' or 'key').
    - reverse: Sort in descending order if True, else in ascending order.

    Returns:
    - A new sorted dictionary.
    '''
    if sort_by == 'key':
        return {k: d[k] for k in sorted(d.keys(), reverse=reverse)}
    elif sort_by == 'value':
        return {k: v for k, v in sorted(d.items(), key=lambda item: item[1], reverse=reverse)}
    else:
        raise ValueError("sort_by must be 'key' or 'value'")


def sort_dict_by_list(d, sort_by):
    '''
    Sort a dictionary's value by the order of elements in a list.

    Parameters:
    - d: The dictionary to sort.
    - sort_by: The list to use for sorting.

    Returns:
    - A new sorted dictionary.
    '''
    return {k: d[k] for k in sort_by if k in d}


def sort_list(l, reverse=False):
    '''
    Sort a list in ascending or descending order.

    Parameters:
    - l: The list to sort.
    - reverse: Sort in descending order if True, else in ascending order.

    Returns:
    - A new sorted list.
    '''
    return sorted(l, reverse=reverse)


def sort_list_as_subsequence(list_to_sort, sort_by, reverse=False):
    '''
    Sort a list (which is a subset of another list) based on the order of the other list.

    Parameters:
    - list_to_sort: The list to sort, which is a subset of 'sort_by'.
    - sort_by: The list to use for sorting.
    - reverse: Sort in descending order if True, else in ascending order.

    Returns:
    - A new list sorted according to the order of elements in 'sort_by'.
    '''
    # Create a dictionary that maps each value in 'sort_by' to its index
    index_map = {value: index for index, value in enumerate(sort_by)}

    # Filter 'list_to_sort' to include only those elements that are present in 'sort_by'
    # Then sort these elements based on their index in 'sort_by'
    sorted_list = sorted(filter(lambda x: x in index_map, list_to_sort),
                         key=lambda x: index_map[x],
                         reverse=reverse)

    return sorted_list


def sort_series(s, sort_by='value', reverse=False, ignore_capitalize=True):
    '''
    Sort a pandas Series by its index or values.

    Parameters:
    - s: The Series to sort.
    - sort_by: Criteria for sorting ('value' or 'index').
    - reverse: Sort in descending order if True, else in ascending order.
    - ignore_capitalize: Ignore capitalization when sorting by index.

    Returns:
    - A new sorted Series.
    '''
    if ignore_capitalize and sort_by == 'index':
        # Convert index to lowercase for case-insensitive sorting
        temp_series = s.copy()
        temp_series.index = s.index.str.lower()
        sorted_series = temp_series.sort_index(ascending=not reverse)
        # Restore original index order based on sorted lower case index
        # This step involves matching the lowercase sorted index with the original index
        # and arranging the original Series according to this order
        original_order_index = dict(zip(temp_series.index, s.index))
        final_index_order = [original_order_index[i] for i in sorted_series.index]
        return s.loc[final_index_order]
    elif sort_by == 'index':
        return s.sort_index(ascending=not reverse)
    elif sort_by == 'value':
        return s.sort_values(ascending=not reverse)
    else:
        raise ValueError("sort_by must be 'index' or 'value'")


def sort_df(df, sort_by='index', axis=0, key=None, reverse=False):
    '''
    Sort a pandas DataFrame by its index, a specific column, or columns.

    Parameters:
    - df: The DataFrame to sort.
    - sort_by: Criteria for sorting ('index', 'values').
    - axis: The axis to sort (0 for rows, 1 for columns). 
             Note: Sorting by columns ('values' with axis=1) is not supported in this function.
    - key: The column name(s) to sort by if sort_by is 'values'. 
           Ignored if sort_by is 'index'. Can be a single label or list of labels.
    - reverse: Sort in descending order if True, else in ascending order.

    Returns:
    - A new sorted DataFrame.

    Raises:
    - ValueError: If `sort_by` is not 'index' or 'values', or if sorting by columns with 'values'.
    '''
    if sort_by == 'index':
        return df.sort_index(axis=axis, ascending=not reverse)
    elif sort_by == 'values':
        if axis == 1:
            raise ValueError(
                "Sorting by columns ('values' with axis=1) is not supported.")
        if key is None:
            raise ValueError("Key must be provided when sorting by 'values'.")
        return df.sort_values(by=key, ascending=not reverse)
    else:
        raise ValueError("sort_by must be 'index' or 'values'")
# endregion


# region 函数处理相关函数
def get_default_param(func):
    sig = inspect.signature(func)
    return {
        k: v.default
        for k, v in sig.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


def get_all_func(module, only_module=True):
    """
    获取指定模块中的所有函数名。

    参数:
    module: Python模块对象，你想要获取函数名的模块。
    only_module: 布尔值，如果为True，只返回模块中自己定义的函数；
                 如果为False，返回模块中的所有函数，包括导入的函数。

    返回值:
    一个列表，包含了指定模块中的所有函数名。函数名按照在源文件中的位置排序。
    """
    if only_module:
        functions = [(name, obj) for name, obj in inspect.getmembers(module) 
                     if isinstance(obj, types.FunctionType) and obj.__module__ == module.__name__]
    else:
        functions = [(name, obj) for name, obj in inspect.getmembers(module) 
                     if isinstance(obj, types.FunctionType)]
    functions.sort(key=lambda x: inspect.getsourcelines(x[1])[1])  # 按照函数在源文件中的位置排序
    return [name for name, _ in functions]


def print_func_source(module, function_name):
    """
    打印指定模块中指定函数的源代码。

    参数:
    module: Python模块对象，你想要获取函数源代码的模块。
    function_name: 字符串，你想要获取源代码的函数的名字。

    返回值:
    无
    """
    function = getattr(module, function_name, None)
    if function and inspect.isfunction(function):
        print(f"Function name: {function_name}")
        print(inspect.getsource(function))
    else:
        print(f"No function named '{function_name}' in module '{module.__name__}'")


def write_func_to_file(module, functions, filename):
    if isinstance(functions, str):
        functions = [functions]
    with open(filename, 'w') as f:
        for func_name in functions:
            func = getattr(module, func_name)  # 获取函数对象
            f.write(f"# Function: {func_name}\n")
            f.write(inspect.getsource(func))
            f.write("\n\n")
# endregion


# region 数据清洗相关函数(nan,inf)
def process_inf(data, inf_policy=INF_POLICY):
    """Processes inf values according to the specified policy."""
    if inf_policy == 'to_nan':
        if isinstance(data, np.ndarray):
            data = np.where(np.isinf(data), np.nan, data)
        elif isinstance(data, list):
            data = np.where(np.isinf(data), np.nan, data).tolist()
        elif isinstance(data, (pd.Series, pd.DataFrame)):
            data = data.replace([np.inf, -np.inf], np.nan)
    elif inf_policy == 'ignore':
        pass
    return data


def process_nan(data, nan_policy=NAN_POLICY, fill_value=0):
    '''
    主要适用于一维数据,对于二维数据,drop interplate等操作不被允许,需要使用者分解为一维数据后再进行操作
    '''
    def df_like_interpolate(data):
        if isinstance(data, np.ndarray):
            return pd.Series(data).interpolate().values
        elif isinstance(data, list):
            return pd.Series(data).interpolate().tolist()
    if isinstance(data, list):
        data = np.array(data)
        if nan_policy == 'propagate':
            result = data
        elif nan_policy == 'fill':
            result = np.where(np.isnan(data), fill_value, data)
        elif nan_policy == 'drop':
            result = data[~np.isnan(data)]
        elif nan_policy == 'interpolate':
            result = df_like_interpolate(data)  # 使用和Pandas一致的插值方法
        else:
            raise ValueError('nan_policy not supported')

        return result.tolist()

    if isinstance(data, np.ndarray):
        if nan_policy == 'propagate':
            result = data
        elif nan_policy == 'fill':
            result = np.where(np.isnan(data), fill_value, data)
        elif nan_policy == 'drop':
            if data.ndim > 1:
                raise ValueError('drop nan not supported for multi-dimensional arrays')
            result = data[~np.isnan(data)]
        elif nan_policy == 'interpolate':
            if data.ndim > 1:
                raise ValueError('interpolate nan not supported for multi-dimensional arrays')
            result = df_like_interpolate(data)  # 使用和Pandas一致的插值方法
        else:
            raise ValueError('nan_policy not supported')

        return result

    if isinstance(data, (pd.Series, pd.DataFrame)):
        if nan_policy == 'propagate':
            result = data
        elif nan_policy == 'fill':
            result = data.fillna(fill_value)
        elif nan_policy == 'drop':
            if isinstance(data, pd.DataFrame):
                if len(data.columns) > 1:
                    raise ValueError('drop nan not supported for multi-column DataFrames')
            result = data.dropna()
        elif nan_policy == 'interpolate':
            if isinstance(data, pd.DataFrame):
                if len(data.columns) > 1:
                    raise ValueError('interpolate nan not supported for multi-column DataFrames')
            result = data.interpolate()  # 线性插值填充NaN
        else:
            raise ValueError('nan_policy not supported')

        return result


def process_special_value(data, nan_policy=NAN_POLICY, fill_value=0, inf_policy=INF_POLICY):
    data = process_inf(data, inf_policy=inf_policy)  # Process inf values first
    data = process_nan(data, nan_policy=nan_policy,
                       fill_value=fill_value)  # Process NaN values
    return data


def sync_special_value(*args, inf_policy=INF_POLICY):
    # Verify that all inputs have the same shape
    shapes = [a.shape if isinstance(
        a, (np.ndarray, pd.Series, pd.DataFrame)) else len(a) for a in args]
    if len(set(shapes)) > 1:
        raise ValueError("All inputs must have the same shape")

    args = [process_inf(a, inf_policy=inf_policy)
            for a in args]  # Process inf values first

    # Convert all inputs to numpy arrays for processing
    args_np = [np.array(a) if not isinstance(
        a, np.ndarray) else a for a in args]

    # Find indices where any input has NaN
    nan_mask = np.any(np.isnan(args_np), axis=0)

    # Apply NaN where any NaNs are found across corresponding elements
    args_np_synced = [np.where(nan_mask, np.nan, arg) for arg in args_np]

    # Convert back to original data types
    args_synced = []
    for original_arg, synced_arg in zip(args, args_np_synced):
        if isinstance(original_arg, list):
            args_synced.append(synced_arg.tolist())
        elif isinstance(original_arg, pd.Series):
            args_synced.append(pd.Series(synced_arg, index=original_arg.index))
        elif isinstance(original_arg, pd.DataFrame):
            args_synced.append(pd.DataFrame(
                synced_arg, index=original_arg.index, columns=original_arg.columns))
        else:  # numpy array
            args_synced.append(synced_arg)

    return args_synced if len(args_synced) > 1 else args_synced[0]


def sync_special_value_along_axis(data, sync_axis=0, inf_policy=INF_POLICY):
    # Input verification
    if not isinstance(data, (np.ndarray, pd.DataFrame, list)):
        raise ValueError("Data must be a numpy array, a list or a pandas DataFrame")

    if isinstance(data, pd.DataFrame):
        if sync_axis == 'row':
            sync_axis = 1
        elif sync_axis == 'column' or sync_axis == 'col':
            sync_axis = 0

    if isinstance(data, np.ndarray):
        local_data = data.copy()
    elif isinstance(data, list):
        local_data = np.array(data)
    elif isinstance(data, pd.DataFrame):
        local_data = data.values

    if not 0 <= sync_axis < local_data.ndim:
        raise ValueError("Invalid sync_axis for the data shape")
    
    # Apply inf_policy
    data_processed = process_inf(local_data, inf_policy=inf_policy)

    # Sync nan and possibly inf values along the specified axis
    if local_data.ndim == 1:
        # For 1D data, simply apply the syncing across the whole array
        nan_mask = np.isnan(data_processed)
    else:
        # For multi-dimensional data, iterate over slices along the sync_axis
        nan_mask = np.any(np.isnan(data_processed), axis=sync_axis, keepdims=True)
    
    # Apply the mask
    data_synced = np.where(nan_mask, np.nan, data_processed)

    # Convert back to original data type if needed
    if isinstance(data, pd.DataFrame):
        data_synced = pd.DataFrame(data_synced, index=data.index, columns=data.columns)
    elif isinstance(data, list):
        data_synced = data_synced.tolist()

    return data_synced
# endregion


# region 数据处理相关函数(缩放、标准化、裁剪)
def get_z_score(data):
    '''
    计算并返回数据集中每个数据点的Z分数。

    :param data: 一个数值型数组，代表数据集
    :return: 一个同样长度的数组，包含每个数据点的Z分数
    '''
    return (np.array(data) - np.nanmean(data)) / np.nanstd(data)


def get_z_score_dict(data_dict):
    '''
    计算并返回数据集中每个数据点的Z分数。

    :param data_dict: 一个字典，包含数据集的名称和数据
    :return: 一个字典，包含每个数据集的名称和Z分数
    '''
    return create_dict(data_dict.keys(), get_z_score(list(data_dict.values())))


def get_z_score_on_column(df, column_name):
    '''
    使用Z分数对数据集的指定列进行缩放。
    '''
    return pd.DataFrame(get_z_score(df[column_name].values), columns=[f'{column_name}_zscore'], index=df.index)


def get_min_max_scaling(data, min_val=0, max_val=1):
    '''
    使用最小-最大缩放对数据集进行缩放。

    :param data: 一个数值型数组，代表数据集
    :param feature_range: 一个包含两个元素的元组，代表缩放后的范围
    :return: 一个同样长度的数组，包含缩放后的数据集
    '''
    data = np.array(data)
    min_data = np.nanmin(data)
    max_data = np.nanmax(data)
    scaled_data = (data - min_data) / (max_data - min_data) * \
        (max_val - min_val) + min_val
    return scaled_data


def get_min_max_scaling_dict(data_dict, min_val=0, max_val=1):
    '''
    使用最小-最大缩放对数据集进行缩放。

    :param data_dict: 一个字典，包含数据集的名称和数据
    :param feature_range: 一个包含两个元素的元组，代表缩放后的范围
    :return: 一个字典，包含每个数据集的名称和缩放后的数据集
    '''
    return create_dict(data_dict.keys(), get_min_max_scaling(list(data_dict.values()), min_val, max_val))


def get_min_max_scaling_on_column(df, column_name, min_val=0, max_val=1):
    '''
    使用最小-最大缩放对数据集的指定列进行缩放。

    :param df: 一个Pandas DataFrame，包含数值型数据
    :param column_name: 一个字符串，指定要缩放的列名
    :param feature_range: 一个包含两个元素的元组，代表缩放后的范围
    :return: 一个新的DataFrame，包含与原始列相同的行索引和一个名为原始列名加上'_scaled'的新列，包含缩放后的数据
    '''
    return pd.DataFrame(get_min_max_scaling(df[column_name].values, min_val, max_val), columns=[f'{column_name}_scaled'], index=df.index)


def clip(data, vmin, vmax):
    """
    将输入数据中的值限制在vmin和vmax之间。支持列表、字典、Pandas DataFrame和Series。

    参数:
    - data: 输入数据，可以是列表,字典,np.array,Pandas DataFrame或Series。
    - vmin: 最小值阈值，数据中小于此值的将被设置为此值。
    - vmax: 最大值阈值，数据中大于此值的将被设置为此值。

    返回:
    - 修改后的数据，类型与输入数据相同。
    """
    if isinstance(data, dict):
        # 对字典的每个值应用clamp操作
        return {k: np.clip(v, vmin, vmax) for k, v in data.items()}
    elif isinstance(data, list):
        # 直接对列表应用np.clip
        return np.clip(data, vmin, vmax).tolist()
    elif isinstance(data, pd.Series) or isinstance(data, pd.DataFrame):
        # 对于Pandas DataFrame和Series，应用clip方法
        return data.clip(lower=vmin, upper=vmax)
    elif isinstance(data, np.ndarray):
        # 对于np.ndarray，直接应用np.clip
        return np.clip(data, vmin, vmax)
    else:
        raise TypeError(
            "Unsupported data type. Supported types are dict, list, np.array, pandas DataFrame, and pandas Series.")
# endregion


# region 数据分析相关函数
def get_distance(points):
    '''
    计算多个点的距离矩阵
    :param points: 二维数组,每行代表一个点的坐标
    :return: 距离矩阵
    '''
    return np.sqrt(np.sum((points[:, np.newaxis] - points[np.newaxis, :])**2, axis=-1))


def get_corr(x, y, nan_policy='drop', fill_value=0, inf_policy=INF_POLICY, sync=True):
    '''
    计算两个数组的相关系数
    sync表示是否将两个数组的nan和inf位置同步处理(假如一个数据有nan,另一个没有,则将两个数据的nan位置都设置为nan)
    '''
    x, y = sync_special_value(x, y, inf_policy=inf_policy) if sync else (x, y)
    return np.corrcoef(process_special_value(x, nan_policy=nan_policy, fill_value=fill_value, inf_policy=inf_policy), process_special_value(y, nan_policy=nan_policy, fill_value=fill_value, inf_policy=inf_policy))[0, 1]


def get_linregress(x, y, nan_policy='drop', fill_value=0, inf_policy=INF_POLICY, sync=True):
    '''
    对给定的数据进行线性回归拟合

    参数:
    x - x数据
    y - y数据

    返回:
    slope - 斜率
    intercept - 截距
    r - 相关系数
    p - p值
    std_err - 标准误差
    residual - 残差(y - y_pred)
    '''
    x, y = sync_special_value(x, y, inf_policy=inf_policy) if sync else (x, y)
    x = np.array(process_special_value(x, nan_policy=nan_policy,
                 fill_value=fill_value, inf_policy=inf_policy))
    y = np.array(process_special_value(y, nan_policy=nan_policy,
                 fill_value=fill_value, inf_policy=inf_policy))

    regress_dict = {}

    # 计算线性回归
    regress_dict['slope'], regress_dict['intercept'], regress_dict['r'], regress_dict['p'], regress_dict['std_err'] = st.linregress(
        x, y)
    regress_dict['residual'] = y - \
        (x * regress_dict['slope'] + regress_dict['intercept'])

    return regress_dict


def get_ks(x, y, nan_policy='drop', fill_value=0, inf_policy=INF_POLICY):
    '''
    计算两个数组的Kolmogorov-Smirnov距离

    参数:
    x - 第一个数组
    y - 第二个数组

    返回:
    ks_distance - 两个数组的KS距离
    '''
    x = process_special_value(x, nan_policy=nan_policy, fill_value=fill_value, inf_policy=inf_policy)
    y = process_special_value(y, nan_policy=nan_policy, fill_value=fill_value, inf_policy=inf_policy)
    return st.ks_2samp(x, y).statistic


def get_ks_and_p(x, y, nan_policy='drop', fill_value=0, inf_policy=INF_POLICY):
    '''
    计算两个数组的Kolmogorov-Smirnov距离及其P值,注意当分布一样时,KS距离为0,而P值为1

    参数:
    x - 第一个数组
    y - 第二个数组

    返回:
    ks_distance - 两个数组的KS距离
    p_value - 对应的P值
    '''
    x = process_special_value(x, nan_policy=nan_policy, fill_value=fill_value, inf_policy=inf_policy)
    y = process_special_value(y, nan_policy=nan_policy, fill_value=fill_value, inf_policy=inf_policy)
    ks_distance, p_value = st.ks_2samp(x, y)
    return ks_distance, p_value


def get_kl_divergence(p, q, base=None, **kwargs):
    '''
    计算两个概率分布的KL散度
    两个概率分布的长度必须相同

    参数:
    p - 第一个概率分布
    q - 第二个概率分布

    返回:
    kl_divergence - 两个概率分布的KL散度
    '''
    return st.entropy(p, q, base=base, **kwargs)


def get_nmf(X, n_components, init='nndsvda', random_state=0, max_iter=200, normalize_components=True, **kwargs):
    '''
    Perform dimensionality reduction using NMF. (对行向量进行的)

    Parameters:
    X - array, the original data.
    n_components - int, the number of components to keep.
    init - {'random', 'nndsvd', 'nndsvda', 'nndsvdar', 'custom'}, default 'random'.
    random_state - int, default 0.
    max_iter - int, default 200.
    normalize_components - bool, default True.

    Returns:
    W - array, the reduced dimensionality data.
    H - array, the components.
    '''
    model = NMF(n_components=n_components, init=init,
                random_state=random_state, max_iter=max_iter, **kwargs)
    W = model.fit_transform(X)
    H = model.components_

    if normalize_components:
        # Normalize each column in H to have unit L2 norm
        norms = np.linalg.norm(H, axis=1)
        H = H / norms[:, np.newaxis]

    return W, H


def get_entropy(p, base=None, **kwargs):
    '''
    计算概率分布的熵

    参数:
    p - 概率分布

    返回:
    entropy - 概率分布的熵
    '''
    return st.entropy(p, base=base, **kwargs)


def get_mutual_information(p, q, joint_pq):
    '''
    计算两个概率分布的互信息
    两个概率分布的长度必须相同

    参数:
    p - 第一个概率分布
    q - 第二个概率分布
    joint_pq - 联合概率分布

    返回:
    mutual_information - 两个概率分布的互信息
    '''
    return get_entropy(p) + get_entropy(q) - get_entropy(joint_pq)


def get_jsd(p, q, base=None, **kwargs):
    '''
    计算两个概率分布的Jensen-Shannon散度
    两个概率分布的长度必须相同

    参数:
    p - 第一个概率分布
    q - 第二个概率分布

    返回:
    jsd - 两个概率分布的Jensen-Shannon散度
    '''
    m = (np.array(p) + np.array(q)) / 2
    return (get_kl_divergence(p, m, base=base, **kwargs) + get_kl_divergence(q, m, base=base, **kwargs)) / 2


def get_pca(data, n_components=2):
    '''
    执行主成分分析 (PCA) 并返回降维后的数据
    自动处理NaN值,通过删除含有NaN的行

    :param data: DataFrame或二维数组,原始数据
    :param n_components: int, 保留的主成分数量
    :return: PCA转换后的DataFrame
    '''
    data_clean = data.dropna()
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(data_clean)
    principal_df = pd.DataFrame(data=principal_components,
                                columns=[f'Principal Component {i+1}' for i in range(n_components)])
    return principal_df


def get_fft(timeseries, T=None, sample_rate=None, nan_policy='interpolate', fill_value=0, inf_policy=INF_POLICY):
    '''
    执行快速傅里叶变换 (FFT) 并返回变换后的信号
    自动处理NaN值,通过线性插值填充NaN
    '''
    if T is None and sample_rate is None:
        raise ValueError("Either T or sample_rate must be provided")
    if T is None:
        T = 1 / sample_rate
    clean_timeseries = process_special_value(
        timeseries, nan_policy=nan_policy, fill_value=fill_value, inf_policy=inf_policy)
    n = len(clean_timeseries)
    yf = scipy.fft.fft(clean_timeseries)
    xf = scipy.fft.fftfreq(n, T)[:n//2]
    yf = 2.0/n * np.abs(yf[0:n//2])
    return xf, yf


def get_acf(timeseries, T=None, sample_rate=None, nlags=None, fft=True, nan_policy='interpolate', fill_value=0, inf_policy=INF_POLICY):
    '''
    计算自相关函数 (ACF) 并返回结果
    自动处理NaN值,通过线性插值填充NaN
    '''
    if T is None and sample_rate is None:
        raise ValueError("Either T or sample_rate must be provided")
    if T is None:
        T = 1 / sample_rate
    if nan_policy == 'interpolate':
        clean_timeseries = timeseries.copy()
        print('to be filled...')
    else:
        clean_timeseries = process_special_value(
            timeseries, nan_policy=nan_policy, fill_value=fill_value, inf_policy=inf_policy)
    return np.arange(nlags+1)*T, acf(clean_timeseries, nlags=nlags, fft=fft)


def get_multi_acf(multi_timeseries, T=None, sample_rate=None, nlags=None, fft=True, nan_policy='interpolate', fill_value=0, inf_policy=INF_POLICY):
    '''
    处理多个时间序列的自相关函数 (ACF) 并返回结果,multi_timeseries的shape为(time_series_num, time_series_length)
    '''
    multi_acf = []
    for timeseries in multi_timeseries:
        lag_times, acf_values = get_acf(timeseries, T=T, sample_rate=sample_rate, nlags=nlags, fft=fft, nan_policy=nan_policy, fill_value=fill_value, inf_policy=inf_policy)
        multi_acf.append(acf_values)
    return lag_times, np.array(multi_acf)


def get_hist(data, bins, stat='probability', nan_policy='drop', fill_value=0, inf_policy=INF_POLICY):
    '''
    Generates a histogram with various normalization options.

    Parameters:
    - data: array-like, the data to be histogrammed.
    - bins: int or array-like, the bin specification.
    - stat: str, the normalization method to use. One of ['count', 'frequency', 'probability', 'proportion', 'percent', 'density'] as sns.histplot.

    Returns:
    - values: array, the values corresponding to the normalization method chosen.
    - bin_edges: array, the edges of the bins.
    - midpoints: array, the midpoints of the bins.
    '''

    local_data = process_special_value(data, nan_policy=nan_policy, fill_value=fill_value, inf_policy=inf_policy)

    hist, bin_edges = np.histogram(local_data, bins=bins)

    # Calculate bin widths
    bin_widths = np.diff(bin_edges)

    if stat == 'count':
        # No change needed, hist already represents the count
        values = hist
    elif stat == 'frequency':
        # Frequency is count divided by bin width
        values = hist / bin_widths
    elif stat == 'probability' or stat == 'proportion':
        # Probability is count divided by total number of observations
        values = hist / hist.sum()
    elif stat == 'percent':
        # Percent is probability times 100
        values = (hist / hist.sum()) * 100
    elif stat == 'density':
        # Density normalizes count by total observations and bin width
        values = hist / (hist.sum() * bin_widths)
    else:
        raise ValueError("Unsupported stat method.")

    return values, bin_edges, get_midpoint(bin_edges)


def get_hist_2d(data_x, data_y, bins_x, bins_y, stat='probability', nan_policy='drop', fill_value=0, inf_policy=INF_POLICY, sync=True):
    '''
    Generates a 2D histogram with various normalization options for two-dimensional data,
    allowing separate bin specifications for each axis.

    Parameters:
    - data_x: array-like, the data for the x-axis to be histogrammed.
    - data_y: array-like, the data for the y-axis to be histogrammed.
    - bins_x: int or array-like, the bin specification for the x-axis.
    - bins_y: int or array-like, the bin specification for the y-axis.
    - stat: str, the normalization method to use. One of ['count', 'frequency', 'probability', 'proportion', 'percent', 'density'].

    Returns:
    - values: 2D array, the values corresponding to the normalization method chosen.
    - bin_edges_x: array, the edges of the bins for the x-axis.
    - bin_edges_y: array, the edges of the bins for the y-axis.
    - midpoints_x: array, the midpoints of the bins for the x-axis.
    - midpoints_y: array, the midpoints of the bins for the y-axis.
    '''

    if sync:
        local_data_x, local_data_y = sync_special_value(data_x, data_y, inf_policy=inf_policy)
    else:
        local_data_x, local_data_y = data_x.copy(), data_y.copy()
    local_data_x = process_special_value(local_data_x, nan_policy=nan_policy, fill_value=fill_value, inf_policy=inf_policy)
    local_data_y = process_special_value(local_data_y, nan_policy=nan_policy, fill_value=fill_value, inf_policy=inf_policy)

    hist, bin_edges_x, bin_edges_y = np.histogram2d(
        local_data_x, local_data_y, bins=[bins_x, bins_y])

    # Calculate bin areas for density calculation
    bin_widths_x = np.diff(bin_edges_x)
    bin_widths_y = np.diff(bin_edges_y)
    bin_areas = np.outer(bin_widths_x, bin_widths_y)

    if stat == 'count':
        values = hist
    elif stat == 'frequency':
        values = hist / bin_areas
    elif stat == 'probability' or stat == 'proportion':
        values = hist / hist.sum()
    elif stat == 'percent':
        values = (hist / hist.sum()) * 100
    elif stat == 'density':
        values = hist / (hist.sum() * bin_areas)
    else:
        raise ValueError("Unsupported stat method.")

    return values, bin_edges_x, bin_edges_y, get_midpoint(bin_edges_x), get_midpoint(bin_edges_y)


def get_kde(data, bw_method='scott', nan_policy='drop', fill_value=0, inf_policy=INF_POLICY, sync=True, **kwargs):
    '''
    计算核密度估计 (KDE) 并返回结果
    :param data: array-like, shape (dims, points) or (points,), the input data.
    '''
    if sync:
        local_data = sync_special_value_along_axis(data=data, sync_axis=0, inf_policy=inf_policy)
    else:
        local_data = data.copy()
    local_data = pure_list(local_data)
    shape = list_shape(local_data)
    if len(shape) > 1:
        for i in range(shape[0]):
            local_data[i] = process_special_value(
                local_data[i], nan_policy=nan_policy, fill_value=fill_value, inf_policy=inf_policy)
    else:
        local_data = process_special_value(
            local_data, nan_policy=nan_policy, fill_value=fill_value, inf_policy=inf_policy)
    return st.gaussian_kde(local_data, bw_method=bw_method, **kwargs)


def get_cdf(data, nan_policy='drop', fill_value=0, inf_policy=INF_POLICY, **kwargs):
    '''
    计算累积分布函数 (CDF) 并返回结果
    :param data: 1-d array-like, the input data.
    '''
    clean_data = process_special_value(
        data, nan_policy=nan_policy, fill_value=fill_value, inf_policy=inf_policy)
    return ECDF(clean_data, **kwargs)


def get_midpoint(x):
    '''
    Calculate the midpoints of a sequence of values.

    Parameters:
    - x: array-like, the values to calculate midpoints for.

    Returns:
    - midpoints: array, the midpoints of the input values.
    '''
    return (x[1:] + x[:-1]) / 2


def repeat_data(data, repeat_times):
    '''
    重复列表中的每个元素指定的次数。

    参数:
    - data: 要重复的数据，可以是列表、numpy数组、字典、Pandas Series或DataFrame。
    - repeat_times: 重复的次数，可以是整数、列表、numpy数组、字典、Pandas Series或DataFrame。

    返回:
    - 所有重复后的元素。
    '''
    print('array 需要改善,dict series需要改善')
    # For lists and numpy arrays
    if isinstance(data, list):
        if isinstance(repeat_times, int):
            return [item for item in data for _ in range(repeat_times)]
        elif isinstance(repeat_times, (list, np.ndarray)) and len(data) == len(repeat_times):
            return [item for item, count in zip(data, repeat_times) for _ in range(count)]
    elif isinstance(data, np.ndarray):
        if isinstance(repeat_times, int):
            return np.repeat(data, repeat_times)
        elif isinstance(repeat_times, (list, np.ndarray)):
            return np.repeat(data, repeat_times)

    # Handling dictionary data
    if isinstance(data, dict) and isinstance(repeat_times, (dict, int)):
        result = []
        if isinstance(repeat_times, int):
            for key, value in data.items():
                if isinstance(value, list):
                    result.extend(
                        [item for item in value for _ in range(repeat_times)])
                else:
                    result.extend([value for _ in range(repeat_times)])
        else:
            for key, value in data.items():
                if isinstance(value, list):
                    result.extend(
                        [item for item in value for _ in range(repeat_times.get(key, 1))])
                else:
                    result.extend(
                        [value for _ in range(repeat_times.get(key, 1))])
        return result

    # Convert Series and DataFrame outputs to list or np.array using pandas.concat
    if isinstance(data, pd.Series) and isinstance(repeat_times, (pd.Series, int)):
        if isinstance(repeat_times, int):
            return data.repeat(repeat_times).tolist()
        else:
            repeated_values = [pd.Series(
                [value] * repeat_times.get(index, 1)) for index, value in data.items()]
            repeated_series = pd.concat(repeated_values).reset_index(drop=True)
            return repeated_series.tolist()
    if isinstance(data, pd.DataFrame) and isinstance(repeat_times, int):
        return data.loc[data.index.repeat(repeat_times)].to_numpy()

    return None
# endregion


# region 稀疏矩阵相关函数
def rm_index_col():
    pass
# endregion


# region 作图相关函数
# region 初级作图函数(matplotlib系列,输入向量使用)
def plt_scatter(ax, x, y, label=None, color=BLUE, vert=True, **kwargs):
    '''
    使用x和y绘制散点图,可以接受plt.scatter的其他参数
    :param ax: matplotlib的轴对象,用于绘制图形
    :param x: x轴的数据
    :param y: y轴的数据
    :param label: 图例标签,默认为None
    :param color: 散点图的颜色,默认为BLUE
    :param kwargs: 其他plt.scatter支持的参数
    '''
    # 画图
    if not vert:
        x, y = y, x
    if 'c' in kwargs:
        return ax.scatter(x, y, label=label, **kwargs)
    else:
        return ax.scatter(x, y, label=label, color=color, **kwargs)


def plt_line(ax, x, y, label=None, color=BLUE, vert=True, **kwargs):
    '''
    使用x和y绘制折线图,可以接受plt.plot的其他参数
    :param ax: matplotlib的轴对象,用于绘制图形
    :param x: x轴的数据
    :param y: y轴的数据
    :param label: 图例标签,默认为None
    :param color: 折线图的颜色,默认为BLUE
    :param kwargs: 其他plt.plot支持的参数
    '''
    # 画图
    if not vert:
        x, y = y, x
    return ax.plot(x, y, label=label, color=color, **kwargs)


def plt_bar(ax, x, y, label=None, color=BLUE, vert=True, equal_space=False, err=None, capsize=PLT_CAP_SIZE, ecolor=BLACK, elabel=None, width=BAR_WIDTH, **kwargs):
    '''
    使用x和y绘制柱状图，可以接受plt.bar的其他参数,此函数的特性是会根据x的值作为bar的位置,当x包含字符串或者equal_space=True时,会自动变成等距离排列。
    :param ax: matplotlib的轴对象，用于绘制图形
    :param x: x轴的数据
    :param y: y轴的数据
    :param label: 图例标签，默认为None
    :param color: 柱状图的颜色，默认为BLUE
    :param vert: 是否为垂直柱状图，默认为True，即纵向
    :param equal_space: 是否将x的值作为字符串处理，这将使得柱子等距排列，默认为False
    :param err: 误差线的数据，默认为None
    :param capsize: 误差线帽大小，默认为PLT_CAP_SIZE
    :param ecolor: 误差线颜色，默认为BLACK
    :param elabel: 误差线图例标签，默认为None
    :param width: 柱子宽度，默认为None(当vert=False时,此参数将自动赋值给height)
    :param kwargs: 其他plt.bar或plt.barh支持的参数
    '''
    if equal_space:
        # 将x的每个元素变为字符串
        x = [str(i) for i in x]

    if err is not None:
        add_errorbar(ax, x, y, err, vert=vert, label=elabel,
                     color=color, capsize=capsize, ecolor=ecolor, **kwargs)
    if vert:
        # 绘制垂直柱状图
        return ax.bar(x, y, label=label, color=color, yerr=err, capsize=capsize, ecolor=ecolor, width=width, **kwargs)
    else:
        # 绘制水平柱状图
        return ax.barh(x, y, label=label, color=color, xerr=err, capsize=capsize, ecolor=ecolor, height=width, **kwargs)


def plt_stem(ax, x, y, label=None, linefmt='-', markerfmt='o', basefmt='-', linecolor=BLUE, markercolor=BLUE, basecolor=BLACK, **kwargs):
    '''
    使用x和y绘制棉棒图,可以接受plt.stem的其他参数
    :param ax: matplotlib的轴对象,用于绘制图形
    :param x: x轴的数据
    :param y: y轴的数据
    :param label: 图例标签,默认为None
    :param linefmt: 折线的格式,默认为'-'
    :param markerfmt: 点的格式,默认为'o'
    :param basefmt: 基线的格式,默认为'k-'
    :param kwargs: 其他plt.stem支持的参数
    '''
    # 画图
    stem_container = ax.stem(x, y, label=label, linefmt=linefmt,
                             markerfmt=markerfmt, basefmt=basefmt, **kwargs)

    stem_container.stemlines.set_color(linecolor)
    # 注意这里可能是markerlines，取决于您的matplotlib版本
    stem_container.markerline.set_color(markercolor)
    stem_container.baseline.set_color(basecolor)
    return stem_container


def plt_box(ax, x, y, label=None, patch_artist=True, boxprops=None, vert=True, **kwargs):
    '''
    使用x和y绘制箱形图,可以接受plt.boxplot的其他参数(和sns_box的输入方式完全不同,请注意并参考示例)
    :param ax: matplotlib的轴对象,用于绘制图形
    :param x: x轴的数据，应为一个列表或数组，其长度与y中的数据集合数量相匹配。示例:x = [1, 2, 3, 4]
    :param y: y轴的数据，每个位置的数据应该是一个列表或数组。示例:y = [[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]]
    :param label: 图例标签,默认为None
    :param patch_artist: 是否使用补丁对象,默认为True
    :param boxprops: 箱形图的属性
    :param vert: 是否为垂直箱形图,默认为True,即纵向
    :param kwargs: 其他plt.boxplot支持的参数
    '''
    if boxprops is None:
        boxprops = dict(facecolor=BLUE)

    # 添加一个高度为0的bar，用于添加图例，调整以支持横向箱形图
    if vert:
        ax.bar(x[0], 0, width=0, bottom=np.nanmean(y[0]),
               color=boxprops['facecolor'], linewidth=0, label=label, **kwargs)
    else:
        ax.barh(x[0], 0, height=0, left=np.nanmean(
            y[0]), color=boxprops['facecolor'], linewidth=0, label=label, **kwargs)

    # 画图
    return ax.boxplot(list(y), positions=x, patch_artist=patch_artist, boxprops=boxprops, vert=vert, **kwargs)


def plt_hist(ax, data, bins=BIN_NUM, label=None, color=BLUE, stat='probability', vert=True, **kwargs):
    '''
    使用数据绘制直方图,可以接受plt.hist的其他参数(不推荐使用,推荐使用sns_hist)
    :param ax: matplotlib的轴对象,用于绘制图形
    :param data: 用于绘制直方图的数据集
    :param bins: 直方图的箱数,默认为None,自动确定箱数
    :param label: 图例标签,默认为None
    :param color: 直方图的颜色,默认为BLUE
    :param stat: 直方图的统计方法,默认为'probability'
    :param vert: 是否为垂直直方图,默认为True,即纵向
    :param kwargs: 其他plt.hist支持的参数
    '''
    hist, edge, mid_point = get_hist(data, bins, stat=stat)
    return plt_bar(ax, mid_point, hist, label=label, color=color, width=edge[1]-edge[0], vert=vert, **kwargs)


def plt_hist_2d(ax, x, y, x_bins=BIN_NUM, y_bins=BIN_NUM, cmap=DENSITY_CMAP, label=None, stat='probability', cbar=True, cbar_position=None, cbar_label='stat', cbar_kwargs=None, vmin=None, vmax=None, **kwargs):
    '''
    使用x和y绘制二维直方图
    :param ax: matplotlib的轴对象,用于绘制图形
    :param x: x轴的数据
    :param y: y轴的数据
    :param x_bins: x轴的箱数,默认为BIN_NUM
    :param y_bins: y轴的箱数,默认为BIN_NUM
    :param cmap: 二维直方图的颜色映射表,默认为DENSITY_CMAP
    :param label: 图例标签,默认为None
    :param stat: 二维直方图的统计方法,默认为'probability'
    :param cbar: 是否添加颜色条,默认为True
    :param cbar_position: 颜色条的位置,默认为None
    :param cbar_kwargs: 其他plt.colorbar支持的参数
    :param vmin: 最小值阈值,数据中小于此值的将被设置为此值,默认为None
    :param vmax: 最大值阈值,数据中大于此值的将被设置为此值,默认为None
    :param kwargs: 其他plt.pcolormesh支持的参数
    '''
    if cbar_kwargs is None:
        cbar_kwargs = {}
    if cbar_label == 'stat':
        cbar_label = stat
    cbar_position = update_dict(CBAR_POSITION, cbar_position)

    h, x_edges, y_edges, x_midpoints, y_midpoints = get_hist_2d(x, y, x_bins, y_bins, stat=stat)

    if vmin is not None:
        h[h < vmin] = None
    if vmax is not None:
        h[h > vmax] = None

    ax.set_xlim(x_edges[0], x_edges[-1])
    ax.set_ylim(y_edges[0], y_edges[-1])

    pc = ax.pcolormesh(x_edges, y_edges, h.T, cmap=cmap, label=label, **kwargs)
    if cbar:
        cbars = add_colorbar(ax, pc, cmap=cmap, cbar_label=stat, cbar_position=cbar_position, **cbar_kwargs)
        return pc, cbars
    else:
        return pc


def plt_contour(ax, x, y, z, levels=None, color=BLUE, cmap=None, label=None, clabel=True, clabel_kwargs=None, **kwargs):
    '''
    使用x, y坐标网格和对应的z值绘制等高线图。
    :param ax: matplotlib的轴对象，用于绘制图形。
    :param x, Y: 定义等高线图网格的二维坐标数组。
    :param z: 在每个(x, y)坐标点上的z值。
    :param levels: 等高线的数量或具体的等高线级别列表，默认为自动确定。
    :param color: 等高线的颜色，默认为BLUE。
    :param cmap: 等高线的颜色映射表，默认为None。
    :param label: 如果不为None，添加一个颜色条并使用该标签。
    :param kwargs: 其他plt.contour支持的参数。
    '''
    clabel_kwargs = update_dict(CLABEL_KWARGS, clabel_kwargs)
    # 如果指定了cmap，则不使用color
    if cmap is not None:
        colors = None
    else:
        colors = [color]
    contour =  ax.contour(x, y, z, levels=levels, colors=colors, cmap=cmap, **kwargs)
    if clabel:
        plt.clabel(contour, **clabel_kwargs)
    if label is not None:
        if cmap is not None:
            print('If use cmap, label will be ignored')
        else:
            plt_line(ax, [], [], label=label, color=color)
    return contour


def plt_contourf(ax, x, y, z, levels=None, cmap=DENSITY_CMAP, contour=True, contour_color=BLACK, clabel=True, clabel_kwargs=None, cbar=True, cbar_kwargs=None, **kwargs):
    '''
    使用X, Y坐标网格和对应的Z值绘制填充等高线图。
    :param ax: matplotlib的轴对象，用于绘制图形。
    :param x, y: 定义等高线图网格的二维坐标数组。
    :param z: 在每个(x, y)坐标点上的z值。
    :param levels: 等高线的数量或具体的等高线级别列表，默认为自动确定。
    :param cmap: 等高线的颜色映射表，默认为None。
    :param contour: 是否绘制等高线，默认为True。
    :param contour_color: 等高线的颜色，默认为BLACK。
    :param clabel: 是否添加等高线标签，默认为True。
    :param clabel_kwargs: 用于自定义等高线标签的参数，例如字体大小和颜色。
    :param cbar: 是否添加颜色条，默认为True。
    :param cbar_kwargs: 用于自定义颜色条的参数，例如位置和标签。
    :param kwargs: 其他plt.contourf支持的参数。
    '''
    if cbar_kwargs is None:
        cbar_kwargs = {}
    clabel_kwargs = update_dict(CLABEL_KWARGS, clabel_kwargs)
    contourf =  ax.contourf(x, y, z, levels=levels, cmap=cmap, **kwargs)
    if contour or clabel:
        plt_contour(ax, x, y, z, levels=levels, color=contour_color, cmap=None, clabel=clabel, **kwargs)
    if cbar:
        add_colorbar(ax, contourf, cmap=cmap, **cbar_kwargs)
    return contourf


def plt_pie(ax, data, labels, colors=CMAP, explode=None, autopct='%1.1f%%', startangle=90, shadow=False, textprops=None, **kwargs):
    '''
    在指定的轴上绘制饼图。
    :param ax: matplotlib的轴对象，用于绘制图形。
    :param data: 数据列表，每个元素对应饼图的一个部分。示例:pie_data = [30, 15, 45, 10]
    :param labels: 标签列表，与数据一一对应。示例:pie_labels = ['Category A', 'Category B', 'Category C', 'Category D']
    :param colors: 每个饼块的颜色列表，示例:pie_colors = ['blue', 'green', 'red', 'purple'];也可以指定cmap,默认为CMAP,然后自动生成颜色序列
    :param explode: 用于强调饼块的偏移量列表，可选。示例:pie_explode = (0, 0.1, 0, 0)
    :param autopct: 自动百分比显示格式。
    :param startangle: 饼图的起始角度。
    :param shadow: 是否显示阴影。
    :param textprops: 用于自定义文本样式的字典，例如字体大小和颜色。
    :param kwargs: 其他matplotlib.pie支持的参数。
    ax.pie()方法绘制饼图，支持通过textprops自定义文本样式。
    '''
    if textprops is None:
        textprops = {'fontsize': LABEL_SIZE, 'color': BLACK}
    if not isinstance(colors, list):
        colors = colors(np.linspace(0, 1, len(data)))

    ax.axis('equal')  # 保持圆形，确保饼图是正圆形。
    return ax.pie(data, labels=labels, colors=colors, explode=explode, autopct=autopct, startangle=startangle, shadow=shadow, textprops=textprops, **kwargs)


def plt_donut(ax, data, labels, colors=CMAP, explode=None, autopct='%1.1f%%', startangle=90, shadow=False, wedgeprops=None, textprops=None, **kwargs):
    '''
    在指定的轴上绘制环形图。
    :param ax: matplotlib的轴对象，用于绘制图形。
    :param data: 数据列表，每个元素对应环形图的一个部分。
    :param labels: 标签列表，与数据一一对应。
    :param colors: 每个环块的颜色列表;也可以指定cmap,默认为CMAP,然后自动生成颜色序列
    :param explode: 用于强调环块的偏移量列表，可选。
    :param autopct: 自动百分比显示格式。
    :param startangle: 环形图的起始角度。
    :param shadow: 是否显示阴影。
    :param wedgeprops: 环形图中心空白部分的属性，如宽度和边缘颜色。
    :param textprops: 用于自定义文本样式的字典，例如字体大小和颜色。
    :param kwargs: 其他matplotlib.pie支持的参数。
    ax.pie()方法绘制环形图，wedgeprops定义环的宽度和边缘颜色，支持通过textprops自定义文本样式。
    '''
    if wedgeprops is None:
        wedgeprops = {'width': 0.3, 'edgecolor': 'w'}
    if textprops is None:
        textprops = {'fontsize': LABEL_SIZE, 'color': BLACK}
    ax.axis('equal')  # 保持圆形，确保环形图是正圆形。
    return plt_pie(ax, data, labels=labels, colors=colors, explode=explode, autopct=autopct, startangle=startangle, shadow=shadow, wedgeprops=wedgeprops, textprops=textprops, **kwargs)
# endregion


# region 初级作图函数(matplotlib系列,三维作图)
def plt_scatter_3d(ax, x, y, z, label=None, color=BLUE, **kwargs):
    '''
    使用x、y和z绘制3D散点图,可以接受ax.scatter的其他参数
    :param ax: matplotlib的3D轴对象,用于绘制图形
    :param x: x轴的数据
    :param y: y轴的数据
    :param z: z轴的数据
    :param label: 图例标签,默认为None
    :param color: 散点图的颜色,默认为BLUE
    :param kwargs: 其他ax.scatter支持的参数
    '''
    # 画图
    return ax.scatter(x, y, z, label=label, color=color, **kwargs)


def plt_line_3d(ax, x, y, z, label=None, color=BLUE, **kwargs):
    '''
    使用x、y和z绘制3D折线图,可以接受ax.plot的其他参数
    :param ax: matplotlib的3D轴对象,用于绘制图形
    :param x: x轴的数据
    :param y: y轴的数据
    :param z: z轴的数据
    :param label: 图例标签,默认为None
    :param color: 折线图的颜色,默认为BLUE
    :param kwargs: 其他ax.plot支持的参数
    '''
    # 画图
    return ax.plot(x, y, z, label=label, color=color, **kwargs)


def plt_bar_3d(ax, x, y, z, dx, dy, dz, label=None, color=BLUE, **kwargs):
    '''
    使用x、y和z绘制3D柱状图,可以接受ax.bar3d的其他参数
    :param ax: matplotlib的3D轴对象,用于绘制图形
    :param x: x轴的数据
    :param y: y轴的数据
    :param z: z轴的数据
    :param label: 图例标签,默认为None
    :param color: 柱状图的颜色,默认为BLUE
    :param kwargs: 其他ax.bar3d支持的参数
    '''
    # 强行加label
    ax.plot([], [], [], color=color, label=label)
    # 画图
    return ax.bar3d(x, y, z, dx, dy, dz, color=color, **kwargs)


def plt_surface_3d(ax, x, y, z, label=None, color=BLUE, **kwargs):
    '''
    使用x、y和z绘制3D曲面图,可以接受ax.plot_surface的其他参数
    :param ax: matplotlib的3D轴对象,用于绘制图形
    :param x: x轴的数据
    :param y: y轴的数据
    :param z: z轴的数据
    :param label: 图例标签,默认为None
    :param color: 曲面图的颜色,默认为BLUE
    :param kwargs: 其他ax.plot_surface支持的参数
    '''
    # 画图
    return ax.plot_surface(x, y, z, label=label, color=color, **kwargs)


def plt_voxel_3d(ax, data, label=None, color=BLUE, **kwargs):
    '''
    使用数据绘制3D体素图,可以接受ax.voxels的其他参数
    :param ax: matplotlib的3D轴对象,用于绘制图形
    :param data: 三维数组,用于绘制3D体素图
    :param label: 图例标签,默认为None
    :param color: 体素图的颜色,默认为BLUE
    :param kwargs: 其他ax.voxels支持的参数
    '''
    # 加label
    ax.plot([], [], [], color=color, label=label)
    # 画图
    return ax.voxels(data, label=None, facecolors=color, **kwargs)


def plt_stem_3d(ax, x, y, z, label=None, linefmt='-', markerfmt='o', basefmt='k-', **kwargs):
    '''
    使用x、y和z绘制3D棉棒图,可以接受ax.stem的其他参数
    :param ax: matplotlib的3D轴对象,用于绘制图形
    :param x: x轴的数据
    :param y: y轴的数据
    :param z: z轴的数据
    :param label: 图例标签,默认为None
    :param linefmt: 折线的格式,默认为'-'
    :param markerfmt: 点的格式,默认为'o'
    :param basefmt: 基线的格式,默认为'k-'
    :param kwargs: 其他ax.stem支持的参数
    '''
    # 画图
    return ax.stem(x, y, z, label=label, linefmt=linefmt, markerfmt=markerfmt, basefmt=basefmt, **kwargs)
# endregion


# region 初级作图函数(sns系列,输入向量使用)
def sns_scatter(ax, x, y, label=None, color=BLUE, linewidth=0, **kwargs):
    '''
    使用x和y绘制散点图,可以接受sns.scatterplot的其他参数
    :param ax: matplotlib的轴对象,用于绘制图形
    :param x: x轴的数据
    :param y: y轴的数据
    :param label: 图例标签,默认为None
    :param color: 散点图的颜色,默认为BLUE
    :param kwargs: 其他sns.scatterplot支持的参数
    '''
    # 画图
    sns.scatterplot(x=x, y=y, ax=ax, label=label, color=color,
                    linewidth=linewidth, **kwargs)


def sns_line(ax, x, y, label=None, color=BLUE, **kwargs):
    '''
    使用x和y绘制折线图,可以接受sns.lineplot的其他参数
    :param ax: matplotlib的轴对象,用于绘制图形
    :param x: x轴的数据
    :param y: y轴的数据
    :param label: 图例标签,默认为None
    :param color: 折线图的颜色,默认为BLUE
    :param kwargs: 其他sns.lineplot支持的参数
    '''
    # 画图
    sns.lineplot(x=x, y=y, ax=ax, label=label, color=color, **kwargs)


def sns_bar(ax, x, y, label=None, bar_width=BAR_WIDTH, color=BLUE, orient='v', **kwargs):
    '''
    使用x和y绘制柱状图,可以接受sns.barplot的其他参数,特别注意,如果x需要展现出数字大小而不是等距,需要使用plt_bar函数
    :param ax: matplotlib的轴对象,用于绘制图形
    :param x: x轴的数据
    :param y: y轴的数据
    :param label: 图例标签,默认为None
    :param bar_width: 柱状图的宽度,默认为0.4
    :param color: 柱状图的颜色,默认为BLUE
    :param orient: 柱状图的方向,默认为'v',即纵向
    :param kwargs: 其他sns.barplot支持的参数
    '''

    # 画图
    if orient == 'v':
        sns.barplot(x=x, y=y, ax=ax, label=label, width=bar_width,
                    color=color, orient=orient, **kwargs)
    if orient == 'h':
        sns.barplot(y=x, x=y, ax=ax, label=label, width=bar_width,
                    color=color, orient=orient, **kwargs)


def sns_box(ax, x, y, label=None, color=BLUE, **kwargs):
    '''
    使用x和y绘制箱形图,可以接受sns.boxplot的其他参数(和plt_box的输入方式完全不同,请注意并参考示例)
    :param ax: matplotlib的轴对象,用于绘制图形
    :param x: x轴的数据，应为一个列表或数组，其长度与y中的数据集合数量相匹配。示例:x = ['A', 'A', 'B', 'B', 'C', 'C', 'C']
    :param y: y轴的数据，每个位置的数据应该是一个列表或数组。示例:y = [1, 2, 1, 2, 1, 2, 3]
    :param label: 图例标签,默认为None
    :param color: 箱形图的颜色,默认为BLUE
    :param kwargs: 其他sns.boxplot支持的参数
    '''
    sns.boxplot(x=x, y=y, ax=ax, color=color, **kwargs)
    if label:
        # 添加一个高度为0的bar，用于添加图例
        ax.bar(x[0], 0, width=0, bottom=np.nanmean(y[0]),
               color=color, linewidth=0, label=label, **kwargs)


def sns_hist(ax, data, bins=BIN_NUM, label=None, color=BLUE, log_scale=False, stat='probability', **kwargs):
    '''
    使用数据绘制直方图,可以接受sns.histplot的其他参数(推荐使用)
    :param ax: matplotlib的轴对象,用于绘制图形
    :param data: 用于绘制直方图的数据(一维数组或列表)
    :param bins: 直方图的箱数,默认为None,自动确定箱数
    :param label: 图例标签,默认为None
    :param color: 直方图的颜色,默认为BLUE
    :param log_scale: 是否使用对数刻度,默认为False
    :param stat: 统计类型,默认为'probability'.'count': show the number of observations in each bin;'frequency': show the number of observations divided by the bin width;'probability' or 'proportion': normalize such that bar heights sum to 1;'percent': normalize such that bar heights sum to 100;'density': normalize such that the total area of the histogram equals 1
    :param kwargs: 其他sns.histplot支持的参数
    '''
    # 画图
    sns.histplot(data, bins=bins, label=label, color=color,
                 ax=ax, log_scale=log_scale, stat=stat, **kwargs)
# endregion


# region 初级作图函数(sns系列,输入pd dataframe或series使用)
def sns_scatter_pd(ax, data, x=None, y=None, label=None, color=BLUE, **kwargs):
    '''
    使用data的x和y列绘制散点图,可以接受sns.scatterplot的其他参数;当x为'index'时,使用DataFrame的索引作为x轴;对于Series,使用索引作为x轴,值作为y轴；当x列有重复,则会自动合并重复的x列并在对应位置绘制多个散点图。
    :param ax: matplotlib的轴对象,用于绘制图形
    :param data: 用于绘制散点图的数据集
    :param x: x轴的列名
    :param y: y轴的列名
    :param label: 图例标签,默认为None
    :param color: 散点图的颜色,默认为BLUE
    :param kwargs: 其他sns.scatterplot支持的参数
    '''
    # 画图
    if isinstance(data, pd.Series):
        # 对于Series，使用索引作为x轴，值作为y轴
        sns.scatterplot(x=data.index, y=data.values, ax=ax,
                        label=label, color=color, **kwargs)
        ax.set_xlabel(data.index.name)
        ax.set_ylabel(data.name)
    elif isinstance(data, pd.DataFrame):
        if x == 'index':
            # 将索引作为一个列用于绘图
            sns.scatterplot(data=data.reset_index(), x='index',
                            y=y, ax=ax, label=label, color=color, **kwargs)
            ax.set_xlabel(data.index.name)
        else:
            sns.scatterplot(data=data, x=x, y=y, ax=ax,
                            label=label, color=color, **kwargs)


def sns_line_pd(ax, data, x=None, y=None, label=None, color=BLUE, **kwargs):
    '''
    使用data的x和y列绘制折线图,可以接受sns.lineplot的其他参数;当x为'index'时,使用DataFrame的索引作为x轴;对于Series,使用索引作为x轴,值作为y轴；当x列有重复,则会自动合并重复的x列并计算y的均值和标准误差，作为折线图的值和误差线。
    :param ax: matplotlib的轴对象,用于绘制图形。
    :param data: 用于绘制折线图的数据集，可以是pd.Series或pd.DataFrame。
    :param x: x轴的列名或'index'，对于pd.DataFrame有效。
    :param y: y轴的列名，对于pd.DataFrame有效。
    :param label: 图例标签，默认为None。
    :param color: 折线图的颜色，默认为BLUE。
    :param kwargs: 其他sns.lineplot支持的参数。
    '''
    if isinstance(data, pd.Series):
        # 对于Series，使用索引作为x轴，值作为y轴
        sns.lineplot(x=data.index, y=data.values, ax=ax,
                     label=label, color=color, **kwargs)
        ax.set_xlabel(data.index.name)
        ax.set_ylabel(data.name)
    elif isinstance(data, pd.DataFrame):
        if x == 'index':
            # 将索引作为一个列用于绘图
            sns.lineplot(data=data.reset_index(), x='index', y=y,
                         ax=ax, label=label, color=color, **kwargs)
            ax.set_xlabel(data.index.name)
        else:
            sns.lineplot(data=data, x=x, y=y, ax=ax,
                         label=label, color=color, **kwargs)


def sns_bar_pd(ax, data, x=None, y=None, label=None, bar_width=BAR_WIDTH, color=BLUE, capsize=SNS_CAP_SIZE, err_kws=None, orient='v', **kwargs):
    '''
    使用data的x和y列或索引和值绘制柱状图,可以接受sns.barplot的其他参数,现支持DataFrame和Series。对于dataframe,假如x列都不重复,则会自动变成等距离排列，假设x列有重复,则会自动合并重复的x列并计算y的均值和标准误差，作为柱状图的值和误差线。假如x为index,则会使用DataFrame的索引作为x轴，由于index是不重复的，所以会自动变成等距离排列。对于Series，自动使用series的index和值作图
    :param ax: matplotlib的轴对象,用于绘制图形
    :param data: 用于绘制柱状图的DataFrame或Series
    :param x: x轴的列名或为'index'时使用DataFrame的索引；对于Series，保持为None
    :param y: y轴的列名；对于Series，此参数不使用
    :param label: 图例标签,默认为None
    :param bar_width: 柱状图的宽度,默认为BAR_WIDTH
    :param color: 柱状图的颜色,默认为BLUE
    :param capsize: 误差线帽大小,默认为SNS_CAP_SIZE
    :param err_kws: 误差线的属性,默认为{'color': BLACK}
    :param kwargs: 其他sns.barplot支持的参数
    '''
    if err_kws is None:
        err_kws = {'color': BLACK}

    if isinstance(data, pd.Series):
        plot_data = data.reset_index()
        plot_data.columns = ['index', data.name or 'value']
        if orient == 'v':
            sns.barplot(x='index', y=data.name or 'value', ax=ax, data=plot_data, width=bar_width,
                        color=color, label=label, capsize=capsize, err_kws=err_kws, **kwargs)
            ax.set_xlabel(data.index.name)
            ax.set_ylabel(data.name)
        else:
            sns.barplot(x=data.name or 'value', y='index', ax=ax, data=plot_data, width=bar_width,
                        color=color, label=label, orient=orient, capsize=capsize, err_kws=err_kws, **kwargs)
            ax.set_ylabel(data.index.name)
            ax.set_xlabel(data.name)
    elif isinstance(data, pd.DataFrame):
        if x == 'index':
            plot_data = data.reset_index()
            if orient == 'v':
                sns.barplot(x='index', y=y, ax=ax, data=plot_data, width=bar_width,
                            color=color, label=label, capsize=capsize, err_kws=err_kws, **kwargs)
                ax.set_xticks(range(len(plot_data['index'])))
                ax.set_xticklabels(plot_data['index'])
                # 重置x轴标签
                ax.set_xlabel('')
            else:
                sns.barplot(x=y, y='index', ax=ax, data=plot_data, width=bar_width, color=color,
                            label=label, orient=orient, capsize=capsize, err_kws=err_kws, **kwargs)
                ax.set_yticks(range(len(plot_data['index'])))
                ax.set_yticklabels(plot_data['index'])
                # 重置y轴标签
                ax.set_ylabel('')
        else:
            if orient == 'v':
                sns.barplot(x=x, y=y, ax=ax, data=data, width=bar_width, color=color,
                            label=label, capsize=capsize, err_kws=err_kws, **kwargs)
            else:
                sns.barplot(x=y, y=x, ax=ax, data=data, width=bar_width, color=color,
                            label=label, orient=orient, capsize=capsize, err_kws=err_kws, **kwargs)


def sns_hist_pd(ax, data, x=None, bins=BIN_NUM, label=None, color=BLUE, **kwargs):
    '''
    使用data的x列绘制直方图,可以接受sns.histplot的其他参数,此函数自动兼容DataFrame和Series,但是series状态需要保证x为None
    :param ax: matplotlib的轴对象,用于绘制图形
    :param data: 用于绘制直方图的数据集
    :param x: x轴的列名
    :param bins: 直方图的箱数,默认为None,自动确定箱数
    :param label: 图例标签,默认为None
    :param color: 直方图的颜色,默认为BLUE
    :param kwargs: 其他sns.histplot支持的参数
    '''
    # 画图
    sns.histplot(data=data, x=x, bins=bins, label=label,
                 color=color, ax=ax, **kwargs)
    if isinstance(data, pd.Series):
        ax.set_xlabel('')


def sns_box_pd(ax, data, x, y, color=BLUE, orient='v', **kwargs):
    '''
    使用data的x和y列绘制箱形图,可以接受sns.boxplot的其他参数
    :param ax: matplotlib的轴对象,用于绘制图形
    :param data: 用于绘制箱形图的数据集
    :param x: x轴的列名
    :param y: y轴的列名
    :param color: 箱形图的颜色,默认为BLUE
    :param kwargs: 其他sns.boxplot支持的参数
    '''
    # 画图
    if orient == 'v':
        sns.boxplot(data=data, x=x, y=y, ax=ax, color=color, **kwargs)
    if orient == 'h':
        sns.boxplot(data=data, x=y, y=x, ax=ax, color=color, **kwargs)
# endregion


# region 初级作图函数(sns系列,可同时接受矩阵和dataframe)
def sns_heatmap(ax, data, cmap=HEATMAP_CMAP, continuous_tick_num=None, discrete=False, discrete_num=None, discrete_label=None, display_edge_ticks=True, square=True, cbar=True, cbar_position=None, cbar_label=None, xtick_rotation=XTICK_ROTATION, ytick_rotation=YTICK_ROTATION, show_xtick=True, show_ytick=True, show_all_xtick=True, show_all_ytick=True, xtick_fontsize=TICK_SIZE, ytick_fontsize=TICK_SIZE, mask=None, mask_color=MASK_COLOR, mask_cbar_ratio=None, mask_tick='mask', vmin=None, vmax=None, cbar_labelsize=CBAR_LABEL_SIZE, cbar_ticksize=CBAR_TICK_SIZE, text_process=None, round_digits=ROUND_DIGITS, round_format_type=ROUND_FORMAT, add_leq=False, add_geq=False, heatmap_kwargs=None, cbar_label_kwargs=None, cbar_kwargs=None):
    '''
    使用数据绘制热图,可以接受sns.heatmap的其他参数;注意,如果要让heatmap按照ax的框架显示,需要将square设置为False
    :param ax: matplotlib的轴对象,用于绘制图形
    :param data: 用于绘制热图的数据矩阵
    :param cmap: 热图的颜色映射，默认为CMAP。可以是离散的或连续的，须与discrete参数相符合。
    :param continuous_tick_num: 连续颜色条的刻度数,默认为None,不作设置
    :param discrete: 是否离散显示,默认为False
    :param discrete_num: 离散显示的颜色数,默认为None,即使用默认颜色数
    :param discrete_label: 离散显示的标签,默认为None
    :param display_edge_ticks: 是否显示边缘刻度,默认为True
    :param square: 是否以正方形显示每个cell,默认为True
    :param cbar: 是否显示颜色条,默认为True
    :param cbar_position: 颜色条的位置,默认为None,即使用默认位置;position参数可选'left', 'right', 'top', 'bottom'
    :param xtick_rotation: x轴刻度标签的旋转角度
    :param ytick_rotation: y轴刻度标签的旋转角度
    :param show_xtick: 是否显示x轴的刻度标签
    :param show_ytick: 是否显示y轴的刻度标签
    :param show_all_xtick: 是否显示所有x轴的刻度标签
    :param show_all_ytick: 是否显示所有y轴的刻度标签
    :param xtick_fontsize: x轴刻度标签的字体大小
    :param ytick_fontsize: y轴刻度标签的字体大小
    :param mask: 用于遮盖的矩阵,默认为None
    :param mask_color: 遮盖矩阵的颜色,默认为'grey'
    :param mask_cbar_ratio: 遮盖矩阵颜色条的比例,默认为None,即使用默认比例
    :param mask_tick: 遮盖矩阵颜色条的标签,默认为'mask'
    :param vmin: 热图的最小值,默认为None,即使用数据的最小值
    :param vmax: 热图的最大值,默认为None,即使用数据的最大值
    :param cbar_labelsize: 颜色条标签的字体大小,默认为CBAR_LABEL_SIZE
    :param cbar_ticksize: 颜色条刻度标签的字体大小,默认为CBAR_TICK_SIZE
    :param text_process: 是否对颜色条标签进行文本处理,默认为TEXT_PROCESS
    :param round_digits: 是否对颜色条标签进行四舍五入,默认为ROUND_DIGITS
    :param add_leq: 是否在颜色条标签中添加小于等于符号,默认为False
    :param add_geq: 是否在颜色条标签中添加大于等于符号,默认为False
    :param heatmap_kwargs: 传递给sns.heatmap的其他参数
    :param cbar_label_kwargs: 传递给颜色条标签的其他参数
    '''
    text_process = update_dict(TEXT_PROCESS, text_process)
    # 如果data是array,则转换为DataFrame
    if isinstance(data, np.ndarray):
        local_data = pd.DataFrame(data)
    else:
        local_data = data.copy()
    if vmin is None:
        vmin = np.nanmin(local_data.values)
    if vmax is None:
        vmax = np.nanmax(local_data.values)
    if heatmap_kwargs is None:
        heatmap_kwargs = {}
    if cbar_label_kwargs is None:
        cbar_label_kwargs = {}
    if cbar_kwargs is None:
        cbar_kwargs = {}

    discrete_num, discrete_label = set_discrete_cmap_param(
        discrete, discrete_num, discrete_label)
    if discrete:
        cmap = discretize_cmap(cmap, discrete_num)

    # 绘制热图
    sns.heatmap(local_data, ax=ax, cmap=cmap,
                square=square, cbar=False, vmin=vmin, vmax=vmax, **heatmap_kwargs)
    # 设置x轴和y轴的刻度标签
    if show_xtick:
        if show_all_xtick:
            ax.set_xticks(np.arange(len(local_data.columns))+0.5)
            ax.set_xticklabels(
                local_data.columns, rotation=xtick_rotation, fontsize=xtick_fontsize)
        else:
            ax.set_xticklabels(ax.get_xticklabels(),
                               rotation=xtick_rotation, fontsize=xtick_fontsize)
        ax.xaxis.set_tick_params(labelbottom=show_xtick)
    else:
        ax.set_xticks([])

    if show_ytick:
        if show_all_ytick:
            ax.set_yticks(np.arange(len(local_data.index))+0.5)
            ax.set_yticklabels(
                local_data.index, rotation=ytick_rotation, fontsize=ytick_fontsize)
        else:
            ax.set_yticklabels(ax.get_yticklabels(),
                               rotation=ytick_rotation, fontsize=ytick_fontsize)
        ax.yaxis.set_tick_params(labelleft=show_ytick)
    else:
        ax.set_yticks([])

    if mask is not None:
        use_mask = True
        mask_cmap = mcolors.ListedColormap([mask_color])
        sns.heatmap(mask, ax=ax, cmap=mask_cmap,
                    cbar=False, mask=mask, square=square)
    else:
        use_mask = False

    # 显示颜色条
    if cbar:
        cbars = add_colorbar(ax, mappable=None, cmap=cmap, continuous_tick_num=continuous_tick_num, discrete=discrete, discrete_num=discrete_num, discrete_label=discrete_label, display_edge_ticks=display_edge_ticks, cbar_position=cbar_position, cbar_label=cbar_label, use_mask=use_mask,
                     mask_color=mask_color, mask_cbar_ratio=mask_cbar_ratio, mask_tick=mask_tick, label_size=cbar_labelsize, tick_size=cbar_ticksize, label_kwargs=cbar_label_kwargs, vmin=vmin, vmax=vmax, text_process=text_process, round_digits=round_digits, round_format_type=round_format_type, add_leq=add_leq, add_geq=add_geq, **cbar_kwargs)
        return cbars
# endregion


# region 初级作图函数(添加errorbar)
def add_errorbar(ax, x, y, err, label=None, color=BLACK, linestyle='None', capsize=PLT_CAP_SIZE, vert=True, equal_space=False, **kwargs):
    '''
    在指定位置添加误差线。当x包含字符串或者equal_space=True时,会自动变成等距离排列。
    :param ax: matplotlib的轴对象，用于绘制图形。
    :param x: 误差线的x坐标。也支持字符串，此时会自动转换为等距的。
    :param y: 误差线的y坐标。
    :param err: 误差线的数据。
    :param label: 误差线的标签，默认为None。
    :param color: 误差线的颜色，默认为BLACK。
    :param linestyle: 误差线的线型，默认为'None'。
    :param capsize: 误差线的线帽大小，默认为PLT_CAP_SIZE。capsize是相对于图的大小的,不会受到xlim和ylim的影响
    :param vert: 是否为垂直误差线，默认为True。
    :param equal_space: 是否将x的值作为字符串处理，这将使得柱子等距排列，默认为False
    :param kwargs: 传递给`ax.errorbar`的额外关键字参数。
    '''
    # 检查x中元素是否包含字符串
    if isinstance(x, (list, np.ndarray)):
        x = list(x)
        if any(isinstance(item, str) for item in x) or equal_space:
            # 如果x中包含字符串，则将其全部转换为字符串
            x = [str(item) for item in x]
            if vert:
                ax.set_xticks(np.arange(len(x)))
                ax.set_xticklabels(x)
            else:
                ax.set_yticks(np.arange(len(x)))
                ax.set_yticklabels(x)
            x = np.arange(len(x))
    if vert:
        return ax.errorbar(x, y, yerr=err, label=label, color=color, linestyle=linestyle, capsize=capsize, **kwargs)
    else:
        return ax.errorbar(y, x, xerr=err, label=label, color=color, linestyle=linestyle, capsize=capsize, **kwargs)
# endregion


# region 初级作图函数(添加ax)
def add_ax(fig, left, right, bottom, top, **kwargs):
    '''
    在指定位置添加一个新的ax。
    :param fig: matplotlib的图形对象，用于绘制图形。
    :param left: 新ax的左边界位置。
    :param right: 新ax的右边界位置。
    :param bottom: 新ax的下边界位置。
    :param top: 新ax的上边界位置。
    :param kwargs: 传递给`fig.add_axes`的其他参数。
    '''
    return fig.add_axes([left, bottom, right-left, top-bottom], **kwargs)


def add_side_ax(ax, position, relative_size, pad, sharex=None, sharey=None, hide_repeat_xaxis=True, hide_repeat_yaxis=True, **kwargs):
    '''
    在指定位置添加一个新的ax，并可以选择共享x轴或y轴。
    :param ax: matplotlib的轴对象，用于绘制图形。
    :param position: 新ax的位置，一个四元组(x0, y0, width, height)。
    :param relative_size: 新ax的宽度或高度，相对于原ax的对应大小。
    :param pad: 新ax与原ax的间距，相对于原ax的对应大小。
    :param sharex: 与新ax共享x轴的ax对象。也可以是True，表示与原ax共享x轴。
    :param sharey: 与新ax共享y轴的ax对象。也可以是True，表示与原ax共享y轴。
    :param hide_repeat_xaxis: 如果共享x轴，是否隐藏重复的x轴标签，默认为True。
    :param hide_repeat_yaxis: 如果共享y轴，是否隐藏重复的y轴标签，默认为True。
    :param kwargs: 传递给`fig.add_axes`的其他参数。
    '''
    if sharex is True:
        sharex = ax
    elif sharex is None or sharex is False:
        sharex = None
    if sharey is True:
        sharey = ax
    elif sharey is None or sharey is False:
        sharey = None

    fig = ax.get_figure()
    pos = ax.get_position()

    # 计算新ax的尺寸和位置
    if position in ['right', 'left']:
        size = pos.width * relative_size
    elif position in ['top', 'bottom']:
        size = pos.height * relative_size

    if position == 'right':
        new_pos = [pos.x0 + pos.width + pad * pos.width, pos.y0, size, pos.height]
    elif position == 'left':
        new_pos = [pos.x0 - size - pad * pos.width, pos.y0, size, pos.height]
    elif position == 'top':
        new_pos = [pos.x0, pos.y0 + pos.height + pad * pos.height, pos.width, size]
    elif position == 'bottom':
        new_pos = [pos.x0, pos.y0 - size - pad * pos.height, pos.width, size]

    # 创建并返回新的ax，可能共享x轴或y轴
    new_ax = fig.add_axes(new_pos, sharex=sharex, sharey=sharey, **kwargs)

    # 如果共享x轴且hide_repeat_xaxis为True，将特定的x轴标签设为不可见
    if sharex is not None and hide_repeat_xaxis:
        if position=='top':
            new_ax.xaxis.set_visible(False)
        elif position=='bottom':
            ax.xaxis.set_visible(False)

    # 如果共享y轴且hide_repeat_yaxis为True，将特定的y轴标签设为不可见
    if sharey is not None and hide_repeat_yaxis:
        if position=='right':
            new_ax.yaxis.set_visible(False)
        elif position=='left':
            ax.yaxis.set_visible(False)

    return new_ax
# endregion


# region 初级作图函数(分割ax)
def split_ax(ax, orientation='horizontal', pad=SIDE_PAD, ratio=0.5, sharex=None, sharey=None, keep_original_ax=False):
    """
    分割给定的轴对象ax为两个部分，根据指定的方向、间距和共享轴设置。

    :param ax: 要分割的轴对象
    :param orientation: 分割方向，'horizontal' 或 'vertical'
    :param pad: 两个新轴之间的间距，相对于原ax的大小
    :param ratio: 分割后两个新轴的比例，相对于原ax的大小，horizontal时为左边的轴的宽度比例，vertical时为上面的轴的高度比例
    :param sharex: 设置是否共享x轴
    :param sharey: 设置是否共享y轴
    :param keep_original_ax: 是否保留原始的ax，默认为False，即隐藏原始的ax,假如不是False,原始的ax会被保留,并且改变大小，保留在keep_original_ax设定的位置
    :return: 分割后的两个轴对象
    """
    fig = ax.figure  # 获取ax所在的figure对象
    pos = ax.get_position()  # 获取原始ax的位置和大小

    if keep_original_ax:
        # 根据orientation创建GridSpec布局
        if orientation == 'horizontal':
            gs = GridSpec(1, 2, figure=fig, left=pos.x0, right=pos.x1, bottom=pos.y0, top=pos.y1, wspace=pad, width_ratios=[ratio, 1 - ratio])
            ax0 = fig.add_subplot(gs[0], sharex=ax if sharex else None, sharey=ax if sharey else None)
            ax1 = fig.add_subplot(gs[1], sharex=ax if sharex else None, sharey=ax if sharey else None)
            if keep_original_ax == 'left':
                ax.set_position(ax0.get_position())
                ax0.set_visible(False)
                new_ax = ax1
            elif keep_original_ax == 'right':
                ax.set_position(ax1.get_position())
                ax1.set_visible(False)
                new_ax = ax0
        elif orientation == 'vertical':
            gs = GridSpec(2, 1, figure=fig, left=pos.x0, right=pos.x1, bottom=pos.y0, top=pos.y1, hspace=pad, height_ratios=[ratio, 1 - ratio])
            ax0 = fig.add_subplot(gs[0], sharex=ax if sharex else None, sharey=ax if sharey else None)
            ax1 = fig.add_subplot(gs[1], sharex=ax if sharex else None, sharey=ax if sharey else None)
            if keep_original_ax == 'top':
                ax.set_position(ax0.get_position())
                ax0.set_visible(False)
                new_ax = ax1
            elif keep_original_ax == 'bottom':
                ax.set_position(ax1.get_position())
                ax1.set_visible(False)
                new_ax = ax0
        else:
            raise ValueError("Invalid orientation value. Use 'horizontal' or 'vertical'.")

        # 当共享轴时，隐藏并排重复的特定轴标签
        if sharex and orientation == 'vertical':
            if keep_original_ax == 'top':
                plt.setp(new_ax.get_xticklabels(), visible=False)
            elif keep_original_ax == 'bottom':
                plt.setp(ax.get_xticklabels(), visible=False)
        if sharey and orientation == 'horizontal':
            if keep_original_ax == 'left':
                plt.setp(new_ax.get_yticklabels(), visible=False)
            elif keep_original_ax == 'right':
                plt.setp(ax.get_yticklabels(), visible=False)

        return new_ax
    else:
        # 隐藏原始的ax，使用新的ax替代
        ax.set_visible(False)

        # 根据orientation创建GridSpec布局
        if orientation == 'horizontal':
            gs = GridSpec(1, 2, figure=fig, left=pos.x0, right=pos.x1, bottom=pos.y0, top=pos.y1, wspace=pad, width_ratios=[ratio, 1 - ratio])
            ax0 = fig.add_subplot(gs[0])
            ax1 = fig.add_subplot(gs[1], sharex=ax0 if sharex else None, sharey=ax0 if sharey else None)
        elif orientation == 'vertical':
            gs = GridSpec(2, 1, figure=fig, left=pos.x0, right=pos.x1, bottom=pos.y0, top=pos.y1, hspace=pad, height_ratios=[ratio, 1 - ratio])
            ax0 = fig.add_subplot(gs[0])
            ax1 = fig.add_subplot(gs[1], sharex=ax0 if sharex else None, sharey=ax0 if sharey else None)
        else:
            raise ValueError("Invalid orientation value. Use 'horizontal' or 'vertical'.")

        # 当共享轴时，隐藏并排重复的特定轴标签
        if sharex and orientation == 'vertical':
            plt.setp(ax0.get_xticklabels(), visible=False)
        if sharey and orientation == 'horizontal':
            plt.setp(ax1.get_yticklabels(), visible=False)

        return ax0, ax1


def split_quadrant_relocate_ax(ax, x_side_ax_position, y_side_ax_position, x_side_ax_size, y_side_ax_size, x_side_ax_pad, y_side_ax_pad):
    '''
    将给定的轴对象ax分割为四个部分，并重新定位原始轴对象。
    '''
    fig = ax.get_figure()
    pos = ax.get_position()
    if x_side_ax_position == 'top':
        height_ratios = [x_side_ax_size, 1 - x_side_ax_size]
    elif x_side_ax_position == 'bottom':
        height_ratios = [1 - x_side_ax_size, x_side_ax_size]
    if y_side_ax_position == 'left':
        width_ratios = [y_side_ax_size, 1 - y_side_ax_size]
    elif y_side_ax_position == 'right':
        width_ratios = [1 - y_side_ax_size, y_side_ax_size]
    gs = GridSpec(2, 2, figure=fig, left=pos.x0, right=pos.x1, bottom=pos.y0, top=pos.y1, wspace=x_side_ax_pad, hspace=y_side_ax_pad, width_ratios=width_ratios, height_ratios=height_ratios)
    if x_side_ax_position == 'top' and y_side_ax_position == 'right':
        new_ax = fig.add_subplot(gs[1, 0])
        x_side_ax = fig.add_subplot(gs[0, 0], sharex=ax)
        y_side_ax = fig.add_subplot(gs[1, 1], sharey=ax)
    if x_side_ax_position == 'top' and y_side_ax_position == 'left':
        new_ax = fig.add_subplot(gs[1, 1])
        x_side_ax = fig.add_subplot(gs[0, 1], sharex=ax)
        y_side_ax = fig.add_subplot(gs[1, 0], sharey=ax)
    if x_side_ax_position == 'bottom' and y_side_ax_position == 'right':
        new_ax = fig.add_subplot(gs[0, 0])
        x_side_ax = fig.add_subplot(gs[1, 0], sharex=ax)
        y_side_ax = fig.add_subplot(gs[0, 1], sharey=ax)
    if x_side_ax_position == 'bottom' and y_side_ax_position == 'left':
        new_ax = fig.add_subplot(gs[0, 1])
        x_side_ax = fig.add_subplot(gs[1, 1], sharex=ax)
        y_side_ax = fig.add_subplot(gs[0, 0], sharey=ax)

    new_ax.set_visible(False)
    ax.set_position(new_ax.get_position())
    return x_side_ax, y_side_ax
# endregion


# region 初级作图函数(获取多个ax的位置的极值)
def get_extreme_ax_position(*args, position):
    '''
    获取多个ax的位置的极值。
    :param args: 多个ax对象。
    :param position: 位置参数，可选'left', 'right', 'top', 'bottom'。
    '''
    if position not in ['left', 'right', 'top', 'bottom']:
        raise ValueError('position参数错误')
    if position == 'left':
        return min([ax.get_position().x0 for ax in args])
    elif position == 'right':
        return max([ax.get_position().x1 for ax in args])
    elif position == 'top':
        return max([ax.get_position().y1 for ax in args])
    elif position == 'bottom':
        return min([ax.get_position().y0 for ax in args])
# endregion


# region 初级作图函数(生成cmap)
def get_cmap(colors, continuous=True):
    '''
    生成颜色映射。
    :param colors: 颜色列表。
    :param continuous: 是否为连续颜色映射，默认为True。
    '''
    if continuous:
        cmap = mcolors.LinearSegmentedColormap.from_list('custom', colors)
    else:
        cmap = mcolors.ListedColormap(colors)
    return cmap
# endregion


# region 初级作图函数(添加colorbar)
def discretize_cmap(cmap, discrete_num):
    '''
    将连续颜色条转换为离散颜色条。
    :param cmap: 连续颜色条的颜色映射。
    :param discrete_num: 离散颜色条的数量。
    '''
    cmap = mcolors.ListedColormap(
        cmap(np.linspace(0, 1, discrete_num, endpoint=True)))
    return cmap


def set_discrete_cmap_param(discrete, discrete_num, discrete_label):
    if discrete:
        if discrete_label is not None and discrete_num is not None:
            if discrete_num != len(discrete_label):
                raise ValueError('discrete_num和discrete_label的长度不一致')
        if discrete_num is None and discrete_label is None:
            raise ValueError('discrete_num和discrete_label不能同时为None')
        if discrete_num is None and discrete_label is not None:
            discrete_num = len(discrete_label)
        if discrete_num is not None and discrete_label is None:
            pass
    else:
        if discrete_num is not None or discrete_label is not None:
            raise ValueError('discrete_num和discrete_label只能在discrete为True时使用')
    return discrete_num, discrete_label


def add_colorbar(ax, mappable=None, cmap=CMAP, continuous_tick_num=None, discrete=False, discrete_num=None, discrete_label=None, display_edge_ticks=True, cbar_position=None, cbar_label=None, use_mask=False, mask_color=MASK_COLOR, mask_cbar_ratio=None, mask_tick='mask', label_size=CBAR_LABEL_SIZE, tick_size=CBAR_TICK_SIZE, label_kwargs=None, vmin=None, vmax=None, text_process=None, round_digits=ROUND_DIGITS, round_format_type=ROUND_FORMAT, add_leq=False, add_geq=False):
    '''
    在指定位置添加颜色条。特别注意，对于离散的cmap，用户一定要提供对应的discrete_num
    :param ax: matplotlib的轴对象，用于绘制图形。
    :param mappable: 用于绘制颜色条的对象，默认为None。
    :param cmap: 颜色条的颜色映射，默认为CMAP。可以是离散的或连续的，须与discrete参数相符合。
    :param continuous_tick_num: 连续颜色条的刻度数量，默认为None,不作设置。
    :param discrete: 是否为离散颜色条，默认为False。
    :param discrete_num: 离散颜色条的数量，默认为5。
    :param discrete_label: 离散颜色条的标签，默认为None。
    :param display_edge_ticks: 是否显示颜色条的边缘刻度，默认为True。(只在离散颜色条下有效)
    :param cbar_position: 颜色条的位置，默认为None,即使用默认位置;position参数可选'left', 'right', 'top', 'bottom'
    :param cbar_label: 颜色条的标签，默认为None。
    :param use_mask: 是否使用mask颜色条，默认为False。
    :param mask_color: mask颜色条的颜色，默认为'grey'。
    :param mask_cbar_ratio: mask颜色条的比例，默认为None。对于连续会自动设置为0.2，对于离散会自动设置为1/(discrete+1)
    :param mask_tick: mask颜色条的标签，默认为'mask'。
    :param label_size: 颜色条标签的字体大小，默认为CBAR_LABEL_SIZE。
    :param tick_size: 颜色条刻度标签的字体大小，默认为CBAR_TICK_SIZE。
    :param label_kwargs: 颜色条标签的其他参数，默认为None。
    :param vmin: 颜色条的最小值，默认为None。
    :param vmax: 颜色条的最大值，默认为None。
    :param text_process: 文本处理函数，默认为TEXT_PROCESS。
    :param round_digits: 刻度标签的小数位数，默认为ROUND_DIGITS。
    :param add_leq: 是否在最小值处添加'<=',默认为False。
    :param add_geq: 是否在最大值处添加'>=',默认为False。
    '''
    discrete_num, discrete_label = set_discrete_cmap_param(
        discrete, discrete_num, discrete_label)
    if discrete:
        cmap = discretize_cmap(cmap, discrete_num)

    if mappable is None:
        # 创建一个默认的ScalarMappable对象，这里使用默认的颜色映射和一个简单的线性规范
        norm = Normalize(vmin=vmin or 0, vmax=vmax or 1)  # 假定默认的数据范围为0到1
        mappable = ScalarMappable(norm=norm, cmap=cmap)
    if label_kwargs is None:
        label_kwargs = {}
    if vmin is None:
        vmin = mappable.norm.vmin
    if vmax is None:
        vmax = mappable.norm.vmax

    mappable.set_clim(vmin, vmax)

    text_process = update_dict(TEXT_PROCESS, text_process)

    if mask_cbar_ratio is None:
        if discrete:
            mask_cbar_ratio = 1/(discrete_num+1)
        else:
            mask_cbar_ratio = 0.2

    cbar_label = format_text(cbar_label, text_process=text_process)
    mask_tick = format_text(mask_tick, text_process=text_process)
    if discrete_label is not None:
        discrete_label = [format_text(
            label, text_process=text_process) for label in discrete_label]

    # 如果用户提供了cbar_position，则更新默认值
    cbar_position = update_dict(CBAR_POSITION, cbar_position)

    side_ax = add_side_ax(ax, cbar_position['position'], cbar_position['size'], cbar_position['pad'])

    if cbar_position['position'] in ['right', 'left']:
        orientation = 'vertical'
    if cbar_position['position'] in ['top', 'bottom']:
        orientation = 'horizontal'


    if use_mask:
        if cbar_position['position'] in ['right', 'left']:
            cbar_ax, mask_ax = split_ax(side_ax, orientation=orientation, pad=0, ratio=1-mask_cbar_ratio)
        if cbar_position['position'] in ['top', 'bottom']:
            mask_ax, cbar_ax = split_ax(side_ax, orientation=orientation, pad=0, ratio=mask_cbar_ratio)

        # 创建一个ScalarMappable对象，使用一个简单的颜色映射和Normalize对象,因为我们要显示一个纯色的colorbar，所以颜色映射范围不重要，只需确保vmin和vmax不同即可
        norm = Normalize(vmin=0, vmax=1)
        sm = ScalarMappable(norm=norm, cmap=mcolors.LinearSegmentedColormap.from_list(
            'pure_mask', [mask_color, mask_color]))

        # 创建colorbar，将ScalarMappable作为输入
        mask_cbar = plt.colorbar(sm, cax=mask_ax)

        # 设置colorbar只显示一个中间的刻度标签
        mask_cbar.set_ticks([0.5])   # 设置刻度位置在正中间
        mask_cbar.set_ticklabels([mask_tick], fontsize=tick_size)  # 设置刻度标签

        # 去掉新ax的坐标轴
        side_ax.axis('off')

        cbar = plt.colorbar(mappable, cax=cbar_ax, orientation=orientation)
    else:
        cbar = plt.colorbar(mappable, cax=side_ax, orientation=orientation)

    # 设置colorbar的标签
    cbar.set_label(cbar_label, fontsize=label_size, **label_kwargs)

    # 假如颜色映射是一个离散的，则自动设置刻度和标签
    if discrete:
        if display_edge_ticks:
            # 生成线性间隔的刻度
            ticks = np.linspace(vmin, vmax, 2*discrete_num+1)
            cbar.set_ticks(ticks)
            # 生成默认的刻度标签
            tick_labels = [str(round_float(tick, round_digits, round_format_type)) for tick in ticks]

            # 如果提供了 discrete_label，则替换相应的标签
            if discrete_label is not None:
                # 每隔一个刻度替换标签，跳过最开始和最末尾的刻度
                tick_labels[1::2] = discrete_label  # 只替换 discrete_label 指定的位置
        else:
            # 生成线性间隔的刻度
            ticks = np.linspace(vmin, vmax, 2*discrete_num+1)
            ticks = ticks[1::2]
            cbar.set_ticks(ticks)
            # 生成默认的刻度标签
            tick_labels = [str(round_float(tick, round_digits, round_format_type)) for tick in ticks]

            # 如果提供了 discrete_label，则替换相应的标签
            if discrete_label is not None:
                # 替换所有的标签
                tick_labels = discrete_label
    else:
        if continuous_tick_num is not None:
            ticks = np.linspace(vmin, vmax, continuous_tick_num, endpoint=True)
            tick_labels = [str(round(tick, round_digits)) for tick in ticks]
            cbar.set_ticks(ticks)

    # 假如tick_labels被定义过，则设置刻度标签
    try:
        # 设置<=和>=
        if add_leq:
            tick_labels[0] = '≤' + tick_labels[0]
        if add_geq:
            tick_labels[-1] = '≥' + tick_labels[-1]

        # 设置刻度标签
        cbar.set_ticklabels(tick_labels, fontsize=tick_size)
    except:
        pass

    # 调整tick的方向
    if use_mask:
        cbar_list = [cbar, mask_cbar]
    else:
        cbar_list = [cbar]
    for cb in cbar_list:
        if cbar_position['position']=='right':
            pass
        elif cbar_position['position']=='left':
            cb.ax.yaxis.set_ticks_position('left')
        elif cbar_position['position']=='top':
            cb.ax.xaxis.set_ticks_position('top')
            cb.ax.set_xticklabels(cb.ax.get_xticklabels(), rotation=90)
        elif cbar_position['position']=='bottom':
            cb.ax.xaxis.set_ticks_position('bottom')
            cb.ax.set_xticklabels(cb.ax.get_xticklabels(), rotation=90)

    # return cbar
    if use_mask:
        return [cbar, mask_cbar]
    else:
        return [cbar]
# endregion


# region 初级作图函数(添加边缘分布)
def add_marginal_distribution(ax, data, side_ax_position, side_ax_pad=SIDE_PAD, side_ax_size=0.3, outside=True, color=BLUE, hist=True, stat='density', bins=BIN_NUM, hist_kwargs=None, kde=True, kde_kwargs=None, rm_tick=True, rm_spine=True, rm_axis=True):
    '''
    在指定位置添加边缘分布。
    :param ax: matplotlib的轴对象，用于绘制图形。
    :param x: x轴的数据。
    :param y: y轴的数据。
    :param color: 边缘分布的颜色，默认为BLUE。
    :param hist_kwargs: 传递给`sns.histplot`的其他参数。
    '''
    if hist_kwargs is None:
        hist_kwargs = {}
    if kde_kwargs is None:
        kde_kwargs = {}
    
    if side_ax_position in ['right', 'left']:
        split_orientation = 'horizontal'
        hist_vert = False
    elif side_ax_position in ['top', 'bottom']:
        split_orientation = 'vertical'
        hist_vert = True

    # 得到边缘分布需要的ax
    if outside:
        if side_ax_position in ['right', 'left']:
            side_ax = add_side_ax(ax, side_ax_position, side_ax_size, side_ax_pad, sharey=True)
        elif side_ax_position in ['top', 'bottom']:
            side_ax = add_side_ax(ax, side_ax_position, side_ax_size, side_ax_pad, sharex=True)
    else:
        if side_ax_position in ['right', 'bottom']:
            side_ax = split_ax(ax, orientation=split_orientation, pad=side_ax_pad, ratio=1-side_ax_size, sharex=True if side_ax_position == 'bottom' else False, sharey=True if side_ax_position == 'right' else False, keep_original_ax='top' if side_ax_position == 'bottom' else 'left')
        elif side_ax_position in ['left', 'top']:
            side_ax = split_ax(ax, orientation=split_orientation, pad=side_ax_pad, ratio=side_ax_size, sharex=True if side_ax_position == 'top' else False, sharey=True if side_ax_position == 'left' else False, keep_original_ax='bottom' if side_ax_position == 'top' else 'right')

    # 绘制边缘分布
    if hist:
        plt_hist(ax=side_ax, data=data, bins=bins, color=color, stat=stat, vert=hist_vert, **hist_kwargs)
    if kde:
        plt_kde(ax=side_ax, data=data, color=color, vert=hist_vert, **kde_kwargs)
    side_ax.set_xlabel('')  # 隐藏x轴标签
    side_ax.set_ylabel('')  # 隐藏y轴标签

    # 隐藏边缘分布的刻度和坐标轴
    if rm_tick:
        if split_orientation == 'horizontal':
            rm_ax_tick(side_ax, 'x')
        elif split_orientation == 'vertical':
            rm_ax_tick(side_ax, 'y')
    if rm_spine:
        rm_ax_spine(side_ax)
    if rm_axis:
        rm_ax_axis(side_ax)

    # 调整边缘分布的方向
    if side_ax_position == 'left':
        side_ax.invert_xaxis()
    if side_ax_position == 'bottom':
        side_ax.invert_yaxis()
    return side_ax


def add_double_marginal_distribution(ax, x, y, outside=True, x_side_ax_position='top', y_side_ax_position='right', x_side_ax_pad=SIDE_PAD, y_side_ax_pad=SIDE_PAD, x_side_ax_size=0.3, y_side_ax_size=0.3, x_color=BLUE, y_color=BLUE, hist=True, stat='density', x_bins=BIN_NUM, y_bins=BIN_NUM, x_hist_kwargs=None, y_hist_kwargs=None, kde=True, x_kde_kwargs=None, y_kde_kwargs=None, rm_tick=True, rm_spine=True, rm_axis=True):
    '''
    在指定位置添加两个方向的边缘分布。
    :param ax: matplotlib的轴对象，用于绘制图形。
    :param x: x轴的数据。
    :param y: y轴的数据。
    :param outside: 是否将边缘分布放在外部，默认为True。
    :param x_side_ax_position: x轴边缘分布的位置，默认为'top'。
    :param y_side_ax_position: y轴边缘分布的位置，默认为'right'。
    :param x_side_ax_pad: x轴边缘分布与原ax的间距，默认为0.05。
    :param y_side_ax_pad: y轴边缘分布与原ax的间距，默认为0.05。
    :param x_side_ax_size: x轴边缘分布的大小，默认为0.3。
    :param y_side_ax_size: y轴边缘分布的大小，默认为0.3。
    :param x_color: x轴边缘分布的颜色，默认为BLUE。
    :param y_color: y轴边缘分布的颜色，默认为BLUE。
    :param hist: 是否绘制直方图，默认为True。
    :param stat: 直方图的统计量，默认为'probability'。
    :param x_bins: x轴直方图的箱数，默认为BIN_NUM。
    :param y_bins: y轴直方图的箱数，默认为BIN_NUM。
    :param x_hist_kwargs: 传递给`plt_hist`的其他参数。
    :param y_hist_kwargs: 传递给`plt_hist`的其他参数。
    :param kde: 是否绘制核密度估计，默认为True。
    :param x_kde_kwargs: 传递给`plt_kde`的其他参数。
    :param y_kde_kwargs: 传递给`plt_kde`的其他参数。
    :param rm_tick: 是否隐藏刻度，默认为True。
    :param rm_spine: 是否隐藏坐标轴，默认为False。
    '''
    if x_hist_kwargs is None:
        x_hist_kwargs = {}
    if y_hist_kwargs is None:
        y_hist_kwargs = {}
    if x_kde_kwargs is None:
        x_kde_kwargs = {}
    if y_kde_kwargs is None:
        y_kde_kwargs = {}
    
    # 添加边缘分布
    if outside:
        x_side_ax = add_marginal_distribution(ax, x, x_side_ax_position, side_ax_pad=x_side_ax_pad, side_ax_size=x_side_ax_size, outside=outside, color=x_color, hist=hist, stat=stat, bins=x_bins, hist_kwargs=x_hist_kwargs, kde=kde, kde_kwargs=x_kde_kwargs, rm_tick=rm_tick, rm_spine=rm_spine, rm_axis=rm_axis)
        y_side_ax = add_marginal_distribution(ax, y, y_side_ax_position, side_ax_pad=y_side_ax_pad, side_ax_size=y_side_ax_size, outside=outside, color=y_color, hist=hist, stat=stat, bins=y_bins, hist_kwargs=y_hist_kwargs, kde=kde, kde_kwargs=y_kde_kwargs, rm_tick=rm_tick, rm_spine=rm_spine, rm_axis=rm_axis)
    else:
        # 当边缘分布在内部时，需要手动分划ax然后获取需要的
        x_side_ax, y_side_ax = split_quadrant_relocate_ax(ax, x_side_ax_position=x_side_ax_position, y_side_ax_position=y_side_ax_position, x_side_ax_size=x_side_ax_size, y_side_ax_size=y_side_ax_size, x_side_ax_pad=x_side_ax_pad, y_side_ax_pad=y_side_ax_pad)
        # 绘制边缘分布
        if hist:
            plt_hist(ax=x_side_ax, data=x, bins=x_bins, color=x_color, stat=stat, vert=True, **x_hist_kwargs)
            plt_hist(ax=y_side_ax, data=y, bins=y_bins, color=y_color, stat=stat, vert=False, **y_hist_kwargs)
        if kde:
            plt_kde(ax=x_side_ax, data=x, color=x_color, vert=True, **x_kde_kwargs)
            plt_kde(ax=y_side_ax, data=y, color=y_color, vert=False, **y_kde_kwargs)

        # 隐藏边缘分布的刻度和坐标轴
        if rm_tick:
            rm_ax_tick(x_side_ax, 'y')
            rm_ax_tick(y_side_ax, 'x')
        if rm_spine:
            rm_ax_spine(x_side_ax)
            rm_ax_spine(y_side_ax)
        if rm_axis:
            rm_ax_axis(x_side_ax)
            rm_ax_axis(y_side_ax)

        # 调整边缘分布的方向
        if x_side_ax_position == 'bottom':
            x_side_ax.invert_yaxis()
        if y_side_ax_position == 'left':
            y_side_ax.invert_xaxis()
    return x_side_ax, y_side_ax
# endregion


# region 初级作图函数(添加star)
def add_star(ax, x, y, label=None, marker=STAR, star_color=RED, star_size=STAR_SIZE, linestyle='None', **kwargs):
    '''
    在指定位置添加五角星标记。
    :param ax: matplotlib的轴对象，用于绘制图形。
    :param x: 五角星的x坐标。
    :param y: 五角星的y坐标。
    :param label: 五角星的标签，默认为None。
    :param marker: 五角星的标记，默认为STAR。
    :param star_color: 五角星的颜色，默认为RED。
    :param star_size: 五角星的大小，默认为STAR_SIZE。
    :param linestyle: 连接五角星的线型，默认为'None'。
    :param kwargs: 传递给`plot`函数的额外关键字参数。
    '''
    # 画图
    return ax.plot(x, y, label=label, marker=marker, color=star_color, markersize=star_size, linestyle=linestyle, **kwargs)


def add_star_heatmap(ax, i, j, label=None, marker=STAR, star_color=RED, star_size=STAR_SIZE, linestyle='None', **kwargs):
    '''
    在热图的指定位置添加五角星标记。

    参数:
    - ax: matplotlib的Axes对象，即热图的绘图区域。
    - i, j: 要在其上添加五角星的行和列索引。(支持同时添加多个五角星，i和j可以是数组)
    - label: 五角星的标签，默认为None。
    - marker: 五角星的标记，默认为STAR。
    - star_color: 五角星的颜色，默认为RED。
    - star_size: 五角星的大小，默认为STAR_SIZE。
    - linestyle: 连接五角星的线型，默认为'None'。
    - kwargs: 传递给`plot`函数的额外关键字参数。
    '''

    # 转换行列索引为坐标位置
    x, y = np.array(j)+0.5, np.array(i)+0.5  # 0.5是为了将五角星放在格子的中心

    # 画图
    return add_star(ax, x, y, label=label, marker=marker, star_color=star_color, star_size=star_size, linestyle=linestyle, **kwargs)
# endregion


# region 初级作图函数(添加辅助线)
def add_vline(ax, x, label=None, color=RED, linestyle=AUXILIARY_LINE_STYLE, linewidth=LINE_WIDTH, **kwargs):
    '''
    在指定位置添加垂直线。
    :param ax: matplotlib的轴对象，用于绘制图形。
    :param x: 垂直线的x坐标。
    :param label: 垂直线的标签，默认为None。
    :param color: 垂直线的颜色，默认为RED。
    :param linestyle: 垂直线的线型，默认为AUXILIARY_LINE_STYLE。
    :param linewidth: 垂直线的线宽，默认为LINE_WIDTH。
    :param kwargs: 传递给`ax.axvline`的额外关键字参数。
    '''
    # 画图
    return ax.axvline(x, label=label, color=color, linestyle=linestyle, linewidth=linewidth, **kwargs)


def add_hline(ax, y, label=None, color=RED, linestyle=AUXILIARY_LINE_STYLE, linewidth=LINE_WIDTH, **kwargs):
    '''
    在指定位置添加水平线。
    :param ax: matplotlib的轴对象，用于绘制图形。
    :param y: 水平线的y坐标。
    :param label: 水平线的标签，默认为None。
    :param color: 水平线的颜色，默认为RED。
    :param linestyle: 水平线的线型，默认为AUXILIARY_LINE_STYLE。
    :param linewidth: 水平线的线宽，默认为LINE_WIDTH。
    :param kwargs: 传递给`ax.axhline`的额外关键字参数。
    '''
    # 画图
    return ax.axhline(y, label=label, color=color, linestyle=linestyle, linewidth=linewidth, **kwargs)
# endregion


# region 初级作图函数(添加箭头)
def add_arrow(ax, x_start, y_start, x_end, y_end, xycoords='data', label=None, fc=RED, ec=RED, linewidth=LINE_WIDTH, arrow_sytle=ARROW_STYLE, head_width=ARROW_HEAD_WIDTH, head_length=ARROW_HEAD_LENGTH, **kwargs):
    '''
    在指定位置添加箭头。更换了使用方式,现在是指定起点和终点,而不是指定增量,并且内部实际调用ax.annotation来保证箭头的头大小相对于ax美观,箭头的终点严格对应xy_end。可以处理单个箭头或多个箭头的情况。支持list,array,单个数字并且不会改变原始数据类型。
    :param ax: matplotlib的轴对象，用于绘制图形。
    :param x_start: 箭头的起始x坐标。
    :param y_start: 箭头的起始y坐标。
    :param x_end: 箭头的终止x坐标。
    :param y_end: 箭头的终止y坐标。
    :param label: 箭头的标签，默认为None。
    :param fc: 箭头的填充颜色，默认为RED。
    :param ec: 箭头的边框颜色，默认为RED。
    :param linestyle: 箭头的线型，默认为'-'。
    :param linewidth: 箭头的线宽，默认为LINE_WIDTH。
    :param arrow_sytle: 箭头的样式，默认为ARROW_STYLE。
    :param head_width: 箭头的宽度，默认为ARROW_HEAD_WIDTH。
    :param head_length: 箭头的长度，默认为ARROW_HEAD_LENGTH。
    :param adjust_end_point: 是否调整箭头的终点,使箭头的end坐标为箭头三角形终点(True)或者箭头直线的终点(False)
    :param kwargs: 传递给`ax.arrow`的额外关键字参数。
    '''

    # 确保起点和终点坐标可以迭代
    x_start, y_start, x_end, y_end = map(
        np.atleast_1d, [x_start, y_start, x_end, y_end])
    n_arrows = len(x_start)

    for i in range(n_arrows):
        add_annotation(ax, '', (x_end[i], y_end[i]), (x_start[i], y_start[i]), xycoords=xycoords, arrowprops={
                       'fc': fc, 'ec': ec, 'linewidth': linewidth, 'arrowstyle': arrow_sytle, 'head_length': head_length, 'head_width': head_width}, **kwargs)

    # 为了添加label而使用的虚拟箭头,恰好arrow的特性是不会因为在那个点画了一个0长度的箭头而改变xy轴的范围
    ax.arrow(0, 0, 0, 0, label=label, fc=fc, ec=ec,
             linewidth=linewidth, head_width=0, head_length=0, **kwargs)
# endregion


# region 初级作图函数(添加注释)
def add_annotation(ax, text, xy, xytext, xycoords='data', fontsize=FONT_SIZE, arrowprops=None, text_process=None, **kwargs):
    '''
    在指定位置添加注释。只支持一个注释。
    :param ax: matplotlib的轴对象，用于绘制图形。
    :param text: 注释的文本。
    :param xy: 注释的位置。即箭头的终止位置。
    :param xytext: 注释文本的位置。即箭头的起始位置。
    :param xycoords: 注释的坐标系，默认为'data'。也可以选择'axes fraction', 'subfigure fraction'等,见https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.annotate.html#matplotlib.axes.Axes.annotate。
    :param fontsize: 注释文本的字体大小，默认为FONT_SIZE。
    :param arrowprops: 箭头的属性
    :param text_process: 文本处理参数，默认为TEXT_PROCESS。
    :param kwargs: 传递给`ax.annotate`的额外关键字参数。
    '''
    text_process = update_dict(TEXT_PROCESS, text_process)
    if arrowprops is None:
        arrowprops = {'fc': RED, 'ec': RED, 'linewidth': LINE_WIDTH,
                      'arrowstyle': ARROW_STYLE, 'head_length': ARROW_HEAD_LENGTH, 'head_width': ARROW_HEAD_WIDTH}

    # 自动调整dict
    if 'head_length' in arrowprops.keys() and 'head_length' not in arrowprops['arrowstyle']:
        arrowprops['arrowstyle'] += ',head_length=' + \
            str(arrowprops['head_length'])
        del arrowprops['head_length']
    if 'head_width' in arrowprops.keys() and 'head_width' not in arrowprops['arrowstyle']:
        arrowprops['arrowstyle'] += ',head_width=' + \
            str(arrowprops['head_width'])
        del arrowprops['head_width']

    if xycoords == 'data':
        # 添加一个s=0的散点,因为annotation函数不会自动调整xy轴的坐标系让箭头可见
        ax.scatter([xy[0], xytext[0]], [xy[1], xytext[1]], s=0)

    # 处理文本
    text = format_text(text, text_process)
    # 画图
    return ax.annotate(text, xy, xytext=xytext, xycoords=xycoords, fontsize=fontsize, arrowprops=arrowprops, **kwargs)
# endregion


# region 初级作图函数(文字)
def add_text(ax, text, x=TEXT_X, y=TEXT_Y, text_process=None, transform='ax', **kwargs):
    '''

    在指定位置添加文字。
    :param ax: matplotlib的轴对象，用于绘制图形。
    :param text: 文本内容。
    :param x: 文本的x坐标，默认为0.05。
    :param y: 文本的y坐标，默认为0.95。
    :param text_process: 文本处理参数，默认为TEXT_PROCESS。
    :param transform: 文本的坐标系，默认为'ax'。也可以选择'fig'或'data'。
    :param kwargs: 传递给`ax.text`的额外关键字参数。
    '''
    text_process = update_dict(TEXT_PROCESS, text_process)

    # 处理文本
    text = format_text(text, text_process)

    if transform == 'ax':
        transform = ax.transAxes
    elif transform == 'fig':
        transform = ax.figure.transFigure
    elif transform == 'data':
        transform = ax.transData

    # 更新默认参数
    kwargs = update_dict(TEXT_KWARGS, kwargs)

    return ax.text(x, y, text, transform=transform, **kwargs)


def adjust_text(fig_or_ax, text, new_position=None, new_text=None, text_kwargs=None, text_process=None):
    """
    调整指定文本对象的文本,位置和对齐方式。
    
    参数:
    - fig_or_ax: matplotlib的Figure或Axes对象，指定在哪里搜索文本对象。
    - text: str, 要搜索和调整的文本内容。
    - new_position: tuple, 新的位置，格式为(x, y)。
    - new_text: str, 新的文本内容，默认为None,表示不修改文本内容。
    - text_kwargs: dict, 传递给`text.update`的其他参数，默认为None。
    """
    text_kwargs = update_dict(TEXT_KWARGS, text_kwargs)
    text_process = update_dict(TEXT_PROCESS, text_process)

    # 确定搜索范围是在Figure级别还是Axes级别
    if isinstance(fig_or_ax, plt.Figure):
        text_objects = fig_or_ax.findobj(plt.Text)
    else:
        text_objects = fig_or_ax.texts

    # 遍历找到的text对象
    for text_object in text_objects:
        if text_object.get_text() == text or text_object.get_text() == format_text(text, text_process):
            if new_position is not None:
                text_object.set_position(new_position)
            if new_text is not None:
                text_object.set_text(format_text(new_text, text_process))
            text_object.update(text_kwargs)
# endregion


# region 初级作图函数(添加tag)
def add_fig_tag(fig, tag, x=FIG_TAG_POS[0], y=FIG_TAG_POS[1], fontsize=TAG_SIZE, va=TAG_VA, ha=TAG_HA, **kwargs):
    '''
    在指定位置添加图的标签。
    :param fig: matplotlib的图对象，用于绘制图形。
    :param tag: 标签的文本。
    :param x: 标签的x坐标，默认为FIG_TAG_POS[0]。
    :param y: 标签的y坐标，默认为FIG_TAG_POS[1]。
    :param fontsize: 标签的字体大小，默认为SUP_TITLE_SIZE。
    :param va: 标签的垂直对齐方式，默认为TAG_VA。
    :param ha: 标签的水平对齐方式，默认为TAG_HA。
    :param kwargs: 传递给`fig.text`的额外关键字参数。
    '''
    # 画图
    return fig.text(x, y, tag, fontsize=fontsize, va=va, ha=ha, **kwargs)


def add_ax_tag(ax, tag, x=AX_TAG_POS[0], y=AX_TAG_POS[1], fontsize=TAG_SIZE, va=TAG_VA, ha=TAG_HA, **kwargs):
    '''
    在指定位置添加轴的标签。
    :param ax: matplotlib的轴对象，用于绘制图形。
    :param tag: 标签的文本。
    :param x: 标签的x坐标，默认为AX_TAG_POS[0]。
    :param y: 标签的y坐标，默认为AX_TAG_POS[1]。
    :param fontsize: 标签的字体大小，默认为TITLE_SIZE。
    :param va: 标签的垂直对齐方式，默认为TAG_VA。
    :param ha: 标签的水平对齐方式，默认为TAG_HA。
    :param kwargs: 传递给`ax.text`的额外关键字参数。
    '''
    # 画图
    return ax.text(x, y, tag, fontsize=fontsize, va=va, ha=ha, **kwargs)
# endregion


# region 初级作图函数(图例)
def rm_legend(ax):
    '''
    删除图例。
    :param ax: matplotlib的轴对象，用于绘制图形。
    '''
    ax.get_legend().remove()
# endregion


# region 复杂作图函数(matplotlib系列,输入向量使用)
def plt_group_bar(ax, x, y, label_list=None, bar_width=None, colors=CMAP, vert=True, **kwargs):
    '''
    绘制分组的柱状图。
    :param ax: matplotlib的轴对象,用于绘制图形
    :param x: x轴的分组标签
    :param y: 一个二维列表或数组,表示每组中柱子的高度
    :param label_list: 每个柱子的标签,例如['a', 'b', 'c']
    :param bar_width: 单个柱子的宽度,默认为None,自动确定宽度
    :param colors: 柱状图的颜色序列,应与label_list的长度相匹配;也可以指定cmap,默认为CMAP,然后根据label_list的长度生成颜色序列
    :param kwargs: 其他plt.bar支持的参数
    '''

    # 假如colors不是一个list,则用colors对应的cmap生成颜色
    if not isinstance(colors, list):
        colors = colors(np.linspace(0, 1, len(label_list)))

    num_groups = len(y)  # 组的数量
    num_bars = len(y[0])  # 每组中柱子的数量，假设每组柱子数量相同

    if bar_width is None:
        bar_width = BAR_WIDTH / num_bars

    # 为每个柱子计算中心位置
    indices = np.arange(num_groups)
    for i, lbl in enumerate(label_list):
        offsets = (np.arange(num_bars) -
                   np.arange(num_bars).mean()) * bar_width
        plt_bar(ax, indices + offsets[i], [y[j][i] for j in range(
            num_groups)], label=lbl, color=colors[i], width=bar_width, vert=vert, **kwargs)

    if vert:
        ax.set_xticks(indices)
        ax.set_xticklabels(x)
    else:
        ax.set_yticks(indices)
        ax.set_yticklabels(x)


def plt_two_side_bar(ax, x, y1, y2, label1=None, label2=None, bar_width=BAR_WIDTH, color1=BLUE, color2=RED, yerr1=None, yerr2=None, capsize=PLT_CAP_SIZE, ecolor1=None, ecolor2=None, elabel1=None, elabel2=None, vert=True, equal_space=False, **kwargs):
    '''
    使用x和y1,y2绘制两组柱状图
    '''
    plt_bar(ax, x, np.array(y1), label=label1, color=color1, vert=vert, equal_space=equal_space,
                 err=yerr1, capsize=capsize, ecolor=ecolor1, elabel=elabel1, width=bar_width, **kwargs)
    plt_bar(ax, x, -np.array(y2), label=label2, color=color2, vert=vert, equal_space=equal_space,
                 err=yerr2, capsize=capsize, ecolor=ecolor2, elabel=elabel2, width=bar_width, **kwargs)

    if vert:
        # 将y轴设置为对称
        ax.set_ylim(-max(max(y1), max(y2)) * 1.2, max(max(y1), max(y2)) * 1.2)

        # 修改y轴标签
        ax.set_yticks(ax.get_yticks())
        ax.set_yticklabels([str(abs(int(y))) for y in ax.get_yticks()])
    else:
        # 将x轴设置为对称
        ax.set_xlim(-max(max(y1), max(y2)) * 1.2, max(max(y1), max(y2)) * 1.2)

        # 修改x轴标签
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels([str(abs(int(x))) for x in ax.get_xticks()])


def plt_group_box():
    pass


def plt_linregress(ax, x, y, label=None, scatter_color=BLUE,
                        line_color=RED, show_list=None, round_digit_list=None, round_format_list=None, text_size=FONT_SIZE, scatter_kwargs=None, line_kwargs=None, text_kwargs=None, regress_kwargs=None):
    '''
    使用线性回归结果绘制散点图和回归线,可以输入scatter的其他参数

    参数:
    ax - 绘图的Axes对象
    x, y - 数据
    label - 数据标签
    x_label, y_label - 坐标轴标签
    title - 图标题
    fontsize - 标签字体大小
    x_sci, y_sci - 是否启用科学记数法
    scatter_color - 散点图颜色
    line_color - 线性回归线颜色
    show_corr - 是否显示相关系数
    show_corr_round - 相关系数小数点位数
    show_p - 是否显示 P 值
    show_p_round - P 值小数点位数
    '''
    if scatter_kwargs is None:
        scatter_kwargs = {}
    if line_kwargs is None:
        line_kwargs = {}
    if regress_kwargs is None:
        regress_kwargs = {}
    if text_kwargs is None:
        text_kwargs = {}
    if show_list is None:
        show_list = ['r', 'p']
    if round_digit_list is None:
        round_digit_list = [ROUND_DIGITS] * len(show_list)
    if round_format_list is None:
        round_format_list = ['general'] * len(show_list)

    # 计算线性回归
    regress_dict = get_linregress(x, y, **regress_kwargs)

    # 绘制散点图
    plt_scatter(ax, x, y, color=scatter_color,
                     label=label, **scatter_kwargs)

    # 绘制回归线
    plt_line(ax, x, x * regress_dict['slope'] +
                  regress_dict['intercept'], color=line_color, **line_kwargs)

    add_linregress_text(ax, regress_dict, show_list=show_list,
                        round_digit_list=round_digit_list, round_format_list=round_format_list, fontsize=text_size, **text_kwargs)


def plt_density_scatter(ax, x, y, label=None, label_cmap_index='max', estimate_type='kde', bw_method='auto', bins_x=BIN_NUM, bins_y=BIN_NUM, cmap=DENSITY_CMAP, cbar=True, cbar_label='density', cbar_position=None, linregress=True, line_color=RED, show_list=None, round_digit_list=None, round_format_list=None, text_size=FONT_SIZE, scatter_kwargs=None, line_kwargs=None, text_kwargs=None, cbar_kwargs=None, regress_kwargs=None):
    '''
    绘制密度散点图。
    :param ax: matplotlib的轴对象,用于绘制图形
    :param x: x轴的数据
    :param y: y轴的数据
    :param label: 图例标签,默认为None
    :param label_cmap_index: 图例颜色,默认为'max',使用密度最大的点作为代表
    :param estimate_type: 密度估计类型,默认为'kde',可选'hist'
    :param bw_method: 密度估计的带宽方法,默认为'auto'
    :param bins_x: x轴的bins,默认为BIN_NUM
    :param bins_y: y轴的bins,默认为BIN_NUM
    :param cmap: 颜色映射,默认为DENSITY_CMAP
    :param cbar: 是否显示颜色条,默认为True
    :param linregress: 是否显示线性回归,默认为True
    :param line_color: 线性回归线的颜色,默认为RED
    :param show_list: 显示的统计量列表,默认为['r', 'p']
    :param round_digit_list: 统计量的小数位数列表,默认为[ROUND_DIGITS, ROUND_DIGITS]
    :param round_format_list: 统计量的格式列表,默认为['general', 'general']
    :param text_size: 文字大小,默认为FONT_SIZE
    :param scatter_kwargs: 散点图的其他参数
    :param line_kwargs: 线性回归线的其他参数
    :param text_kwargs: 文字的其他参数
    :param cbar_kwargs: 颜色条的其他参数
    '''
    if show_list is None:
        show_list = ['r', 'p']
    if round_digit_list is None:
        round_digit_list = [ROUND_DIGITS, ROUND_DIGITS]
    if round_format_list is None:
        round_format_list = ['general', 'general']
    if scatter_kwargs is None:
        scatter_kwargs = {}
    if line_kwargs is None:
        line_kwargs = {}
    if text_kwargs is None:
        text_kwargs = {}
    if cbar_kwargs is None:
        cbar_kwargs = {}
    if regress_kwargs is None:
        regress_kwargs = {}
    cbar_position = update_dict(CBAR_POSITION, cbar_position)

    x_array = np.array(x)
    y_array = np.array(y)

    xy = np.vstack([x_array, y_array])

    if estimate_type == 'kde':
        if bw_method == 'auto':
            z = gaussian_kde(xy)(xy)
        else:
            z = gaussian_kde(xy, bw_method=bw_method)(xy)
    if estimate_type == 'hist':
        hist, xedges, yedges, _, _ = get_hist_2d(
            x_array, y_array, bins_x=bins_x, bins_y=bins_y, stat='density')
        ix = np.digitize(x_array, xedges, right=False) - 1
        iy = np.digitize(y_array, yedges, right=False) - 1

        ix = np.clip(ix, 0, hist.shape[0] - 1)
        iy = np.clip(iy, 0, hist.shape[1] - 1)
        z = hist[ix, iy]

    # 选择代表性点
    if label_cmap_index == 'max':
        selected_index = np.argmax(z)
    elif label_cmap_index == 'min':
        selected_index = np.argmin(z)
    elif label_cmap_index == 'median':
        selected_index = np.argmin(np.abs(z - np.median(z)))

    # 画出代表性点,这里要特别设定color参数为None,否则会和z的颜色和cmap冲突
    local_z = get_min_max_scaling(z)
    plt_scatter(ax, x_array[selected_index], y_array[selected_index], c=cmap(
        local_z[selected_index]), label=label, color=None, **scatter_kwargs)

    # 画出所有点,这里要特别设定color参数为None,否则会和z的颜色和cmap冲突
    sc = plt_scatter(ax, x_array, y_array, c=z,
                          cmap=cmap, color=None, **scatter_kwargs)

    if cbar:
        add_colorbar(ax, mappable=sc, cbar_label=cbar_label, **cbar_kwargs)

    if linregress:
        # 设置scatter_color为透明,防止覆盖density
        plt_linregress(ax, x_array, y_array, label=None, scatter_color='#FF573300',
                            line_color=line_color, show_list=show_list, round_digit_list=round_digit_list, round_format_list=round_format_list, text_size=text_size, scatter_kwargs=scatter_kwargs, line_kwargs=line_kwargs, text_kwargs=text_kwargs, regress_kwargs=regress_kwargs)


def plt_marginal_density_scatter(ax, x, y, density_scatter_kwargs=None, marginal_kwargs=None):
    '''
    绘制密度散点图和边缘分布。
    :param ax: matplotlib的轴对象,用于绘制图形
    :param x: x轴的数据
    :param y: y轴的数据
    :param density_scatter_kwargs: 传递给plt_density_scatter的额外关键字参数
    :param marginal_kwargs: 传递给add_double_marginal_distribution的额外关键字参数
    '''
    density_scatter_kwargs = update_dict(get_default_param(plt_density_scatter), density_scatter_kwargs)
    density_scatter_kwargs['cbar_position'] = update_dict(CBAR_POSITION, density_scatter_kwargs['cbar_position'])

    marginal_kwargs = update_dict(get_default_param(add_double_marginal_distribution), marginal_kwargs)

    # 此处先不添加cbar
    hist = plt_density_scatter(ax, x, y, **update_dict(density_scatter_kwargs, {'cbar': False}))

    x_side_ax, y_side_ax = add_double_marginal_distribution(ax, x, y, **marginal_kwargs)

    if density_scatter_kwargs['cbar']:
        if density_scatter_kwargs['cbar_position']['position'] == marginal_kwargs['x_side_ax_position']:
            density_scatter_kwargs['cbar_position']['pad'] += marginal_kwargs['x_side_ax_size'] + marginal_kwargs['x_side_ax_pad']
        if density_scatter_kwargs['cbar_position']['position'] == marginal_kwargs['y_side_ax_position']:
            density_scatter_kwargs['cbar_position']['pad'] += marginal_kwargs['y_side_ax_size'] + marginal_kwargs['y_side_ax_pad']

        cbars = add_colorbar(ax, mappable=hist, cmap=density_scatter_kwargs['cmap'], cbar_label=density_scatter_kwargs['cbar_label'], cbar_position=density_scatter_kwargs['cbar_position'], **update_dict({}, density_scatter_kwargs['cbar_kwargs']))
        return hist, x_side_ax, y_side_ax, cbars
    else:
        return hist, x_side_ax, y_side_ax


def plt_errorbar_line(ax, x, y, err, line_label=None, elabel=None, line_color=BLUE, ecolor=BLACK, capsize=PLT_CAP_SIZE, vert=True, line_kwargs=None, error_kwargs=None):
    '''
    使用x和y绘制折线图，并添加误差线
    :param ax: matplotlib的轴对象,用于绘制图形
    :param x: x轴的数据
    :param y: y轴的数据
    :param err: 误差线的数据
    :param line_label: 折线图的标签,默认为None
    :param elabel: 误差线的标签,默认为None
    :param line_color: 折线图的颜色,默认为BLUE
    :param ecolor: 误差线的颜色,默认为BLACK
    :param capsize: 误差线的线帽大小,默认为PLT_CAP_SIZE
    :param vert: 是否是垂直的误差线,默认为True
    :param line_kwargs: 传递给plt_line的额外关键字参数
    :param error_kwargs: 传递给add_errorbar的额外关键字参数
    '''
    if line_kwargs is None:
        line_kwargs = {}
    if error_kwargs is None:
        error_kwargs = {}
    # 画图
    if vert:
        return plt_line(ax, x, y, label=line_label, color=line_color, **line_kwargs), add_errorbar(ax, x, y, err, label=elabel, color=ecolor, capsize=capsize, vert=vert, **error_kwargs)
    else:
        return plt_line(ax, y, x, label=line_label, color=line_color, **line_kwargs), add_errorbar(ax, x, y, err, label=elabel, color=ecolor, capsize=capsize, vert=vert, **error_kwargs)


def plt_fill_between_line(ax, x, y1, y2, label=None, color=BLUE, alpha=FAINT_ALPHA, vert=True, **kwargs):
    '''
    使用x和y1,y2绘制填充区域
    :param ax: matplotlib的轴对象,用于绘制图形
    :param x: x轴的数据
    :param y1: y轴的数据
    :param y2: y轴的数据
    :param label: 填充区域的标签,默认为None
    :param color: 填充区域的颜色,默认为BLUE
    :param alpha: 填充区域的透明度,默认为FAINT_ALPHA
    :param kwargs: 传递给plt.fill_between的额外关键字参数
    '''
    # 画图
    if vert:
        return ax.fill_between(x, y1, y2, label=label, color=color, alpha=alpha, **kwargs)
    else:
        return ax.fill_betweenx(x, y1, y2, label=label, color=color, alpha=alpha, **kwargs)


def plt_band_line(ax, x, y, bandwidth, line_label=None, line_color=BLUE, fill_label=None, fill_color=BLUE, alpha=FAINT_ALPHA, vert=True, line_kwargs=None, fill_kwargs=None):
    '''
    使用x和y绘制折线图，并添加一个表示误差带的区域
    :param ax: matplotlib的轴对象,用于绘制图形
    :param x: x轴的数据
    :param y: y轴的数据
    :param bandwidth: 误差带的数据
    :param label: 图例标签,默认为None
    :param color: 折线图的颜色,默认为BLUE
    :param alpha: 误差带的透明度,默认为FAINT_ALPHA
    :param line_kwargs: 传递给plt_line的额外关键字参数
    :param fill_kwargs: 传递给plt_fill_between_line的额外关键字参数
    '''
    if line_kwargs is None:
        line_kwargs = {}
    if fill_kwargs is None:
        fill_kwargs = {}
    # 画图
    if vert:
        return plt_line(ax, x, y, label=line_label, color=line_color, **line_kwargs), plt_fill_between_line(ax, x, np.array(y)-np.array(bandwidth), np.array(y)+np.array(bandwidth), label=fill_label, color=fill_color, alpha=alpha, vert=vert, **fill_kwargs)
    else:
        return plt_line(ax, y, x, label=line_label, color=line_color, **line_kwargs), plt_fill_between_line(ax, x, np.array(y)-np.array(bandwidth), np.array(y)+np.array(bandwidth), label=fill_label, color=fill_color, alpha=alpha, vert=vert, **fill_kwargs)


def plt_kde(ax, data, label=None, color=BLUE, vert=True, x=None, kde_kwargs=None, **kwargs):
    '''
    使用数据绘制核密度估计图,可以接受plt.plot的其他参数
    :param ax: matplotlib的轴对象,用于绘制图形
    :param data: 用于绘制核密度估计图的数据集
    :param label: 图例标签,默认为None
    :param color: 核密度估计图的颜色,默认为BLUE
    :param bw_method: 核密度估计的带宽,默认为None
    :param kwargs: 其他plt.plot支持的参数
    '''
    if kde_kwargs is None:
        kde_kwargs = {}
    kde = get_kde(data, **kde_kwargs)
    if x is None:
        x = np.linspace(np.nanmin(data), np.nanmax(data), 1000, endpoint=True)
    return plt_line(ax, x, kde(x), label=label, color=color, vert=vert, **kwargs)


def plt_kde_contour(ax, x, y, label=None, color=BLUE, cmap=None, levels=None, gridsize=100, xmin=None, xmax=None, ymin=None, ymax=None, kde_kwargs=None, contour_kwargs=None):
    '''
    使用x和y绘制核密度估计的轮廓图
    :param ax: matplotlib的轴对象,用于绘制图形
    :param x: x轴的数据
    :param y: y轴的数据
    :param label: 图例标签,默认为None
    :param color: 轮廓图的颜色,默认为BLUE
    :param cmap: 轮廓图的颜色映射,默认为None
    :param levels: 轮廓图的等高线数量,默认为None
    :param kde_kwargs: 传递给get_kde的额外关键字参数
    :param contour_kwargs: 传递给plt.contour的额外关键字参数
    '''
    if kde_kwargs is None:
        kde_kwargs = {}
    if contour_kwargs is None:
        contour_kwargs = {}

    kde = get_kde(np.vstack([x, y]), **kde_kwargs)

    # Generate a Grid
    if xmin is None:
        xmin = np.nanmin(x) - (np.nanmax(x) - np.nanmin(x)) * 0.1
    if xmax is None:
        xmax = np.nanmax(x) + (np.nanmax(x) - np.nanmin(x)) * 0.1
    if ymin is None:
        ymin = np.nanmin(y) - (np.nanmax(y) - np.nanmin(y)) * 0.1
    if ymax is None:
        ymax = np.nanmax(y) + (np.nanmax(y) - np.nanmin(y)) * 0.1
    xx, yy = np.mgrid[xmin:xmax:complex(gridsize), ymin:ymax:complex(gridsize)]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    
    # Evaluate the KDE
    z = np.reshape(kde(positions).T, xx.shape)

    return plt_contour(ax, xx, yy, z, label=label, color=color, cmap=cmap, levels=levels, **contour_kwargs)


def plt_kde_contourf(ax, x, y, cmap=DENSITY_CMAP, levels=None, gridsize=100, xmin=None, xmax=None, ymin=None, ymax=None, kde_kwargs=None, contourf_kwargs=None):
    '''
    使用x和y绘制核密度估计的填充轮廓图
    :param ax: matplotlib的轴对象,用于绘制图形
    :param x: x轴的数据
    :param y: y轴的数据
    :param label: 图例标签,默认为None
    :param cmap: 轮廓图的颜色映射,默认为DENSITY_CMAP
    :param levels: 轮廓图的等高线数量,默认为None
    :param kde_kwargs: 传递给get_kde的额外关键字参数
    :param contourf_kwargs: 传递给plt.contourf的额外关键字参数
    '''
    if kde_kwargs is None:
        kde_kwargs = {}
    if contourf_kwargs is None:
        contourf_kwargs = {}

    kde = get_kde(np.vstack([x, y]), **kde_kwargs)

    # Generate a Grid
    if xmin is None:
        xmin = np.nanmin(x) - (np.nanmax(x) - np.nanmin(x)) * 0.1
    if xmax is None:
        xmax = np.nanmax(x) + (np.nanmax(x) - np.nanmin(x)) * 0.1
    if ymin is None:
        ymin = np.nanmin(y) - (np.nanmax(y) - np.nanmin(y)) * 0.1
    if ymax is None:
        ymax = np.nanmax(y) + (np.nanmax(y) - np.nanmin(y)) * 0.1
    xx, yy = np.mgrid[xmin:xmax:complex(gridsize), ymin:ymax:complex(gridsize)]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    
    # Evaluate the KDE
    z = np.reshape(kde(positions).T, xx.shape)

    return plt_contourf(ax, xx, yy, z, cmap=cmap, levels=levels, **contourf_kwargs)


def plt_cdf(ax, data, label=None, color=BLUE, vert=True, **kwargs):
    '''
    使用数据绘制累积分布函数图,可以接受plt.plot的其他参数
    :param ax: matplotlib的轴对象,用于绘制图形
    :param data: 用于绘制累积分布函数图的数据集
    :param label: 图例标签,默认为None
    :param color: 累积分布函数图的颜色,默认为BLUE
    :param kwargs: 其他plt.plot支持的参数
    '''
    cdf = get_cdf(data)
    return plt_line(ax, cdf.x, cdf.y, label=label, color=color, vert=vert, **kwargs)


def plt_compare_cdf_ks(ax, data1, data2, label1=None, label2=None, color1=BLUE, color2=RED, vert=True, text_x=TEXT_X, text_y=TEXT_Y, round_digit_list=None, round_format_list=None, show_list=None, text_kwargs=None, cdf_kwargs=None):
    '''
    使用两组数据绘制累积分布函数图,并计算K-S检验的p值
    :param ax: matplotlib的轴对象,用于绘制图形
    :param data1: 第一组数据
    :param data2: 第二组数据
    :param label1: 第一组数据的标签
    :param label2: 第二组数据的标签
    :param color1: 第一组数据的颜色
    :param color2: 第二组数据的颜色
    :param kwargs: 传递给plt_cdf的额外关键字参数
    '''
    if show_list is None:
        show_list = ['KS', 'P']
    if round_digit_list is None:
        round_digit_list = [ROUND_DIGITS] * len(show_list)
    if round_format_list is None:
        round_format_list = ['general'] * len(show_list)
    if text_kwargs is None:
        text_kwargs = {}
    if cdf_kwargs is None:
        cdf_kwargs = {}

    plt_cdf(ax, data1, label=label1, color=color1, vert=vert, **cdf_kwargs)
    plt_cdf(ax, data2, label=label2, color=color2, vert=vert, **cdf_kwargs)

    ks, p = get_ks_and_p(data1, data2)
    text = []
    for show, round_digit, round_format in zip(show_list, round_digit_list, round_format_list):
        if show == 'KS':
            text.append(f'KS: {round_float(ks, round_digit, round_format)}')
        if show == 'P':
            text.append(f'P: {round_float(p, round_digit, round_format)}')
    text = '\n'.join(text)
    add_text(ax, text, x=text_x, y=text_y, **text_kwargs)


def plt_marginal_hist_2d(ax, x, y, stat='probability', cbar_label=None, hist_kwargs=None, marginal_kwargs=None):
    '''
    带有边缘分布的二维直方图
    :param ax: matplotlib的轴对象,用于绘制图形
    :param x: x轴的数据
    :param y: y轴的数据
    :param stat: 统计量,默认为'probability'
    :param cbar_label: 颜色条的标签,默认和stat相同
    :param hist_kwargs: 传递给plt_hist_2d的额外关键字参数
    :param marginal_kwargs: 传递给add_double_marginal_distribution的额外关键字参数
    '''
    if cbar_label is None:
        cbar_label = stat
    hist_kwargs = update_dict(get_default_param(plt_hist_2d), hist_kwargs)
    hist_kwargs['cbar_position'] = update_dict(CBAR_POSITION, hist_kwargs['cbar_position'])

    marginal_kwargs = update_dict(get_default_param(add_double_marginal_distribution), marginal_kwargs)

    # 此处先不添加cbar
    hist = plt_hist_2d(ax, x, y, **update_dict(hist_kwargs, {'cbar': False, 'stat': stat}))

    x_side_ax, y_side_ax = add_double_marginal_distribution(ax, x, y, **marginal_kwargs)

    if hist_kwargs['cbar']:
        if hist_kwargs['cbar_position']['position'] == marginal_kwargs['x_side_ax_position']:
            hist_kwargs['cbar_position']['pad'] += marginal_kwargs['x_side_ax_size'] + marginal_kwargs['x_side_ax_pad']
        if hist_kwargs['cbar_position']['position'] == marginal_kwargs['y_side_ax_position']:
            hist_kwargs['cbar_position']['pad'] += marginal_kwargs['y_side_ax_size'] + marginal_kwargs['y_side_ax_pad']

        cbars = add_colorbar(ax, mappable=hist, cmap=hist_kwargs['cmap'], cbar_label=cbar_label, cbar_position=hist_kwargs['cbar_position'], **update_dict({}, hist_kwargs['cbar_kwargs']))
        return hist, x_side_ax, y_side_ax, cbars
    else:
        return hist, x_side_ax, y_side_ax


def plt_vector_input_bar(ax, x, y, label=None, color=BLUE, vert=True, equal_space=False, err=None, capsize=PLT_CAP_SIZE, ecolor=BLACK, elabel=None, width=BAR_WIDTH, bar_kwargs=None, ebar_kwargs=None):
    '''
    使用x和y绘制柱状图，可以接受plt.bar的其他参数,此函数的特性是会根据x的值作为bar的位置,当x包含字符串或者equal_space=True时,会自动变成等距离排列。这个函数的特殊性是可以接受向量值的width，color，capsize，ecolor
    :param ax: matplotlib的轴对象，用于绘制图形
    :param x: x轴的数据
    :param y: y轴的数据
    :param label: 图例标签，默认为None
    :param color: 柱状图的颜色，默认为BLUE
    :param vert: 是否为垂直柱状图，默认为True，即纵向
    :param equal_space: 是否将x的值作为字符串处理，这将使得柱子等距排列，默认为False
    :param err: 误差线的数据，默认为None
    :param capsize: 误差线帽大小，默认为PLT_CAP_SIZE
    :param ecolor: 误差线颜色，默认为BLACK
    :param elabel: 误差线图例标签，默认为None
    :param width: 柱子宽度，默认为None(当vert=False时,此参数将自动赋值给height)
    :param kwargs: 其他plt.bar或plt.barh支持的参数
    '''
    if bar_kwargs is None:
        bar_kwargs = {}
    if ebar_kwargs is None:
        ebar_kwargs = {}

    if equal_space:
        # 将x的每个元素变为字符串
        xtick_label = [str(i) for i in x]
        xtick = np.arange(len(xtick))
        ax.set_xticklabels(xtick_label)
    else:
        xtick = x.copy()

    if not isinstance(color, (list, np.ndarray)):
        color = [color] * len(x)
    if not isinstance(width, (list, np.ndarray)):
        width = [width] * len(x)
    if not isinstance(err, (list, np.ndarray)):
        err = [err] * len(x)
    if not isinstance(capsize, (list, np.ndarray)):
        capsize = [capsize] * len(x)
    if not isinstance(ecolor, (list, np.ndarray)):
        ecolor = [ecolor] * len(x)

    for i in range(len(x)):
        if i==0:
            local_label = label
        else:
            local_label = None
        plt_bar(ax, xtick[i], y[i], label=local_label, color=color[i], vert=vert, equal_space=False, width=width[i], **bar_kwargs)
        add_errorbar(ax, xtick[i], y[i], err[i], vert=vert, label=elabel, color=color[i], capsize=capsize[i], ecolor=ecolor[i], **ebar_kwargs)


def plt_xlog_bar(*args, **kwargs):
    print('如果要使用xlog的bar,建议先把x数据取好log后使用plt_bar')
# endregion


# region 复杂作图函数(matplotlib系列,输入dataframe使用)
def plt_group_bar_df(ax, df, bar_width=None, group_by='columns', colors=CMAP, vert=True, **kwargs):
    '''
    基于DataFrame绘制分组的柱状图，根据分组是按照行还是列来自动处理。
    :param ax: matplotlib的轴对象,用于绘制图形
    :param df: 数据存放的DataFrame
    :param bar_width: 单个柱子的宽度,默认为None,自动确定宽度
    :param group_by: 指定分组数据是按行('rows')还是按列('columns')，默认为'columns'
    :param colors: 每个分组的颜色列表;也可以指定cmap,默认为CMAP,然后自动生成颜色序列
    :param vert: 是否是垂直的柱状图,默认为True
    :param kwargs: 其他plt.bar支持的参数
    '''
    if group_by == 'columns':
        x = df.index.tolist()  # x轴标签
        y = df.values.T.tolist()  # 每组的柱状图值
        label_list = df.columns.tolist()  # 每个柱状图的标签
    else:  # 'rows'
        x = df.columns.tolist()  # x轴标签
        y = df.values.tolist()  # 每组的柱状图值
        label_list = df.index.tolist()  # 每个柱状图的标签

    plt_group_bar(ax=ax, x=x, y=y, label_list=label_list,
                       bar_width=bar_width, colors=colors, vert=vert, **kwargs)


def plt_two_side_bar_df():
    pass


def plt_group_box_df(ax, df, group_by='columns', colors=None, **kwargs):
    print('需要加入横纵选项')
    pass


def plt_split_violine_df(ax, df, group_by='columns', colors=None, **kwargs):
    '''
    see https://seaborn.pydata.org/examples/grouped_violinplots.html
    '''
    print('需要加入横纵选项')
    pass


def read_sns():
    print('please see sns gallery and add new function')
# endregion

    
# region 复杂作图函数(sns系列,输入向量使用)
def sns_band_line(ax, x, y, band_width, label=None, color=BLUE, alpha=FAINT_ALPHA, **kwargs):
    '''
    使用x和y绘制折线图，并添加一个表示误差带的区域
    :param ax: matplotlib的轴对象,用于绘制图形
    :param x: x轴的数据
    :param y: y轴的数据
    :param band_width: 误差带宽度
    :param label: 图例标签,默认为None
    :param color: 折线图的颜色,默认为蓝色
    :param alpha: 误差带的透明度
    :param kwargs: 其他sns.lineplot支持的参数
    '''
    # 绘制基础折线图
    sns_line(ax, x, y, label=label, color=color, **kwargs)

    # 计算误差带的上下界
    y_upper = np.array(y) + np.array(band_width)
    y_lower = np.array(y) - np.array(band_width)

    # 绘制误差带
    ax.fill_between(x, y_lower, y_upper, color=color, alpha=alpha)
# endregion


# region 复杂作图函数(sns系列,输入dataframe使用)
def sns_marginal_heatmap(ax, data, outside=True, x_side_ax_position='top', y_side_ax_position='right', x_side_ax_pad=SIDE_PAD, y_side_ax_pad=SIDE_PAD, x_side_ax_size=0.3, y_side_ax_size=0.3, x_color=BLUE, y_color=BLUE, heatmap_kwargs=None, bar_kwargs=None, rm_tick=True, rm_spine=True, rm_axis=True):
    '''
    '''
    heatmap_kwargs = update_dict(get_default_param(sns_heatmap), heatmap_kwargs)
    heatmap_kwargs['cbar_position'] = update_dict(CBAR_POSITION, heatmap_kwargs['cbar_position'])
    bar_kwargs = update_dict(get_default_param(plt_bar), bar_kwargs)
    bar_kwargs = update_dict(bar_kwargs, {'width': 1})

    if isinstance(data, np.ndarray):
        local_data = pd.DataFrame(data)
    else:
        local_data = data.copy()
    
    # 绘制热图,先不显示cbar
    sns_heatmap(ax, local_data, **update_dict(heatmap_kwargs, {'cbar': False}))

    # 获得边缘分布需要的ax
    if outside:
        x_side_ax = add_side_ax(ax, position=x_side_ax_position, relative_size=x_side_ax_size, pad=x_side_ax_pad, sharex=ax)
        y_side_ax = add_side_ax(ax, position=y_side_ax_position, relative_size=y_side_ax_size, pad=y_side_ax_pad, sharey=ax)
    else:
        # 当边缘分布在内部时，需要手动分划ax然后获取需要的
        x_side_ax, y_side_ax = split_quadrant_relocate_ax(ax, x_side_ax_position=x_side_ax_position, y_side_ax_position=y_side_ax_position, x_side_ax_size=x_side_ax_size, y_side_ax_size=y_side_ax_size, x_side_ax_pad=x_side_ax_pad, y_side_ax_pad=y_side_ax_pad)

    # 绘制边缘分布
    plt_bar(x_side_ax, x=np.arange(len(local_data.columns)) + 0.5, y=col_sum_df(local_data), **update_dict(bar_kwargs, {'color': x_color}))
    plt_bar(y_side_ax, x=np.arange(len(local_data.index)) + 0.5, y=row_sum_df(local_data), **update_dict(bar_kwargs, {'color': y_color, 'vert': False}))

    # 添加cbar
    if heatmap_kwargs['cbar']:
        if heatmap_kwargs['cbar_position']['position'] == x_side_ax_position:
            heatmap_kwargs['cbar_position']['pad'] += x_side_ax_size + x_side_ax_pad
        if heatmap_kwargs['cbar_position']['position'] == y_side_ax_position:
            heatmap_kwargs['cbar_position']['pad'] += y_side_ax_size + y_side_ax_pad
        add_colorbar(ax, mappable=ax.collections[0], cbar_label=heatmap_kwargs['cbar_label'], cbar_position=heatmap_kwargs['cbar_position'], **update_dict({}, heatmap_kwargs['cbar_kwargs']))

    # 隐藏边缘分布的刻度和坐标轴
    if rm_tick:
        rm_ax_tick(x_side_ax, 'x')
        rm_ax_tick(y_side_ax, 'y')
    if rm_spine:
        rm_ax_spine(x_side_ax)
        rm_ax_spine(y_side_ax)
    if rm_axis:
        rm_ax_axis(x_side_ax)
        rm_ax_axis(y_side_ax)

    # 调整边缘分布的方向
    if x_side_ax_position == 'bottom':
        x_side_ax.invert_yaxis()
    if y_side_ax_position == 'left':
        y_side_ax.invert_xaxis()

    return x_side_ax, y_side_ax
# endregion


# region 复杂作图函数(添加star)
def add_star_extreme_value(ax, x, y, extreme_type, label=None, marker=STAR, star_color=None, star_size=STAR_SIZE, linestyle='None', **kwargs):
    '''
    在min或者max位置添加星号
    :param extreme_type: 极值类型,可以是'max'或者'min'
    '''
    # 找到数据中的最大值和最小值位置
    if extreme_type == 'max':
        # 计算最大值
        max_value = np.nanmax(y)
        # 找到所有最大值的位置
        pos = np.where(y == max_value)[0]
    elif extreme_type == 'min':
        # 计算最小值
        min_value = np.nanmin(y)
        # 找到所有最小值的位置
        pos = np.where(y == min_value)[0]

    # 给label自动赋值
    if label is None:
        label = extreme_type

    # 给star_color自动赋值
    if star_color is None:
        if extreme_type == 'max':
            star_color = RED
        if extreme_type == 'min':
            star_color = GREEN

    # 画图
    return add_star(ax, np.array(x)[pos], np.array(y)[pos], label=label, marker=marker, star_color=star_color, star_size=star_size, linestyle=linestyle, **kwargs)


def add_star_extreme_value_heatmap(ax, data, extreme_type, label=None, marker=STAR, star_color=None, star_size=STAR_SIZE, linestyle='None', **kwargs):
    '''
    在热图的极值位置添加星号
    :param extreme_type: 极值类型,可以是'max'或者'min'
    '''
    # 找到数据中的最大值和最小值位置
    if extreme_type == 'max':
        # 找到最大值
        extreme_value = np.nanmax(data)
    elif extreme_type == 'min':
        # 找到最小值
        extreme_value = np.nanmin(data)

    # 找到所有极值的位置
    pos = np.where(data == extreme_value)

    # 给label自动赋值
    if label is None:
        label = extreme_type

    # 给star_color自动赋值
    if star_color is None:
        if extreme_type == 'max':
            star_color = RED
        if extreme_type == 'min':
            star_color = GREEN

    # 画图
    return add_star_heatmap(ax, pos[1], pos[0], label=label, marker=marker, star_color=star_color, star_size=star_size, linestyle=linestyle, **kwargs)
# endregion


# region 复杂作图函数(添加辅助线)
def add_vline_extreme_value(ax, x, y, extreme_type, label=None, color=None, linestyle=AUXILIARY_LINE_STYLE, linewidth=LINE_WIDTH, **kwargs):
    '''
    在min或者max位置添加垂直线
    :param extreme_type: 极值类型,可以是'max'或者'min'
    '''
    # 找到数据中的最大值和最小值位置
    if extreme_type == 'max':
        # 计算最大值
        max_value = np.nanmax(y)
        # 找到所有最大值的位置
        pos = np.where(y == max_value)[0]
    elif extreme_type == 'min':
        # 计算最小值
        min_value = np.nanmin(y)
        # 找到所有最小值的位置
        pos = np.where(y == min_value)[0]

    # 给label自动赋值
    if label is None:
        label = extreme_type

    # 给color自动赋值
    if color is None:
        if extreme_type == 'max':
            color = RED
        if extreme_type == 'min':
            color = GREEN

    # 画图
    return add_vline(ax, np.array(x)[pos], label=label, color=color, linestyle=linestyle, linewidth=linewidth, **kwargs)
# endregion


# region 复杂作图函数(添加文字)
def add_linregress_text(ax, regress_dict, show_list=None, round_digit_list=None, round_format_list=None, fontsize=FONT_SIZE, text_process=None, text_kwargs=None):
    '''
    Displays the results of linear regression on a plot.

    :param ax: Matplotlib axis object to draw the graph
    :param regress_dict: dict, results of the linear regression
    :param show_list: list, names of the results to display
    :param round_digit_list: list, number of decimal places to round the results
    :param round_format_list: list, format of the rounded results
    :param fontsize: int, font size for the displayed results
    '''
    if show_list is None:
        show_list = ['r', 'p']
    if round_digit_list is None:
        round_digit_list = [ROUND_DIGITS, ROUND_DIGITS]
    if round_format_list is None:
        round_format_list = ['general', 'general']
    text_process = update_dict(TEXT_PROCESS, text_process)
    if text_kwargs is None:
        text_kwargs = {}

    text_str = ''
    for i, key in enumerate(show_list):
        if key in regress_dict:
            # Format the key name
            local_key = format_text(key, text_process=text_process)
            value = regress_dict[key]

            # Round the value
            round_value = round_float(
                value, round_digit_list[i], round_format_list[i])
            text_str += f'{local_key}: {round_value}\n'

    # Display the linear regression results string
    if text_str:
        add_text(ax, text_str, fontsize=fontsize, **text_kwargs)
# endregion


# region 通用图片设定函数
# region inch, width, height, figsize, axsize
def inch_to_point(inch):
    '''
    将plt中使用的inch单位转换为points单位
    '''
    return inch * 72


def get_ax_size(ax):
    """
    获取给定轴的尺寸（以英寸为单位）。
    
    参数:
    - ax: matplotlib轴对象。
    
    返回:
    - ax_width: 轴的宽度（英寸）。
    - ax_height: 轴的高度（英寸）。
    """
    # 获取图形的尺寸（英寸）
    fig = ax.get_figure()
    fig_width, fig_height = fig.get_size_inches()
    
    # 获取轴的边界框，这是相对于图形大小的
    bbox = ax.get_position()
    
    # 计算轴的实际尺寸（英寸）
    ax_width = fig_width * bbox.width
    ax_height = fig_height * bbox.height
    
    return ax_width, ax_height
# endregion


# region xy_lim spine tick axis aspect设置函数
def share_ax(*axes, sharex=True, sharey=True):
    '''
    通过在原先位置强行创建新的ax来共享轴。
    '''
    print('to be teset')
    fig = axes[0].get_figure()
    new_axes = []
    for i, ax in enumerate(axes):
        pos = ax.get_position()
        if i==0:
            new_axes.append(add_ax(fig, pos.x0, pos.x0 + pos.width, pos.y0, pos.y0 + pos.height))
        else:
            new_axes.append(add_ax(fig, pos.x0, pos.x0 + pos.width, pos.y0, pos.y0 + pos.height, sharex=axes[0] if sharex else None, sharey=axes[0] if sharey else None))
        ax.set_visible(False)
    return new_axes


def rm_ax_spine(ax, spines_to_remove=None):
    '''
    移除轴的边框。

    参数:
    - ax: matplotlib的Axes对象
    - spines_to_remove: 要移除的边框列表
    '''
    if spines_to_remove is None:
        spines_to_remove = ['top', 'right', 'left', 'bottom']

    for spine in spines_to_remove:
        ax.spines[spine].set_visible(False)


def rm_ax_tick(ax, ticks_to_remove=None):
    '''
    移除轴的刻度。

    参数:
    - ax: matplotlib的Axes对象
    - ticks_to_remove: 要移除的刻度列表
    '''
    if ticks_to_remove is None or ticks_to_remove == 'both':
        ticks_to_remove = ['x', 'y']

    for tick in ticks_to_remove:
        getattr(ax, f'set_{tick}ticks')([])


def rm_ax_axis(ax):
    '''
    移除轴的坐标轴。

    参数:
    - ax: matplotlib的Axes对象
    '''
    ax.axis('off')


def equal_ax_aspect(ax, aspect=1):
    '''
    设置轴的纵横比。

    参数:
    - ax: matplotlib的Axes对象
    - aspect: 纵横比
    '''
    ax.set_aspect(aspect)
# endregion


# region ax视角相关函数
def set_ax_view_3d(ax, elev=ELEV, azim=AZIM):
    '''
    对单个或多个3D子图Axes应用统一的视角设置。

    参数:
    - ax: 单个Axes实例或包含多个Axes实例的数组。
    - elev: 视角的高度。
    - azim: 视角的方位角。
    '''
    # 如果ax是Axes实例的列表或数组
    if isinstance(ax, (list, np.ndarray)):
        for ax_i in np.ravel(ax):  # 使用np.ravel确保可以迭代所有可能的结构
            if isinstance(ax_i, Axes3D):
                ax_i.view_init(elev=elev, azim=azim)
    # 如果ax是单个Axes实例
    elif isinstance(ax, Axes3D):
        ax.view_init(elev=elev, azim=azim)
# endregion


# region ax位置相关函数
def set_ax_position(ax, left, right, bottom, top):
    '''
    设置轴的位置。

    参数:
    - ax: matplotlib的Axes对象
    - left: 左边界的位置
    - right: 右边界的位置
    - bottom: 下边界的位置
    - top: 上边界的位置
    '''
    ax.set_position([left, bottom, right - left, top - bottom])


def set_relative_ax_position(ax, nrows=1, ncols=1, margin=None):
    '''
    自动设置subplot的位置，使其在等分的图像中按照给定的比例占据空间。

    参数:
    - nrows: 子图的行数。
    - ncols: 子图的列数。
    - ax: 一个或一组matplotlib的Axes对象。
    - margin: 一个字典，定义了图像边缘的留白，包括left, right, bottom, top。
    '''
    if margin is None:
        margin = MARGIN.copy()

    # 计算每个子图的宽度和高度(相对于整个图像的宽度和高度,所以是比例值)
    subplot_width = 1 / ncols
    subplot_height = 1 / nrows
    ax_width = (margin['right'] - margin['left']) * subplot_width
    ax_height = (margin['top'] - margin['bottom']) * subplot_height

    if nrows > 1 and ncols > 1:
        # 对于每个子图，计算其位置并设置
        for row in range(nrows):
            for col in range(ncols):
                left = margin['left'] / ncols + col * subplot_width
                bottom = margin['bottom'] / nrows + \
                    (nrows - row - 1) * subplot_height

                # 设置子图的位置
                ax[row, col].set_position(
                    [left, bottom, ax_width, ax_height])
    if nrows == 1 and ncols > 1:
        for col in range(ncols):
            left = margin['left'] / ncols + col * subplot_width
            bottom = margin['bottom']

            # 设置子图的位置
            ax[col].set_position([left, bottom, ax_width, ax_height])
    if nrows > 1 and ncols == 1:
        for row in range(nrows):
            left = margin['left']
            bottom = margin['bottom'] / nrows + (nrows - row - 1) * subplot_height

            # 设置子图的位置
            ax[row].set_position([left, bottom, ax_width, ax_height])
    if nrows == 1 and ncols == 1:
        left = margin['left']
        bottom = margin['bottom']

        # 设置子图的位置
        ax.set_position([left, bottom, ax_width, ax_height])

    return ax


def align_ax(axs, align_ax, align_type='horizontal'):
    '''
    将axs中ax的position对齐到align_ax的position
    align_type:
        'horizontal', 'vertical' - 改变高度和宽度,horizontal则左右对齐(ax的上下框对齐),vertical则上下对齐(ax的左右框对齐)
        'left', 'right', 'top', 'bottom' - 不改变高度和宽度,只改变位置去对齐到需要的位置
    '''
    if align_type == 'horizontal':
        for ax in axs:
            pos = align_ax.get_position()
            ax.set_position([ax.get_position().x0, pos.y0, ax.get_position().width, pos.height])
    elif align_type == 'vertical':
        for ax in axs:
            pos = align_ax.get_position()
            ax.set_position([pos.x0, ax.get_position().y0, pos.width, ax.get_position().height])
    elif align_type == 'left':
        for ax in axs:
            pos = align_ax.get_position()
            ax.set_position([pos.x0, ax.get_position().y0, ax.get_position().width, ax.get_position().height])
    elif align_type == 'right':
        for ax in axs:
            pos = align_ax.get_position()
            ax.set_position([pos.x0 + pos.width - ax.get_position().width, ax.get_position().y0, ax.get_position().width, ax.get_position().height])
    elif align_type == 'top':
        for ax in axs:
            pos = align_ax.get_position()
            ax.set_position([ax.get_position().x0, pos.y0 + pos.height - ax.get_position().height, ax.get_position().width, ax.get_position().height])
    elif align_type == 'bottom':
        for ax in axs:
            pos = align_ax.get_position()
            ax.set_position([ax.get_position().x0, pos.y0, ax.get_position().width, ax.get_position().height])
# endregion


# region title, label, tick调整函数
def label_title_process(x_label, y_label, z_label, title, text_process=None):
    '''
    处理标签和标题
    '''
    text_process = update_dict(TEXT_PROCESS, text_process)
    return format_text(x_label, text_process), format_text(y_label, text_process), format_text(z_label, text_process), format_text(title, text_process)


def set_fig_title(fig, title, text_process=None, title_size=SUP_TITLE_SIZE):
    '''
    设置图形的标题

    参数:
    - fig: matplotlib的Figure对象
    - title: 图形的标题
    - text_process: 文本处理参数
    - title_size: 标题字体大小
    '''
    text_process = update_dict(TEXT_PROCESS, text_process)

    # 处理标题
    local_title = format_text(title, text_process)

    # 设置标题
    fig.suptitle(local_title, fontsize=title_size)


def suitable_tick_size(num_ticks, plt_size, proportion=0.9):
    '''
    Adjusts the font size of the tick labels based on the number of ticks and the size of the axis.
    '''
    # suitable_tick_size = 50 * plt_size / num_ticks # 50 是一个经验值，用于调整字体大小
    suitable_tick_size = inch_to_point(plt_size) / num_ticks * proportion
    return min(suitable_tick_size, TICK_SIZE)


def adjust_ax_tick(ax, xtick_rotation=XTICK_ROTATION, ytick_rotation=YTICK_ROTATION, proportion=0.9):
    '''
    x轴和y轴的刻度标签字体大小根据刻度数量和轴的大小进行调整。(要写在set_ax之后,否则会被覆盖)
    自动旋转x轴刻度标签。
    '''
    ax_width, ax_height = get_ax_size(ax)
    # 获取x轴和y轴的刻度数量
    num_ticks_x = len(ax.get_xticks())
    num_ticks_y = len(ax.get_yticks())
    if num_ticks_x > 0:
        ax.tick_params(axis='x', labelsize=suitable_tick_size(
            num_ticks_x, ax_width, proportion))
    if num_ticks_y > 0:
        ax.tick_params(axis='y', labelsize=suitable_tick_size(
            num_ticks_y, ax_height, proportion))

    # 旋转x轴刻度标签
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(ax.get_xticklabels(), rotation=xtick_rotation)

    # 旋转y轴刻度标签
    ax.set_yticks(ax.get_yticks())
    ax.set_yticklabels(ax.get_yticklabels(), rotation=ytick_rotation)


def set_ax(ax, x_label=None, y_label=None, z_label=None, x_labelpad=LABEL_PAD, y_labelpad=LABEL_PAD, z_labelpad=LABEL_PAD, title=None, title_pad=TITLE_PAD, text_process=None, title_size=TITLE_SIZE, label_size=LABEL_SIZE, tick_size=TICK_SIZE, xtick_size=None, ytick_size=None, ztick_size=None, legend_size=LEGEND_SIZE, xlim=None, ylim=None, zlim=None, x_sci=None, y_sci=None, z_sci=None, x_log=False, y_log=False, z_log=False, elev=None, azim=None, legend_loc=LEGEND_LOC, bbox_to_anchor=None, tight_layout=False):
    '''
    设置图表的轴、标题、范围和图例

    参数:
    ax - 绘图的Axes对象
    x_label, y_label, z_label - x轴、y轴和z轴的标签
    x_labelpad, y_labelpad, z_labelpad - x轴、y轴和z轴标签的间距
    title - 图标题
    title_pad - 标题的间距
    text_process - 文本处理参数
    title_size - 标题字体大小
    label_size - 标签字体大小
    tick_size - 刻度标签字体大小
    xtick_size, ytick_size - x轴和y轴刻度标签字体大小
    legend_size - 图例字体大小
    xlim, ylim - 坐标轴范围
    x_sci, y_sci - 是否启用科学记数法
    legend_loc - 图例位置
    bbox_to_anchor - 图例的位置参数(示例:(1, 1), 配合legend_loc='upper left'表示图例框的upper left角放在坐标(1, 1)处)
    tight_layout - 是否启用紧凑布局,默认为False,如果输入dict,则作为tight_layout的参数
    '''
    text_process = update_dict(TEXT_PROCESS, text_process)

    is_3d = isinstance(ax, Axes3D)

    # 尝试获取x_label和y_label
    x_label = ax.get_xlabel() if x_label is None else x_label
    y_label = ax.get_ylabel() if y_label is None else y_label
    if is_3d:
        z_label = ax.get_zlabel() if z_label is None else z_label

    # 处理标签和标题
    local_x_label, local_y_label, local_z_label, local_title = label_title_process(
        x_label, y_label, z_label, title, text_process)

    # 设置标签和标题
    ax.set_xlabel(local_x_label, fontsize=label_size, labelpad=x_labelpad)
    ax.set_ylabel(local_y_label, fontsize=label_size, labelpad=y_labelpad)
    if is_3d:
        ax.set_zlabel(local_z_label, fontsize=label_size, labelpad=z_labelpad)
    ax.set_title(local_title, fontsize=title_size, pad=title_pad)

    # 设置字体
    if xtick_size is None:
        ax.tick_params(axis='x', labelsize=tick_size)
    else:
        ax.tick_params(axis='x', labelsize=xtick_size)

    if ytick_size is None:
        ax.tick_params(axis='y', labelsize=tick_size)
    else:
        ax.tick_params(axis='y', labelsize=ytick_size)

    if is_3d:
        if ztick_size is None:
            ax.tick_params(axis='z', labelsize=tick_size)
        else:
            ax.tick_params(axis='z', labelsize=ztick_size)

    # 设置坐标轴格式
    if x_sci:
        ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    if y_sci:
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    if is_3d and z_sci:
        ax.zaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='sci', axis='z', scilimits=(0, 0))

    # 设置对数坐标轴
    if x_log:
        ax.set_xscale('log')
    if y_log:
        ax.set_yscale('log')
    if is_3d and z_log:
        ax.set_zscale('log')

    # 设置坐标轴范围
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    if is_3d:
        ax.set_zlim(zlim)

    # 检查是否有图例标签
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        labels = [format_text(label, text_process) for label in labels]
        ax.legend(handles, labels, loc=legend_loc,
                  fontsize=legend_size, bbox_to_anchor=bbox_to_anchor)

    if tight_layout:
        plt.tight_layout(**tight_layout)

    if is_3d:
        set_ax_view_3d(ax, elev=elev, azim=azim)
# endregion


# region 创建fig, ax函数
def get_fig_ax(nrows=1, ncols=1, ax_width=AX_WIDTH, ax_height=AX_HEIGHT, fig_width=None, fig_height=None, sharex=False, sharey=False, subplots_params=None, adjust_params=False, ax_box_params=None, margin=None):
    '''
    创建一个图形和轴对象，并根据提供的参数调整布局和轴的方框边缘。推荐的方式是设定ax_width和ax_height，而不是fig_width和fig_height。
    如果想要先把fig分成等分,然后在每个等分里设置框的位置,使用margin
    如果想要最外层的图有自己单独的距离图像边框的范围,使用adjust_params,示例:adjust_params={'left': 0.2, 'right': 0.8, 'top': 0.8, 'bottom': 0.2, 'wspace': 0.5, 'hspace': 0.5}
    Parameters:
    - figsize: 元组，指定图形的宽度和高度。
    - adjust_params: 字典，包含用于调用subplots_adjust的参数。
    - ax_box_params: 字典，包含用于设置轴方框边缘样式的参数（如边框颜色和宽度）。

    Returns:
    - fig, ax: 创建的图形和轴对象。
    '''
    if subplots_params is None:
        subplots_params = {}
    if margin is None:
        margin = MARGIN.copy()
    if fig_width is None:
        fig_width = ax_width / ( margin['right'] - margin['left'] ) * ncols
    if fig_height is None:
        fig_height = ax_height / ( margin['top'] - margin['bottom'] ) * nrows
    figsize = (fig_width, fig_height)

    fig, ax = plt.subplots(nrows, ncols, figsize=figsize,
                           sharex=sharex, sharey=sharey, **subplots_params)

    # 如果提供了轴方框边缘参数，则应用这些样式
    if ax_box_params:
        for spine in ax.spines.values():
            spine.set(**ax_box_params)

    # 设置ax的位置
    set_relative_ax_position(ax, nrows, ncols, margin=margin)

    # 如果提供了调整参数，则应用这些参数
    if adjust_params:
        fig.subplots_adjust(**adjust_params)

    return fig, ax


def get_fig_ax_3d(**kwargs):
    '''
    创建一个3D图形和轴对象,提供cerate_fig_ax的参数作为**kwargs
    '''
    # Ensure 'subplots_params' exists in 'kwargs'
    if 'subplots_params' not in kwargs:
        kwargs['subplots_params'] = {}

    # Ensure 'subplot_kw' exists in 'subplots_params'
    if 'subplot_kw' not in kwargs['subplots_params']:
        kwargs['subplots_params']['subplot_kw'] = {}

    # Update the 'projection' in 'subplot_kw'
    kwargs['subplots_params']['subplot_kw']['projection'] = '3d'
    return get_fig_ax(**kwargs)
# endregion


# region 保存图像函数
def save_fig(fig, filename, formats=None, dpi=SAVEFIG_DPI, close=True, bbox_inches=BBOX_INCHES, pad_inches=PAD_INCHES, filename_process=None, pkl=True, **kwargs):
    '''
    保存图形到指定的文件格式(搭配concat_str使用,concat_str可以用于生成文件名)

    参数:
    filename - 保存文件的基础名（不建议包含扩展名）
    formats - 要保存的文件格式列表,默认为 [SAVEFIG_FORMAT]
    dpi - 图像的分辨率,默认为 SAVEFIG_DPI
    close - 是否在保存后关闭图形,默认为 True
    bbox_inches - 设置边界框(bounding box)尺寸,默认为 BBOX_INCHES
    '''
    if formats is None:
        formats = [SAVEFIG_FORMAT]
    filename_process = update_dict(FILENAME_PROCESS, filename_process)

    # 对filename进行处理
    filename = format_filename(filename, filename_process)

    # 从filename中获取文件夹名并创建文件夹
    mkdir(os.path.dirname(filename))

    if filename.endswith(('.png', '.pdf', '.eps')):
        # 假如filename以'.png','.pdf'或'.eps'结尾,将后缀名添加到保存的格式列表中
        formats.append(filename.split('.')[-1])
        filename = filename[:-4]
        
    # 按照formats列表中的格式保存
    for fmt in formats:
        fig.savefig(f'{filename}.{fmt}', dpi=dpi,
                    bbox_inches=bbox_inches, pad_inches=pad_inches, **kwargs)

    # 保存fig到pkl
    if pkl:
        save_pkl(fig, f'{filename}.pkl')

    if close:
        plt.close(fig)


def save_fig_lite(fig, filename, formats=None, dpi=SAVEFIG_DPI/3, close=True, bbox_inches=BBOX_INCHES, pad_inches=PAD_INCHES, filename_process=None, pkl=False, **kwargs):
    '''
    轻量级的保存图形函数,只保存位图,降低dpi,不保存pkl
    '''
    formats = [SAVEFIG_RASTER_FORMAT]
    save_fig(fig, filename, formats=formats, dpi=dpi, close=close, bbox_inches=bbox_inches, pad_inches=pad_inches, filename_process=filename_process, pkl=pkl, **kwargs)


def save_fig_3d(fig, filename, elev_list=None, azim_list=np.arange(0, 180, 30), formats=None, dpi=SAVEFIG_DPI, close=True, bbox_inches=BBOX_INCHES, pkl=True, generate_video=False, frame_rate=FRAME_RATE, delete_figs=False, video_formats=None, savefig_kwargs=None):
    '''
    保存3D图形的多个视角，对于每个视角，都会生成图片（并在文件名上加入角度），然后将图片合成视频。
    对于filename,如果以'.png','.pdf'或'.eps'结尾,则按照后缀名保存,否则按照formats列表中的格式保存。
    '''
    if elev_list is None:
        elev_list = [ELEV]
    if formats is None:
        if generate_video:
            # 如果要生成视频,且formats没有指定，只保存位图
            formats = [SAVEFIG_RASTER_FORMAT]
        else:
            # 如果不生成视频,且formats没有指定，只保存矢量图
            formats = [SAVEFIG_VECTOR_FORMAT]
    if video_formats is None:
        video_formats = ['mp4', 'gif']
    if savefig_kwargs is None:
        savefig_kwargs = {}

    if filename.endswith(('.png', '.pdf', '.eps')):
        # 假如filename以'.png','.pdf'或'.eps'结尾,将后缀名添加到保存的格式列表中
        formats.append(filename.split('.')[-1])
        filename = filename[:-4]

    ax = fig.get_axes()

    fig_paths_dict = {}
    fig_paths_list = []
    for elev in elev_list:
        for azim in azim_list:
            set_ax_view_3d(ax, elev=elev, azim=azim)
            local_filename = f'{filename}_elev_{str(int(elev))}_azim_{str(int(azim))}'
            # 注意这里不能close,并且不重复存储pkl
            save_fig(fig=fig, filename=local_filename, formats=formats,
                        dpi=dpi, close=False, bbox_inches=bbox_inches, pkl=False, **savefig_kwargs)
            fig_paths_dict[(elev, azim)] = fig_paths_dict.get((elev, azim), []) + [local_filename]
            fig_paths_list.append(local_filename)

    # 保存fig到pkl
    if pkl:
        save_pkl(fig, f'{filename}.pkl')

    if close:
        plt.close(fig)

    if generate_video:
        fig_to_video(fig_paths_list, filename, frame_rate=frame_rate, delete_figs=delete_figs, formats=video_formats)
        
    if delete_figs:
        for elev in elev_list:
            for azim in azim_list:
                for fmt in formats:
                    os.remove(f'{filename}_elev_{elev}_azim_{azim}.{fmt}')

    return fig_paths_dict, fig_paths_list


def save_fig_3d_lite(fig, filename, elev_list=None, azim_list=np.arange(0, 180, 60), formats=None, dpi=SAVEFIG_DPI/3, close=True, bbox_inches=BBOX_INCHES, pkl=False, generate_video=False, frame_rate=FRAME_RATE, delete_figs=False, video_formats=None, savefig_kwargs=None):
    '''
    轻量级的保存3D图形的多个视角,只保存位图,降低dpi,不保存pkl,不生成视频,减少了azim_list的数量
    '''
    formats = [SAVEFIG_RASTER_FORMAT]
    save_fig_3d(fig, filename, elev_list=elev_list, azim_list=azim_list, formats=formats, dpi=dpi, close=close, bbox_inches=bbox_inches, pkl=pkl, generate_video=generate_video, frame_rate=frame_rate, delete_figs=delete_figs, video_formats=video_formats, savefig_kwargs=savefig_kwargs)
# endregion
# endregion


# region 图片格式转换函数
def convert_fig(input_file_path, output_format):
    input_format = input_file_path.split('.')[-1].lower()
    output_file_path = input_file_path[:-len(input_format)] + output_format

    if input_format == 'png' and output_format == 'pdf':
        with Image.open(input_file_path) as img:
            img.convert('RGB').save(output_file_path,
                                    'PDF', resolution=SAVEFIG_DPI)

    elif input_format == 'png' and output_format == 'eps':
        print('png to eps is not supported')

    elif input_format == 'eps' and output_format == 'png':
        # # EPS to PNG conversion
        with Image.open(input_file_path) as img:
            img.load(scale=10)  # 不知道这个有什么用,但是有了就会清晰,没有就模糊
            # img.save(output_file_path, 'PNG', dpi=(SAVEFIG_DPI, SAVEFIG_DPI))
            # 创建一个白色背景的画布
            background = Image.new('RGBA', img.size, (255, 255, 255))
            # 将EPS图像合并到背景画布上
            composite = Image.alpha_composite(background, img.convert('RGBA'))
            # 保存为PNG，这时不需要指定DPI，因为我们已经在背景画布上处理了图像
            composite.save(output_file_path, 'PNG',
                           dpi=(SAVEFIG_DPI, SAVEFIG_DPI))

    # PDF to PNG (requires pdf2fig)
    elif input_format == 'pdf' and output_format == 'png':
        # Increased DPI for better quality
        figs = convert_from_path(input_file_path, dpi=SAVEFIG_DPI)
        for i, fig in enumerate(figs):
            fig.save(output_file_path, 'PNG')
            break  # Assuming saving only the first page

    # PDF to EPS (PDF -> PNG -> EPS)
    elif input_format == 'pdf' and output_format == 'eps':
        temp_fig = convert_from_path(input_file_path, dpi=SAVEFIG_DPI)[
            0]  # Increased DPI for better quality
        basedir = os.path.dirname(input_file_path)
        temp_fig_path = os.path.join(basedir, 'temp.png')
        temp_fig.save(temp_fig_path, 'PNG')
        with Image.open(temp_fig_path) as img:
            img.save(output_file_path, 'EPS')
        os.remove(temp_fig_path)  # Clean up the temporary file

    # EPS to PDF (EPS -> PNG -> PDF)
    elif input_format == 'eps' and output_format == 'pdf':
        with Image.open(input_file_path) as img:
            img.load(scale=10)  # 不知道这个有什么用,但是有了就会清晰,没有就模糊
            basedir = os.path.dirname(input_file_path)
            temp_fig_path = os.path.join(basedir, 'temp.png')
            img.save(temp_fig_path, 'PNG', dpi=(SAVEFIG_DPI, SAVEFIG_DPI))
        with Image.open(temp_fig_path) as temp_img:
            temp_img.convert('RGB').save(output_file_path,
                                         'PDF', resolution=SAVEFIG_DPI)
        os.remove(temp_fig_path)  # Clean up the temporary file

    else:
        raise ValueError(
            f"Conversion from {input_format} to {output_format} is not supported.")
# endregion
# endregion


# region 拼图相关函数
def concat_fig(fig_paths_grid, filename, formats=None, background='transparent'):
    '''
    根据二维图片路径列表自动拼接成网格布局的图片。

    参数:
        fig_paths_grid (list of list): 二维图片路径列表，每个内部列表代表一行。(如果输入一维,自动转换为二维,并横向排列,如果想要纵向排列,输入时使用[[path1],[path2]])
        filename (str): 拼接后图片的保存路径。
    '''
    if formats is None:
        formats = [SAVEFIG_RASTER_FORMAT]
    
    # 创建文件夹
    mkdir(os.path.dirname(filename))

    # 如果输入一维,自动转换为二维
    if not isinstance(fig_paths_grid[0], list):
        fig_paths_grid = [fig_paths_grid]

    rows = len(fig_paths_grid)
    cols = max(len(row) for row in fig_paths_grid)

    figs_grid = [[None for _ in range(cols)] for _ in range(rows)]
    max_widths = [0] * cols
    max_heights = [0] * rows

    # 加载所有图片并计算每列最大宽度和每行最大高度
    for row_idx, row in enumerate(fig_paths_grid):
        for col_idx, fig_path in enumerate(row):
            valid_fig_path, fig_exist = find_fig(fig_path, order=['.png', '.pdf', '.eps'])
            if fig_exist:
                if not valid_fig_path.endswith('.png'):
                    convert_fig(valid_fig_path, 'png')
                    valid_fig_path = valid_fig_path[:-3] + 'png'
            else:
                print(f'Please check the path {fig_path} and try again.')
                return
            fig = Image.open(valid_fig_path)
            figs_grid[row_idx][col_idx] = fig
            max_widths[col_idx] = max(max_widths[col_idx], fig.width)
            max_heights[row_idx] = max(max_heights[row_idx], fig.height)

    total_width = sum(max_widths)
    total_height = sum(max_heights)

    # 根据背景选择模式和颜色
    if background == 'transparent':
        new_fig = Image.new('RGBA', (total_width, total_height), (0, 0, 0, 0))
    elif background == 'white':
        new_fig = Image.new('RGB', (total_width, total_height), (255, 255, 255))
    else:
        raise ValueError(f'Background mode {background} is not supported.')

    y_offset = 0
    for row_idx, row in enumerate(figs_grid):
        x_offset = 0
        for col_idx, fig in enumerate(row):
            if fig:  # 如果该位置有图片
                new_fig.paste(fig, (x_offset, y_offset))
            x_offset += max_widths[col_idx]
        y_offset += max_heights[row_idx]

    if filename.endswith(('.png', '.pdf', '.eps')):
        # 假如filename以'.png','.pdf'或'.eps'结尾,将后缀名添加到保存的格式列表中
        formats.append(filename.split('.')[-1])
        filename = filename[:-4]
        
    for fmt in formats:
        if fmt == 'png':
            new_fig.save(f'{filename}.{fmt}')
        else:
            new_fig.save(f'{filename}.png')
            convert_fig(f'{filename}.png', fmt)


def concat_fig_with_tag(figs, filename, tags=None, formats=None, background='transparent', close=True, tag_size=TAG_SIZE, tag_color=BLACK, tag_position=FIG_TAG_POS, tag_kwargs=None, auto_tag_params=None):
    '''
    根据图片和标签自动拼接成网格布局的图片。

    参数:
        figs (list): 图片路径列表。
        labels (list): 标签列表。
        filename (str): 拼接后图片的保存路径。
    '''
    if formats is None:
        formats = [SAVEFIG_RASTER_FORMAT]
    if tag_kwargs is None:
        tag_kwargs = {}
    if auto_tag_params is None:
        auto_tag_params = {}

    if filename.endswith(('.png', '.pdf', '.eps')):
        # 假如filename以'.png','.pdf'或'.eps'结尾,将后缀名添加到保存的格式列表中
        formats.append(filename.split('.')[-1])
        filename = filename[:-4]

    fig_paths_grid = []
    flatten_figs = flatten_list(figs)
    if tags is None:
        flatten_tags = get_tag(len(flatten_figs), **auto_tag_params)
    else:
        flatten_tags = flatten_list(tags)
    for fig, tag in zip(flatten_figs, flatten_tags):
        add_fig_tag(fig, tag, x=tag_position[0], y=tag_position[1], fontsize=tag_size, color=tag_color, **tag_kwargs)
        save_fig(fig, concat_str([filename, tag]), formats=formats, close=close)
        fig_paths_grid.append(concat_str([filename, tag]))
    fig_paths_grid = rebuild_list(figs, fig_paths_grid)
    concat_fig(fig_paths_grid, filename, formats=formats, background=background)
# endregion


# region 合成视频相关函数
def fig_to_video(fig_paths, filename, frame_rate=FRAME_RATE, delete_figs=False, formats=None):
    '''
    Combine PNG figs into MP4 videos and/or GIF figs, checking for the existence of each fig.
    假如fig_paths中的图片没有后缀，则会自动寻找有无对应的.png,.pdf或.eps文件,如果有则转换为.png文件，并生成视频。
    Parameters:
    - fig_paths: List of paths to the input figs.
    - filename: Base path for the output video or GIF fig. Extension will be added based on format.
    - frame_rate: Frame rate of the output video (ignored for GIF).
    - delete_figs: Flag to indicate whether to delete the input figs after creating the output.
    - formats: List of formats for the output, including 'mp4' and/or 'gif'.
    '''
    if formats is None:
        formats = ['mp4', 'gif']

    # 创建文件夹
    mkdir(os.path.dirname(filename))

    valid_fig_paths = []

    # Check for the existence of figs and accumulate valid paths
    for fig_path in fig_paths:
        valid_fig_path, fig_exist = find_fig(fig_path, order=['.png', '.pdf', '.eps'])
        if fig_exist:
            if valid_fig_path.endswith('.png'):
                valid_fig_paths.append(valid_fig_path)
            else:
                convert_fig(valid_fig_path, 'png')
                valid_fig_path = valid_fig_path[:-3] + 'png'
                valid_fig_paths.append(valid_fig_path)
        else:
            print(f'Please check the path {fig_path} and try again.')
            return

    # Ensure there are valid figs to process
    if not valid_fig_paths:
        print("No valid figs to process.")
        return

    if 'mp4' in formats:
        # MP4 Video output
        video_filename = f"{filename}.mp4"
        frame = cv2.imread(valid_fig_paths[0])
        height, width, layers = frame.shape
        video_size = (width, height)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_filename, fourcc,
                              frame_rate, video_size)

        for fig_path in valid_fig_paths:
            fig = cv2.imread(fig_path)
            if fig.shape[1] != video_size[0] or fig.shape[0] != video_size[1]:
                fig = cv2.resize(
                    fig, video_size, interpolation=cv2.INTER_LANCZOS4)
            out.write(fig)
        out.release()

    if 'gif' in formats:
        # GIF output
        gif_filename = f"{filename}.gif"
        figs = [Image.open(fig_path) for fig_path in valid_fig_paths]
        resized_figs = [fig.resize(
            (figs[0].width, figs[0].height), Image.LANCZOS) for fig in figs]
        resized_figs[0].save(gif_filename, save_all=True,
                               append_images=resized_figs[1:], duration=1000/frame_rate, loop=0)

    if delete_figs:
        # Delete figs after processing all formats
        for fig_path in valid_fig_paths:
            os.remove(fig_path)
        print("All valid figs deleted.")

    cv2.destroyAllWindows()
# endregion