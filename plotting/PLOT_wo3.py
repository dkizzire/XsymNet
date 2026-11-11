import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from io_utils import utility_functions as util
from xsymnet import xsymnet_funcs as xf

rcParams.update({
    "text.usetex": False,  # Use True for LaTeX rendering if desired
    "font.family": "serif",
    "font.size": 14,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "legend.fontsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "figure.dpi": 300,
    "figure.figsize": (8.5, 4.25),  # 1-column journal width

})

# use current working directory as base and build paths portably
cwd = os.getcwd()
data_dir = os.path.join(cwd, "wo3")

do_neutron = True
do_xray = True
do_maxabs = True

alpha = 0.6

if do_neutron:
    neutron_tetragonal_file = os.path.join(cwd, 'wo3', 'rietveld_str_0032_neutron_powder_calc', '20260226094955022558_2.dat')
    neutron_monoclinic_file = os.path.join(cwd, 'wo3', 'rietveld_str_0094_neutron_powder_calc', '20260226091805609723_2.dat')

    x1, y1 = util.load_xy(neutron_tetragonal_file)
    x2, y2 = util.load_xy(neutron_monoclinic_file)

    y2 = torch.from_numpy(y2)
    y1 = torch.from_numpy(y1)
    y2, y1, _ = xf.MinMaxScale_2d(y2, y1, scale = 100)

    fig, ax = plt.subplots()
    ax.plot(x2, y2, '-', lw=0.9, color='#ED3030FF', label=r'WO$_{3}$ 1123 K')
    ax.plot([],[], ' ', label= '130 P4/ncc:2')
    
    ax.plot(x1, y1, '-', lw=0.9, color='#000000FF', label=r'WO$_{3}$ 1043 K', alpha=alpha)
    ax.plot([],[], ' ', label= '14 P121/c1')

    ax.set_xlim(np.min(x1), np.max(x1))
    ax.set_xlabel(r'ToF ($\mu$s)')
    ax.set_ylabel('Normalized Intensity (a.u.)')
    ax.legend(loc = 'upper right')

    plt.savefig(os.path.join(cwd, 'wo3', "wo3_neutron_simulation_1043_1123.svg"), dpi=600, bbox_inches='tight')
    plt.show()

if do_xray:
    file_names = ['d8_03901_t1043_plot.xy', 'd8_03901_1123K_850c_trim.xy']
    xray_tetragonal_file = os.path.join(data_dir, file_names[0])
    xray_monoclinic_file = os.path.join(data_dir, file_names[1])
    x1, y1 = util.load_xy(xray_tetragonal_file)
    x2, y2 = util.load_xy(xray_monoclinic_file)

    y2 = torch.from_numpy(y2)
    y1 = torch.from_numpy(y1)
    y1, y2, _ = xf.MinMaxScale_2d(y1, y2, scale = 100)

    fig, ax = plt.subplots()
    ax.plot(x2, y2, '-', lw=0.9, color='#ED3030FF', label=r'WO$_{3}$ 1123 K')
    ax.plot([],[], ' ', label= '130 P4/ncc:2')
    
    ax.plot(x1, y1, '-', lw=0.9, color='#000000FF', label=r'WO$_{3}$ 1043 K', alpha=alpha)
    ax.plot([],[], ' ', label= '14 P121/c1')


    # add inset axes (zoomed region between 52 and 62)
    axins = fig.add_axes([0.30, 0.45, 0.33, 0.36])  # [left, bottom, width, height]
    axins.plot(x2, y2, '-', lw=0.9, color='#ED3030FF')
    axins.plot(x1, y1, '-', lw=0.9, color='#000000FF', alpha=alpha)


    x_min, x_max = 52, 62
    axins.set_xlim(x_min, x_max)

    # determine y-limits from data in the zoom window with a small margin
    mask2 = (x2 >= x_min) & (x2 <= x_max)
    mask1 = (x1 >= x_min) & (x1 <= x_max)
    ys = np.concatenate([y2[mask2], (y1*1.2)[mask1]])
    if ys.size > 0:
        ymin, ymax = ys.min(), ys.max()
        yrange = ymax - ymin if ymax > ymin else 1.0
        axins.set_ylim(ymin - 0.06 * yrange, ymax + 0.06 * yrange)

    axins.set_xticks([52, 57, 62])
    axins.set_yticks([])


    ax.set_xlim(21, 90)
    ax.set_xlabel(r'2$\theta$ (°)')
    ax.set_ylabel('Normalized Intensity (a.u.)')
    ax.legend()

    plt.savefig(os.path.join(cwd, 'wo3', "wo3_xray_comparison_1043_1123.svg"), dpi=600, bbox_inches='tight')
    plt.show()

if do_maxabs:
    file_names = ['X_expt_maxabs_wo3_1043K_expt_1043K.xy', 'X_expt_maxabs_wo3_1043K_expt_1123K.xy']
    xray_tetragonal_file = os.path.join(data_dir, file_names[0])
    xray_monoclinic_file = os.path.join(data_dir, file_names[1])
    x1, y1 = util.load_xy(xray_tetragonal_file)
    x2, y2 = util.load_xy(xray_monoclinic_file)

    fig, ax = plt.subplots()
    ax.plot(x2, y2, '-', lw=0.9, color='#ED3030FF', label=r'WO$_{3}$ 1123 K')
    ax.plot([],[], ' ', label= '130 P4/ncc:2')
    
    ax.plot(x1, y1, '-', lw=0.9, color='#000000FF', label=r'WO$_{3}$ 1043 K', alpha=alpha)
    ax.plot([],[], ' ', label= '14 P121/c1')

    ax.set_xlim(21, 90)
    ax.set_xlabel(r'2$\theta$ (°)')
    ax.set_ylabel('Feature-Wise Scaled Intensity (a.u.)')
    ax.legend(loc='upper right')
    

    plt.savefig(os.path.join(cwd, 'wo3', "wo3_xray_Feature_comparison_1043_1123.svg"), dpi=600, bbox_inches='tight')
    plt.show()