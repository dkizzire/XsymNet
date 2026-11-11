import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from io_utils import utility_functions as util
from xsymnet import xsymnet_funcs as funcs


rcParams.update({
    "text.usetex": False,  # Use True for LaTeX rendering if desired
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 300,
    "figure.figsize": (8.5, 4.25),  # 1-column journal width

})

# use current working directory as base and build paths portably
cwd = os.getcwd()
material_name = 'wo3'
filename1 = os.path.join(cwd, material_name, f'X_expt_maxabs_wo3_1043K_expt_1043K.xy')
filename2 = os.path.join(cwd, material_name, f'X_expt_maxabs_wo3_1043K_expt_1123K.xy')
x1,y1 = funcs.load_xy_data(filename1)
x2,y2 = funcs.load_xy_data(filename2)
# label = r'$\beta$ Bi$_{2}Sn$_{2}$O$_{7}$ MaxAbs'
label1 = r'WO$_{3}$ 1043K Feature-Wise'
label2 = r'WO$_{3}$ 1123K Feature-Wise'
fig, ax = plt.subplots()
ax.plot(x1, y1, label=label1, color='blue', linewidth=1)
ax.plot(x2, y2, label=label2, color='red', linewidth=1)
ax.set_xlabel('2θ (degreesm )')
# ax.set_xlabel(r'ToF ($\mu$s)')
ax.set_ylabel('Intensity (a.u.)')
# ax.set_title('Experimental Diffraction Pattern')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(cwd, material_name, f'X_expt_maxabs_{material_name}_1043K_vs_1123K.png'), dpi=300)

ydiff = y2 - y1
fig1, ax1 = plt.subplots()
ax1.plot(x1, ydiff, label='Difference (1123K - 1043K)', color='green', linewidth=1)
ax1.set_xlabel('2θ (degrees)')
ax1.set_ylabel('Intensity Difference (a.u.)')
ax1.legend()
plt.tight_layout()
plt.savefig(os.path.join(cwd, material_name, f'X_expt_maxabs_{material_name}_1043K_vs_1123K_diff.png'), dpi=300)