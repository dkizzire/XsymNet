import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import utility_functions as util


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

    # "savefig.bbox": "tight"

# use current working directory as base and build paths portably
cwd = os.getcwd()
data_dir = os.path.join(cwd, "wo3")
file_names = ['d8_03901_t1043.xy', 'd8_03901_t1085.xy']

x1, y1 = util.load_xy(os.path.join(data_dir, file_names[0]))
x2, y2 = util.load_xy(os.path.join(data_dir, file_names[1]))

# Normalize both datasets to 0-100 scale for better comparison
maxy = np.max(y1)
miny = np.min(y1)
scale = 83
y1 = (y1 - miny) / (maxy - miny) * scale
y2 = (y2 - miny) / (maxy - miny) * scale

# Compute optimal scale factor
a_opt = np.dot(y1, y2) / np.dot(y1, y1)
# Compute minimized error
mse = np.mean((a_opt * y1 - y2) ** 2)
print("Optimal scale factor a =", a_opt)
print("Minimized mean squared error =", mse)

fig, ax = plt.subplots()
# fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x2, y2, '-', lw=0.9, color='#000000FF', label=r'WO$_{3}$ 1085 K')
ax.plot([],[], ' ', label= 'SG# 130 P4/ncc:2')

ax.plot(x1, y1*1.2, '-', lw=0.5, color='#D2572EFF', label=r'WO$_{3}$ 1043 K')
ax.plot([],[], ' ', label= 'SG# 14 P121/c1')

# add inset axes (zoomed region between 52 and 62)
axins = fig.add_axes([0.30, 0.45, 0.33, 0.36])  # [left, bottom, width, height]
axins.plot(x2, y2, '-', lw=0.9, color='#000000FF')
axins.plot(x1, y1*1.2, '-', lw=0.5, color='#D2572EFF')
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
# axins.set_title('Zoom 52–62°', fontsize=9)

# # indicate zoomed area on the main axes
# rect = plt.Rectangle((x_min, ax.get_ylim()[0]), x_max - x_min, ax.get_ylim()[1] - ax.get_ylim()[0],
#                      linewidth=0.8, edgecolor='gray', facecolor='none', linestyle='--')
# ax.add_patch(rect)


ax.set_xlim(21, 90)
ax.set_xlabel(r'2$\theta$ (degrees)')
ax.set_ylabel('Intensity (a.u.)')
ax.legend()

plt.savefig(os.path.join(cwd, 'wo3', "wo3_diffraction_comparison.png"), dpi=300, bbox_inches='tight')
plt.show()