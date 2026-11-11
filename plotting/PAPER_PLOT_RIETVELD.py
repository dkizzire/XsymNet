
"""
Plot Reitveld refinement .xy files from ./material/Rietveld_top_5/phase/*.xy  each in its own subplot against experimental data,
Saves PNGs next to the input files and (optionally) shows them.
"""
import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

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
    "savefig.bbox": "tight"
})

material = "bso"
phase = "beta"
# experimental_label = r'WO$_{3}$ 1123 K'
# experimental_label = r'WO$_{3}$ 1043 K'
# experimental_label = r'PbZr$_{0.1}$Ti$_{0.1}$O$_{3}$ 300 K'
# experimental_label = r'$\alpha$ BSO 298 K'
experimental_label = r'$\beta$ BSO 473 K'

x_axis_label = r'ToF ($\mu$s)'
# x_axis_label = r'2$\theta$ (°)'
ground_truth_id = '0152'
# vertical offset for difference curve (number is converted to negative offset in code)
# difference_curve_offset = 2000  # wo3
difference_curve_offset = 15  # bso
# Directories and input files
cwd = os.getcwd()
data_dir = os.path.join(cwd, material, "Rietveld_top_5_new", phase)
# exp_file = os.path.join(cwd, material, "d8_03901_1123K_850c_trim.xy") # Experimental data file wo3 1123K
# exp_file = os.path.join(cwd, material, "d8_03901_t1085_plot.xy") # Experimental data file wo3 1085K
# exp_file = os.path.join(cwd, material, "d8_03901_t1043_plot.xy") # Experimental data file wo3 1043K
# exp_file = os.path.join(cwd, material, "pzt90_raw_exp_xy.xy") # Experimental data file pzt90
exp_file = os.path.join(cwd, material, "Hrp27190_bs_NoComments_dmin_1p0.xy") # Experimental data file bso beta
# exp_file = os.path.join(cwd, material, "hrp27188_bs_alpha_neutron_NoComments_dmin_1p0.xy") # Experimental data file bso alpha

SAVE_PNG = True
SHOW_PLOTS = False

def load_xy(path):
    """Load two-column XY data robustly. Returns (x, y)."""
    try:
        data = np.loadtxt(path)
    except Exception:
        raise RuntimeError(f"Failed reading {path}")
    if data.size == 0:
        raise RuntimeError(f"No data in {path}")
    data = np.atleast_2d(data)
    if data.shape[1] < 2:
        raise RuntimeError(f"File {path} has fewer than 2 columns")
    return data[:, 0], data[:, 1]

def extract_info_from_out(data_dir):
    """Extract r_wp, space group number, and H-M name from .inp files in data_dir.

    Returns a dict mapping 4-digit IDs to dicts with keys: 'rwp', 'sg_number', 'hm_name'.
    """
    info_dict = {}
    inp_files = glob.glob(os.path.join(data_dir, "*.out"))

    # Pattern to find 4-digit ID in filename (not followed by k/K to avoid matching temperatures like 1043K)
    id_pattern = re.compile(r'(\d{4})(?![kK])')
    # Pattern to extract r_wp value (r_wp followed by whitespace and float)
    rwp_pattern = re.compile(r'r_wp\s+([\d.]+)')
    # Pattern to extract space_group number (number before colon)
    sg_pattern = re.compile(r'space_group\s+(\d+):')

    for inp_file in inp_files:
        basename = os.path.basename(inp_file)
        id_match = id_pattern.search(basename)
        if not id_match:
            continue
        file_id = id_match.group(1)

        try:
            with open(inp_file, 'r') as f:
                lines = f.readlines()
            content = ''.join(lines)

            info = {'rwp': None, 'sg_number': None, 'hm_name': None}

            # Extract r_wp
            rwp_match = rwp_pattern.search(content)
            if rwp_match:
                info['rwp'] = float(rwp_match.group(1))

            # Find space_group line and extract number + H-M name from line above
            for i, line in enumerate(lines):
                sg_match = sg_pattern.search(line)
                if sg_match:
                    info['sg_number'] = sg_match.group(1)
                    # Get H-M name from line above (strip ' prefix and whitespace)
                    if i > 0:
                        hm_line = lines[i - 1].strip()
                        if hm_line.startswith("'"):
                            info['hm_name'] = hm_line[1:].strip()
                    break

            info_dict[file_id] = info
        except Exception as e:
            print(f"Warning: Could not read info from {inp_file}: {e}")

    return info_dict

def get_id_from_xy_filename(filename):
    """Extract 4-digit ID from an .xy filename (not followed by k/K to avoid matching temperatures)."""
    id_pattern = re.compile(r'(\d{4})(?![kK])')
    match = id_pattern.search(os.path.basename(filename))
    return match.group(1) if match else None

def main():
    if not os.path.isfile(exp_file):
        raise SystemExit(f"Experimental file not found: {exp_file}")

    exp_x, exp_y = load_xy(exp_file)
    files = sorted(glob.glob(os.path.join(data_dir, "*.xy")))
    if not files:
        raise SystemExit(f"No .xy files found in {data_dir}")

    # Extract r_wp, space group, and H-M name from .inp files
    info_dict = extract_info_from_out(data_dir)

    n_files = min(len(files), 5)  # make sure there are at least 5 datasets
    fig, axes = plt.subplots(
        n_files, 1, figsize=(6.5, 7),
        sharex=True, constrained_layout=True
    )

    if n_files == 1:
        axes = [axes]  # make iterable

    for fpath, ax in zip(files[:n_files], axes):
        try:
            x, y = load_xy(fpath)
        except Exception as e:
            print(f"Skipping {fpath}: {e}")
            continue

        y = y[:len(exp_y)]
        x = x[:len(exp_x)]

        # Get info for this file based on its ID
        file_id = get_id_from_xy_filename(fpath)
        sglabel = f'{file_id} Rietveld Refinement'
        if file_id and file_id in info_dict:
            info = info_dict[file_id]
            rwp_label = rf'R$_{{wp}}$={info["rwp"]:.2f}' if info['rwp'] else r'R$_{wp}$=N/A'
            # Build annotation text: file_id - space group number H-M name
            sg_num = info['sg_number'] or '?'
            hm_name = info['hm_name'] or '?'
            annotation_text = f"{file_id} - {sg_num} {hm_name}"
        else:
            rwp_label = r'R$_{wp}$=N/A'
            annotation_text = f"{file_id or '?'}"

        # Main experimental and refined patterns
        ax.plot(exp_x, exp_y, linestyle='None', marker='.', markersize=3, color='k', label=experimental_label)
        ax.plot(x, y, '-', lw=1.0, color="#ED3030FF", label=sglabel)
        ax.plot(x, (y - exp_y - difference_curve_offset), '-', lw=1.0, color="#0A7D2EFF", label='Difference')
        ax.plot([], [], ' ', label=rwp_label)  # dummy for legend

        # Add annotation text in upper left corner (highlight if ground truth)
        is_ground_truth = (file_id == ground_truth_id)
        text_kwargs = dict(
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            horizontalalignment='left'
        )
        if is_ground_truth:
            text_kwargs['bbox'] = dict(boxstyle='round,pad=0.3', facecolor='yellow', edgecolor='black', alpha=0.8)
            # text_kwargs['fontweight'] = 'bold'
        ax.text(0.02, 0.95, annotation_text, **text_kwargs)

        ax.set_ylabel('Intensity (a.u.)')
        ax.legend(loc='upper right', frameon=False, handlelength=1.5)

    # Shared x-axis label
    axes[-1].set_xlabel(x_axis_label)

    # Optional: add global title
    # fig.suptitle(r'$\beta$-BSO Rietveld Refinements', fontsize=12)

    # Save combined figure
    if SAVE_PNG:
        out_path = os.path.join(data_dir, f"combined_rietveld_top5_plots_{material}_{phase}_new.png")
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"Saved combined figure: {out_path}")

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)

if __name__ == "__main__":
    main()
    
# import os
# import glob
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import rcParams

# rcParams.update({
#     "text.usetex": False,  # Use True for LaTeX rendering if desired
#     "font.family": "serif",
#     "font.size": 10,
#     "axes.labelsize": 11,
#     "axes.titlesize": 11,
#     "legend.fontsize": 9,
#     "xtick.labelsize": 9,
#     "ytick.labelsize": 9,
#     "figure.dpi": 300,
#     "figure.figsize": (5.3, 2.5),  # 1-column journal width
#     "savefig.bbox": "tight"
# })


# # use current working directory as base and build paths portably
# cwd = os.getcwd()
# data_dir = os.path.join(cwd, "bso", "Rietveld_top_5", "beta")
# exp_file = os.path.join(cwd, "bso", "Hrp27190_bs_NoComments_dataonly.xy")
# SAVE_PNG = True     # set False to avoid saving PNGs
# SHOW_PLOTS = True  # set True to call plt.show() at end (blocks)

# def load_xy(path):
#     """Load two-column XY data robustly. Returns (x, y)."""
#     try:
#         data = np.loadtxt(path)
#     except Exception:
#         raise RuntimeError(f"Failed reading {path}")
#     if data.size == 0:
#         raise RuntimeError(f"No data in {path}")
#     data = np.atleast_2d(data)
#     if data.shape[1] < 2:
#         raise RuntimeError(f"File {path} has fewer than 2 columns")
#     x = data[:, 0]
#     y = data[:, 1]
#     return x, y

# def main():
#     if not os.path.isfile(exp_file):
#         raise SystemExit(f"Experimental file not found: {exp_file}")
#     expt = np.loadtxt(exp_file)
#     exp_x = expt[:, 0]
#     exp_y = expt[:, 1]

#     pattern = os.path.join(data_dir, "*.xy")
#     # files = sorted(glob.glob(os.path.join(data_dir, f'*.xy')))
#     # direcotry = os.path.join(cwd, 'bso','Rietveld_top_5', 'beta',  f'rietveld_0152_xy.xy')
#     # print (direcotry)
#     # newfile = f'{cwd}/bso/Rietveld_top_5/beta/rietveld_0152_xy.xy'
#     # files = sorted(glob.glob('./bso/Rietveld_top_5/beta/rietveld_0152_xy.xy'))
#     # files = sorted(glob.glob(os.path.join(cwd, 'bso','Rietveld_top_5', 'beta',  f'rietveld_0152_xy.xy')))
#     files = sorted(glob.glob(os.path.join(cwd, 'bso','Rietveld_top_5', 'beta',  f'*.xy')))
#     # print(files)
#     # exclude experimental file if it's in the same folder
#     # files = [f for f in files if os.path.abspath(f) != os.path.abspath(exp_file)]
#     # if not files:
#     #     raise SystemExit(f"No .xy files found in {data_dir}")
#     rwp_labels = [r'R$_{wp}$=3.932308', r'R$_{wp}$=7.983865', r'R$_{wp}$=8.333912', r'R$_{wp}$=5.225279', r'R$_{wp}$=4.761396']
#     count = 0
#     for fpath in files:
#         try:
#             x, y = load_xy(fpath)
#         except Exception as e:
#             print(f"Skipping {fpath}: {e}")
#             continue
#         ylen = len(exp_y)
#         y = y[:ylen]  # truncate to match experimental data length
#         x = x[:ylen]
#         exp_x = exp_x[:ylen]
#         sglabel = os.path.basename(fpath).strip('rietveld_').strip('_xy.xy')
#         sglabel = sglabel + ' Rietveld Refinement'
#         fig, ax = plt.subplots(figsize=(8, 5))
#         ax.plot(exp_x, exp_y, linestyle='None', marker='.', markersize=3, color='k', label=r'$\beta$ BSO 473 K')
#         ax.plot(x, y, '-', lw=1.0, color="#9ED1E4FF", label=sglabel)
#         ax.plot(x, (y - exp_y - 15) , '-', lw=1.0, color="#0A7D2EFF", label='Difference')
#         ax.plot([], [], ' ', label=rwp_labels[count])  # dummy for legend
#         ax.set_xlabel(r'ToF ($\mu$s)')
#         # ax.set_xlabel(r'2$\theta$ (degrees)')
#         ax.set_ylabel('Intensity (a.u.)')
#         # ax.set_ylim(-5, 100)
#         # ax.set_title(os.path.basename(fpath))
#         ax.legend()
#         fig.tight_layout()  # tight box

#         if SAVE_PNG:
#             out = os.path.splitext(fpath)[0] + "_plot.png"
#             fig.savefig(out, dpi=300, bbox_inches='tight')
#             print(f"Saved: {out}")

#         if not SHOW_PLOTS:
#             plt.close(fig)

#         count += 1
#     if SHOW_PLOTS:
#         plt.show()

# if __name__ == "__main__":
#     main()

