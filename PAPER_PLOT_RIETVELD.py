
"""
Plot all .xy files in /bso/Rietveld_top_5/beta each in its own figure,
overlaying experimental data from /bso/Hrp27190_bs_NoComments_dataonly.xy.
Saves PNGs next to the input files and (optionally) shows them.
"""
import os
import glob
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

# Directories and input files
cwd = os.getcwd()
beta_dir = os.path.join(cwd, "bso", "Rietveld_top_5", "beta")
exp_file = os.path.join(cwd, "bso", "Hrp27190_bs_NoComments_dataonly.xy")

SAVE_PNG = True
SHOW_PLOTS = True

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

def main():
    if not os.path.isfile(exp_file):
        raise SystemExit(f"Experimental file not found: {exp_file}")

    exp_x, exp_y = load_xy(exp_file)
    files = sorted(glob.glob(os.path.join(beta_dir, "*.xy")))
    if not files:
        raise SystemExit(f"No .xy files found in {beta_dir}")

    rwp_labels = [
        r'R$_{wp}$=3.932308', r'R$_{wp}$=7.983865',
        r'R$_{wp}$=8.333912', r'R$_{wp}$=5.225279',
        r'R$_{wp}$=4.761396'
    ]

    n_files = min(len(files), 5)  # make sure there are at least 5 datasets
    fig, axes = plt.subplots(
        n_files, 1, figsize=(8.5, 11),
        sharex=True, constrained_layout=True
    )

    if n_files == 1:
        axes = [axes]  # make iterable

    for count, (fpath, ax) in enumerate(zip(files[:n_files], axes)):
        try:
            x, y = load_xy(fpath)
        except Exception as e:
            print(f"Skipping {fpath}: {e}")
            continue

        y = y[:len(exp_y)]
        x = x[:len(exp_x)]

        sglabel = os.path.basename(fpath).replace("rietveld_", "").replace("_xy.xy", "")
        sglabel = sglabel + ' Rietveld Refinement'

        # Main experimental and refined patterns
        ax.plot(exp_x, exp_y, linestyle='None', marker='.', markersize=3,
                color='k', label=r'$\beta$ BSO 473 K')
        ax.plot(x, y, '-', lw=1.0, color="#9ED1E4FF", label=sglabel)
        ax.plot(x, (y - exp_y - 15), '-', lw=1.0, color="#0A7D2EFF", label='Difference')
        ax.plot([], [], ' ', label=rwp_labels[count])  # dummy for legend

        ax.set_ylabel('Intensity (a.u.)')
        ax.legend(loc='upper right', frameon=False, handlelength=1.5)

    # Shared x-axis label
    axes[-1].set_xlabel(r'ToF ($\mu$s)')

    # Optional: add global title
    fig.suptitle(r'$\beta$-BSO Rietveld Refinements', fontsize=12)

    # Save combined figure
    if SAVE_PNG:
        out_path = os.path.join(beta_dir, "combined_rietveld_5plots.png")
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
# beta_dir = os.path.join(cwd, "bso", "Rietveld_top_5", "beta")
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

#     pattern = os.path.join(beta_dir, "*.xy")
#     # files = sorted(glob.glob(os.path.join(beta_dir, f'*.xy')))
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
#     #     raise SystemExit(f"No .xy files found in {beta_dir}")
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

