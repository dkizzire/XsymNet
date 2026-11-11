import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

cwd = os.getcwd()
# =========================
# Set Plot Style for Journal
# =========================
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
    "figure.figsize": (5.3, 2.5),  # 1-column journal width
    "savefig.bbox": "tight"
})

# =========================
# Load the .pt Tensor File
# =========================
# file_path = os.path.join(cwd, 'wo3', 'ml_data','1043K','topas0032_wo3_1043K_rs.pt')  # Replace with your actual file
file_path = os.path.join(cwd, 'wo3', 'P1_wo3_1085K_xy.pt')  # Replace with your actual file
tensor = torch.load(file_path)
tensor = tensor.unsqueeze(0)
tensor_np = tensor.cpu().numpy()
num_patterns = tensor_np.shape[0]

# file_path = os.path.join(cwd, 'wo3', 'experimental_refinement', 'pawley', 'P1_wo3_1085K_xy.xy')  # Replace with your actual file
file_path = os.path.join(cwd, 'wo3', 'P1_wo3_1085K_xy.xy')  # Replace with your actual file
tensor_xy = np.loadtxt(file_path)
x = tensor_xy[:,0]
y = tensor_xy[:,1]
# =========================
# Plot the First 5 Patterns with Offsets
# =========================
# x = np.arange(tensor_np.shape[1])  # Replace with real x-axis if available
# offset = 1.05 * np.max(tensor_np[:5])  # Offset amount based on data range
offset = 1.05 * np.max(tensor_np[0])  # Offset amount based on data range

fig, ax = plt.subplots()
colors = plt.cm.tab10.colors

# for i in range(min(5, num_patterns)):
for i in range(1):
    # j = i
    # y = tensor_np[j] + j * offset  # Add vertical offset

    # ax.plot(x, y, label=f'Pattern {i+1}', color=colors[i % len(colors)], linewidth=1)
    # ax.plot(x, y, label=f'Pattern {i+1}', color='#156082', linewidth=1)
    ax.plot(x, y, label=f'Pattern {i+1}', color='#B95E5E', linewidth=1)

# =========================
# Final Touches
# =========================
ax.set_xlabel("2θ (degrees)")  # Update to actual x-axis unit if needed
ax.set_ylabel("Intensity (a.u.)")
# ax.set_yticks([])  # Remove y-axis ticks for cleaner look
# ax.set_xticks([])  # Remove x-axis ticks for cleaner look
# ax.legend(loc='upper right', frameon=False)
# ax.grid(True, linestyle='--', alpha=0.4)

plt.tight_layout()
# plt.savefig("stacked_xrd_patterns.svg", format="svg")  # Save as vector image
plt.savefig("stacked_xrd_patterns.png")  # Save as vector image
plt.show()
