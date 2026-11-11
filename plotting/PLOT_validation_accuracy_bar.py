import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Data
samples = [50, 100, 150, 200, 225]
x = np.arange(len(samples))  # positions on the x-axis
x = x / 2
# Validation accuracies
pzt90_5 = [95.86, 98.76, 99.17, 99.03, 99.59]  
pzt90_10 = [99.72, 100.0, 100.0, 100.0, 100.0]

bso_5 = [83.48, 90.7, 93.79, 94.49, 95.38]
bso_10 = [94.34, 97.5, 98.5, 98.52, 98.92]

wo3_5 = [83.85, 89.12, 92.31, 94.35, 96.07]
wo3_10 = [93.94, 96.8, 98.27, 99.1, 99.22]

# Bar width
bar_width = 0.08
# hatches = ['', '', '', '///', '///', '///']  # Only BSO bars get hatching
hatches = ['', '', '']  # Only BSO bars get hatching

# Offsets for each bar group
# offsets = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]  # for 4 bars per group
offsets = [-1., 0, 1.]  # for 4 bars per group
# colors = ['#4D4D4D', '#1B9E77', "#1F2199", '#999999', '#66C2A5', "#0C10E9"]
# colors = ["#B95E5E", '#1B9E77', "#3C6C9C" ] #, '#999999', '#66C2A5', "#0C10E9"]
colors = ["#D49595FF", '#A8DD94FF', "#9ED1E4FF" ] #, '#999999', '#66C2A5', "#0C10E9"]
# labels = [r'PbZr$_{0.1}$Ti$_{0.9}$O$_3$ Top 5', r'BiSn$_{2}$O$_{7}$ Top 5', r'WO$_{3}$ Top 5', r'PbZr$_{0.1}$Ti$_{0.9}$O$_3$ Top 10', r'BiSn$_{2}$O$_{7}$ Top 10', r'WO$_{3}$ Top 10']
labels = [r'PbZr$_{0.1}$Ti$_{0.9}$O$_3$', r'BiSn$_{2}$O$_{7}$', r'WO$_{3}$']#, r'PbZr$_{0.1}$Ti$_{0.9}$O$_3$ Top 10', r'BiSn$_{2}$O$_{7}$ Top 10', r'WO$_{3}$ Top 10']
# data_groups = [pzt90_5, bso_5, wo3_5, pzt90_10, bso_10, wo3_10]
data_groups = [pzt90_5, bso_5, wo3_5]

# Create plot
plt.figure(figsize=(4.0, 3))
for i, (offset, data, color, hatch, label) in enumerate(zip(offsets, data_groups, colors, hatches, labels)):
    plt.bar(x + offset * bar_width, data, width=bar_width, color=color, hatch=hatch, label=label)

plt.xlabel('Samples per Subgroup')
plt.ylabel('Validation Accuracy (%)')
plt.xticks(x, samples)
plt.ylim(70, 101)
plt.yticks(np.arange(70, 110, 10))
# plt.legend(frameon=True, loc='upper left')
plt.tight_layout()
plt.savefig('validation_accuracy_vs_samples_bar.png', dpi=300)
plt.show()