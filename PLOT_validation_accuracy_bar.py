import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Data
samples = [50, 100, 150, 200, 225]
x = np.arange(len(samples))  # positions on the x-axis
x = x / 2
# Validation accuracies
pzt90_5 = [85, 90.8, 90.2, 92.2, 92.3]  
pzt90_10 = [98.5, 99.1, 99.3, 99.4, 99.7]

bso_5 = [78.8, 77.71, 86.7, 90, 91]
bso_10 = [91, 89.8, 95.1, 96.8, 97.3]

wo3_5 = [74.9, 76.1, 83.6, 83.5, 85.2]
wo3_10 = [88.3, 89.5, 93.9, 93.5, 94.5]

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
plt.ylim(70, 95)
plt.yticks(np.arange(70, 100, 10))
# plt.legend(frameon=True, loc='upper left')
plt.tight_layout()
plt.savefig('validation_accuracy_vs_samples_bar.png', dpi=300)
plt.show()