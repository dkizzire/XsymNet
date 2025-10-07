import os 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# plot of the validation accuracy vs. number of samples in the training set 

pzt90_1 = [38, 41.7, 44.8, 49.2, 48.7]
pzt90_5 = [85, 90.8, 90.2, 92.2, 92.3]
pzt90_10 = [98.5, 99.1, 99.3, 99.4, 99.7]

wo3_1 = [32.5, 33.6, 43.5, 44, 48]
wo3_5 = [74.9, 76.1, 83.6, 83.5, 85.2]
wo3_10 = [88.3, 89.5, 93.9, 93.5, 94.5]

bso_1 = [37.2, 37.3, 46.8, 51.27, 53.5]
bso_5 = [78.8, 77.71, 86.7, 90, 91]
bso_10 = [91, 89.8, 95.1, 96.8, 97.3]

samples = [50, 100, 150, 200, 225]

plt.figure(figsize=(10, 6))
# plt.plot(samples, pzt90_1, color='blue', marker='o', linestyle='--', label='PZT90 Top 1')
plt.plot(samples, pzt90_5, color='blue', marker='o', linestyle='--', label='PZT90 Top 5')
plt.plot(samples, pzt90_10, color='blue', marker='^', linestyle='--', label='PZT90 Top 10')
# plt.plot(samples, wo3_1, color='red', marker='o', linestyle='--', label='WO3 Top 1')
# plt.plot(samples, wo3_5, color='red', marker='o', linestyle='--', label='WO3 Top 5')
# plt.plot(samples, wo3_10, color='red', marker='^', linestyle='--', label='WO3 Top 10')
# plt.plot(samples, bso_1, color='green', marker='o', linestyle='--', label='BSO Top 1')
plt.plot(samples, bso_5, color='green', marker='o', linestyle='--', label='BSO Top 5')
plt.plot(samples, bso_10, color='green', marker='^', linestyle='--', label='BSO Top 10')
plt.xlabel('Samples of Each Subgroup in Training Set')
plt.ylabel('Validation Accuracy (%)')
plt.legend()
plt.ylim(70, 105)
plt.xlim(25, 250)
plt.yticks(np.arange(70, 110, 10))
plt.savefig('validation_accuracy_vs_samples.png', dpi=300)
plt.show()

