import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
import torch

cwd = os.getcwd()

# import xy file to get tof/2theta x axis
xy_file = os.path.join(cwd, 'wo3', 'experimental_refinement', 'height', 'P1_wo3_1103K_xy.xy')
expt = pd.read_csv(xy_file, sep='\s+', names=['x','y'])


fig, ax = plt.subplots()

torch_data = ['0001', '0032', '0093']
torch_path = os.path.join(cwd, 'wo3', 'ml_data', 'three_temps')
data_rows = [0, 1, 2]

torch_files = [glob.glob(os.path.join(torch_path, f'*{torch_data[x]}*.pt')) for x in range(len(torch_data))]


data = {}
data_list = []
data_label = []
x_exp = torch.tensor(expt['x'])
y_exp = torch.tensor(expt['y'])

data_list.append(x_exp)
data_label.append(os.path.basename(xy_file))
for i in range(len(torch_files)):
    temp = torch.load(torch_files[i][0])
    for j in range(len(data_rows)):
        data_list.append(temp[j])
        data_label.append(f'{torch_data[i]}_{data_rows[j]}')

print('check')

for i in range(len(data_list)):
    line, = ax.plot(x_exp, data_list[i], label=data_label[i])
legend = ax.legend(loc='upper right')
plt.show()
