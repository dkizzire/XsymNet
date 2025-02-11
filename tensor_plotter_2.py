import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
import torch

# Define MinMax scaling function
def MinMaxScale_2d(tensor_2d, expt, scale=100):
    min_val = torch.min(tensor_2d)
    max_val = torch.max(tensor_2d)
    train_scaled = ((tensor_2d - min_val) / (max_val - min_val)) * scale
    print(f'data_min: {min_val}, data_max: {max_val}')
    expt_min = torch.min(expt)
    expt_max = torch.max(expt)
    expt_scaled = ((expt - min_val) / (max_val - min_val)) * scale

    return train_scaled, expt_scaled

# Set up the current working directory
cwd = os.getcwd() 

# Import xy file to get tof/2theta x-axis
xy_file = os.path.join(cwd, 'wo3', 'experimental_refinement', 'height', 'P1_wo3_1103K_xy.xy')
expt = pd.read_csv(xy_file, sep=r'\s+', names=['x', 'y'])  # Use regex for multiple spaces

# Define torch data and paths
torch_path = os.path.join(cwd, 'wo3', 'ml_data', 'three_temps_0bkg')
#!torch_data = ['0001', '0032', '0093']
data_rows = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# Gather PyTorch files
#!torch_files = [glob.glob(os.path.join(torch_path, f'*{torch_data[x]}*.pt')) for x in range(len(torch_data))]

abc = sorted(glob.glob(os.path.join(torch_path, '*.pt')))
torch_files = abc[25:30]
torch_data = [torch_files[x].strip('topas').strip('_wo3_three_temps_0bkg_rs.pt') for x in range(len(torch_files))]

# Prepare x-axis and experimental data
x_exp = torch.tensor(expt['x'])
y_exp = torch.tensor(expt['y'])

# Create subplots: one per torch_data file
fig, axes = plt.subplots(len(torch_data), 1, figsize=(8, 4 * len(torch_data)), sharex=True)

# Plot experimental data and rows for each torch_data file
for i, ax in enumerate(axes):
    # Load PyTorch data for the current file
    #! torch_file = torch_files[i][0]  # Assuming one file matches per torch_data
    torch_file = torch_files[i]  # Assuming one file matches per torch_data
    temp = torch.load(torch_file)  # Load tensor file
    
    # MinMax scale the current file's te5nsor data and experimental data
    scaled_tensor, scaled_expt = MinMaxScale_2d(temp, y_exp)
    # scaled_tensor = temp
    # scaled_expt = y_exp
    # Plot experimental data
    ax.plot(x_exp, scaled_expt, label='Experimental Data', color='black', linewidth=1.5)
    
    # Plot selected rows from the current file
    for j in data_rows:
        if j < temp.shape[0]:  # Check if row index is valid
            ax.plot(x_exp, scaled_tensor[j], label=f'Row {j}', linestyle='--', color=f'C{j}')
    
    ax.set_ylabel('Scaled Intensity')
    ax.legend(loc='upper right')
    ax.set_title(f'Torch Data: {os.path.basename(torch_data[i])}')  # Add title for each subplot
    ax.grid(True)

axes[-1].set_xlabel('X (tof/2theta)')  # Set x-label only on the last subplot

# Adjust layout
plt.tight_layout()
plt.show()
