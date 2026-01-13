import os
import numpy as np
import torch
import matplotlib.pyplot as plt

cwd = os.getcwd()
material_name = 'bso'
phase_name = 'beta_5_diff'

# data_file = os.path.join(cwd, material_name, 'ml_data', phase_name, 'topas0088_bso_alpha_5_rs.pt')
data_file = os.path.join(cwd, material_name, 'ml_data', phase_name, 'cif0152_randomstates_bragg.pt')
# expt_file = os.path.join(cwd, material_name, 'hrp27188_bs_alpha_neutron_60-120_trim.xy')
# expt_file = os.path.join(cwd, material_name, 'bso_alpha_P1_trunc_je_xy.xy')
expt_file = os.path.join(cwd, material_name, 'beta_exp_xy.xy')

data = torch.load(data_file)
expt_data = np.loadtxt(expt_file)
print(f'Loaded data shape: {data.shape}')
print(f'Experimental data shape: {expt_data.shape}')

avg_data = torch.mean(data, dim=0)
print(f'Average data shape: {avg_data.shape}')
x = np.arange(len(avg_data))
fig, ax = plt.subplots(figsize=(16, 8))

ax.plot(x, avg_data.numpy(), label=f'Average Data - {phase_name} 0152', color='blue')
ax.plot(x, data[0].numpy(), label='Sample Data 1', color='orange', alpha=0.5)
ax.plot(x, data[1].numpy(), label='Sample Data 2', color='green', alpha=0.5)
# ax.plot(expt_data[:, 0], expt_data[:, 1], label='Experimental Data', color='red', alpha=0.7)
ax.plot(x, expt_data[:, 1], label=f'Experimental Data - {os.path.basename(expt_file)}', color='red', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(cwd, material_name, f'{material_name}_{phase_name}_comparison_plot.png'),dpi=300, bbox_inches='tight')
plt.show()

