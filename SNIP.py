import os 
import torch
import glob

cwd = os.getcwd()
material_name = 'bso'
phase_name = 'beta_6'
data_folder = os.path.join(cwd, material_name, 'ml_data', phase_name)
data_files = sorted(glob.glob(os.path.join(data_folder, 'topas0152_bso_beta_6_rs.pt')))

start_index = 5389
end_index = -1
x = torch.load(data_files[0])
X_snip = x[:, start_index:end_index]

torch.save(X_snip, os.path.join(data_folder, 'topas0152_bso_beta_6_rs_snip.pt'))