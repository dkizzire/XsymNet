import os
import torch

cwd = os.getcwd()
# file_path = os.path.join(cwd, 'bso', 'ml_data', 'alpha_5', 'topas0088_bso_alpha_5_rs.pt')
file_path = os.path.join(cwd, 'bso', 'ml_data', 'beta_3', 'cif0088_randomstates_bragg.pt')

data = torch.load(file_path, map_location='cpu')

print(data)
print(data.shape)