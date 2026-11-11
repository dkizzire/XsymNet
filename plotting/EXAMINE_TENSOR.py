import os
import torch
import numpy as np

cwd = os.getcwd()

std = True
bkg = False

if std:
    file_path = os.path.join(cwd, 'pzt_90', 'ml_data', '300K', 'cif0001_randomstates_bragg.pt')
    data = torch.load(file_path, map_location='cpu')
    print(data)
    print(data.shape)
    print(torch.min(data))
    print(torch.max(data))

if bkg:
    # file_path = os.path.join(cwd, 'bso', 'ml_data', 'alpha_5', 'topas0088_bso_alpha_5_rs.pt')
    file_path = os.path.join(cwd, 'wo3', 'ml_data', '1085K', 'topas0032_wo3_1085K_rs.pt')
    bkg_path = os.path.join(cwd, 'wo3', 'experimental_refinement', 'pawley', 'P1_wo3_1085K_bkg.xy')

    data = torch.load(file_path, map_location='cpu')
    bkg = np.loadtxt(bkg_path)
    bkg_x = torch.tensor(bkg[:,0], dtype=torch.float32)
    bkg_y = torch.tensor(bkg[:,1], dtype=torch.float32)

    bkgsub = data[0] - bkg_y
    print(data)
    print(bkgsub)
    print(data.shape)


