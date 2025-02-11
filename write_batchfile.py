
import os
import glob
from pathlib import Path, PureWindowsPath
cwd = os.getcwd()
material_name = 'lxs'
phase_name = '6040'


inp_files = sorted(glob.glob(os.path.join(cwd, material_name, 'ml_data_generation', phase_name, 'RS_SGT_INP', '*.inp')))
inp_winpath = [PureWindowsPath(Path(x).absolute()) for x in inp_files]
with open(os.path.join(cwd, material_name, 'ml_data_generation', phase_name, 'RS_SGT_INP', 'topas_batch.bat'), 'w') as f:
    f.write('cd C:\TOPAS6 \n')
    for i in range(len(inp_files)):
        f.write(f'tc {inp_winpath[i]}\n')