import os
import glob
import numpy as np
import write_TOPAS_funcs as funcs

cwd = os.getcwd()

topas_strs = sorted(glob.glob(os.path.join(cwd, 'TOPAS', 'topas_strs', '*.str')))
beq_dict = {
        'Bi1': [0.0, 1.5],
        'Sn1': [0.0, 1.5],
        'O1': [0.0, 2.5], 
        'O2': [0.0, 2.5]
    }


all_atom_list = []
for i in range(len(topas_strs)):
    mode_lines= funcs.generate_symmetry_mode_lines_from_topas_str(topas_strs[i], refine_beq=True, beq_dict=beq_dict, use_vc_ma_macro=False)
    for j in range(len(mode_lines)):
        print(mode_lines[j])
    print(f'iteration {i}')