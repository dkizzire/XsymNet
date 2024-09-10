import os
import glob
import numpy as np
import write_TOPAS_funcs as funcs


cwd = os.getcwd()
cif_files = sorted(glob.glob(os.path.join(cwd, 'subgroup_cifs_nvpf', '*')))
str_files = sorted(glob.glob(os.path.join(cwd, 'TOPAS', 'topas_strs_nvpf', '*')))

for i in range(len(cif_files)):
    str_latparams, str_sgo = funcs.get_str_info(str_files[i], 'strain')
    # print(str_latparams)
    # print(str_sgo)
    # print()

    cif_latparams, cif_sgo = funcs.get_cif_info(cif_files[i])
    # print(cif_latparams)
    # print(cif_sgo)
    # print()

    if str_latparams != cif_latparams:
        print(f'latparam mismatch')
        print(f'{os.path.basename(cif_files[i])}')
        print(f'{os.path.basename(str_files[i])}')
        print(f'{cif_latparams}')
        print(f'{str_latparams}')
        print(f'{cif_sgo}')
        print(f'{str_sgo}')

    # if str_sgo != cif_sgo:
    #     print(f'sgo mismatch')
    #     print(f'{os.path.basename(cif_files[i])}')
    #     print(f'{os.path.basename(str_files[i])}')
    #     print(f'{cif_latparams}')
    #     print(f'{str_latparams}')
    #     print(f'{cif_sgo}')
    #     print(f'{str_sgo}')