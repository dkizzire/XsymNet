import numpy as np
import utility_functions as util
import glob 
import os


phase_name = 'pzt_90'
multiphase = True
cwd = os.getcwd()
str_files = sorted(glob.glob(os.path.join(cwd, 'TOPAS', f'topas_strs_{phase_name}', '*.str')))


ml_dirs = sorted(glob.glob(os.path.join(cwd, 'TOPAS', phase_name, 'ml_data_generation', 'randomstates_strain', '*')))
dirnames = [os.path.basename(x).split('_randomstates')[0] for x in ml_dirs]
dirnames = [x.strip('cif') for x in dirnames]
print(dirnames)

dof_dict = {}

for i in range(len(str_files)):
    dof = util.count_refinable_parameters_in_topas_strs(str_files[i])
    dof_dict[dirnames[i]] = dof

multi_list = []
if multiphase == True:
    for i in range(len(dirnames)):
        if 'mixed' in dirnames[i]:
            multi_list.append(dirnames[i])
    
    for i in range(len(multi_list)):
        subs = multi_list[i].split('_')
        subs.pop(0) # removes word 'mixed' from list
        print(subs)
        count = 0
        for j in subs:
            count += dof_dict[j]
        
        dof_dict[multi_list[i]] = count
    
util.save_dict_to_json(dof_dict, f'./{phase_name}_dof_counts.json')