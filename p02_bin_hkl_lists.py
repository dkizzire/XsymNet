import os
import time
import numpy as np
import pandas as pd
import glob
from pathlib import Path
import torch
from sklearn.preprocessing import MaxAbsScaler
import json

import utility_functions as util

cwd = os.getcwd()
# torch.set_printoptions(precision=6)
phase_name = 'alpha'
experimental_hkl_tensor_filename = 'd_Ip_expt_VC_P1_latparams.pt'
load_tensor_directory = 'hkl_tensors_notbinned_unscaled'
save_tensor_dirctory = 'hkl_tensors_binned_scaled'
data_suffix = '*hkl.pt'
# data_suffix = '*hkl.pt'

# set c = delta_d / d = TOF instrument resolution factor; instrument and frame dependent  
c = 0.0008

# populate with your info
number_of_subgroup_members = 546
number_of_random_states = 200

#---------------------------------------------------------------------------------------------
# Determine if running on GPU or CPU and set tensor map locations
#---------------------------------------------------------------------------------------------
t0 = time.time()

gpu_index = 3
maplocation = util.determine_gpu_or_cpu(gpu_index)

#---------------------------------------------------------------------------------------------
# Collect Tensors of hkl's : define smallest and largest d values to set ends of binning
#---------------------------------------------------------------------------------------------

# get cif IDs
cif_dirs = [cwd, 'subgroup_cifs', '*.cif']
cif_files = sorted(glob.glob(os.path.join(*cif_dirs)))
cif_filenames = [os.path.basename(x) for x in cif_files]
cif_noext = [os.path.splitext(x)[0] for x in cif_filenames]
subgroup_nums = [x[3:] for x in cif_noext]

# collect hkl tensors for determining the smallest and largest d values in preparation to create binning tensors
hkl_dirs = [cwd, 'TOPAS', phase_name, 'ml_data_generation', load_tensor_directory, data_suffix]
hkl_di_tensors = sorted(glob.glob(os.path.join(*hkl_dirs)))

hkl_expt_dirs =  [cwd, 'TOPAS', phase_name, 'experimental_refinement', experimental_hkl_tensor_filename]
hkl_di_expt_tensor = sorted(glob.glob(os.path.join(*hkl_expt_dirs)))

# get the smallest and largest d_spacing values
lowest_d_value, largest_d_value = util.get_d_bounds_for_binning(hkl_di_tensors, hkl_di_expt_tensor)
print(f'Lowest d value = {lowest_d_value}')
print(f'Largest d value = {largest_d_value}')

# create bin_edges.txt / bin_centers.txt 
bin_edges = util.create_bin_edges(phase_name, c, lowest_d_value, largest_d_value)
bin_edges = torch.tensor(bin_edges, dtype=torch.double, device=maplocation)

# reshape hkl_di_tensors (a list of filepaths) to number_of_subgroup_members x number_of_random_states
np_hkl_di_tensors = np.array(hkl_di_tensors)
hkl_di_tensors_2d = np.reshape(np_hkl_di_tensors, (number_of_subgroup_members, number_of_random_states)).tolist()

#---------------------------------------------------------------------------------------------
# Group hkl reflection intensities into bins as defined above 
# save each cifID's random states into tensor (working memory conservation + speedup)
#---------------------------------------------------------------------------------------------

intermediates_dirs = [cwd, 'TOPAS', phase_name, 'ml_data_generation', save_tensor_dirctory]
path_to_write_intermediates = os.path.join(*intermediates_dirs)
if not os.path.exists(path_to_write_intermediates): os.makedirs(path_to_write_intermediates)

running_time = 0
for i in range(len(hkl_di_tensors_2d)):
    random_states_tensor_list = []
    t2=time.time()
    for j in range(len(hkl_di_tensors_2d[i])):
        
        di_tensor = torch.load(hkl_di_tensors_2d[i][j], map_location=maplocation)
        # initialize sums_binned_tensor with values : will be overwritten and concatenated : bin_centers has correct .size
        sums_binned_tensor = torch.zeros_like(bin_edges, device=maplocation)
        bucket_tensor = torch.bucketize(di_tensor[:,0], bin_edges, right=True)

        for index,value in zip(bucket_tensor, di_tensor[:,1]):
            sums_binned_tensor[index] += value

        random_states_tensor_list.append(sums_binned_tensor)

    # we save each folder of random_states to a tensor since appending them to a giant database and keeping them in memory makes the program take forever
    random_state_folder_tensor = torch.vstack(random_states_tensor_list)
    torch.save(random_state_folder_tensor, os.path.join(path_to_write_intermediates, f'{cif_noext[i]}.pt'))
    t3 = time.time()
    iteration_time = t3-t2
    running_time = running_time + iteration_time
    print(f'Folder {i}  took {t3-t2} seconds to run ')
    # print()

print(f'Running time: {running_time/60} minutes')
print()




