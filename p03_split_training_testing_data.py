import os
import glob
import json
import pandas as pd
import torch
import utility_functions as util


phase_name = 'alpha'
number_of_random_states = 200
number_of_random_states_for_testing = 20

training_data_filename = 'unscaled_X_train_hkl_binned_C0p0008.pt'
testing_data_filename = 'unscaled_X_test_hkl_binned_C0p0008.pt'
training_labels_filename = 'y_train_subgroup_labels_C0p0008.pt'
testing_labels_filename = 'y_test_subgroup_labels_C0p0008.pt'

# get binned random state hkl tensor filepaths created by p02_bin_hkl_lists.py
cwd = os.getcwd()
binned_tensors_dirs = [cwd, 'TOPAS', phase_name, 'ml_data_generation', 'hkl_tensors_binned', '*.pt']
all_data_filenames = sorted(glob.glob(os.path.join(*binned_tensors_dirs)))

#---------------------------------------------------------------------------------------------
# Determine if running on GPU or CPU and set tensor map locations
#---------------------------------------------------------------------------------------------
gpu_index = 2
maplocation = util.determine_gpu_or_cpu(gpu_index)
#---------------------------------------------------------------------------------------------
# Split data and labels into training and testing groups
# Create labels (subgroup and spacegroup) for data
#---------------------------------------------------------------------------------------------

# split data into training and testing
X_train_list = []
X_test_list = []
y_train_list = []
y_test_list = []
split_index = number_of_random_states - number_of_random_states_for_testing
for i in range(len(all_data_filenames)): # for the number of subgroup cifs in your tree
    temp_tensor = torch.load(all_data_filenames[i], map_location=maplocation)
    count = 0
    for j in range(temp_tensor.shape[0]):
        if count < split_index:
            X_train_list.append(temp_tensor[j])
            y_train_list.append(i)
        else: 
            X_test_list.append(temp_tensor[j])
            y_test_list.append(i)
        count += 1

# Stack lists tensors of X_train and X_test
X_train_tensor = torch.vstack(X_train_list)
X_test_tensor = torch.vstack(X_test_list)


# save unscaled training and testing data
training_data_path = os.path.join(cwd, 'ml_training_data', phase_name)
if not os.path.exists(training_data_path): os.makedirs(training_data_path)

torch.save(X_train_tensor, os.path.join(training_data_path, training_data_filename))
torch.save(X_test_tensor, os.path.join(training_data_path, testing_data_filename))

# save labels
torch.save(torch.tensor(y_train_list), os.path.join(training_data_path, training_labels_filename))
torch.save(torch.tensor(y_test_list), os.path.join(training_data_path, testing_labels_filename))
