import os
import glob
import json
import numpy as np
import torch
from sklearn.preprocessing import MaxAbsScaler


def save_kernel_weights(model, epoch, save_file="kernel_weights.txt"):
    """Appends convolutional layer weights to a text file, grouped by kernel with epochs appended."""
    if epoch % 50 != 0:
        return  # Only save every 50 epochs
    
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    
    conv1_weights = model.conv1.weight.detach().cpu().numpy()
    conv2_weights = model.conv2.weight.detach().cpu().numpy()
    
    existing_data = {}
    
    # Read existing data if the file already exists
    try:
        with open(save_file, "r") as f:
            lines = f.readlines()
            kernel_name = None
            for line in lines:
                if line.startswith("conv"):
                    kernel_name = line.strip()
                    existing_data[kernel_name] = []
                elif kernel_name and line.strip():
                    existing_data[kernel_name].append(line.strip())
    except FileNotFoundError:
        pass  # No previous data exists, so we start fresh
    
    with open(save_file, "w") as f:
        for layer_name, weights in zip(["conv1", "conv2"], [conv1_weights, conv2_weights]):
            for i, kernel in enumerate(weights):
                kernel_key = f"{layer_name} kernel {i+1}"
                if kernel_key not in existing_data:
                    f.write(f"{kernel_key}\n")
                
                weights_str = ", ".join(map(str, kernel.flatten()))
                existing_data.setdefault(kernel_key, []).append(f"epoch {epoch} {weights_str}")
        
        # Write back grouped data
        for kernel_key, records in existing_data.items():
            f.write(f"{kernel_key}\n")
            f.write("\n".join(records) + "\n\n")
    
    print(f"Kernel weights appended for epoch {epoch} in {save_file}")




def configure_bragg_data_trunc(path2braggdata, n_randomstates, n_testing, scaling_dict={}, start_index:int = 0, end_index:int = -1):
    
    filepaths = sorted(glob.glob(os.path.join(path2braggdata, '*.pt')))

    xtrain_list = []
    xtest_list = []
    ytrain_list = []
    ytest_list = []

    split_index = n_randomstates - n_testing

    keys = list(scaling_dict.keys())

    # expt_min = torch.min(expt)
    # expt_max = torch.max(expt)
    # print(f'\n\nConfigure Bragg PreScaling:')
    # print(f'Expt MAX:{expt_max:.2f}\nExpt MIN:{expt_min:.2f}\n')


    if len(keys) > 0:
        print(('Applying scaling dict to data:\n'))
        
    for i in range(len(filepaths)):
        temp = torch.load(filepaths[i])
        #! Turn ManualScale on/off here
        # temp = ManualScale(temp, expt_max, i)
        # temp = ManualScaleConstant(temp, scalefactor=0.001)
        #^ here we apply the optimal scale factor for each subgroup
        if len(keys) > 0:
            temp_max = torch.max(temp)
            temp = temp*scaling_dict[keys[i]]
            scaled_max = torch.max(temp)
            print(f'SGT {os.path.splitext(os.path.basename(filepaths[i]))[0]} max: {temp_max:.2f}   scaled: {scaled_max:.2f}')

        count = 0
        for j in range(temp.shape[0]):
            if count < split_index:
                # xtrain_list.append(temp[j,start_index:end_index])
                xtrain_list.append(temp[j])
                ytrain_list.append(i)
            else:
                # xtest_list.append(temp[j,start_index:end_index])
                xtest_list.append(temp[j])
                ytest_list.append(i)
            count+=1

    xtrain = torch.vstack(xtrain_list)
    xtest = torch.vstack(xtest_list)
    ytrain = torch.tensor(ytrain_list)
    ytest = torch.tensor(ytest_list)

    print(f'Post Configure Bragg:')
    # print(f'Expt MAX:{expt_max:.2f}\nExpt MIN:{expt_min:.2f}')
    print(f'Train_MAX: {torch.max(xtrain):.2f}\nTrain_MIN: {torch.min(xtrain):.2f}')
    print(f'Test_MAX: {torch.max(xtest):.2f}\nTest_MIN: {torch.min(xtest):.2f}\n')


    # train_max = torch.max(xtrain)
    # test_max = torch.max(xtest)

    # whole_max = max(train_max, test_max)
    # # Compute the scaling factor to match the order of magnitude
    # order_of_mag_simulated = 10 ** (len(str(int(whole_max))) - 1)
    # order_of_mag_experimental = 10 ** (len(str(int(expt_max))) - 1)
    # scaling_factor = order_of_mag_experimental / order_of_mag_simulated 

    # scaled_xtrain = xtrain*scaling_factor
    # scaled_xtest = xtest*scaling_factor

    # print(f'scaled_xtrain max: {torch.max(scaled_xtrain)}')
    # print(f'scaled_xtest max: {torch.max(scaled_xtest)}')

    return xtrain, xtest, ytrain, ytest



def MinMaxScale_2d(train, test, expt=None, scale=1):

    expt_min = 0
    expt_max = 0
   
    min_val = torch.min(train)
    max_val = torch.max(train)
    
    if expt != None:
        # expt_scale = 1/0.000295037729
        # expt = expt*expt_scale
        expt_min = torch.min(expt)
        expt_max = torch.max(expt)
        print(f'Expt MAX:{expt_max:.2f}\nExpt MIN:{expt_min:.2f}\n')

    print(f'Normalization with MinMaxScale2d:')

    train_scaled = ((train-min_val) / (max_val - min_val))*scale
    test_scaled = ((test - min_val) / (max_val - min_val))*scale
    
    if expt != None:
        expt_scaled = ((expt - min_val) / (max_val - min_val))*scale 
        #^ make sure that after scaling, the values don't go negative
        # print('MIN clamping EXPT after scaling to prevent negative values\n')
        # expt_scaled = torch.clamp(expt_scaled, min=0)
        #^ debug clamp and check output
        # expt_scaled_write = expt_scaled.t().numpy()
        # np.savetxt('./TESTINGEXPTTENSOR.txt', expt_scaled_write, newline='\n')
    else:
        expt_scaled = None
        
    if expt != None:
        print(f'Expt MAX:{torch.max(expt_scaled):.2f}\nExpt MIN:{torch.min(expt_scaled):.2f}\n')
    print(f'Train_MAX:{torch.max(train_scaled):.2f}\nTrain_MIN:{torch.min(train_scaled):.2f}\n')
    print(f'Test_MAX:{torch.max(test_scaled):.2f}\nTest_MIN:{torch.min(test_scaled):.2f}\n')
    return train_scaled, test_scaled, expt_scaled



def MaxAbsScale(X_train_load, X_test_load):
    '''
    Purpose: 
        Uses sklearn's MaxAbsScaler to build scaling transformer from X_train 
        Then scale X_train and X_test with said transformer
    
    Inputs:
        X_train, X_test : tensors

    Returns: 
        X_train, X_test : tensors - scaled to bounds 0 to 1
    '''

    training_local = X_train_load.numpy(force=True)
    testing_local = X_test_load.numpy(force=True)

    transformer = MaxAbsScaler().fit(training_local)
    training_scaled = transformer.transform(training_local)
    testing_scaled = transformer.transform(testing_local)

    X_train = torch.tensor(training_scaled)
    X_test = torch.tensor(testing_scaled)
    
    return X_train, X_test


def MinMaxScale(tensor):
    scaled_tensor = torch.zeros_like(tensor)
    for i in range(len(tensor)):
        min_val = torch.min(tensor[i])
        max_val = torch.max(tensor[i])
        temp_tensor = (tensor[i]-min_val) / (max_val - min_val)
        scaled_tensor[i] = temp_tensor
    return scaled_tensor





def get_subgroup_info(subgroup_num, cif_folder_path):
    '''
    Parameters
    ----------
    subgroup_num: str  EX: 0054
        
    Returns:
        List containing the spacegroup and the lattice parameters
        [subgroup_line, a, b, c, alpha, beta, gamma, spacegroup, origin_choice]

    '''
    cif_path = os.path.join(cif_folder_path, f'cif{subgroup_num}.cif')
    with open(cif_path, 'r') as f:
        for line in f:
            if '# Subgroup ' in line:
                subgroup_line = line
            if '_cell_length_a' in line:
                a = float(line.split()[-1])
            if '_cell_length_b' in line:
                b = float(line.split()[-1])
            if '_cell_length_c' in line:
                c = float(line.split()[-1])
            if '_cell_angle_alpha' in line:
                alpha = float(line.split()[-1])
            if '_cell_angle_beta' in line:
                beta = float(line.split()[-1])
            if '_cell_angle_gamma' in line:
                gamma = float(line.split()[-1])
            if '_symmetry_Int_Tables_number' in line: 
                spacegroup_int = int(line.split()[-1])
            if '_symmetry_space_group_name_H-M' in line:
                sline = line.split('"') 
                hm_name = sline[1]

        return_list = [subgroup_line, a,b,c,alpha,beta,gamma, hm_name, spacegroup_int]
    return return_list


def configure_bragg_data(path2braggdata, n_randomstates, n_testing):
    
    filepaths = sorted(glob.glob(os.path.join(path2braggdata, '*.pt')))

    xtrain_list = []
    xtest_list = []
    ytrain_list = []
    ytest_list = []

    split_index = n_randomstates - n_testing
    for i in range(len(filepaths)):
        temp = torch.load(filepaths[i])
        count = 0
        for j in range(temp.shape[0]):
            if count < split_index:
                xtrain_list.append(temp[j])
                ytrain_list.append(i)
            else:
                xtest_list.append(temp[j])
                ytest_list.append(i)
            count+=1

    xtrain = torch.vstack(xtrain_list)
    xtest = torch.vstack(xtest_list)
    ytrain = torch.tensor(ytrain_list)
    ytest = torch.tensor(ytest_list)

    return xtrain, xtest, ytrain, ytest


# def ManualScale(data, expt, num_sgs, n_randomstates):

#     expt_max = torch.max(expt)
#     print(f'\n\nexpt_max: {expt_max}\n\n')
#     scaled_data = data
#     for i in range(num_sgs):

#         a = i*n_randomstates
#         b = a+n_randomstates
#         temp_max = torch.max(data[a:b])
#         scaling = expt_max/temp_max
#         scaled_data[a:b] = data[a:b]*scaling
#         temp2_max = torch.max(data[a:b])
#         print(f'subgroup {i} max: {temp_max}, {temp2_max}, {scaling}')
#     return scaled_data


def ManualScale(data, expt_max, i):

    temp_max = torch.max(data)
    scaling = expt_max/temp_max
    scaled_data = data*scaling
    
    print(f'subgroup {i} max: {temp_max}, {torch.max(scaled_data)}, {scaling}')

    return scaled_data

def ManualScaleConstant(data, scalefactor):

    scaled_data = data*scalefactor
    
    return scaled_data





def configure_hkl_data_trunc(path2braggdata, trunc_index, n_randomstates, n_testing, maplocation):
    
    filepaths = sorted(glob.glob(os.path.join(path2braggdata, '*.pt')))

    xtrain_list = []
    xtest_list = []
    ytrain_list = []
    ytest_list = []

    split_index = n_randomstates - n_testing
    for i in range(len(filepaths)):
        temp = torch.load(filepaths[i], map_location=maplocation)
        count = 0
        for j in range(temp.shape[0]):
            if count < split_index:
                xtrain_list.append(temp[j,trunc_index:])
                ytrain_list.append(i)
            else:
                xtest_list.append(temp[j,trunc_index:])
                ytest_list.append(i)
            count+=1

    xtrain = torch.vstack(xtrain_list)
    xtest = torch.vstack(xtest_list)
    ytrain = torch.tensor(ytrain_list)
    ytest = torch.tensor(ytest_list)

    return xtrain, xtest, ytrain, ytest


def increment_dict_value(dictionary, key):
    value = dictionary.get(key, 0)
    value += 1
    return value


def sort_dict_by_values_desc(dictionary):
    sorted_items = sorted(dictionary.items(), key=lambda x: x[1], reverse=True)
    sorted_dict = dict(sorted_items)
    return sorted_dict


def sort_dict_by_values_ascend(dictionary):
    sorted_items = sorted(dictionary.items(), key=lambda x: x[1], reverse=False)
    sorted_dict = dict(sorted_items)
    return sorted_dict


def save_dict_to_json(dictionary, filename):
    with open(filename, 'w') as f:
        json.dump(dictionary, f, indent=4)






