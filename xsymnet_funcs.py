import os
import glob
import json
import numpy as np
import torch
from sklearn.preprocessing import MaxAbsScaler




def configure_bragg_data_trunc(path2braggdata, expt, n_randomstates, n_testing, scaling_dict={}, start_index:int = 0, end_index:int = -1):
    
    filepaths = sorted(glob.glob(os.path.join(path2braggdata, '*.pt')))

    xtrain_list = []
    xtest_list = []
    ytrain_list = []
    ytest_list = []

    split_index = n_randomstates - n_testing

    keys = list(scaling_dict.keys())

    expt_min = torch.min(expt)
    expt_max = torch.max(expt)
    print(f'\n\nIn Configure Bragg PreScaling:\nExpt Pawley MAX:{expt_max:.2f}\nExpt Pawley MIN:{expt_min:.2f}\n')

    for i in range(len(filepaths)):
        temp = torch.load(filepaths[i])
        #! Turn ManualScale on/off here
        #temp = ManualScale(temp, expt_max, i)
        #temp = ManualScaleConstant(temp, scalefactor=1)
        #^ here we apply the optimal scale factor for each subgroup
        if len(keys) > 0:
            temp_max = torch.max(temp)
            temp = temp*scaling_dict[keys[i]]
            scaled_max = torch.max(temp)
            print(f'SGT {os.path.splitext(os.path.basename(filepaths[i]))[0]} max: {temp_max:.2f}   scaled: {scaled_max:.2f}')

        count = 0
        for j in range(temp.shape[0]):
            if count < split_index:
                xtrain_list.append(temp[j,start_index:end_index])
                ytrain_list.append(i)
            else:
                xtest_list.append(temp[j,start_index:end_index])
                ytest_list.append(i)
            count+=1

    xtrain = torch.vstack(xtrain_list)
    xtest = torch.vstack(xtest_list)
    ytrain = torch.tensor(ytrain_list)
    ytest = torch.tensor(ytest_list)
    print('End of Congigure Bragg Data Truncated\n\n')
    return xtrain, xtest, ytrain, ytest


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


def MinMaxScale_2d(train, test, expt=None, scale=1):

    expt_min = 0
    expt_max = 0
    if expt != None:
        expt_min = torch.min(expt)
        expt_max = torch.max(expt)
    
    min_val = torch.min(train)
    max_val = torch.max(train)
    print(f'MinMaxScale2d prescale:\nExpt Pawley MAX:{expt_max:.2f}\nExpt Pawley:{expt_min:.2f}\n')
    print(f'Training MAX:{max_val:.2f}\nTraining MIN:{min_val:.2f}\n')
    train_scaled = ((train-min_val) / (max_val - min_val))*scale
    test_scaled = ((test - min_val) / (max_val - min_val))*scale
    if expt != None:
        expt_scaled = ((expt - min_val) / (max_val - min_val))*scale
    
    print(f'MinMaxScale2d postscale:\nExpt Pawley MAX:{torch.max(expt_scaled):.2f}\nExpt Pawley MIN:{torch.min(expt_scaled):.2f}\n')
    print(f'Training MAX:{torch.max(train_scaled):.2f}\nTraining MIN:{torch.min(train_scaled):.2f}\n')
    return train_scaled, test_scaled, expt_scaled


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






