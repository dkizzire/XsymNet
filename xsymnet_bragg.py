import os
import time
import numpy as np
import pandas as pd
import json
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR, StepLR, MultiStepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms 
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import MaxAbsScaler

import utility_functions as util
import xsymnet_funcs as funcs
import xsymnet_classes as xsym

#--------------------------------------------------------------------------------------------
# Input phase
#--------------------------------------------------------------------------------------------
phase_name = 'nvpf'
train_model = True
classify_experiment = True
# ground_truth_cif = '0152'
# ground_truth_cif = '0009'
ground_truth_cif = '0070'
confidence_iterations = 100
number_of_classes = 134
t0 = time.time()
cwd = os.getcwd()
#--------------------------------------------------------------------------------------------
# Determine if running on GPU or CPU and set tensor map locations
#--------------------------------------------------------------------------------------------
gpu_index = 5
maplocation = util.determine_gpu_or_cpu(gpu_index)

# -------------------------------------------------------------------------------------------
#* Load training & testing data
# -------------------------------------------------------------------------------------------

# Load training and testing data as tensors 
# data_folder = os.path.join(cwd, 'ml_data', phase_name, 'bragg_randomstates_strain_tensors_unscaled')
data_folder = os.path.join(cwd, 'ml_data', phase_name, 'bragg_randomstates_tensors')
number_of_random_states = 500
number_for_testing = 200
data_start_index = 0
data_end_index = 8640 #661 #TOF 23750 for NVPF  ...... # 1181 TOF 36000  for PZT_90

X_train_unscaled, X_test_unscaled, y_train, y_test = funcs.configure_bragg_data_trunc(data_folder, number_of_random_states, number_for_testing, start_index=data_start_index, end_index=data_end_index)
expt_data_unscaled = torch.load(os.path.join(cwd, 'ml_data', phase_name, f'{phase_name}_bragg_expt.pt'))
expt_data_unscaled = expt_data_unscaled[0,data_start_index:data_end_index]
expt_data_unscaled = expt_data_unscaled.unsqueeze(0)

# with open('./subgroup_ML_labels_to_cifIDs.json', 'r') as f: 
#     label_mapper = json.load(f)

with open(f'./dof_counts_{phase_name}.json', 'r') as f:
    param_mapper = json.load(f)

class_names = list(param_mapper.keys())
label_mapper = {}
for i in range(len(class_names)):
    label_mapper[f"{i}"] = class_names[i]


# -------------------------------------------------------------------------------------------
#* Scale training & testing data, send to GPU
# -------------------------------------------------------------------------------------------

X_train, X_test, X_expt = funcs.MinMaxScale_2d(X_train_unscaled, X_test_unscaled, expt_data_unscaled, scale=100)

# X_train = X_train.to(maplocation)
# X_test = X_test.to(maplocation)
X_expt = X_expt.to(maplocation)

y_train = y_train.to(maplocation)
y_test = y_test.to(maplocation)

# # create weights tensor to fix class imbalance
# class_weights_spacegroup = compute_class_weight('balanced', classes=np.unique(y_train_mapped), y=y_train_mapped)
# class_weights_spacegroup = torch.tensor(class_weights_spacegroup).float()

# # create an inverse mapping dict to retrieve space group numbers from integer labels
# inv_map = {v: k for k, v in int_labels_in_range_map.items()}


# ------------------------------------------------------------------------------------------------------
#* Package testing and training data with our custom Dataset class : initialize DataLoader
# ------------------------------------------------------------------------------------------------------

train_data_subgroup = xsym.simulated_data(X_train, y_train)
test_data_subgroup = xsym.simulated_data(X_test, y_test)

# load training and testing data with torch.DataLoader
train_loader_subgroup = DataLoader(train_data_subgroup, batch_size=64, shuffle=True)
test_loader_subgroup = DataLoader(test_data_subgroup, batch_size=64, shuffle=True)


# ------------------------------------------------------------------------------------------------------
#* Initialize CNN model 
# ------------------------------------------------------------------------------------------------------   


train_loader  = train_loader_subgroup
test_loader   = test_loader_subgroup

load_pretrained_model = False

if load_pretrained_model == True:
    model = xsym.CNN_bragg(num_classes=number_of_classes)
    criterion = nn.CrossEntropyLoss()
    checkpoint = torch.load('/netdisk/dx6/saved_models/beta_e100_bragg_trunc_1_pawley.pt', map_location=maplocation)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(maplocation)
    scheduler = MultiStepLR(optimizer, milestones=[75, 125], gamma=0.1)
    # epoch = checkpoint['epoch']
    # loss = checkpoint['loss']

if load_pretrained_model == False:
    model = xsym.CNN_bragg(num_classes=number_of_classes)
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = MultiStepLR(optimizer, milestones=[75, 125], gamma=0.1)


model.to(maplocation)
criterion.to(maplocation)

# ------------------------------------------------------------------------------------------------------
# * Train & Test model
# ------------------------------------------------------------------------------------------------------   

if train_model == True:

    number_of_top_results = min(20, number_of_classes)
    epochs = 150

    for epoch in range(epochs):
        count = 0
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            count = count+1
            optimizer.zero_grad()
            inputs = inputs.to(maplocation)
            outputs = model(inputs.float())
            #top = torch.topk(outputs, number_of_top_results, dim=1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()

        print("Epoch {} loss: {}".format(epoch + 1, running_loss / len(train_loader)))

    '''
    Evaluate the model: loop over the test data and compute the model's predictions
    then compare the predictions to the ground truth labels and compute the overall accuracy
    '''

    number_of_top_results = min(10, number_of_classes)
    correct = [0]*number_of_top_results
    incorrect = []
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            model.eval
            inputs = inputs.to(maplocation)
            predicted = model(inputs.float())
            total += labels.size(0)
            top = torch.topk(predicted, number_of_top_results, dim=1) # top is tuple with (values, indicies) where indicies can be used to check with labels for accuracy
            
            for i in range(len(labels)):
                temp_incorrect = []
                for j in range(len(correct)):
                    k = j+1
                    if labels[i] in top[1][i][:k]:
                        correct[j] = correct[j] + 1
                        temp_incorrect.append(float('NaN'))
                    else:
                        temp_incorrect.append(labels[i].tolist())

                incorrect.append(temp_incorrect)

    print()
    for i in range(len(correct)):
        print(f'Top {i+1} Accuracy: {100 * correct[i]/total}')
    print()
    t1 = time.time()
    print(f'Runtime: {(t1-t0)/60} minutes')

    #! ------------ SAVE THE MODEL ------------
    path = f'/netdisk/dx6/saved_models/{phase_name}_e{epochs}_bragg.pt'
    torch.save({
        'epoch':epoch, 
        'model_state_dict':model.state_dict(), 
        'optimizer_state_dict':optimizer.state_dict(), 
        'loss':loss,}, path)

# ------------------------------------------------------------------------------------------------------
# * Run XsymNet against experimental data
# ------------------------------------------------------------------------------------------------------   

model_filename = f'{phase_name}_e150_bragg.pt'

if classify_experiment == True:

    
    if train_model == False:
        pretrained_model = os.path.join('/netdisk', 'dx6', 'saved_models', model_filename)
        model = xsym.CNN_bragg(num_classes=number_of_classes)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        checkpoint = torch.load(pretrained_model)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        model.to(maplocation)
        criterion.to(maplocation)

    number_of_top_results = min(10, number_of_classes)
    confidence_results = {}
    weighted_confidence_results = {}
    ranked_confidence_results = {}
    param_weight = [10,9,8,7,6,5,4,3,2,1]
    
    with torch.no_grad():
        for i in range(confidence_iterations):
            model.eval
            predicted = model(X_expt.float())
            top = torch.topk(predicted, number_of_top_results, dim=1) # top is tuple with (values, indicies) where indicies can be used to check with labels for accuracy

            top_indicies = top[1].tolist()[0] # put [0] at end since without it returns 1d nested list
            top_logits = top[0].tolist()[0]
            predicted_subgroups = [label_mapper.get(str(x)) for x in top_indicies]
            for j in range(len(predicted_subgroups)):
                if predicted_subgroups[j] not in confidence_results.keys():
                    confidence_results[predicted_subgroups[j]] = 1
                    weighted_confidence_results[predicted_subgroups[j]] = 0
                    ranked_confidence_results[predicted_subgroups[j]] = param_weight[j]
                else: 
                    confidence_results[predicted_subgroups[j]] = funcs.increment_dict_value(confidence_results, predicted_subgroups[j])
                    ranked_confidence_results[predicted_subgroups[j]] += param_weight[j]

        for key in weighted_confidence_results.keys():
            weighted_confidence_results[key] = param_mapper.get(key)
        
        weighted_confidence_results = funcs.sort_dict_by_values_ascend(weighted_confidence_results)

    # save confidence results to json
    model_name_noext = os.path.splitext(model_filename)[0]
    confidence_results = funcs.sort_dict_by_values_desc(confidence_results)
    funcs.save_dict_to_json(confidence_results, f'./z_{model_name_noext}.json')
    print(f'subgroup: top10 rating, DOF, place socre, ')
    for key, value in confidence_results.items():
        print(f'{key}: {value}, {weighted_confidence_results[key]}, {ranked_confidence_results[key]/confidence_iterations}')

# ------------------------------------------------------------------------------------------------------
# * Print Output and write output files
# ------------------------------------------------------------------------------------------------------   

    # get subgroup info for predicted subgroups and write to file

        
    # print()
    # cif_path = os.path.join(cwd, f'subgroup_cifs_{phase_name}' )
    # model_name_noext = os.path.splitext(model_filename)[0]
    # keys = list(confidence_results.keys())
    # with open(f'./z_{model_name_noext}.txt', 'a') as f:
    #     f.write('------------------\n')
    #     f.write(f'Ground Truth Info: \n')
    #     subgroup_info = funcs.get_subgroup_info(ground_truth_cif, cif_path)
    #     f.write(f'{subgroup_info[0]}')
    #     f.write(f'{subgroup_info[1]},  {subgroup_info[2]},  {subgroup_info[3]},  {subgroup_info[4]},  {subgroup_info[5]},  {subgroup_info[6]}\n')
    #     f.write('------------------\n')
    #     f.write(f'Prediction Results: \n')
    #     f.write(f'Evaluated Model: {model_filename}\n')
    #     f.write(f'{keys}\n\n')
    #     for i in range(len(keys)):
    #         subgroup_info = funcs.get_subgroup_info(keys[i], cif_path)
    #         f.write(f'{subgroup_info[0]}')
    #         f.write(f'{subgroup_info[1]},  {subgroup_info[2]},  {subgroup_info[3]},  {subgroup_info[4]},  {subgroup_info[5]},  {subgroup_info[6]}\n')
    #         f.write(f'100 iteration score: {confidence_results[keys[i]]}\n\n')
    #     f.write(f'\n-----------------------------------------------------------------------------\n\n')
