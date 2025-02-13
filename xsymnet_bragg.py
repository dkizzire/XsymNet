import os
import glob
import time
import numpy as np
import pandas as pd
import json
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR, StepLR, MultiStepLR
from torch.utils.data import DataLoader, Dataset
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import MaxAbsScaler
from collections import OrderedDict


import utility_functions as util
import xsymnet_funcs as funcs
import xsymnet_classes as xsym

import optuna

t0 = time.time()
cwd = os.getcwd()

#============================================================================================
# Determine if running on GPU or CPU and set tensor map locations
#============================================================================================

# gpu_index = 7
# device = util.determine_gpu_or_cpu(gpu_index)

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"#, 1, 2, 3, 4, 5, 6, 7"
# selected_gpus = [0]#, 1, 2, 3, 4, 5, 6, 7]
#^ os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"#, 4, 5"#, 6, 7"
#^ selected_gpus = [0, 1, 2, 3]#, 4, 5]#, 6, 7]
#^ torch.cuda.set_device(selected_gpus[0])
#^ device = f'cuda:{selected_gpus[0]}'

# Set available GPUs
selected_gpus = [0, 1, 6, 7]
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, selected_gpus))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Now, selected_gpus are remapped as [0, 1, 2, 3]
logical_gpus = list(range(len(selected_gpus)))

#============================================================================================
# Input phase
#============================================================================================
material_name = 'wo3'
phase_name = '1085K'
epochs = 150
training_batchsize = 512
confidence_iterations = 100
scheduler_milestones = [40, 130]

train_model = True
path_to_save_model = f'/netdisk/dx6/saved_models/{material_name}_{phase_name}_e{epochs}_multiheadattention_02.pt'
# path_to_save_model = f'/netdisk/dx6/saved_models/{material_name}_{phase_name}_e{epochs}_1103K_multihead_02.pt'

classify_experiment = True
# experimental_p1_filename = 'd8_03901_t1103.xy'
experimental_p1_filename = 'P1_wo3_1085K_xy.xy'

warmstart_model = False
# warmstart_path = f'/netdisk/dx6/saved_models/wo3_p4nmm_parent_e25_manualscaling.pt'
warmstart_path = f'/netdisk/dx6/saved_models/wo3_1103K_e250_1103K_multihead_01.pt'
# warmstart_path = f'/netdisk/dx6/saved_models/wo3_1243K_scale_base_e150_prescale.pt'
# warmstart_path = f'/netdisk/dx6/saved_models/bso_beta_5_e150_bragg.pt'

load_pretrained_model = False
pretrain_path = f'/netdisk/dx6/saved_models/wo3_1085K_e150_multiheadattention_01.pt'


ground_truth_cif = '0032' # wo3_1085K   
# ground_truth_cif = '0093' # wo3_1043K
# ground_truth_cif = '0001' # wo3_1243K
# ground_truth_cif = '0009' # pzt_90
# ground_truth_cif = '0088' # bso_alpha
# ground_truth_cif = '0152' # bso_beta

# #^ scaling json
# scale_dict_path = os.path.join(cwd, material_name, 'ml_data', '1243K_scale_bkgsub', '1243K_scale_bkgsub_scale_dict.json')
# # scale_dict_path = os.path.join(cwd, material_name, 'ml_data', phase_name, f'{phase_name}_scale_dict.json')
# with open(scale_dict_path) as f:
#     scale_dict = json.load(f)

#^ convert diffraction xy to tensor (.pt)    
if experimental_p1_filename[-3:] != '.pt':
    converted_expt_data = util.convert_bragg_to_tensors(os.path.join(cwd, material_name, 'experimental_p1s', experimental_p1_filename))
    new_filename = f'{os.path.splitext(experimental_p1_filename)[0]}.pt'
    torch.save(converted_expt_data, os.path.join(cwd, material_name, 'experimental_p1s', new_filename))
    experimental_p1_filename = new_filename



#============================================================================================
#* Load training & testing data
#============================================================================================

# Load training and testing data as tensors 
# data_folder = os.path.join(cwd, 'ml_data', material_name, 'bragg_randomstates_strain_tensors_unscaled')
# data_folder = os.path.join(cwd, 'ml_data', material_name, phase_name, 'bragg_randomstates_tensors')
data_folder = os.path.join(cwd, material_name, 'ml_data', phase_name)
data_files = sorted(glob.glob(os.path.join(data_folder, '*.pt')))
number_of_classes = len(data_files)
first_file = torch.load(data_files[0])
number_of_random_states = first_file.size()[0]
number_for_testing = int(number_of_random_states*0.1)

data_start_index = 0 #70  #661 #TOF 23750 for NVPF  ...... # 1181 TOF 36000  for PZT_90  ....... TOF 60000 i=5390 for BSO
data_end_index = -1 #3841
# data_end_index = -1

expt_data_unscaled = torch.load(os.path.join(cwd, material_name, 'experimental_p1s', experimental_p1_filename))
expt_data_unscaled = expt_data_unscaled[0,data_start_index:data_end_index]
X_expt_unscaled = expt_data_unscaled.unsqueeze(0)

X_train_unscaled, X_test_unscaled, y_train, y_test = funcs.configure_bragg_data_trunc(data_folder, expt_data_unscaled, number_of_random_states, number_for_testing, start_index=data_start_index, end_index=data_end_index)
# X_train_unscaled, X_test_unscaled, y_train, y_test = funcs.configure_bragg_data_trunc(data_folder, expt_data_unscaled, number_of_random_states, number_for_testing, scaling_dict=scale_dict, start_index=data_start_index, end_index=data_end_index)



#============================================================================================
#* create dof counts for topas strs and create label mappers
#============================================================================================

dof_count_path = os.path.join(cwd, material_name, f'dof_counts_{material_name}.json')
topas_strs_path = os.path.join(cwd, material_name, f'topas_strs_{material_name}')
if os.path.isfile(dof_count_path):
    print(f'dof counts already created\n')
else:
    util.count_dofs_topas_strs_dir(topas_strs_path, dof_count_path)

with open(dof_count_path, 'r') as f:
    param_mapper = json.load(f)

class_names = list(param_mapper.keys())
label_mapper = {}
for i in range(len(class_names)):
    label_mapper[f"{i}"] = class_names[i]

# with open('./subgroup_ML_labels_to_cifIDs.json', 'r') as f: 
#     label_mapper = json.load(f)

#============================================================================================
#* Scale data & send to GPU
#============================================================================================
scale_data = 100
X_train, X_test, X_expt = funcs.MinMaxScale_2d(X_train_unscaled, X_test_unscaled, X_expt_unscaled, scale=scale_data)

X_train = X_train.to(device)
X_test = X_test.to(device)
X_expt = X_expt.to(device)

y_train = y_train.to(device)
y_test = y_test.to(device)


#============================================================================================
#* Package testing and training data with our custom Dataset class : initialize DataLoader
#============================================================================================

train_data = xsym.simulated_data(X_train, y_train)
test_data = xsym.simulated_data(X_test, y_test)

# load training and testing data with torch.DataLoader
train_loader = DataLoader(train_data, batch_size=training_batchsize, shuffle=True)
test_loader = DataLoader(test_data, batch_size=training_batchsize, shuffle=False)


#^============================================================================================
#* Initialize CNN model: Pretrained, Warm Start, or New Model
#^============================================================================================   

# model = xsym.CNN_bragg(num_classes=number_of_classes)
# model = xsym.CNN_GlobalSelfAttention(num_classes=number_of_classes)
# model = xsym.CNN_LocalAttention(num_classes=number_of_classes)
model = xsym.CNN_MultiHeadAttention(num_classes=number_of_classes)
# model = xsym.CNN_bragg(num_classes=number_of_classes)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.00001)
scheduler = MultiStepLR(optimizer, milestones=scheduler_milestones, gamma=0.1)

# Use multiple GPUs if available
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model.to(device)  
    dummy_input = torch.randn(1, X_train.shape[1]).to(device)  # Initialize Lazy Layers
    model(dummy_input)  
    model = nn.DataParallel(model, device_ids=logical_gpus)

model = model.to(device)
criterion = criterion.to(device)

# Load Pretrained Model
if load_pretrained_model:
    checkpoint = torch.load(pretrain_path, map_location=device)
    
    new_state_dict = OrderedDict()
    for k, v in checkpoint["model_state_dict"].items():
        new_key = k.replace("module.", "") if "module." in k and torch.cuda.device_count() < 2 else k
        new_state_dict[new_key] = v

    model.load_state_dict(new_state_dict)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Move optimizer states to GPU
    for param_state in optimizer.state.values():
        for k, v in param_state.items():
            if isinstance(v, torch.Tensor):
                param_state[k] = v.to(device)

# Warmstart Model
if warmstart_model:
    checkpoint = torch.load(warmstart_path, map_location=device)
    model.load_state_dict(checkpoint, strict=False)


# def objective(trial):
#     # Suggest hyperparameters
#     filter_size = trial.suggest_int("filter_size", 15, 51, step=2)  # e.g., odd numbers between 3 and 15
#     stride = trial.suggest_int("stride", 1, 5)
    
#     model = xsym.CNN_MultiHeadAttention(num_classes = number_of_classes, filter_size=filter_size, stride=stride, num_classes=10).to("cuda")
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#     criterion = nn.CrossEntropyLoss()

#     for epoch in range(epochs):
#         model.train()
#         count = 0
#         running_loss = 0.0
#         for inputs, labels in train_loader:
#             count = count+1
#             inputs, labels = inputs.to(device), labels.to(device)
#             optimizer.zero_grad(set_to_none=True)
#             inputs = inputs.to(device)
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item()
#         scheduler.step()
#         if epoch % 10 ==0:
#             print("Epoch {} loss: {}".format(epoch + 1, running_loss / len(train_loader)))


#============================================================================================
# * Train & Test model
#============================================================================================   

if train_model == True:
    #^ Training loop
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            inputs = inputs.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()
        # if epoch % 10 ==0:
        #     print("Epoch {} loss: {}".format(epoch + 1, running_loss / len(train_loader)))
        print("Epoch {} loss: {}".format(epoch + 1, running_loss / len(train_loader)))

    #^ Validation on test data loop
    number_of_top_results = min(10, number_of_classes)
    correct = [0]*number_of_top_results

    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            model.eval()
            inputs, labels = inputs.to(device), labels.to(device)
            predicted = model(inputs)
            total += labels.size(0)
            top = torch.topk(predicted, number_of_top_results, dim=1) # top is tuple with (values, indicies) where indicies can be used to check with labels for accuracy
            
            for i in range(len(labels)):

                for j in range(len(correct)):
                    k = j+1
                    if labels[i] in top[1][i][:k]:
                        correct[j] = correct[j] + 1


    print()
    correct_list = []
    for i in range(len(correct)):
        iteration = i+1
        accuracy = round(100 * (correct[i]/total), 2)
        print(f'Top {i+1} Accuracy: {100 * (correct[i]/total):.2f}')
        
        correct_list.append([iteration, accuracy])

    print()
    t1 = time.time()
    runtime = f'Runtime: {(t1-t0)/60} minutes'
    print(runtime)

    #! ------------ SAVE THE MODEL ------------
    
    torch.save({
        'epoch':epoch, 
        'model_state_dict':model.state_dict(), 
        'optimizer_state_dict':optimizer.state_dict(), 
        'loss':loss,}, path_to_save_model)

#============================================================================================
#* Run XsymNet against experimental data
#============================================================================================   

if classify_experiment == True:

    number_of_top_results = min(10, number_of_classes)
    confidence_results = {}
    weighted_confidence_results = {}
    ranked_confidence_results = {}
    param_weight = [10,9,8,7,6,5,4,3,2,1]
    
    with torch.no_grad():
        for i in range(confidence_iterations):
            model.eval()
            predicted = model(X_expt)
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
    model_name_noext = os.path.splitext(os.path.basename(path_to_save_model))[0]
    confidence_results = funcs.sort_dict_by_values_desc(confidence_results)
    print(f'subgroup: top10 rating, DOF, place socre, ')
    for key, value in confidence_results.items():
        out_confidence = f'{key}: {value}, {weighted_confidence_results[key]}, {ranked_confidence_results[key]/confidence_iterations}'
        print(out_confidence)

    confidence_results['material_name']=material_name
    confidence_results['phase_name']=phase_name
    confidence_results['model_name']=model.__class__.__name__
    confidence_results['epochs']=epochs
    confidence_results['scheduler_milestones']=scheduler_milestones
    confidence_results['train_batchsize']=training_batchsize
    confidence_results['data_start_index']=data_start_index
    confidence_results['data_end_index']=data_end_index
    if warmstart_model == True:
        confidence_results['warmstart_from']=warmstart_path
    if load_pretrained_model == True: 
        confidence_results['pretrained_model']=pretrain_path
    confidence_results['training_data']= data_folder
    confidence_results['classified_data']=experimental_p1_filename
    confidence_results['minmax2d_scale']=scale_data
    confidence_results['saved_model']=path_to_save_model
    confidence_results['validation_accuracy']=correct_list
    confidence_results['runtime']=runtime
    confidence_results['num_gpus']=len(logical_gpus)



    


        #^ confidence_results[key] = out_confidence



    funcs.save_dict_to_json(confidence_results, os.path.join(cwd, material_name, f'z_{model_name_noext}.json'))
#============================================================================================
# * Print Output and write output files
#============================================================================================   

    # get subgroup info for predicted subgroups and write to file

        
    # print()
    # cif_path = os.path.join(cwd, f'subgroup_cifs_{material_name}' )
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
