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

import utility_functions as util
import xsymnet_funcs as funcs
import xsymnet_classes as xsym

t0 = time.time()
cwd = os.getcwd()


# torch.cuda.init()
# torch.cuda.is_available()
# torch.cuda.device_count()

#--------------------------------------------------------------------------------------------
# Determine if running on GPU or CPU and set tensor map locations
#--------------------------------------------------------------------------------------------

# gpu_index = 7
# device = util.determine_gpu_or_cpu(gpu_index)

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"#, 1, 2, 3, 4, 5, 6, 7"
# selected_gpus = [0]#, 1, 2, 3, 4, 5, 6, 7]
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"#, 4, 5"#, 6, 7"
selected_gpus = [0, 1, 2, 3]#, 4, 5]#, 6, 7]
torch.cuda.set_device(selected_gpus[0])
device = f'cuda:{selected_gpus[0]}'

#--------------------------------------------------------------------------------------------
# Input phase
#--------------------------------------------------------------------------------------------
material_name = 'wo3'
phase_name = '1243K_scale_bkgsub'
epochs = 150
training_batchsize = 512
confidence_iterations = 100

use_encoder = True
path_to_save_model = f'/netdisk/dx6/saved_models/{material_name}_{phase_name}_e{epochs}_encoder_01.pt'

train_model = False
path_to_save_model = f'/netdisk/dx6/saved_models/{material_name}_{phase_name}_e{epochs}_warmstart_03_1325test.pt'

classify_experiment = True
experimental_p1_filename = 'bkg_sub_P1_wo3_1243K_xy.xy'

warmstart_model = False
# warmstart_path = f'/netdisk/dx6/saved_models/wo3_p4nmm_parent_e25_manualscaling.pt'
warmstart_path = f'/netdisk/dx6/saved_models/wo3_1243K_scale_bkgsub_e200_warmstart_02.pt'
# warmstart_path = f'/netdisk/dx6/saved_models/wo3_1243K_scale_base_e150_prescale.pt'
# warmstart_path = f'/netdisk/dx6/saved_models/bso_beta_5_e150_bragg.pt'

load_pretrained_model = False
pretrain_path = f'/netdisk/dx6/saved_models/wo3_p4nmm_parent_e25_manualscaling.pt'


# ground_truth_cif = '0032' # wo3_1003K
ground_truth_cif = '0001' # wo3_1243K
# ground_truth_cif = '0009' # pzt_90
# ground_truth_cif = '0088' # bso_alpha
# ground_truth_cif = '0152' # bso_beta

#^ scaling json
scale_dict_path = os.path.join(cwd, material_name, 'ml_data', '1325K_scale_bkgsub', '1325K_scale_bkgsub_scale_dict.json')
# scale_dict_path = os.path.join(cwd, material_name, 'ml_data', phase_name, f'{phase_name}_scale_dict.json')
with open(scale_dict_path) as f:
    scale_dict = json.load(f)

#^ convert diffraction xy to tensor (.pt)    
if experimental_p1_filename[-3:] != '.pt':
    converted_expt_data = util.convert_bragg_to_tensors(os.path.join(cwd, material_name, 'experimental_p1s', experimental_p1_filename))
    new_filename = f'{os.path.splitext(experimental_p1_filename)[0]}.pt'
    torch.save(converted_expt_data, os.path.join(cwd, material_name, 'experimental_p1s', new_filename))
    experimental_p1_filename = new_filename



# -------------------------------------------------------------------------------------------
#* Load training & testing data
# -------------------------------------------------------------------------------------------

# Load training and testing data as tensors 

data_folder = os.path.join(cwd, material_name, 'ml_data', phase_name)
data_files = sorted(glob.glob(os.path.join(data_folder, '*.pt')))
number_of_classes = len(data_files)
first_file = torch.load(data_files[0])
number_of_random_states = first_file.size()[0]
number_for_testing = int(number_of_random_states*0.1)

data_start_index = 0  #661 #TOF 23750 for NVPF  ...... # 1181 TOF 36000  for PZT_90  ....... TOF 60000 i=5390 for BSO
data_end_index = -1

expt_data_unscaled = torch.load(os.path.join(cwd, material_name, 'experimental_p1s', experimental_p1_filename))
expt_data_unscaled = expt_data_unscaled[0,data_start_index:data_end_index]
X_expt_unscaled = expt_data_unscaled.unsqueeze(0)

#^ Split data into training and testing & scaling if using scale_dict 
X_train_unscaled, X_test_unscaled, y_train, y_test = funcs.configure_bragg_data_trunc(data_folder, expt_data_unscaled, number_of_random_states, number_for_testing, scaling_dict=scale_dict, start_index=data_start_index, end_index=data_end_index)



# -------------------------------------------------------------------------------------------
#* create dof counts for topas strs and create label mappers
# -------------------------------------------------------------------------------------------

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



# -------------------------------------------------------------------------------------------
#* Scale data & send to GPU
# -------------------------------------------------------------------------------------------
scale_data = 10
X_train, X_test, X_expt = funcs.MinMaxScale_2d(X_train_unscaled, X_test_unscaled, expt_data_unscaled, scale=scale_data)



X_train = X_train.to(device)
X_test = X_test.to(device)
X_expt = X_expt.to(device)

y_train = y_train.to(device)
y_test = y_test.to(device)


# ------------------------------------------------------------------------------------------------------
#* Package testing and training data with our custom Dataset class : initialize DataLoader
# ------------------------------------------------------------------------------------------------------

train_encoder_data = xsym.simulated_data(X_train, y_train)
test_encoder_data = xsym.simulated_data(X_test, y_test)

# load training and testing data with torch.DataLoader
train_encoder_loader = DataLoader(train_encoder_data, batch_size=training_batchsize, shuffle=True)
test_encoder_loader = DataLoader(test_encoder_data, batch_size=training_batchsize, shuffle=False)

#^ ------------------------------------------------------------------------------------------------------
#* Initialize Autoencoder
#^ ------------------------------------------------------------------------------------------------------   
print("Initial feature shape for X_train:", X_train.shape)
# Initialize Autoencoder
encoded_dim = 1000
encoder_model = xsym.XRD_Autoencoder(input_size=X_train.size()[1], encoded_dim=encoded_dim)
#! device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder_model.to(device)

# Loss & optimizer
encoder_criterion = nn.MSELoss().to(device)
encoder_optimizer = torch.optim.Adam(encoder_model.parameters(), lr=0.0005)

# Encoder Model Training loop
num_epochs = 100
for epoch in range(num_epochs):
    print(f'Encoder epoch {epoch}/{num_epochs}')
    encoder_model.train()
    train_loss = 0
    for batch in train_encoder_loader:
        inputs = batch[0].to(device)
        encoder_optimizer.zero_grad()
        encoded, decoded = encoder_model(inputs)
        loss = encoder_criterion(decoded, inputs)
        loss.backward()
        encoder_optimizer.step()
        train_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss / len(train_encoder_loader):.4f}")

print("Training complete.")

# Extract encoded features
with torch.no_grad():
    encoder_model.eval()
    X_train_encoded = torch.cat([encoder_model(batch[0].to(device))[0] for batch in train_encoder_loader]).cpu().numpy()
    X_val_encoded = torch.cat([encoder_model(batch[0].to(device))[0] for batch in test_encoder_loader]).cpu().numpy()

print("Encoded feature shape for X_train:", X_train_encoded.shape)

# Convert features to tensors for ML training
X_train_encoded_tensor = torch.tensor(X_train_encoded, dtype=torch.float32)
X_val_encoded_tensor = torch.tensor(X_val_encoded, dtype=torch.float32)

# Package encoded data and labels
train_ml_data = xsym.simulated_data(X_train_encoded_tensor, y_train)
test_ml_data = xsym.simulated_data(X_val_encoded_tensor, y_test)

# Create DataLoader for ML model
train_ml_loader = DataLoader(train_ml_data, batch_size=training_batchsize, shuffle=True)
test_ml_loader = DataLoader(test_ml_data, batch_size=training_batchsize, shuffle=False)

#^ making new loaders for training and validation data
#* examine size of data 
#* visually examine new tensors - look at variance, mean, std_dev, 
#* research encoder-decoders 

# ------------------------------------------------------------------------------------------------------
#* Initialize CNN model: Pretrained, Warm Start, or New Model
# ------------------------------------------------------------------------------------------------------   

if use_encoder == True: 
    # Train Classifier
    # model = xsym.XRD_Classifier(input_dim=encoded_dim, num_classes=number_of_classes).to(device)
    # criterion = nn.CrossEntropyLoss().to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # scheduler = MultiStepLR(optimizer, milestones=[75, 125], gamma=0.1)
    model = xsym.CNN_bragg(num_classes=number_of_classes)
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = MultiStepLR(optimizer, milestones=[50, 150], gamma=0.1)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # run a dummy forward pass to initialize model and lazy linear layers
        dummy_input = torch.randn(training_batchsize, X_train_encoded_tensor.shape[1])  # Adjust shape as needed
        #^ dummy_input = torch.randn(training_batchsize, X_train.shape[1])  # Adjust shape as needed
        model(dummy_input)
        model = nn.DataParallel(model, device_ids=selected_gpus)
        model = model.to(f'cuda:{selected_gpus[0]}')
        
    else:
        model = model.to(device)
    
    criterion = criterion.to(device)
#*-------------
#^ Make new function for if torch.cuda.device_count() > 1
#*-------------

#^ ------------------------------------------------------------------------------------------------------
# * Train & Test model
#^ ------------------------------------------------------------------------------------------------------   
if use_encoder == True:

    # # Train Classifier
    # num_epochs_cls = 100
    # for epoch in range(num_epochs_cls):
    #     model.train()
    #     train_loss_cls = 0
    #     for batch in train_ml_loader:
    #         inputs, labels = batch[0].to(device), batch[1].to(device)
    #         optimizer.zero_grad()
    #         outputs = model(inputs)
    #         loss = criterion(outputs, labels)
    #         loss.backward()
    #         optimizer.step()
    #         train_loss_cls += loss.item()

    #     print(f"Epoch [{epoch+1}/{num_epochs_cls}], Classifier Loss: {train_loss_cls / len(train_ml_loader):.4f}")

    # print("Classifier training complete.")
    #^ Training loop
    
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_ml_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            inputs = inputs.to(device)
            outputs = model(inputs)
            #top = torch.topk(outputs, number_of_top_results, dim=1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()

        print("Epoch {} loss: {}".format(epoch + 1, running_loss / len(train_ml_loader)))
    #^ Validation on test data
    print('Encoder Classifier on Validation Data')

    number_of_top_results = min(10, number_of_classes)
    correct = [0]*number_of_top_results

    total = 0
    with torch.no_grad():
        for inputs, labels in test_ml_loader:
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
    for i in range(len(correct)):
        print(f'Top {i+1} Accuracy: {100 * (correct[i]/total)}')
    print()
    t1 = time.time()
    print(f'Runtime: {(t1-t0)/60} minutes')

    #! ------------ SAVE THE MODEL ------------
    
    torch.save({
        'epoch':epoch, 
        'model_state_dict':model.state_dict(), 
        'optimizer_state_dict':optimizer.state_dict(), 
        'loss':loss,}, path_to_save_model)



# ------------------------------------------------------------------------------------------------------
# * Run XsymNet against experimental data
# ------------------------------------------------------------------------------------------------------   

if classify_experiment == True:

    number_of_top_results = min(10, number_of_classes)
    confidence_results = {}
    weighted_confidence_results = {}
    ranked_confidence_results = {}
    param_weight = [10,9,8,7,6,5,4,3,2,1]
    
    if use_encoder == True: 
        with torch.no_grad():
            encoder_model.eval()
            X_expt = X_expt[None, :]
            X_expt_encoded = encoder_model(X_expt)[0]
            X_expt_original = X_expt
            X_expt = X_expt_encoded
            print("X_expt original shape: ",X_expt_original.shape)
            print("X_expt encoded shape: ",X_expt.shape)

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
        print(f'{key}: {value}, {weighted_confidence_results[key]}, {ranked_confidence_results[key]/confidence_iterations}')

    confidence_results['material_name']=material_name
    confidence_results['phase_name']=phase_name
    confidence_results['epochs']=epochs
    confidence_results['train_batchsize']=training_batchsize
    confidence_results['data_start_index']=data_start_index
    confidence_results['data_end_index']=data_end_index
    if warmstart_model == True:
        confidence_results['warmstart_from']=warmstart_path
    if load_pretrained_model == True: 
        confidence_results['pretrained_model']=pretrain_path
    confidence_results['training_data']= data_folder
    confidence_results['minmax2d_scale']=scale_data
    confidence_results['scaling_json']=scale_dict_path
    confidence_results['saved_model']=path_to_save_model


    funcs.save_dict_to_json(confidence_results, os.path.join(cwd, material_name, f'z_{model_name_noext}.json'))
# ------------------------------------------------------------------------------------------------------
# * Print Output and write output files
# ------------------------------------------------------------------------------------------------------   

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
