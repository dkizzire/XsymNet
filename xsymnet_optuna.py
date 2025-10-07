import os
import glob
import time
import json
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from collections import OrderedDict, defaultdict

import utility_functions as util
import xsymnet_funcs as funcs
import xsymnet_classes as xsym

import optuna

t0 = time.time()
cwd = os.getcwd()

#============================================================================================
# Determine if running on GPU or CPU and set tensor map locations
#============================================================================================

# Set available GPUs
# selected_gpus = [0, 1, 2, 3] #, 4, 5, 6, 7]
selected_gpus = [7] 

os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, selected_gpus))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'\nUsing device: {device}\n')
logical_gpus = list(range(len(selected_gpus)))
print(f'Selected GPU IDs: {selected_gpus}, are mapped to Logical GPU IDs: {logical_gpus}\n')

#^============================================================================================
#* Input phase
#^============================================================================================
material_name = 'wo3'
phase_name = '1043K'
epochs = 150
training_batchsize = 256
confidence_iterations = 100
scheduler_milestones = [80, 175]

#* Slice data to region of interest
data_start_index = 0 #70  #661 #TOF 23750 for NVPF  ...... # 1181 TOF 36000  for PZT_90  ....... TOF 60000 i=5390 for BSO
data_end_index = -1 #6765 #:6061-110k:# #:6765-118k:# #3841

percent_for_testing = 0.2 # 0.2 for 20% testing, 80% training

train_model = True
path_to_save_model = f'/netdisk/dx6/saved_models/{material_name}_{phase_name}_e{epochs}_optuna_1Conv_01.pt'

#* Classify Experimental Data
classify_experiment = True
# experimental_files = sorted(glob.glob(os.path.join(cwd, material_name, 'experimental_p1s', '*.xy')))
experimental_files = sorted(glob.glob(os.path.join(cwd, material_name, 'experimental_p1s', 'P1_wo3_1043K_xy.xy')))
# experimental_files = sorted(glob.glob(os.path.join(cwd, material_name, 'experimental_p1s', 'beta_exp_xy.xy')))
# experimental_files = sorted(glob.glob(os.path.join(cwd, material_name, 'experimental_p1s', 'pzt90_exp_xy.xy')))
assert experimental_files, f"Experimental file(s) not found in {os.path.join(cwd, material_name, 'experimental_p1s')} \n --> Check path and file name. <--"
experimental_names = [os.path.splitext(os.path.basename(f))[0] for f in experimental_files]

#^ get beta trunc je xy 

#* Warmstart Control
warmstart_model = False
warmstart_path = f'/netdisk/dx6/saved_models/bso_alpha_5_e150_multiheadattention_14.pt'

#* Load Pretrained Model Control
load_pretrained_model = False
pretrain_path = f'/netdisk/dx6/saved_models/bso_beta_3_e150_no_mha_avgcam.pt'

#* Create CAM Control
create_cam = True
# subgroup_cam_target = '0093' # wo3_1043K
subgroup_cam_target = '0152' # bso_beta
# subgroup_cam_target = '0009' # pzt_90_300K
cam_save_basename = f'{material_name}_{phase_name}_avgcam_optuna_1Conv_01'
cam_x_units_label = 'TOF (μs)'
# cam_x_units_label = '2Theta (°)'  # '2Theta (°)' or 'TOF (μs)'
cam_legend_label = r'WO$_{3} 1043 K$'  # legend label for CAM plot
# cam_legend_label = r'$\beta$ BiSn$_{2}$O$_{7}$'  # legend label for CAM plot
# cam_legend_label = r'PbZr$_{0.1}$Ti$_{0.9}$O$_{3}$'  # legend label for CAM plot


# ground_truth_cif = '0088' # bso_alpha
# ground_truth_cif = '0152' # bso_beta
# ground_truth_cif = '0032' # wo3_1085K   
# ground_truth_cif = '0093' # wo3_1043K
# ground_truth_cif = '0001' # wo3_1243K
ground_truth_cif = '0009' # pzt_90


#^ convert diffraction xy to tensor (.pt)
for i in range(len(experimental_files)):    
    if experimental_files[i][-3:] != '.pt':
        converted_expt_data, expt_x_axis = util.convert_bragg_to_tensors(os.path.join(cwd, material_name, 'experimental_p1s', experimental_files[i]))
        new_filename = f'{os.path.splitext(experimental_files[i])[0]}.pt'
        torch.save(converted_expt_data, os.path.join(cwd, material_name, 'experimental_p1s', new_filename))
        experimental_files[i] = new_filename



#============================================================================================
#* Load training & testing data
#============================================================================================

#^ Load training and testing data as tensors 
data_folder = os.path.join(cwd, material_name, 'ml_data', phase_name)
data_files = sorted(glob.glob(os.path.join(data_folder, '*.pt')))
num_classes = len(data_files)
first_file = torch.load(data_files[0])
number_of_random_states = first_file.size()[0]
number_for_testing = int(number_of_random_states*percent_for_testing)


#^ load experimental data
expt_data = [torch.load(f) for f in experimental_files]
if len(expt_data) > 1:
    expt_data = torch.vstack(expt_data)
else:
    expt_data = expt_data[0]


#^ split data into training and testing sets & create labels
X_train_unscaled, X_test_unscaled, y_train, y_test = funcs.train_test_split(data_folder, number_of_random_states, number_for_testing)


#^ slice training, testing, expt data to region of interest
X_train_unscaled = X_train_unscaled[:, data_start_index:data_end_index]
X_test_unscaled = X_test_unscaled[:, data_start_index:data_end_index]
if expt_data.ndim > 1:
    expt_data_unscaled = expt_data[:,data_start_index:data_end_index]
else:
    expt_data_unscaled = expt_data[data_start_index:data_end_index]
    expt_data_unscaled = expt_data_unscaled.unsqueeze(0)  # Add a dimension if it's a single tensor

expt_x_axis = expt_x_axis[data_start_index:data_end_index]


#============================================================================================
#* Scale data & send to GPU
#============================================================================================
scale_data = 100
X_train, X_test, X_expt = funcs.MinMaxScale_2d(X_train_unscaled, X_test_unscaled, expt_data_unscaled, scale=scale_data)

# X_train = torch.log10(X_train_unscaled)
# X_test = torch.log10(X_test_unscaled)
# X_expt = torch.log10(expt_data_unscaled)


# scaler = Normalizer()
# scaler = MinMaxScaler(feature_range=(0,100))
# X_train = scaler.fit_transform(X_train_unscaled)
# X_train = scaler.transform(X_train_unscaled)
# X_test = scaler.transform(X_test_unscaled)
# X_expt = scaler.transform(expt_data_unscaled)

# X_train, X_test, X_expt = funcs.MaxAbsScale(X_train_unscaled, X_test_unscaled, X_expt_load=expt_data_unscaled)

X_train = torch.Tensor(X_train)
X_test = torch.Tensor(X_test)
X_expt = torch.Tensor(X_expt)

print(f'X_train MAX:{torch.max(X_train):.2f}\nX_train MIN:{torch.min(X_train):.2f}\n')
print(f'X_test  MAX:{torch.max(X_test):.2f}\nX_test  MIN:{torch.min(X_test):.2f}\n')
print(f'X_expt  MAX:{torch.max(X_expt):.2f}\nX_expt  MIN:{torch.min(X_expt):.2f}\n')

y_train = y_train
y_test = y_test
# print(f'X_train {X_train.shape}')
# print(f'X_test  {X_test.shape}')
# print(f'X_expt  {X_expt.shape}')
# print(f'y_train {y_train.shape}')
# print(f'y_test  {y_test.shape}\n')
# print()

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

#* Create label mapper
label_mapper = {i: class_names[i] for i in range(len(class_names))} 

#^============================================================================================
# Optuna Objective with Early Stopping
#^============================================================================================
def objective_factory(train_loader, val_loader, num_classes, device, logical_gpus):
    def objective(trial):
        # Hyperparameter search space
        channel_1 = trial.suggest_categorical("channel_1", [8, 16, 32, 64])
        channel_2 = trial.suggest_categorical("channel_2", [16, 32, 64, 128])
        kernel_1 = trial.suggest_int("kernel_1", 7, 31, step=2)
        kernel_2 = trial.suggest_int("kernel_2", 7, 31, step=2)
        stride_1 = trial.suggest_categorical("stride_1", [1, 2, 3])
        stride_2 = trial.suggest_categorical("stride_2", [1, 2, 3])
        mha_num_heads = trial.suggest_categorical("mha_num_heads", [2, 4, 8])
        linear_1 = trial.suggest_categorical("linear_1", [256, 512, 1000])

        if channel_2 % mha_num_heads != 0:
            print(f"Trial {trial.number} pruned due to invalid MHA configuration with channel_2 = {channel_2} and num_heads = {mha_num_heads}.")
            raise optuna.exceptions.TrialPruned()

        model = xsym.CNN_MultiHeadAttention(
            num_classes=num_classes,
            channel_1=channel_1,
            channel_2=channel_2,
            kernel_1=kernel_1,
            kernel_2=kernel_2,
            stride_1=stride_1,
            stride_2=stride_2,
            num_heads=mha_num_heads,
            linear_1=linear_1
        )
        # model = xsym.CNN_MultiHeadAttention_1Conv(
        #     num_classes=num_classes,
        #     channel_2=channel_2,
        #     kernel_2=kernel_2,
        #     stride_2=stride_2,
        #     num_heads=mha_num_heads,
        #     linear_1=linear_1
        # )

        print(f"Trial {trial.number} parameters: channel_2={channel_2}, kernel_2={kernel_2}, stride_2={stride_2}, mha_num_heads={mha_num_heads}, linear_1={linear_1}")
        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        # Ensure the model is properly initialized before wrapping in DataParallel
        model.eval()
        dummy_input = torch.randn(1, X_train.shape[1]).to(device)
        model(dummy_input)  # Run a forward pass to initialize lazy layers

        #* Wrap in DataParallel if multiple GPUs are available
        if torch.cuda.device_count() > 1:
            print(f"\nUsing {torch.cuda.device_count()} GPUs!\n")
            model = nn.DataParallel(model, device_ids=logical_gpus)

        model.to(device)

        # Training loop with early stopping
        best_val_acc = 0
        patience, counter = 2, 0
        max_epochs = 50

        for epoch in range(max_epochs):
            # Training
            running_loss = 0.0
            val_running_loss = 0.0
            model.train()
            tr_correct1, tr_correct3, tr_correct5, tr_total = 0, 0, 0, 0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                preds = model(inputs)
                loss = criterion(preds, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                # Top-1 Top-3 and Top-5
                tr_correct1 += (preds.argmax(1) == labels).sum().item()
                tr_correct3 += (preds.topk(3, dim=1).indices == labels.unsqueeze(1)).any(dim=1).sum().item()
                tr_correct5 += (preds.topk(5, dim=1).indices == labels.unsqueeze(1)).any(dim=1).sum().item()
                tr_total += labels.size(0)

            # Validation
            model.eval()
            val_correct1, val_correct3, val_correct5, val_total = 0, 0, 0, 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    preds = model(inputs)
                    val_loss = criterion(preds, labels)
                    val_running_loss += val_loss.item()
                    # Top-1 Top-3 and Top-5
                    val_correct1 += (preds.argmax(1) == labels).sum().item()
                    val_correct3 += (preds.topk(3, dim=1).indices == labels.unsqueeze(1)).any(dim=1).sum().item()
                    val_correct5 += (preds.topk(5, dim=1).indices == labels.unsqueeze(1)).any(dim=1).sum().item()
                    val_total += labels.size(0)

            val_acc1 = val_correct1 / val_total
            val_acc3 = val_correct3 / val_total
            val_acc5 = val_correct5 / val_total
            print(f"Trial {trial.number} Epoch {epoch}: Top-1 Acc: {val_acc1:.2f}, Top-3 Acc: {val_acc3:.2f}, Top-5 Acc: {val_acc5:.2f}, Tr-Loss: {running_loss/len(train_loader):.4f}, Val-Loss: {val_running_loss/len(val_loader):.4f}")
            # Report to Optuna
            # trial.report(val_acc1, epoch)
            trial.report(val_acc5, epoch)

            #^ Early stopping
            # if val_acc1 > best_val_acc:
            #     best_val_acc = val_acc1
            #     counter = 0
            if val_acc5 > best_val_acc:
                best_val_acc = val_acc5
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    break

            if trial.should_prune():
                print(f"Trial {trial.number} pruned at epoch {epoch} by optuna.")
                raise optuna.exceptions.TrialPruned()

        # Log final metrics for this trial
        trial.set_user_attr("val_acc_top1", val_acc1)
        trial.set_user_attr("val_acc_top3", val_acc3)
        trial.set_user_attr("val_acc_top5", val_acc5)
        print(f"Trial {trial.number}: Top-1 Acc: {val_acc1:.2f}, Top-3 Acc: {val_acc3:.2f}, Top-5 Acc: {val_acc5:.2f}, Tr-Loss: {running_loss/len(train_loader):.4f}, Val-Loss: {val_running_loss/len(val_loader):.4f}")


        # --- save best model so far ---

        # if trial.number !=0 and val_acc1 > trial.study.best_value:
        if trial.number !=0 and val_acc5 > trial.study.best_value:
            model.eval()
            model_save_dict = OrderedDict()
            if isinstance(model, nn.DataParallel):
                model_save_dict['model_state_dict'] = model.module.state_dict()
            else:
                model_save_dict['model_state_dict'] = model.state_dict()
            
            # model_save_dict['optimizer_state_dict'] = optimizer.state_dict()
            # model_save_dict['epoch'] = epochs
            # model_save_dict['loss'] = loss

            torch.save(model_save_dict, path_to_save_model)

        return best_val_acc
    return objective


#! ============================================================================================
#! Run Optuna Study
#! ============================================================================================
#============================================================================================
#* Package testing and training data with our custom Dataset class : initialize DataLoader
#============================================================================================

train_data = xsym.simulated_data(X_train, y_train)
test_data = xsym.simulated_data(X_test, y_test)

# load training and testing data with torch.DataLoader
train_loader = DataLoader(train_data, batch_size=training_batchsize, shuffle=True)
val_loader = DataLoader(test_data, batch_size=training_batchsize, shuffle=False)

objective = objective_factory(train_loader, val_loader, num_classes, device, logical_gpus)
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)

print("Best Trial:", study.best_trial.params)
print("Top-1:", study.best_trial.user_attrs["val_acc_top1"])
print("Top-3:", study.best_trial.user_attrs["val_acc_top3"])
print("Top-5:", study.best_trial.user_attrs["val_acc_top5"])

topt = time.time()
print(f'\nOptuna tuning time: {(topt - t0)/60:.2f} minutes\n')

#============================================================================================
#* Load the best model from the study for final evaluation
#============================================================================================
model = xsym.CNN_MultiHeadAttention(
    num_classes=num_classes,
    channel_1=study.best_params["channel_1"],
    channel_2=study.best_params["channel_2"],
    kernel_1=study.best_params["kernel_1"],
    kernel_2=study.best_params["kernel_2"],
    stride_1=study.best_params["stride_1"],
    stride_2=study.best_params["stride_2"],
    num_heads=study.best_params["mha_num_heads"],
    linear_1=study.best_params["linear_1"],
).to(device)

# model = xsym.CNN_MultiHeadAttention_1Conv(
#     num_classes=num_classes,
#     channel_2=study.best_params["channel_2"],
#     kernel_2=study.best_params["kernel_2"],
#     stride_2=study.best_params["stride_2"],
#     num_heads=study.best_params["mha_num_heads"],
#     linear_1=study.best_params["linear_1"],
# ).to(device)

# checkpoint = torch.load(path_to_save_model, map_location=device)
# model_check.load_state_dict(torch.load(path_to_save_model)['model_state_dict'], strict=True)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
scheduler = ReduceLROnPlateau(optimizer)
previous_epochs = 0
model.eval()
dummy_input = torch.randn(1, X_train.shape[1]).to(device)
model(dummy_input)  # Run a forward pass to initialize lazy layers

#* Wrap in DataParallel if multiple GPUs are available
if torch.cuda.device_count() > 1:
    print(f"\nUsing {torch.cuda.device_count()} GPUs!\n")
    model = nn.DataParallel(model, device_ids=logical_gpus)

# Move model to device again to ensure it's properly placed
model.to(device)
    
#============================================================================================
# * Train & Test model
#============================================================================================   

if train_model == True:
    # Training loop with early stopping
    best_val_acc = 0
    patience, counter = 2, 0
    model.train()

    for epoch in range(epochs):
        # funcs.save_kernel_weights(model, epoch, save_file="kernel_weights.txt")
        running_loss = 0.0
        val_running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            # optimizer.zero_grad(set_to_none=True)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step(loss)
        
        # Validation
        model.eval()
        val_correct1, val_correct3, val_correct5, val_total = 0, 0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                preds = model(inputs)
                val_loss = criterion(preds, labels)
                val_running_loss += val_loss.item()
                # Top-1 Top-3 and Top-5
                val_correct1 += (preds.argmax(1) == labels).sum().item()
                val_correct3 += (preds.topk(3, dim=1).indices == labels.unsqueeze(1)).any(dim=1).sum().item()
                val_correct5 += (preds.topk(5, dim=1).indices == labels.unsqueeze(1)).any(dim=1).sum().item()
                val_total += labels.size(0)

        val_acc1 = val_correct1 / val_total
        val_acc3 = val_correct3 / val_total
        val_acc5 = val_correct5 / val_total
        print(f"Epoch {epoch}: Top-1 Acc: {val_acc1:.2f}, Top-3 Acc: {val_acc3:.2f}, Top-5 Acc: {val_acc5:.2f}, Tr-Loss: {running_loss/len(train_loader):.4f}, Val-Loss: {val_running_loss/len(val_loader):.4f}")
        # Early stopping
        # if val_acc1 > best_val_acc:
        #     best_val_acc = val_acc1
        #     counter = 0
        if val_acc5 > best_val_acc:
            best_val_acc = val_acc5
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                break

    #!============================================================================================
    #* Save the model and optimizer
    #!============================================================================================
    # Save Model
    if load_pretrained_model:
        epochs += previous_epochs
    # Make sure to put model in eval mode before saving
    model.eval()
    model_save_dict = OrderedDict()
    if isinstance(model, nn.DataParallel):
        model_save_dict['model_state_dict'] = model.module.state_dict()
    else:
        model_save_dict['model_state_dict'] = model.state_dict()
    
    model_save_dict['optimizer_state_dict'] = optimizer.state_dict()
    model_save_dict['epoch'] = epochs
    model_save_dict['loss'] = loss

    torch.save(model_save_dict, path_to_save_model)



#^ Run Validation
number_of_top_results = min(10, num_classes)
total = 0
correct_top_k = torch.zeros(number_of_top_results, device=device)  # Store counts for Top-1 to Top-k
#* Dictionary to store misclassified labels

misclassified_counts = defaultdict(int)

with torch.no_grad():
    for inputs, labels in val_loader:
        model.eval()
        inputs, labels = inputs.to(device), labels.to(device)
        predicted = model(inputs)
        total += labels.size(0)
        # Get top-k predictions
        top_k_vals, top_k_indices = torch.topk(predicted, number_of_top_results, dim=1)  # (values, indices)
        
        # Check if the correct labels are in the top-k predictions
        correct_matrix = top_k_indices.eq(labels.view(-1, 1).expand_as(top_k_indices))  # Shape: (batch_size, k)

        # Accumulate counts for Top-1 to Top-k
        for k in range(number_of_top_results):
            correct_top_k[k] += correct_matrix[:, : (k + 1)].sum().item()

        # Track misclassifications (if the label is never in the top-10)
        for i in range(len(labels)):
            label = labels[i].item()  # Convert label tensor to int
            if not correct_matrix[i].any():  # If correct label is not in top-10
                misclassified_counts[label] += 1  # Increment count for this label

# Print misclassification summary
print("\nMisclassification Summary (Labels not in Top 10):")
for label, count in misclassified_counts.items():
    class_name = label_mapper.get(label, f"Unknown ({label})")
    print(f"Class '{class_name}' (Label {label}): {count} times")

# Print Top-k Accuracy
print("\nTop-k Accuracy:")
validation_list = []
for k in range(number_of_top_results):
    accuracy = round(100 * (correct_top_k[k].item() / total), 2)
    print(f'Top {k+1} Accuracy: {accuracy:.2f}%')
    validation_list.append([k+1, accuracy])

print()
t1 = time.time()
runtime = f'Runtime: {(t1-t0)/60} minutes'
print(runtime)

# #============================================================================================
# #* Run XsymNet against experimental data
# #============================================================================================   

if classify_experiment == True:
    number_of_top_results = min(5, num_classes)
    param_weight = list(range(number_of_top_results, 0, -1))  

    # Dictionaries to store confidence metrics
    confidence_results = defaultdict(int)
    ranked_confidence_results = defaultdict(int)
    weighted_confidence_results = {}
    logit_results = defaultdict(list)
    predicted_subgroups_dof = []

    #============================================================================================
    #* Get top-k predictions for each experimental pattern
    #* For use with ILLUMINE PROJECT - uncommment this block and comment out below block to turn on
    #============================================================================================
    # with torch.no_grad():
    #     for i in range(X_expt.shape[0]):
    #         model.eval()
    #         predicted = model(X_expt[i].unsqueeze(0))  # Add batch dimension
    #         # predicted = model(X_expt)
    #         top_logits, top_indices = torch.topk(predicted, number_of_top_results, dim=1)

    #         # Convert to lists
    #         top_indices = top_indices[0].tolist()  # Extract first batch element
    #         top_logits = top_logits[0].tolist()  # Extract first batch element
    #         # round each number in top_logits to have 8 decimal places
    #         top_logits = [round(x, 8) for x in top_logits]
    #         predicted_subgroups = [label_mapper[x] for x in top_indices]  # Map indices to class names
    #         results = {'subgroups': predicted_subgroups, 'logits': top_logits}#, 'indices': top_indices}
    #         logit_results.update({experimental_names[i]:results})

    # # Save to JSON
    # json_path = os.path.join(cwd, material_name, 'logit_results.json')
    # funcs.save_dict_to_json(logit_results, json_path) 

    # # Save logit results to CSV
    # save_logits_path = os.path.join(cwd, material_name, f'{phase_name}_logit_results.csv')
    # save_subgroups_path = os.path.join(cwd, material_name, f'{phase_name}_subgroup_results.csv')
    # util.save_logits_table(logit_results, number_of_top_results, save_logits_path, save_subgroups_path)
    #============================================================================================
    #============================================================================================


    with torch.no_grad():
        model.eval()
        X_expt = X_expt.to(device)
        predicted = model(X_expt)
        top_logits, top_indices = torch.topk(predicted, number_of_top_results, dim=1)
        # Convert to lists
        top_indices = top_indices[0].tolist()  # Extract first batch element
        top_logits = top_logits[0].tolist()  # Extract first batch element
        top_logits = [round(x, 4) for x in top_logits]
        predicted_subgroups = [label_mapper[x] for x in top_indices]  # Map indices to class names
    
    for key in predicted_subgroups:
        predicted_subgroups_dof.append(param_mapper.get(key))


    top_pred_sg = []
    top_pred_HM = []
    for i in range(len(predicted_subgroups)):
        top_pred_file = glob.glob(os.path.join(topas_strs_path, f'*{predicted_subgroups[i]}.str'))
        _, pred_sgo = funcs.get_str_info(top_pred_file[0], 'strain')
        top_pred_sg.append(f'{pred_sgo[0]}:{pred_sgo[1]}')
        top_pred_HM.append(f'{pred_sgo[2]}')
        


    # zip predicted_subgroups and their dof counts together
    predicted_with_dof = list(zip(predicted_subgroups, predicted_subgroups_dof))
    print(f'\nPredicted Subgroups and DOF Counts: \n')
    print(f'Rank, Subgroup, DOF Count, Logit')
    for i in range(len(predicted_subgroups)):
        print(f'{i:<2}: {predicted_subgroups[i]:<4} {top_pred_sg[i]:<6} {top_pred_HM[i]:<8} {predicted_subgroups_dof[i]:<4} {top_logits[i]:<8}\n')

    with open(os.path.join(cwd, material_name, f'zz_{os.path.splitext(os.path.basename(path_to_save_model))[0]}.txt'), 'w') as f:
        f.write(f'material: {material_name}\n')
        f.write(f'phase: {phase_name}\n')
        f.write(f'Experimental Data: {experimental_names}\n\n')
        f.write(f'Top Predicted Subgroups, Space Group:Origin, Logits, DOF Count:\n')
        for i in range(len(predicted_subgroups)):
            f.write(f'{predicted_subgroups[i]}, {top_pred_sg[i]:<6}, {top_pred_HM[i]:<8}, {top_logits[i]}, {predicted_subgroups_dof[i]}\n')
        f.write(f'\nModel Information:\n')
        f.write(f'Epochs: {epochs}\n')
        f.write(f'Training Batch Size: {training_batchsize}\n')
        f.write(f'Training Data: {data_folder}\n')
        f.write(f'Data Start/End Indices: {data_start_index} to {data_end_index}\n')
        f.write(f'MinMax2d Scale Factor: {scale_data}\n')
        f.write(f'Saved Model: {path_to_save_model}\n')
        if warmstart_model:
            f.write(f'Warmstarted From: {warmstart_path}\n')
        if load_pretrained_model:
            f.write(f'Pretrained From: {pretrain_path}\n')

        f.write(f'\nValidation Top-k Accuracy: Prediction Rank (k), % Accuracy \n')
        for i in range(len(validation_list)):
            f.write(f'Top {validation_list[i][0]} accuracy, {validation_list[i][1]} \n')

        f.write(f'\nRuntime: {(t1-t0)/60} minutes\n')
        f.write(f'Number of GPUs Used: {len(logical_gpus)}\n')
        f.write(f'\nModel Structure:\n {str(model)}\n')
        for item in study.best_params.items():
            f.write(f'{item}\n')
        

# #^============================================================================================
# #* Create CAM (Class Activation Map)
# #^============================================================================================   
if create_cam: 

    # Initialize GradCAM
    # target_layer = model.module.attention if isinstance(model, torch.nn.DataParallel) else model.attention
    target_layer = model.module.conv2 if isinstance(model, torch.nn.DataParallel) else model.conv2
    grad_cam = xsym.GradCAM(model, target_layer=target_layer)

    for tsg in predicted_subgroups:
        #^ get cam data, raise error if not found, and slice to region of interest
        cam_data_path = ''
        for i in range(len(data_files)):
            if tsg in data_files[i]:
                cam_data_path = data_files[i]
                break
        if create_cam and cam_data_path == '':
            raise ValueError(f"Subgroup {tsg} not found in data files. Please check the subgroup name or data files.")

        cam_data = torch.load(cam_data_path)
        cam_data = cam_data.unsqueeze(0) if cam_data.ndim == 1 else cam_data  # Ensure it's 2D
        cam_data = cam_data[:, data_start_index:data_end_index]  # Slice to region of interest

        print(f'Scaling CAM data: Test Max is max of {tsg} training data\n')
        X_train_dummy, cam_data_scaled, X_expt_dummy = funcs.MinMaxScale_2d(X_train_unscaled, cam_data, expt_data_unscaled, scale=scale_data)
        cam_data_scaled = torch.Tensor(cam_data_scaled).to(device)

        # Subset training data for target class (e.g., class 5)
        input_batch = cam_data_scaled  # shape: (N, seq_len)
        target_class = label_mapper.get(tsg)  # Convert subgroup name to label index

        cam_save_name = f'{cam_save_basename}_{tsg}.png'

        # Compute average CAM
        avg_cam = grad_cam.generate_average_cam(input_batch, target_class)

        # Plot over experimental pattern
        xsym.plot_avg_cam_with_expt(xrd_expt=X_expt.cpu().numpy(), 
                                    avg_cam=avg_cam, 
                                    x_axis=expt_x_axis, 
                                    x_units = cam_x_units_label, 
                                    save_name= cam_save_name, 
                                    legend_label = cam_legend_label)


