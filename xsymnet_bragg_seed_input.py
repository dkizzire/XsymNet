import sys
import os

# ====================================================================================
# *** UPDATED: Command Line Argument Parsing ***
# ====================================================================================
def get_command_line_args():
    """
    Reads command-line arguments for config file path and seed number.
    
    Usage: python script.py <CONFIG_FILE> <SEED_NUMBER>
    
    Returns:
        tuple: (config_file_path, seed_value)
    """
    if len(sys.argv) < 3:
        print("Error: Please provide both config file path and seed number as arguments.")
        print(f"Usage: python {sys.argv[0]} <CONFIG_FILE> <SEED_NUMBER>")
        print(f"\nExample: python {sys.argv[0]} config_wo3_1043K.txt 48")
        sys.exit(1)
    
    config_file = sys.argv[1]
    
    # Check if config file exists
    if not os.path.exists(config_file):
        print(f"Error: Configuration file not found: {config_file}")
        sys.exit(1)
    
    try:
        seed_value = int(sys.argv[2])
        return config_file, seed_value
    except ValueError:
        print(f"Error: The seed argument '{sys.argv[2]}' is not a valid integer.")
        sys.exit(1)


# ====================================================================================
# *** Get Command Line Arguments ***
# ====================================================================================
try:
    CONFIG_FILE, SEED = get_command_line_args()
except Exception as e:
    print(f"An error occurred while parsing arguments: {e}")
    sys.exit(1)

print(f"\n{'='*10}")
print(f"XSymNet Training Script")
print(f"{'='*10}")
print(f"Config File: {CONFIG_FILE}")
print(f"Seed Value: {SEED}")
print(f"{'='*10}\n")

# ====================================================================================
# *** Set Environment Variables for Determinism ***
# ====================================================================================
# --- Determinism / CUDA env vars BEFORE importing torch ---
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["PYTORCH_CUDA_FUSED_ATTENTION"] = "0"
os.environ["FLASH_ATTENTION"] = "0"

# Choose which physical GPUs to use, then remap them to logical [0..N-1]
# selected_gpus = [1, 2, 3, 5]  # change to [1,2] etc., as needed
selected_gpus = [0, 1, 2, 3]  # change to [1,2] etc., as needed
# selected_gpus = [6, 7]  # change to [1,2] etc., as needed
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, selected_gpus))


import glob
import time
import numpy as np
import pandas as pd
import json
import re  # for subgroup id parsing
import random

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
from torch.utils.data import DataLoader  # , Dataset
# from sklearn.preprocessing import MaxAbsScaler, StandardScaler, QuantileTransformer, RobustScaler, MinMaxScaler, Normalizer
from sklearn.metrics import f1_score, multilabel_confusion_matrix, precision_recall_fscore_support
from collections import OrderedDict, defaultdict
import matplotlib.pyplot as plt

import utility_functions as util
import xsymnet_funcs as funcs
import xsymnet_classes as xsym
import config_reader as config_reader
# ====================================================================================
# *** Import Configuration Reader ***
# ====================================================================================
from config_reader import read_config_file, apply_config_to_globals

#============================================================================================
# Global timing, cwd, set device
#============================================================================================
t0 = time.time()
cwd = os.getcwd()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'\nUsing device: {device}\n')
logical_gpus = list(range(len(selected_gpus)))
print(f'Selected GPU IDs: {selected_gpus}, are mapped to Logical GPU IDs: {logical_gpus}\n')

#============================================================================================
# Global seeding & deterministic-ish settings
#============================================================================================
seed = SEED

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# Disable TF32 (can introduce nondeterministic matmul kernels on some GPUs)
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

# This generator will be used for deterministic DataLoader shuffling
data_loader_generator = torch.Generator()
data_loader_generator.manual_seed(seed)

#============================================================================================
# Input phase - Load from Configuration File
#============================================================================================
print("Loading configuration from file...")
config_dict = config_reader.read_config_file(CONFIG_FILE, SEED)
print(config_dict)

material_name = config_dict.get('material_name', 'not_specified')
phase_name = config_dict.get('phase_name', 'not_specified')
max_epochs = config_dict.get('max_epochs', 100)
training_batchsize = config_dict.get('training_batchsize', 1024)
scheduler_milestones = config_dict.get('scheduler_milestones', [30, 100, 150])
scheduler_patience = config_dict.get('scheduler_patience', 5)   # patience before reducing LR
early_stop_patience = config_dict.get('early_stop_patience', 12)  # Total patience before breaking

#* Slice data to region of interest
#70  #661 #TOF 23750 for NVPF  ...... # 1181 TOF 36000  for PZT_90  ....... TOF 60000 i=5390 for BSO & 3542
data_start_index = config_dict.get('data_start_index', 0)
#6765 #:6061-110k:# #:6765-118k:# #3841
data_end_index = config_dict.get('data_end_index', -1)

# 0.2 for 20% testing, 80% training
percent_for_testing = config_dict.get('percent_for_testing', 0.2)
scaling_method = config_dict.get('scaling_method', 'MinMax2d')
do_zscore = config_dict.get('do_zscore', True)

train_model = config_dict.get('train_model', True)
save_model = config_dict.get('save_model', False)
# path_to_save_model = f'/netdisk/dx6/saved_models/{material_name}_{phase_name}_Zscale_expt_1085_seed_{seed}_c8c16.pt'
path_to_save_model = config_dict.get('path_to_save_model', f'/netdisk/dx6/saved_models/{material_name}_{phase_name}_seed_{seed}.pt')
# path_to_save_model = path_to_save_model.replace('SEED', str(seed))

#* Classify Experimental Data
classify_experiment = config_dict.get('classify_experiment', True)
experimental_files = config_dict.get('experimental_files', 'not_specified')

if isinstance(experimental_files, str):
    experimental_files = sorted(glob.glob(experimental_files))
# if isinstance(experimental_files, list):
#     get_filenames = []
#     for i in range() = sorted(glob.glob(f)) f in experimental_files]


# experimental_files = sorted(glob.glob(os.path.join(cwd, material_name, 'experimental_xy', '*.xy')))
# experimental_files = sorted(glob.glob(os.path.join(cwd, material_name, 'experimental_xy', 'P1_wo3_1085K_xy.xy')))
# experimental_files = sorted(glob.glob(os.path.join(cwd, material_name, 'experimental_xy', 'beta_exp_xy.xy')))
# experimental_files = sorted(glob.glob(os.path.join(cwd, material_name, 'experimental_xy', 'hrp27188_bs_alpha_neutron_NoComments_dmin_1p0.xy')))
# experimental_files = sorted(glob.glob(os.path.join(cwd, material_name, 'experimental_xy', 'bso_alpha_P1_trunc_je_xy.xy')))
# experimental_files = sorted(glob.glob(os.path.join(cwd, material_name, 'experimental_xy', 'alpha_exp_NEW_xy.xy')))
# experimental_files = sorted(glob.glob(os.path.join(cwd, material_name, 'experimental_xy', 'pzt90_exp_xy.xy')))
assert experimental_files, f"Experimental file(s) not found in {experimental_files} \n --> Check path and file name. <--"
experimental_names = [os.path.splitext(os.path.basename(f))[0] for f in experimental_files]

#* Warmstart Control
warmstart_model = config_dict.get('warmstart_model', False)
warmstart_path = config_dict.get('warmstart_path', f'/netdisk/dx6/saved_models/{material_name}_{phase_name}_seed_42_c8c16.pt')

#* Load Pretrained Model Control
load_pretrained_model = config_dict.get('load_pretrained_model', False)
pretrain_path = config_dict.get('pretrain_path', f'/netdisk/dx6/saved_models/{material_name}_{phase_name}_seed_42_c816.pt')
# pretrain_path = f'/netdisk/dx6/saved_models/bso_alpha_e150_paper_01.pt'

#* Create CAM Control
create_cam = config_dict.get('create_cam', False)
subgroup_cam_target = config_dict.get('subgroup_cam_target', '0001')  # subgroup ID for CAM generation
cam_save_basename = config_dict.get('cam_save_basename', f'CAM_{material_name}_{phase_name}_seed_{seed}')
# '2Theta (°)' or 'TOF (μs)'
cam_x_units_label = config_dict.get('cam_x_units_label', 'TOF (μs)')

# cam_legend_label = r'WO$_{3} 1043 K$'  # legend label for CAM plot
# cam_legend_label = r'$\alpha$ BiSn$_{2}$O$_{7}$'  # legend label for CAM plot
# cam_legend_label = r'PbZr$_{0.1}$Ti$_{0.9}$O$_{3}$'  # legend label for CAM plot
cam_legend_label = config_dict.get('cam_legend_label', f'{material_name} {phase_name}')

# ground_truth_cif = '0088' # bso_alpha
# ground_truth_cif = '0152' # bso_beta
# ground_truth_cif = '0032' # wo3_1085K   
# ground_truth_cif = '0093' # wo3_1043K
# ground_truth_cif = '0001' # wo3_1243K
# ground_truth_cif = '0009' # pzt_90

expt_data = []
for i in range(len(experimental_files)):
    expt_x_axis, y_data = funcs.load_xy_data(experimental_files[i])
    expt_data.append(y_data)

if len(expt_data) > 1:
    expt_data = torch.vstack(expt_data)
else:
    expt_data = expt_data[0]

#=======================================================================================================
#* Load training & testing data 
#* build class index -> label mapper
#* trim training/testing/expt data
#=======================================================================================================

#^ Load training and testing data as tensors 
# data_folder = os.path.join(cwd, material_name, 'ml_data', phase_name)
# data_files = sorted(glob.glob(os.path.join(data_folder, '*.pt')))

data_path = config_dict.get('train_data_dir', 'INCORRECTLY_SPECIFIED')
data_files = sorted(glob.glob(os.path.join(data_path, '*.pt')))
assert len(data_files) > 0, f"Training/Testing data file(s) not found in {data_path} \n --> Check path and file name. <--"
num_classes = len(data_files)

#^ Build class-index -> subgroup-id mapping from ml_data filenames
subgroup_ids = []
for fpath in data_files:
    base = os.path.splitext(os.path.basename(fpath))[0]
    # e.g. "cif0001_randomstates_bragg"
    m = re.search(r'(\d{4})', base)
    if not m:
        raise ValueError(f"Could not parse subgroup ID from filename: {base}")
    sg_id = m.group(1)  # "0001"
    subgroup_ids.append(sg_id)

label_mapper = {i: subgroup_ids[i] for i in range(len(subgroup_ids))}  # 0 -> "0001", etc.
subgroup_to_label = {sg: i for i, sg in label_mapper.items()}         # "0001" -> 0, etc.

#^ split data into training and testing sets & create labels
first_file = torch.load(data_files[0])
number_of_random_states = first_file.size()[0]
number_for_testing = int(number_of_random_states*percent_for_testing)

# n_training = config_dict.get('n_training', 0) # only for samples per subgroup testing to keep same number of validation data throughout tests
# n_testing = config_dict.get('n_testing', 0) # only for samples per subgroup testing to keep same number of validation data throughout tests

X_train_unscaled, X_test_unscaled, y_train, y_test = funcs.train_test_split(data_path, number_of_random_states, number_for_testing)
# X_train_unscaled, X_test_unscaled, y_train, y_test = funcs.train_test_split_number(data_path, number_of_random_states, n_training, n_testing) # only for samples per subgroup testing to keep same number of validation data throughout tests

if data_end_index == -1:
    data_end_index = X_train_unscaled.shape[1]+1

#^ slice training, testing, expt data to region of interest
X_train_unscaled = X_train_unscaled[:, data_start_index:data_end_index]
X_test_unscaled = X_test_unscaled[:, data_start_index:data_end_index]
if expt_data.ndim > 1:
    X_expt_unscaled = expt_data[:,data_start_index:data_end_index]
else:
    X_expt_unscaled = expt_data[data_start_index:data_end_index]
    X_expt_unscaled = X_expt_unscaled.unsqueeze(0)  # Add a dimension if it's a single tensor

expt_x_axis = expt_x_axis[data_start_index:data_end_index]
print('Loaded training, testing, and experimental data.\n')

print('Unscaled training, testing, and experimental data.\n')
print(f'X_train MAX:{torch.max(X_train_unscaled):.2f}\nX_train MIN:{torch.min(X_train_unscaled):.2f}\n')
print(f'X_test  MAX:{torch.max(X_test_unscaled):.2f}\nX_test  MIN:{torch.min(X_test_unscaled):.2f}\n')
print(f'X_expt  MAX:{torch.max(X_expt_unscaled):.2f}\nX_expt  MIN:{torch.min(X_expt_unscaled):.2f}\n\n')
print(f'X_train shape: {X_train_unscaled.shape}\nX_test shape: {X_test_unscaled.shape}\nX_expt shape: {X_expt_unscaled.shape}\n')

#============================================================================================
#* Scale data
#============================================================================================
scale_data = 100


expt_scale = config_dict.get('expt_scale_factor', 1.0)

X_train = X_train_unscaled
X_test  = X_test_unscaled 
X_expt  = X_expt_unscaled * expt_scale

if scaling_method == 'MinMax2d':
    X_train, X_test, X_expt = funcs.MinMaxScale_2d(X_train, X_test, X_expt, scale=scale_data)
    
if scaling_method == 'MaxAbs':
    X_train, X_test, X_expt = funcs.MaxAbsScale(X_train, X_test, X_expt)
    X_train = X_train * scale_data
    X_test = X_test * scale_data
    X_expt = X_expt * scale_data

if scaling_method == 'Sqrt':
    X_train, X_test, X_expt = funcs.sqrt_scale_diffraction(X_train, X_test, X_expt)

xexpt_save = X_expt.squeeze()
xexpt_filename = os.path.join(cwd, material_name, f'X_expt_maxabs_{material_name}_{phase_name}_expt_1123K.xy')
with open(xexpt_filename, 'w') as f:
    for i in range(xexpt_save.shape[0]):
        f.write(f"{expt_x_axis[i]:.4f} {xexpt_save[i]:.6f}\n")


def zscore_per_pattern(t, eps=1e-8):
    mean = t.mean(dim=1, keepdim=True)
    std = t.std(dim=1, keepdim=True)
    return (t - mean) / (std + eps)

if do_zscore:
    X_train = zscore_per_pattern(X_train)
    X_test  = zscore_per_pattern(X_test)
    X_expt  = zscore_per_pattern(X_expt)

# X_train = zscore_per_pattern(X_train)
# X_test  = zscore_per_pattern(X_test)
# X_expt  = zscore_per_pattern(X_expt)

print('Scaled training, testing, and experimental data.\n')
print(f'X_train MAX:{torch.max(X_train):.2f}\nX_train MIN:{torch.min(X_train):.2f}\n')
print(f'X_test  MAX:{torch.max(X_test):.2f}\nX_test  MIN:{torch.min(X_test):.2f}\n')
print(f'X_expt  MAX:{torch.max(X_expt):.2f}\nX_expt  MIN:{torch.min(X_expt):.2f}\n\n') # type: ignore

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

# Optional: sanity check that every subgroup in ml_data has a DOF entry
for sg in subgroup_ids:
    if sg not in param_mapper:
        raise KeyError(f"Subgroup {sg} from ml_data has no DOF entry in {dof_count_path}")
    
# Per-class DOF tensor for DOF-aware regularization
dof_per_subgroup = torch.zeros(num_classes, dtype=torch.float32, device=device)
for class_idx, sg in label_mapper.items():
    dof_per_subgroup[class_idx] = float(param_mapper[sg])

dof_mean = dof_per_subgroup.mean()
dof_std  = dof_per_subgroup.std().clamp(min=1e-6)
dof_per_subgroup = (dof_per_subgroup - dof_mean) / dof_std

#============================================================================================
#* Package testing and training data with our custom Dataset class : initialize DataLoader
#============================================================================================

train_data = xsym.simulated_data(X_train, y_train)
test_data = xsym.simulated_data(X_test, y_test)

# load training and testing data with torch.DataLoader
train_loader = DataLoader(train_data, batch_size=training_batchsize, shuffle=True)
val_loader = DataLoader(test_data, batch_size=training_batchsize, shuffle=False)


#^============================================================================================
#* Initialize CNN model: Pretrained, Warm Start, or New Model
#^============================================================================================   



#* Initialize model, criterion, optimizer
# model = xsym.CNN_bragg(num_classes=num_classes)
# model = xsym.CNN_GlobalSelfAttention(num_classes=num_classes)
# model = xsym.CNN_LocalAttention(num_classes=num_classes)

# channel_1 = 8
# channel_2 = 16
# kernel_1 = 31
# kernel_2 = 31
# stride_1 = 2
# stride_2 = 2
# num_heads = 8
# linear_1 = 1024

channel_1 = 16
channel_2 = 64
kernel_1 = 31
kernel_2 = 7
stride_1 = 2
stride_2 = 1
num_heads = 4
linear_1 = 1024


model = xsym.CNN_MultiHeadAttention(
    num_classes=num_classes,
    channel_1=channel_1,
    channel_2=channel_2,
    kernel_1=kernel_1,
    kernel_2=kernel_2,
    stride_1=stride_1,
    stride_2=stride_2,
    num_heads=num_heads,
    linear_1=linear_1
)

# attention embedding dim must be divisible by num_heads
if model.channel_2 % model.num_heads != 0:
    print(
        f"Invalid MHA configuration "
        f"with channel_2 = {model.channel_2} and num_heads = {model.num_heads}.\n"
        f"Requirement -> channel_2 % num_heads = 0\n"
    )
    raise ValueError("Invalid MHA configuration.")

learning_rate = 0.001
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# scheduler = MultiStepLR(optimizer, milestones=scheduler_milestones, gamma=0.1)
# scheduler = ReduceLROnPlateau(optimizer)

# Move model to device first
model.to(device)
criterion = criterion.to(device)
model.eval()

#* Load Pretrained Model
if load_pretrained_model:
    checkpoint = torch.load(pretrain_path, map_location=device)
    model.load_state_dict(checkpoint, strict=True)
    # optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

# #* Load Pretrained Model
# if load_pretrained_model:
#     checkpoint = torch.load(pretrain_path, map_location=device)
#     model.load_state_dict(checkpoint["model_state_dict"], strict=True)
#     optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

#* Warmstart Model
if warmstart_model:
    checkpoint = torch.load(warmstart_path, map_location=device)
    model.load_state_dict(checkpoint, strict=False)

# Ensure the model is properly initialized before wrapping in DataParallel
model.eval()
dummy_input = torch.randn(1, X_train.shape[1]).to(device)
model(dummy_input)  # Run a forward pass to initialize lazy layers

#* Wrap in DataParallel if multiple GPUs are available
if torch.cuda.device_count() > 1:
    print(f"\nUsing {torch.cuda.device_count()} GPUs!\n")
    model = nn.DataParallel(model, device_ids=logical_gpus)

# Move model to device again to ensure it's properly placed
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=scheduler_patience, min_lr=1e-5)
model.to(device)
# print(model)

#============================================================================================
# * Train & Test model
#============================================================================================  
epochs_completed = 0 
# Training loop with early stopping
if train_model == True:
    best_val_loss = float('inf')
    best_val_acc = 0
    stop_counter = 0

    
    for epoch in range(max_epochs):
        # ------------------ Training ------------------
        model.train()
        running_loss = 0.0
        tr_correct1 = tr_correct3 = tr_correct5 = tr_total = 0
        epochs_completed += 1

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            preds = model(inputs)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            tr_total += labels.size(0)

            # Top-1, Top-3, Top-5 training accuracy (for monitoring)
            tr_correct1 += (preds.argmax(1) == labels).sum().item()
            tr_correct3 += (preds.topk(3, dim=1).indices == labels.unsqueeze(1)).any(dim=1).sum().item()
            tr_correct5 += (preds.topk(5, dim=1).indices == labels.unsqueeze(1)).any(dim=1).sum().item()

        avg_tr_loss = running_loss / tr_total
        tr_acc1 = tr_correct1 / tr_total
        

        # ------------------ Validation ------------------
        model.eval()
        val_running_loss = 0.0
        val_correct1 = val_correct3 = val_correct5 = val_total = 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                preds = model(inputs)
                val_loss = criterion(preds, labels)
                val_running_loss += val_loss.item() * inputs.size(0)

                # Top-1, Top-3, Top-5 validation accuracy
                val_correct1 += (preds.argmax(1) == labels).sum().item()
                val_correct3 += (preds.topk(3, dim=1).indices == labels.unsqueeze(1)).any(dim=1).sum().item()
                val_correct5 += (preds.topk(5, dim=1).indices == labels.unsqueeze(1)).any(dim=1).sum().item()
                val_total += labels.size(0)

                all_preds.extend(preds.argmax(1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_acc1 = val_correct1 / val_total
        val_acc3 = val_correct3 / val_total
        val_acc5 = val_correct5 / val_total
        avg_val_loss = val_running_loss / val_total
        # scheduler.step(val_acc1)
        previous_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        if current_lr < previous_lr:
            print(f"\n[Scheduler] Learning rate reduced from {previous_lr:.1e} to {current_lr:.1e}")

        # Compute F1 scores
        f1_macro = f1_score(all_labels, all_preds, average='macro')
        precision, recall, f1_micro, _ = precision_recall_fscore_support(all_labels, all_preds, average='micro')
        # confusion_matrix = multilabel_confusion_matrix(all_labels, all_preds)

        # print(f'Epoch {epoch+1}: Top-1 Acc: {val_acc1:.2f}, Top-3 Acc: {val_acc3:.2f}, Top-5 Acc: {val_acc5:.2f}, Tr-Loss: {running_loss/len(train_loader):.4f}, Val-Loss: {val_running_loss/len(val_loader):.4f}')
        print(
        f"Epoch {epoch}: \n"
        f"Top-1, 3, 5 Acc: {val_acc1:.2f}, {val_acc3:.2f}, {val_acc5:.2f}\n"
        f"F1_macro: {f1_macro:.3f}, "
        f"Tr-Loss: {avg_tr_loss:.4f}, "
        f"Val-Loss: {avg_val_loss:.4f}\n"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            stop_counter = 0
            # torch.save(model.state_dict(), path_to_save_model)
            # Check if model is DataParallel, if so, save model.module
            state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(state_dict, path_to_save_model)

            print(f"Epoch {epoch}: New best loss {avg_val_loss:.4f}. Saved model.\n")
        else:
            stop_counter += 1
            print(f"Epoch {epoch}: No improvement. Stop counter: {stop_counter}/{early_stop_patience}\n")
            
        if stop_counter >= early_stop_patience:
            print("Early stopping triggered.\n")
            break

        # if val_acc1 > best_val_acc:
        #     best_val_acc = val_acc1
        #     stop_counter = 0
        #     # torch.save(model.state_dict(), path_to_save_model)
        #     # Check if model is DataParallel, if so, save model.module
        #     state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
        #     torch.save(state_dict, path_to_save_model)

        #     print(f"Epoch {epoch}: New best accuracy {val_acc1:.4f}. Saved model.")
        # else:
        #     stop_counter += 1
        #     print(f"Epoch {epoch}: No improvement. Stop counter: {stop_counter}/{early_stop_patience}")
            
        # if stop_counter >= early_stop_patience:
        #     print("Early stopping triggered.")
        #     break

# ============================================================================================
#^ Run Validation
# ============================================================================================
print("\nRunning validation on test set...")

if train_model or load_pretrained_model:
    if not load_pretrained_model:
        # Re-initialize model for evaluation with best weights
        model = xsym.CNN_MultiHeadAttention(
            num_classes=num_classes,
            channel_1=channel_1,
            channel_2=channel_2,
            kernel_1=kernel_1,
            kernel_2=kernel_2,
            stride_1=stride_1,
            stride_2=stride_2,
            num_heads=num_heads,
            linear_1=linear_1
        )
        model.to(device)
        # Ensure the model is properly initialized before wrapping in DataParallel
        model.eval()
        dummy_input = torch.randn(1, X_train.shape[1]).to(device)
        model(dummy_input)  # Run a forward pass to initialize lazy layers
        # load best model weights
        state_dict = torch.load(path_to_save_model, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        #* Wrap in DataParallel if multiple GPUs are available
        if torch.cuda.device_count() > 1:
            print(f"\nUsing {torch.cuda.device_count()} GPUs!\n")
            model = nn.DataParallel(model, device_ids=logical_gpus)
            model.to(device)


number_of_top_results = min(10, num_classes)
total = 0
correct_top_k = torch.zeros(number_of_top_results, device=device)  # Store counts for Top-1 to Top-k
#* Dictionary to store misclassified labels

misclassified_counts = defaultdict(int)

with torch.no_grad():
    val_running_loss = 0.0
    val_correct1 = val_correct3 = val_correct5 = val_total = 0
    all_preds, all_labels = [], []
    for inputs, labels in val_loader:
        model.eval()
        inputs, labels = inputs.to(device), labels.to(device)
        predicted = model(inputs)
        total += labels.size(0)
        val_loss = criterion(predicted, labels)
        val_running_loss += val_loss.item() * inputs.size(0)

        # Top-1, Top-3, Top-5 validation accuracy
        val_correct1 += (predicted.argmax(1) == labels).sum().item()
        val_correct3 += (predicted.topk(3, dim=1).indices == labels.unsqueeze(1)).any(dim=1).sum().item()
        val_correct5 += (predicted.topk(5, dim=1).indices == labels.unsqueeze(1)).any(dim=1).sum().item()
        val_total += labels.size(0)

        all_preds.extend(predicted.argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
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
    
# Compute F1 scores
f1_macro = f1_score(all_labels, all_preds, average='macro')
precision, recall, f1_micro, _ = precision_recall_fscore_support(all_labels, all_preds, average='micro')

val_acc1 = val_correct1 / val_total
val_acc3 = val_correct3 / val_total
val_acc5 = val_correct5 / val_total
avg_val_loss = val_running_loss / val_total
print(
f"Validation Results: \n"
f"Top-1, 3, 5 Acc: {val_acc1:.2f}, {val_acc3:.2f}, {val_acc5:.2f}\n"
f"Val-Loss: {avg_val_loss:.4f}\n"
)
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

# Print misclassification summary
# print("\nMisclassification Summary (Labels not in Top 10):")
# for label, count in misclassified_counts.items():
#     class_name = label_mapper.get(label, f"Unknown ({label})")
#     print(f"Class '{class_name}' (Label {label}): {count} times")

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
    number_of_top_results = min(10, num_classes)
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
    
    config_lines = []
    with open(CONFIG_FILE, 'r') as f:
        config_lines = f.readlines()

    with open(os.path.join(cwd, material_name, f'zz_{os.path.splitext(os.path.basename(path_to_save_model))[0]}.txt'), 'w') as f:
        f.write(f'material: {material_name}\n')
        f.write(f'phase: {phase_name}\n')
        f.write(f'Experimental Data: {experimental_names}\n\n')
        f.write(f'Top Predicted Subgroups, Logits, DOF Count:\n')
        for i in range(len(predicted_subgroups)):
            f.write(f'{predicted_subgroups[i]}, {top_pred_sg[i]:<6}, {top_pred_HM[i]:<8}, {top_logits[i]}, {predicted_subgroups_dof[i]}\n')
        f.write(f'\nModel Information:\n')
        f.write(f'epochs_completed: {epochs_completed}\n')
        f.write(f'Training Batch Size: {training_batchsize}\n')
        f.write(f'Training Data: {data_path}\n')
        f.write(f'Data Start/End Indices: {data_start_index} to {data_end_index}\n')
        f.write(f'MinMax2d Scale Factor: {scale_data}\n')
        f.write(f'Saved Model: {path_to_save_model}\n')
        if warmstart_model:
            f.write(f'Warmstarted From: {warmstart_path}\n')
        if load_pretrained_model:
            f.write(f'Pretrained From: {pretrain_path}\n')
        f.write(f'\nValidation Results:\n')
        f.write(f'Total Validation Samples: {total}\n')
        f.write(f'F1 (micro): {f1_micro:.3f}\n')
        f.write(f'F1 (macro): {f1_macro:.3f}\n')
        f.write(f'Precision: {precision:.3f}\n')
        f.write(f'Recall: {recall:.3f}\n')
        f.write(f'Top-1 Accuracy: {val_acc1:.2f}\n')
        f.write(f'Top-3 Accuracy: {val_acc3:.2f}\n')
        f.write(f'Top-5 Accuracy: {val_acc5:.2f}\n')

        f.write(f'\nValidation Top-k Accuracy: Prediction Rank (k), % Accuracy \n')
        for i in range(len(validation_list)):
            f.write(f'Top {validation_list[i][0]} accuracy, {validation_list[i][1]} \n')

        f.write(f'\nRuntime: {(t1-t0)/60} minutes\n')
        f.write(f'Number of GPUs Used: {len(logical_gpus)}\n')
        f.write(f'\nModel Structure:\n {str(model)}\n')
        f.write(f'\nConfiguration File Contents:\n')
        for line in config_lines:
            f.write(line)

        # f.write(confusion_matrix.__str__())
        

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
        X_train_dummy, cam_data_scaled, X_expt_dummy = funcs.MinMaxScale_2d(X_train_unscaled, cam_data, X_expt_unscaled, scale=scale_data)
        cam_data_scaled = torch.Tensor(cam_data_scaled).to(device)

        # Subset training data for target class (e.g., class 5)
        input_batch = cam_data_scaled  # shape: (N, seq_len)
        target_class = subgroup_to_label.get(tsg)  # Convert subgroup name to label index

        cam_save_name = f'{cam_save_basename}_{tsg}.png'

        # Compute average CAM
        avg_cam = grad_cam.generate_average_cam(input_batch, target_class)
        # Plot over experimental pattern
        # xsym.plot_avg_cam_with_expt(xrd_expt=X_expt.cpu().numpy(), 
        #                             avg_cam=avg_cam, 
        #                             x_axis=expt_x_axis, 
        #                             x_units = cam_x_units_label, 
        #                             save_name= cam_save_name, 
        #                             legend_label = cam_legend_label)
        
        xsym.plot_avg_cam_with_expt_zoom_in(xrd_expt=X_expt.cpu().numpy(), 
                                    avg_cam=avg_cam, 
                                    x_axis=expt_x_axis, 
                                    x_units = cam_x_units_label, 
                                    save_name= cam_save_name, 
                                    legend_label = cam_legend_label)