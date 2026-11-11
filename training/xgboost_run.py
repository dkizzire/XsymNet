import os
import glob
import time
import numpy as np
import pandas as pd
import json
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR, StepLR, MultiStepLR, LRScheduler, ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import MaxAbsScaler, StandardScaler, QuantileTransformer, RobustScaler, MinMaxScaler, Normalizer
from collections import OrderedDict, defaultdict
import matplotlib.pyplot as plt

from io_utils import utility_functions as util
from xsymnet import xsymnet_funcs as funcs
from xsymnet import xsymnet_classes as xsym

# import xgboost as xgb
from xsymnet import xgboost_class as xgbc
import cupy as cp


t0 = time.time()
cwd = os.getcwd()

#============================================================================================
# Determine if running on GPU or CPU and set tensor map locations
#============================================================================================

# Set available GPUs
selected_gpus = [4]
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

#* Slice data to region of interest
data_start_index = 0 #70  #661 #TOF 23750 for NVPF  ...... # 1181 TOF 36000  for PZT_90  ....... TOF 60000 i=5390 for BSO
data_end_index = -1 #6765 #:6061-110k:# #:6765-118k:# #3841

percent_for_testing = 0.1 # 0.2 for 20% testing, 80% training

train_model = True
path_to_save_model = f'/netdisk/dx6/saved_models/{material_name}_{phase_name}_xgb_01.pt'

#* Classify Experimental Data
classify_experiment = True
# experimental_files = sorted(glob.glob(os.path.join(cwd, material_name, 'experimental_p1s', '*.xy')))
experimental_files = sorted(glob.glob(os.path.join(cwd, material_name, 'experimental_p1s', 'P1_wo3_1043K_xy.xy')))
assert experimental_files, f"Experimental file(s) not found in {os.path.join(cwd, material_name, 'experimental_p1s')} \n --> Check path and file name. <--"
experimental_names = [os.path.splitext(os.path.basename(f))[0] for f in experimental_files]


#^ convert diffraction xy to tensor (.pt)
for i in range(len(experimental_files)):    
    if experimental_files[i][-3:] != '.pt':
        converted_expt_data, expt_x_axis = util.convert_bragg_to_tensors(os.path.join(cwd, material_name, 'experimental_p1s', experimental_files[i]))
        new_filename = f'{os.path.splitext(experimental_files[i])[0]}.pt'
        torch.save(converted_expt_data, os.path.join(cwd, material_name, 'experimental_p1s', new_filename))
        experimental_files[i] = new_filename

#============================================================================================
#* create dof counts for topas strs and create label mappers
#============================================================================================

dof_count_path = os.path.join(cwd, material_name, f'dof_counts_{material_name}.json')
topas_strs_path = os.path.join(cwd, material_name, f'topas_strs_{material_name}')
util.count_dofs_topas_strs_dir(topas_strs_path, dof_count_path)

with open(dof_count_path, 'r') as f:
    param_mapper = json.load(f)

class_names = list(param_mapper.keys())

#* Create label mapper
label_mapper = {i: class_names[i] for i in range(len(class_names))} 
#============================================================================================
#* Load training & testing data
#============================================================================================

#^ Load training and testing data as tensors 
data_folder = os.path.join(cwd, material_name, 'ml_data', phase_name)
data_files = sorted(glob.glob(os.path.join(data_folder, '*.pt')))
number_of_classes = len(data_files)
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

print(f'Expt_MAX: {expt_data_unscaled.max():.2f} \nExpt_MIN: {expt_data_unscaled.min():.2f}\n')

#============================================================================================
#* Scale data & send to GPU
#============================================================================================
scale_data = 100
X_train, X_test, X_expt = funcs.MinMaxScale_2d(X_train_unscaled, X_test_unscaled, expt_data_unscaled, scale=scale_data)

# X_train = X_train_unscaled
# X_test = X_test_unscaled
# X_expt = expt_data_unscaled

X_train = torch.Tensor(X_train).to(device)
X_test = torch.Tensor(X_test).to(device)
X_expt = torch.Tensor(X_expt).to(device)

y_train = y_train.to(device)
y_test = y_test.to(device)

X_train = X_train.cpu().numpy()
X_test = X_test.cpu().numpy()
X_expt = X_expt.cpu().numpy()
y_train = y_train.cpu().numpy()
y_test = y_test.cpu().numpy()

offset = 0.0001
X_train = np.log(X_train + offset)
X_test = np.log(X_test + offset)
X_expt = np.log(X_expt + offset)

print(f'\nAfter natural log scaling')
print(f'X_train max: {X_train.max():.4f}, min: {X_train.min():.4f}')
print(f'X_test max: {X_test.max():.4f}, min: {X_test.min():.4f}')
print(f'X_expt max: {X_expt.max():.4f}, min: {X_expt.min():.4f}\n')

# ----------------------------
# Initialize wrapper
# ----------------------------
wrapper = xgbc.XGBoostWrapper(
    device='cuda',
    objective="multi:softprob",
    num_class=number_of_classes,
    n_estimators=200,
    learning_rate=0.1,
    max_depth=4,
    random_state=42,
)

# ----------------------------
# Train classifier
# ----------------------------
wrapper.fit(
    X_train,
    y_train,
    eval_set=[(X_test, y_test)],
    early_stopping_rounds=10,
    eval_metric="mlogloss",
    verbose_eval=10
)



# ----------------------------
# Predict
# ----------------------------

y_pred = wrapper.predict(X_test)
proba = wrapper.predict_proba(X_test)


# ----------------------------
# Show validation results
# ----------------------------
# y_pred_reshape = y_pred.reshape(number_of_classes, number_for_testing)
# y_test_reshape = y_test.reshape(number_of_classes, number_for_testing)
# # calculate number incorrect between predicted and true labels
# num_incorrect = []
# for i in range(len(y_pred_reshape)):
#     num_incorrect.append(np.sum(y_pred_reshape[i] != y_test_reshape[i]))

# for i in range(len(y_pred_reshape)):
#     print(f'sg {class_names[i]}: {num_incorrect[i]}')

# print(f'{np.sum(y_pred_reshape != y_test_reshape)} incorrect out of {number_for_testing*number_of_classes} ({(np.sum(y_pred_reshape != y_test_reshape)/(number_for_testing*number_of_classes))*100:.2f}%)')

t1 = time.time()
print(f'\nTraining time: {(t1 - t0)/60:.2f} minutes\n')


evaluation = xgbc.evaluate_classification(y_test, y_pred, y_proba=proba, class_names=class_names)
print("\nEvaluation Metrics:")
print("Top-1 Accuracy:", evaluation["accuracy_top1"])
print("Top-3 Accuracy:", evaluation["accuracy_top3"])
print("Top-5 Accuracy:", evaluation["accuracy_top5"])
print("F1 Macro:", evaluation["f1_macro"])
print("F1 Weighted:", evaluation["f1_weighted"])
print("Precision Macro:", evaluation["precision_macro"])
print("Recall Macro:", evaluation["recall_macro"])

# Optionally, get predicted probabilities

# ----------------------------
# Predict experimental data
# ----------------------------

topk = 5
expt_prob = wrapper.predict_proba(X_expt)
expt_topk = np.argsort(expt_prob, axis=1)[:, -topk:]
topk_labels = [[label_mapper[idx] for idx in row[::-1]] for row in expt_topk][0]  # Reverse to have highest prob first
print(f'\nTop {topk} predicted subgroups for experimental data:')
for rank, label in enumerate(topk_labels, start=1):
    print(f'Rank {rank}: {label}')