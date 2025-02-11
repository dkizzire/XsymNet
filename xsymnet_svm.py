import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
import numpy as np

import time
import os
import glob
import json
import utility_functions as util
import xsymnet_funcs as funcs
import xsymnet_classes as xsym


#============================================================================================
# 1️⃣ CNN Feature Extractor (NO BatchNorm)
#============================================================================================
class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 256, kernel_size=31, stride=15, padding=15)
        self.maxpool_1 = nn.MaxPool1d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv1d(256, 512, kernel_size=31, stride=15, padding=15)
        self.maxpool_2 = nn.MaxPool1d(kernel_size=3, stride=2)
        self.fc1 = nn.LazyLinear(1000)  # Feature representation

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.conv1(x)
        x = self.maxpool_1(x)
        x = self.conv2(x)
        x = self.maxpool_2(x)
        x = torch.flatten(x, 1)  # Flatten before fully connected layers
        x = F.relu(self.fc1(x))  # Extract feature representation
        return x  # Return extracted features


#============================================================================================
# 2️⃣ Load & Prepare Data
#============================================================================================
t0 = time.time()
cwd = os.getcwd()

# ============================================================================================
# Set available GPUs
# ============================================================================================

selected_gpus = [0, 1, 2, 3]
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, selected_gpus))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#============================================================================================
# Input phase
#============================================================================================
material_name = 'wo3'
phase_name = '1103K'
epochs = 250
training_batchsize = 512
confidence_iterations = 100
scheduler_milestones = [175]

train_model = True
path_to_save_model = f'/netdisk/dx6/saved_models/{material_name}_{phase_name}_e{epochs}_1103K_localattention_03.pt'
# path_to_save_model = f'/netdisk/dx6/saved_models/{material_name}_{phase_name}_e{epochs}_1103K_multihead_02.pt'

classify_experiment = True
experimental_p1_filename = 'P1_wo3_1103K_xy.xy'

warmstart_model = False
warmstart_path = f'/netdisk/dx6/saved_models/wo3_1103K_e250_1103K_multihead_01.pt'

load_pretrained_model = False
pretrain_path = f'/netdisk/dx6/saved_models/wo3_1103K_e250_1103K_multihead_01.pt'

ground_truth_cif = '0032' # wo3_1103K

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
data_folder = os.path.join(cwd, material_name, 'ml_data', phase_name)
data_files = sorted(glob.glob(os.path.join(data_folder, '*.pt')))
number_of_classes = len(data_files)
first_file = torch.load(data_files[0])
number_of_random_states = first_file.size()[0]
number_for_testing = int(number_of_random_states*0.1)

data_start_index = 0 #70  #661 #TOF 23750 for NVPF  ...... # 1181 TOF 36000  for PZT_90  ....... TOF 60000 i=5390 for BSO
data_end_index = -1 #3841

expt_data_unscaled = torch.load(os.path.join(cwd, material_name, 'experimental_p1s', experimental_p1_filename))
expt_data_unscaled = expt_data_unscaled[0,data_start_index:data_end_index]
X_expt_unscaled = expt_data_unscaled.unsqueeze(0)

X_train_unscaled, X_test_unscaled, y_train, y_test = funcs.configure_bragg_data_trunc(data_folder, expt_data_unscaled, number_of_random_states, number_for_testing, start_index=data_start_index, end_index=data_end_index)

#============================================================================================
#* Scale data & send to GPU
#============================================================================================
scale_data = 100
X_train, X_test, X_expt = funcs.MinMaxScale_2d(X_train_unscaled, X_test_unscaled, X_expt_unscaled, scale=scale_data)

# # Split into train & test sets
# X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# # Create DataLoaders
# batch_size = 32
# train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

train_data = xsym.simulated_data(X_train, y_train)
test_data = xsym.simulated_data(X_test, y_test)

# load training and testing data with torch.DataLoader
train_loader = DataLoader(train_data, batch_size=training_batchsize, shuffle=True)
test_loader = DataLoader(test_data, batch_size=training_batchsize, shuffle=False)

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


#============================================================================================
# 3️⃣ Train CNN
#============================================================================================
def train_cnn(model, dataloader, optimizer, loss_fn, epochs=5):
    model.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(epochs):
        for X_batch, _ in dataloader:  # We only extract features, so ignore labels
            X_batch = X_batch.to(device)
            optimizer.zero_grad()
            _ = model(X_batch)  # Forward pass
            loss = torch.tensor(0.0, requires_grad=True)  # No classification loss
            loss.backward()
            optimizer.step()

# Initialize and train CNN
cnn_model = CNNFeatureExtractor()
optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)
train_cnn(cnn_model, train_loader, optimizer, loss_fn=None, epochs=5)

#============================================================================================
# 4️⃣ Extract Features from CNN
#============================================================================================
def extract_features(model, dataloader):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    features, labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            deep_features = model(X_batch).cpu().numpy()  # Extract features
            features.extend(deep_features)
            labels.extend(y_batch.numpy())

    return np.array(features), np.array(labels)

# Extract training & test features
X_train_features, y_train_labels = extract_features(cnn_model, train_loader)
X_test_features, y_test_labels = extract_features(cnn_model, test_loader)

#============================================================================================
# 5️⃣ Train SVM on Extracted Features
#============================================================================================
svm_classifier = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1, gamma="scale"))
svm_classifier.fit(X_train_features, y_train_labels)

# Evaluate SVM
accuracy = svm_classifier.score(X_test_features, y_test_labels)
print(f"\n🚀 SVM Accuracy on Test Set: {accuracy:.4f}\n")
