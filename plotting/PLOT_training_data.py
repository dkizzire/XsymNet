from xsymnet import xsymnet_funcs as funcs
import os
import torch
import glob
import re
import matplotlib.pyplot as plt

cwd = os.getcwd()
#=======================================================================================================
#* User parameters
#=======================================================================================================
material_name = 'bso'          # e.g., 'bso'
phase_name    = 'beta_5'      # e.g., 'beta_5'
data_start_index = 0         # e.g., 200
data_end_index   = -1        # e.g., 1200
scaling_method   = 'MinMax2d'  # 'MinMax2d'
percent_for_testing = 0.2      # e.g., 0.2 for
subgroup_id_to_plot = '0152'  # e.g., '0152' (for plotting later)
random_state_to_plot = 4     # e.g., 3 (for plotting later)
#=======================================================================================================
#* Load experimental data
#=======================================================================================================

experimental_files = sorted(glob.glob(os.path.join(cwd, material_name, 'experimental_refinement', 'beta_exp_xy.xy')))
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
data_folder = os.path.join(cwd, material_name, 'ml_data', phase_name)
data_files = sorted(glob.glob(os.path.join(data_folder, '*.pt')))
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
number_for_training = number_of_random_states - number_for_testing

X_train_unscaled, X_test_unscaled, y_train, y_test = funcs.train_test_split(data_folder, number_of_random_states, number_for_testing)

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

#============================================================================================
#* Scale data
#============================================================================================
scale_data = 100

X_train = X_train_unscaled
X_test  = X_test_unscaled
X_expt  = X_expt_unscaled

if scaling_method == 'MinMax2d':
    X_train, X_test, X_expt = funcs.MinMaxScale_2d(X_train, X_test, X_expt, scale=scale_data)
if scaling_method == 'MaxAbs':
    X_train, X_test, X_expt = funcs.MaxAbsScale(X_train, X_test, X_expt)

print('SCALED training, testing, and experimental data.\n')
print(f'X_train MAX:{torch.max(X_train):.2f}\nX_train MIN:{torch.min(X_train):.2f}\n')
print(f'X_test  MAX:{torch.max(X_test):.2f}\nX_test  MIN:{torch.min(X_test):.2f}\n')
print(f'X_expt  MAX:{torch.max(X_expt):.2f}\nX_expt  MIN:{torch.min(X_expt):.2f}\n\n')


def zscore_per_pattern(t, eps=1e-8):
    mean = t.mean(dim=1, keepdim=True)
    std = t.std(dim=1, keepdim=True)
    return (t - mean) / (std + eps)

# calculate z-score normalized data after MinMax scaling
X_train_z = zscore_per_pattern(X_train)
X_test_z  = zscore_per_pattern(X_test)
X_expt_z  = zscore_per_pattern(X_expt)

# calculate z-score normalized data before MinMax scaling
X_train_unscaled_z = zscore_per_pattern(X_train_unscaled)
X_test_unscaled_z  = zscore_per_pattern(X_test_unscaled)
X_expt_unscaled_z  = zscore_per_pattern(X_expt_unscaled)

print('Z-score training, testing, and experimental data.\n')
print(f'X_train MAX:{torch.max(X_train):.2f}\nX_train MIN:{torch.min(X_train):.2f}\n')
print(f'X_test  MAX:{torch.max(X_test):.2f}\nX_test  MIN:{torch.min(X_test):.2f}\n')
print(f'X_expt  MAX:{torch.max(X_expt):.2f}\nX_expt  MIN:{torch.min(X_expt):.2f}\n\n')

#============================================================================================
#* Plot comparison: Expt vs Training (scaled and z-score)
#============================================================================================
# Calculate tr_id from subgroup_id_to_plot and random_state_to_plot
# Since subgroup IDs may not be continuous (e.g., 0001, 0002, 0005),
# we use subgroup_to_label to get the correct class index
class_label = subgroup_to_label[subgroup_id_to_plot]
tr_id = class_label * number_for_training + random_state_to_plot

print(f"\nPlotting comparison:")
print(f"  Subgroup ID: {subgroup_id_to_plot} -> class label: {class_label}")
print(f"  Random state: {random_state_to_plot}")
print(f"  Training index (tr_id): {tr_id}")

# Convert x-axis to numpy for plotting
x_axis = expt_x_axis.numpy() if isinstance(expt_x_axis, torch.Tensor) else expt_x_axis

# Create figure with 4 subplots
fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

# Plot 1: X_expt (scaled experimental data)
axes[0].plot(x_axis, X_expt[0].numpy(), 'b-', linewidth=0.8)
axes[0].set_ylabel('Intensity')
axes[0].set_title(f'Experimental Data (scaled)')
axes[0].grid(True, alpha=0.3)

# Plot 2: X_expt_z (z-score normalized experimental data)
axes[1].plot(x_axis, X_expt_z[0].numpy(), 'b-', linewidth=0.8)
axes[1].set_ylabel('Z-Score')
axes[1].set_title(f'Experimental Data (z-score)')
axes[1].grid(True, alpha=0.3)

# Plot 3: X_train[tr_id] (scaled training data)
axes[2].plot(x_axis, X_train[tr_id].numpy(), 'r-', linewidth=0.8)
axes[2].set_ylabel('Intensity')
axes[2].set_title(f'Training Data: subgroup {subgroup_id_to_plot}, random state {random_state_to_plot} (scaled)')
axes[2].grid(True, alpha=0.3)

# Plot 4: X_train_z[tr_id] (z-score normalized training data)
axes[3].plot(x_axis, X_train_z[tr_id].numpy(), 'r-', linewidth=0.8)
axes[3].set_ylabel('Z-Score')
axes[3].set_title(f'Training Data: subgroup {subgroup_id_to_plot}, random state {random_state_to_plot} (z-score)')
axes[3].grid(True, alpha=0.3)
axes[3].set_xlabel('TOF (microseconds)')

# # Plot 5: X_expt_unscaled_z (z-score normalized unscaled experimental data)
# axes[4].plot(x_axis, X_expt_unscaled_z[0].numpy(), 'b-', linewidth=0.8)
# axes[4].set_ylabel('Z-Score')
# axes[4].set_title(f'Experimental Unscaled Data (z-score)')
# axes[4].grid(True, alpha=0.3)

# # Plot 4: X_train_unscaled_z[tr_id] (z-score normalized unscaled training data)
# axes[5].plot(x_axis, X_train_unscaled_z[tr_id].numpy(), 'r-', linewidth=0.8)
# axes[5].set_ylabel('Z-Score')
# axes[5].set_title(f'Training Data: subgroup {subgroup_id_to_plot}, random state {random_state_to_plot} (z-score)')
# axes[5].grid(True, alpha=0.3)
# # axes[5].set_xlabel('TOF (microseconds)')

plt.tight_layout()

# Save figure
save_path = os.path.join(cwd, f'comparison_expt_vs_train_{material_name}_{phase_name}_{subgroup_id_to_plot}_rs{random_state_to_plot}.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"\nSaved plot to: {save_path}")

plt.show()

#============================================================================================
#* Plot 2: Max intensity per class from training data
#============================================================================================
# Calculate max intensity for each class from unscaled training data
max_intensities = []
for class_idx in range(num_classes):
    start_idx = class_idx * number_for_training
    end_idx = (class_idx + 1) * number_for_training
    class_data = X_train_unscaled[start_idx:end_idx]
    max_val = torch.max(class_data).item()
    max_intensities.append(max_val)

# # Create bar plot
# fig2, ax2 = plt.subplots(figsize=(max(10, num_classes * 0.5), 6))

# x_positions = range(num_classes)
# bars = ax2.bar(x_positions, max_intensities, color='steelblue', edgecolor='black', linewidth=0.5)

# # Use subgroup IDs as x-tick labels
# ax2.set_xticks(x_positions)
# ax2.set_xticklabels(subgroup_ids, rotation=90, ha='right')

# ax2.set_xlabel('Subgroup ID')
# ax2.set_ylabel('Max Intensity (unscaled)')
# ax2.set_title(f'Max Intensity per Class - {material_name}/{phase_name}')
# ax2.grid(True, alpha=0.3, axis='y')

# # Add experimental max as horizontal line for reference
# expt_max = torch.max(X_expt_unscaled).item()
# ax2.axhline(y=expt_max, color='red', linestyle='--', linewidth=1.5, label=f'Expt max: {expt_max:.2f}')
# ax2.legend()

# plt.tight_layout()

# # Save figure
# save_path2 = os.path.join(cwd, f'max_intensity_per_class_{material_name}_{phase_name}.png')
# plt.savefig(save_path2, dpi=150, bbox_inches='tight')
# print(f"\nSaved plot to: {save_path2}")

# plt.show()

#============================================================================================
#* Plot 3: Max intensity per class from training data
#============================================================================================
fig3, ax3 = plt.subplots(figsize=(12, 10))
x_positions = range(num_classes)
ax3.scatter(x_positions, max_intensities, color='darkorange', s=100, edgecolor='black')
plt.show()