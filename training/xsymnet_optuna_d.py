import os

# === Single source of truth for the seed ===
SEED = 100  # 46457#42#1337

# --- Determinism-related env vars BEFORE importing torch ---
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# Disable fused/flash attention kernels that can be nondeterministic
os.environ["PYTORCH_CUDA_FUSED_ATTENTION"] = "0"
os.environ["FLASH_ATTENTION"] = "0"

import glob
import time
import json
import random
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
from torch.utils.data import DataLoader
from collections import OrderedDict, defaultdict
from sklearn.metrics import f1_score, multilabel_confusion_matrix, precision_recall_fscore_support

from io_utils import utility_functions as util
from xsymnet import xsymnet_funcs as funcs
from xsymnet import xsymnet_classes_d as xsym

import optuna
import re  # for subgroup id parsing

#============================================================================================
# Global timing, cwd
#============================================================================================
t0 = time.time()
cwd = os.getcwd()

#============================================================================================
# Global seeding & deterministic settings
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
torch.use_deterministic_algorithms(True, warn_only=False)

# Disable TF32 (can introduce nondeterministic matmul kernels on some GPUs)
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

# For extra determinism on CPU
# In interactive environments (Spyder, Jupyter) parallel work may already be started,
# so we guard these to avoid RuntimeError on re-runs.
try:
    torch.set_num_threads(1)
except RuntimeError as e:
    print(f"[warn] torch.set_num_threads(1) failed: {e}")

try:
    # This one is especially picky about being called before any parallel work
    torch.set_num_interop_threads(1)
except (AttributeError, RuntimeError) as e:
    print(f"[warn] torch.set_num_interop_threads(1) skipped: {e}")

#============================================================================================
# Determine if running on GPU or CPU and set tensor map locations
#============================================================================================

# Set available GPUs (single GPU for reproducibility)
selected_gpus = [1,2]  # , 4, 5, 6, 7

os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, selected_gpus))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'\nUsing device: {device}\n')
logical_gpus = list(range(len(selected_gpus)))
print(f'Selected GPU IDs: {selected_gpus}, are mapped to Logical GPU IDs: {logical_gpus}\n')

# This generator will be used for deterministic DataLoader shuffling
data_loader_generator = torch.Generator()
data_loader_generator.manual_seed(seed)

#^============================================================================================
#* Input phase
#^============================================================================================
material_name = 'wo3'  # 'bso'
phase_name = '1043K'       # 'alpha_5'

epochs = 150
training_batchsize = 128
confidence_iterations = 100
scheduler_milestones = [10, 175]

#* Slice data to region of interest
data_start_index = 0
data_end_index = -1

percent_for_testing = 0.2  # 0.2 for 20% testing, 80% training

train_model = True
path_to_save_model = f'./{material_name}_{phase_name}_e{epochs}_new_optuna_d_01.pt'

#* Classify Experimental Data
classify_experiment = True
experimental_files = sorted(
    glob.glob(os.path.join(cwd, material_name, 'experimental_p1s', 'P1_wo3_1043K_xy.xy'))
)
assert experimental_files, (
    f"Experimental file(s) not found in "
    f"{os.path.join(cwd, material_name, 'experimental_p1s')} \n --> Check path and file name. <--"
)
experimental_names = [os.path.splitext(os.path.basename(f))[0] for f in experimental_files]

#* Warmstart Control
warmstart_model = False
warmstart_path = f'./saved_models/{material_name}_{phase_name}_e{epochs}_multiheadattention_14.pt'

#* Load Pretrained Model Control
load_pretrained_model = False
pretrain_path = f'./saved_models/{material_name}_{phase_name}_e{epochs}_no_mha_avgcam.pt'

#* Create CAM Control
create_cam = True
subgroup_cam_target = '0093'  # pzt_90_300K
cam_save_basename = f'{material_name}_{phase_name}_avgcam_new_optuna_d_01'
cam_x_units_label = 'TOF (μs)'
cam_legend_label = r'PbZr$_{0.1}$Ti$_{0.9}$O$_{3}$'

ground_truth_cif = '0093'  # pzt_90

#^ convert diffraction xy to tensor (.pt)
for i in range(len(experimental_files)):
    if experimental_files[i][-3:] != '.pt':
        converted_expt_data, expt_x_axis = util.convert_bragg_to_tensors(
            os.path.join(cwd, material_name, 'experimental_p1s', experimental_files[i])
        )
        new_filename = f'{os.path.splitext(experimental_files[i])[0]}.pt'
        torch.save(
            converted_expt_data,
            os.path.join(cwd, material_name, 'experimental_p1s', new_filename)
        )
        experimental_files[i] = new_filename

#============================================================================================
#* Load training & testing data
#============================================================================================

#^ Load training and testing data as tensors
data_folder = os.path.join(cwd, material_name, 'ml_data', phase_name)
data_files = sorted(glob.glob(os.path.join(data_folder, '*.pt')))
num_classes = len(data_files)

# Build class-index -> subgroup-id mapping from ml_data filenames
subgroup_ids = []
for fpath in data_files:
    base = os.path.splitext(os.path.basename(fpath))[0]
    # e.g. "cif0001_randomstates_bragg"
    m = re.search(r'(\d{4})', base)
    if not m:
        raise ValueError(f"Could not parse subgroup ID from filename: {base}")
    sg_id = m.group(1)  # "0001"
    subgroup_ids.append(sg_id)

# This is now THE mapping used everywhere
label_mapper = {i: subgroup_ids[i] for i in range(len(subgroup_ids))}  # 0 -> "0001", etc.
subgroup_to_label = {sg: i for i, sg in label_mapper.items()}         # "0001" -> 0, etc.

first_file = torch.load(data_files[0])
number_of_random_states = first_file.size()[0]
number_for_testing = int(number_of_random_states * percent_for_testing)

#^ load experimental data
expt_data = [torch.load(f) for f in experimental_files]
if len(expt_data) > 1:
    expt_data = torch.vstack(expt_data)
else:
    expt_data = expt_data[0]

#^ split data into training and testing sets & create labels
X_train_unscaled, X_test_unscaled, y_train, y_test = funcs.train_test_split(
    data_folder, number_of_random_states, number_for_testing
)

#^ slice training, testing, expt data to region of interest
X_train_unscaled = X_train_unscaled[:, data_start_index:data_end_index]
X_test_unscaled = X_test_unscaled[:, data_start_index:data_end_index]
if expt_data.ndim > 1:
    expt_data_unscaled = expt_data[:, data_start_index:data_end_index]
else:
    expt_data_unscaled = expt_data[data_start_index:data_end_index]
    expt_data_unscaled = expt_data_unscaled.unsqueeze(0)  # Add a dimension if it's a single tensor

expt_x_axis = expt_x_axis[data_start_index:data_end_index]
print('Loaded training, testing, and experimental data.\n')
#============================================================================================
#* Scale data & send to GPU
#============================================================================================
scale_data = 100
# X_train, X_test, X_expt = funcs.MinMaxScale_2d(
#     X_train_unscaled, X_test_unscaled, expt_data_unscaled, scale=scale_data
# )

def normalize_per_pattern(t, eps=1e-8):
    """
    t: (N, L) tensor
    Normalizes each pattern by its max over L.
    """
    max_vals, _ = t.max(dim=1, keepdim=True)
    min_vals, _ = t.min(dim=1, keepdim=True)
    max_vals = max_vals - min_vals
    return (t - min_vals) / (max_vals + eps)
def zscore_per_pattern(t, eps=1e-8):
    mean = t.mean(dim=1, keepdim=True)
    std = t.std(dim=1, keepdim=True)
    return (t - mean) / (std + eps)

X_train = zscore_per_pattern(X_train_unscaled)
X_test =   zscore_per_pattern(X_test_unscaled)
X_expt = zscore_per_pattern(expt_data_unscaled)
# funcs.MinMaxScale_2d(
#     X_train_unscaled, X_test_unscaled, expt_data_unscaled, scale=scale_data
# )





print("\nAfter zscore_per_pattern :")
print(f'X_train MAX:{torch.max(X_train):.2f}\nX_train MIN:{torch.min(X_train):.2f}\n')
print(f'X_test  MAX:{torch.max(X_test):.2f}\nX_test  MIN:{torch.min(X_test):.2f}\n')
print(f'X_expt  MAX:{torch.max(X_expt):.2f}\nX_expt  MIN:{torch.min(X_expt):.2f}\n')

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
# label_mapper: 0 -> "0001", 1 -> "0009", ...
dof_per_subgroup = torch.zeros(num_classes, dtype=torch.float32, device=device)
for class_idx, sg in label_mapper.items():
    dof_per_subgroup[class_idx] = float(param_mapper[sg])

# (optional but usually helps conditioning)
dof_mean = dof_per_subgroup.mean()
dof_std  = dof_per_subgroup.std().clamp(min=1e-6)
dof_per_subgroup = (dof_per_subgroup - dof_mean) / dof_std

#============================================================================================
# Attention regularization: proto-class suppression / entropy (minimal-API)
#============================================================================================

LAMBDA_PROTO = 1e-2   # weight for proto-uniformity regularizer
LAMBDA_ENTROPY = 1e-2   # weight for entropy regularizer (0.0 = off)
LAMBDA_DOF= 1e-2   #weight for physics DOF regularizer
def proto_uniformity_loss(attn_weights):
    """
    Penalize batch-averaged attention drifting toward a fixed template.
    attn_weights: (B, H, Q, K) tensor or None.
    """
    if attn_weights is None:
        return torch.tensor(0.0, device=device)

    # mean over batch dimension -> (H, Q, K)
    mean_attn = attn_weights.mean(dim=0)
    K = mean_attn.size(-1)
    uniform = 1.0 / float(K)

    reg = (mean_attn - uniform).pow(2).mean()
    return reg

def attention_entropy_loss(attn_weights):
    """
    Mean entropy of attention distributions.
    """
    if attn_weights is None:
        return torch.tensor(0.0, device=device)

    eps = 1e-8
    p = attn_weights.clamp(min=eps)
    entropy_per_row = -(p * p.log()).sum(dim=-1)
    return entropy_per_row.mean()

#^============================================================================================
# Optuna Objective with Early Stopping (deterministic, BraggResNetMHA)
#^============================================================================================
def objective_factory(train_loader, val_loader, num_classes, device, logical_gpus):
    def objective(trial):
    
        # ---------------------------------------------------------------------
        # Hyperparameter search space (keep as you have it or tweak separately)
        # ---------------------------------------------------------------------
        # channel_1 = trial.suggest_categorical("channel_1", [32, 64, 96])
        # channel_2 = trial.suggest_categorical("channel_2", [64, 96, 128, 160])
        
        # kernel_1  = trial.suggest_categorical("kernel_1", [11, 15, 19, 23, 27])
        # kernel_2  = trial.suggest_categorical("kernel_2", [3, 5, 7, 9, 11])
        
        # stride_1  = trial.suggest_categorical("stride_1", [1, 2])
        # stride_2  = trial.suggest_categorical("stride_2", [1, 2])
        
        # mha_num_heads = trial.suggest_categorical("mha_num_heads", [2, 4, 8])
        # linear_1      = trial.suggest_categorical("linear_1", [512, 1024, 2048])
        # if (stride_2 == 2):
        #     return objective
        # Hyperparameter search space (keep as you have it or tweak separately)
        # ---------------------------------------------------------------------
        channel_1 = trial.suggest_categorical("channel_1", [64])
        channel_2 = trial.suggest_categorical("channel_2", [96])
        
        kernel_1  = trial.suggest_categorical("kernel_1", [19])
        kernel_2  = trial.suggest_categorical("kernel_2", [5])
        
        stride_1  = trial.suggest_categorical("stride_1", [2])
        stride_2  = trial.suggest_categorical("stride_2", [1])
        
        mha_num_heads = trial.suggest_categorical("mha_num_heads", [8])
        linear_1      = trial.suggest_categorical("linear_1", [512, 1024])
        #attention embedding dim must be divisible by num_heads
        if channel_2 % mha_num_heads != 0:
            print(
                f"Trial {trial.number} pruned due to invalid MHA configuration "
                f"with channel_2 = {channel_2} and num_heads = {mha_num_heads}."
            )
            raise optuna.exceptions.TrialPruned()

        # ---------------------------------------------------------------------
        # Model
        # ---------------------------------------------------------------------
        model = xsym.BraggResNetMHA(
            num_classes=num_classes,
            channel_1=channel_1,
            channel_2=channel_2,
            kernel_1=kernel_1,
            kernel_2=kernel_2,
            stride_1=stride_1,
            stride_2=stride_2,
            num_heads=mha_num_heads,
            linear_1=linear_1,
        )

        print(
            f"\nTrial {trial.number} parameters: "
            f"channel_1={channel_1}, channel_2={channel_2}, "
            f"kernel_1={kernel_1}, kernel_2={kernel_2}, "
            f"stride_1={stride_1}, stride_2={stride_2}, "
            f"mha_num_heads={mha_num_heads}, linear_1={linear_1}"
        )

        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        criterion = criterion.to(device)
        # ---------------------------------------------------------------------
        # Initialize LazyLinear (fc1) with a dummy forward
        # ---------------------------------------------------------------------
        model.eval()
        # dummy_input = torch.randn(1, X_train.shape[1], device=device)
        dummy_input = torch.randn(1, X_train.shape[1]).to(device)
        # with torch.no_grad():
        #     _ = model(dummy_input)
        dummy_input = torch.randn(1, X_train.shape[1]).to(device)
        model(dummy_input)  # Run a forward pass to initialize lazy layers
        # Wrap in DataParallel if multiple GPUs are available
        if torch.cuda.device_count() > 1:
            print(f"\nUsing {torch.cuda.device_count()} GPUs!\n")
            model = nn.DataParallel(model, device_ids=logical_gpus)

        model.to(device)

        # ---------------------------------------------------------------------
        # Training loop with early stopping on Top-1 accuracy
        # ---------------------------------------------------------------------
        best_val_acc = 0.0          # this will now track BEST Top-1
        patience = 5
        counter = 0
        max_epochs = 50

        minibatch_counter = 0
        for epoch in range(max_epochs):
            # ------------------ Training ------------------
            model.train()
            running_loss = 0.0
            tr_correct1 = tr_correct3 = tr_correct5 = tr_total = 0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                print('right before preds')
                preds = model(inputs)
                print(f'preds shape: {preds.shape}, labels shape: {labels.shape}')
                ce_loss = criterion(preds, labels)

                # Attention-based regularization (if model exposes last_attn_weights)
                if isinstance(model, nn.DataParallel):
                    attn_weights = getattr(model.module, "last_attn_weights", None)
                    print(f'Using DataParallel: attn_weights shape {attn_weights.shape if attn_weights is not None else None}')
                else:
                    attn_weights = getattr(model, "last_attn_weights", None)

                if (attn_weights is not None) and LAMBDA_PROTO != 0.0:
                    reg_proto = proto_uniformity_loss(attn_weights)
                else:
                    reg_proto = 0.0

                if (attn_weights is not None) and LAMBDA_ENTROPY != 0.0:
                    reg_ent = attention_entropy_loss(attn_weights)
                else:
                    reg_ent = 0.0

                loss = ce_loss + LAMBDA_PROTO * reg_proto + LAMBDA_ENTROPY * reg_ent
                loss.backward()
                optimizer.step()
                minibatch_counter += 1
                print(f'minibatch {minibatch_counter} loss: {loss.item():.4f}')
                running_loss += loss.item()

                # Top-1, Top-3, Top-5 training accuracy (for monitoring)
                tr_correct1 += (preds.argmax(1) == labels).sum().item()
                tr_correct3 += (
                    preds.topk(3, dim=1).indices == labels.unsqueeze(1)
                ).any(dim=1).sum().item()
                tr_correct5 += (
                    preds.topk(5, dim=1).indices == labels.unsqueeze(1)
                ).any(dim=1).sum().item()
                tr_total += labels.size(0)

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
                    val_running_loss += val_loss.item()

                    # Top-1, Top-3, Top-5 validation accuracy
                    val_correct1 += (preds.argmax(1) == labels).sum().item()
                    val_correct3 += (
                        preds.topk(3, dim=1).indices == labels.unsqueeze(1)
                    ).any(dim=1).sum().item()
                    val_correct5 += (
                        preds.topk(5, dim=1).indices == labels.unsqueeze(1)
                    ).any(dim=1).sum().item()
                    val_total += labels.size(0)

                    all_preds.extend(preds.argmax(1).cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            val_acc1 = val_correct1 / val_total
            val_acc3 = val_correct3 / val_total
            val_acc5 = val_correct5 / val_total

            # F1 metrics
            f1_macro = f1_score(all_labels, all_preds, average='macro')
            precision, recall, f1_micro, _ = precision_recall_fscore_support(
                all_labels, all_preds, average='micro'
            )

            print(
                f"Trial {trial.number} Epoch {epoch}: "
                f"Top-1, 3, 5 Acc: {val_acc1:.2f}, {val_acc3:.2f}, {val_acc5:.2f}, "
                f"F1_macro: {f1_macro:.3f}, "
                f"Tr-Loss: {running_loss/len(train_loader):.4f}, "
                f"Val-Loss: {val_running_loss/len(val_loader):.4f}\n"
            )
            print(
                f"Trial {trial.number} Epoch {epoch}: "
                f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1_micro: {f1_micro:.3f}\n"
            )

            # Report Top-1 to Optuna (this becomes trial.value)
            trial.report(val_acc1, epoch)

            # Early stopping logic on Top-1 accuracy
            if val_acc1 > best_val_acc:
                best_val_acc = val_acc1
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    break

            # Let Optuna prune bad trials
            if trial.should_prune():
                print(f"Trial {trial.number} pruned at epoch {epoch} by Optuna.")
                raise optuna.exceptions.TrialPruned()

        # ---------------------------------------------------------------------
        # Log final metrics to the trial
        # ---------------------------------------------------------------------
        trial.set_user_attr("val_acc_top1", val_acc1)
        trial.set_user_attr("val_acc_top3", val_acc3)
        trial.set_user_attr("val_acc_top5", val_acc5)
        trial.set_user_attr("f1_macro", f1_macro)

        print(
            f"Trial {trial.number} final: "
            f"Top-1,3,5 Acc: {val_acc1:.2f}, {val_acc3:.2f}, {val_acc5:.2f}, "
            f"F1_macro: {f1_macro:.3f}, "
            f"Tr-Loss: {running_loss/len(train_loader):.4f}, "
            f"Val-Loss: {val_running_loss/len(val_loader):.4f}"
        )

        # ---------------------------------------------------------------------
        # Save best model so far by Top-1
        # ---------------------------------------------------------------------
        # trial.study.best_value may not exist for trial 0 → special-case it
        should_save = False
        if trial.number == 0:
            should_save = True
        else:
            try:
                if val_acc1 > trial.study.best_value:
                    should_save = True
            except ValueError:
                # just in case, fall back to saving
                should_save = True

        if should_save:
            model.eval()
            model_save_dict = OrderedDict()
            if isinstance(model, nn.DataParallel):
                model_save_dict["model_state_dict"] = model.module.state_dict()
            else:
                model_save_dict["model_state_dict"] = model.state_dict()

            torch.save(model_save_dict, path_to_save_model)

        # This is what Optuna sees as "trial.value"
        return best_val_acc

    return objective


#! ============================================================================================
#! ============================================================================================
#! Run Optuna Study
#! ============================================================================================
#! ============================================================================================
#============================================================================================
#* Package testing and training data with our custom Dataset class : initialize DataLoader
#============================================================================================

augment = xsym.BraggAugment(
    p=0.7,
    intensity_noise_std=0.02,
    offset_std=0.02,
    shift_bins=1,
    q_scale_eps=0.01  # 1%
)
train_data = xsym.simulated_data(X_train, y_train, transform=augment)
test_data  = xsym.simulated_data(X_test, y_test, transform=None)

# Deterministic DataLoader: generator=data_loader_generator for shuffle=True
train_loader = DataLoader(
    train_data,
    batch_size=training_batchsize,
    shuffle=True,
    generator=data_loader_generator
)
val_loader = DataLoader(
    test_data,
    batch_size=training_batchsize,
    shuffle=False
)

objective = objective_factory(train_loader, val_loader, num_classes, device, logical_gpus)

# Deterministic Optuna sampler
sampler = optuna.samplers.TPESampler(seed=seed)
study = optuna.create_study(direction="maximize", sampler=sampler)
study.optimize(objective, n_trials=2)

print("Best Trial:", study.best_trial.params)
print("Top-1:", study.best_trial.user_attrs["val_acc_top1"])
print("Top-3:", study.best_trial.user_attrs["val_acc_top3"])
print("Top-5:", study.best_trial.user_attrs["val_acc_top5"])
print(f"F1 Score: {study.best_trial.user_attrs['f1_macro']:.3f}")

topt = time.time()
print(f'\nOptuna tuning time: {(topt - t0)/60:.2f} minutes\n')

#! ============================================================================================
#! ============================================================================================
#* Load the best model from the study for final evaluation
#! ============================================================================================
#! ============================================================================================
model = xsym.BraggResNetMHA(
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

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
scheduler = MultiStepLR(optimizer, milestones=scheduler_milestones, gamma=0.1)

previous_epochs = 0
model.eval()
dummy_input = torch.randn(1, X_train.shape[1]).to(device)
model(dummy_input)  # Run a forward pass to initialize lazy layers

#* Wrap in DataParallel if multiple GPUs are available
if torch.cuda.device_count() > 1:
    print(f"\nUsing {torch.cuda.device_count()} GPUs!\n")
    model = nn.DataParallel(model, device_ids=logical_gpus)

model.to(device)

#============================================================================================
# * Train & Test model
#============================================================================================

if train_model:
    best_val_acc = 0
    patience, counter = 7, 0
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        val_running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)

            ce_loss = criterion(outputs, labels)

            if isinstance(model, nn.DataParallel):
                attn_weights = getattr(model.module, "last_attn_weights", None)
            else:
                attn_weights = getattr(model, "last_attn_weights", None)

            reg_proto = proto_uniformity_loss(attn_weights) if LAMBDA_PROTO != 0.0 else 0.0
            reg_ent = attention_entropy_loss(attn_weights) if LAMBDA_ENTROPY != 0.0 else 0.0

            loss = ce_loss + LAMBDA_PROTO * reg_proto + LAMBDA_ENTROPY * reg_ent

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()

        # Validation
        model.eval()
        val_correct1, val_correct3, val_correct5, val_total = 0, 0, 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                preds = model(inputs)
                val_loss = criterion(preds, labels)
                val_running_loss += val_loss.item()
                # Top-1 Top-3 and Top-5
                val_correct1 += (preds.argmax(1) == labels).sum().item()
                val_correct3 += (
                    preds.topk(3, dim=1).indices == labels.unsqueeze(1)
                ).any(dim=1).sum().item()
                val_correct5 += (
                    preds.topk(5, dim=1).indices == labels.unsqueeze(1)
                ).any(dim=1).sum().item()
                val_total += labels.size(0)

                all_preds.extend(preds.argmax(1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_acc1 = val_correct1 / val_total
        val_acc3 = val_correct3 / val_total
        val_acc5 = val_correct5 / val_total

        f1_macro = f1_score(all_labels, all_preds, average='macro')
        precision, recall, f1_micro, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='micro'
        )

        print(f"Epoch {epoch}: Top-1, 3, 5 Acc: {val_acc1:.2f}, {val_acc3:.2f}, {val_acc5:.2f}")
        print(f'F1_macro: {f1_macro:.3f}, F1_micro: {f1_micro:.3f}')
        print(
            f'Tr-Loss: {running_loss/len(train_loader):.4f}, '
            f'Val-Loss: {val_running_loss/len(val_loader):.4f}'
        )

        # Early stopping on Top-5
        if val_acc1 > best_val_acc:
            best_val_acc = val_acc1
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                break

    confusion_matrix = multilabel_confusion_matrix(all_labels, all_preds)

    #!============================================================================================
    #* Save the model and optimizer
    #!============================================================================================
    if load_pretrained_model:
        epochs += previous_epochs

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
number_of_top_results = min(25, num_classes)
total = 0
correct_top_k = torch.zeros(number_of_top_results, device=device)
misclassified_counts = defaultdict(int)

with torch.no_grad():
    for inputs, labels in val_loader:
        model.eval()
        inputs, labels = inputs.to(device), labels.to(device)
        predicted = model(inputs)
        total += labels.size(0)
        top_k_vals, top_k_indices = torch.topk(predicted, number_of_top_results, dim=1)

        correct_matrix = top_k_indices.eq(labels.view(-1, 1).expand_as(top_k_indices))

        for k in range(number_of_top_results):
            correct_top_k[k] += correct_matrix[:, : (k + 1)].sum().item()

        for i in range(len(labels)):
            label = labels[i].item()
            if not correct_matrix[i].any():
                misclassified_counts[label] += 1

print("\nMisclassification Summary (Labels not in Top 10):")
for label, count in misclassified_counts.items():
    class_name = label_mapper.get(label, f"Unknown ({label})")
    print(f"Class '{class_name}' (Label {label}): {count} times")

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

if classify_experiment:
    number_of_top_results = min(10, num_classes)
    param_weight = list(range(number_of_top_results, 0, -1))

    confidence_results = defaultdict(int)
    ranked_confidence_results = defaultdict(int)
    weighted_confidence_results = {}
    logit_results = defaultdict(list)
    predicted_subgroups_dof = []

    with torch.no_grad():
        model.eval()
        X_expt = X_expt.to(device)
        predicted = model(X_expt)
        top_logits, top_indices = torch.topk(predicted, number_of_top_results, dim=1)
        ### FIX: keep top_indices as a 1D Python list of indices
        top_indices = top_indices[0].tolist()
        top_logits = top_logits[0].tolist()
        top_logits = [round(x, 4) for x in top_logits]
        predicted_subgroups = [label_mapper[x] for x in top_indices]

    for key in predicted_subgroups:
        predicted_subgroups_dof.append(param_mapper.get(key))

    top_pred_sg = []
    top_pred_HM = []
    for i in range(len(predicted_subgroups)):
        top_pred_file = glob.glob(
            os.path.join(topas_strs_path, f'*{predicted_subgroups[i]}.str')
        )
        _, pred_sgo = funcs.get_str_info(top_pred_file[0], 'strain')
        top_pred_sg.append(f'{pred_sgo[0]}:{pred_sgo[1]}')
        top_pred_HM.append(f'{pred_sgo[2]}')

    predicted_with_dof = list(zip(predicted_subgroups, predicted_subgroups_dof))
    print(f'\nPredicted Subgroups and DOF Counts: \n')
    print(f'Rank, Subgroup, DOF Count, Logit')
    for i in range(len(predicted_subgroups)):
        print(
            f'{i:<2}: {predicted_subgroups[i]:<4} {top_pred_sg[i]:<6} {top_pred_HM[i]:<8} '
            f'{predicted_subgroups_dof[i]:<4} {top_logits[i]:<8}\n'
        )

    with open(
        os.path.join(
            cwd,
            material_name,
            f'zz_{os.path.splitext(os.path.basename(path_to_save_model))[0]}.txt'
        ), 'w'
    ) as f:
        f.write(f'material: {material_name}\n')
        f.write(f'phase: {phase_name}\n')
        f.write(f'Experimental Data: {experimental_names}\n\n')
        f.write(
            f'Top Predicted Subgroups, Space Group:Origin, Logits, DOF Count:\n'
        )
        for i in range(len(predicted_subgroups)):
            f.write(
                f'{predicted_subgroups[i]}, {top_pred_sg[i]:<6}, '
                f'{top_pred_HM[i]:<8}, {top_logits[i]}, {predicted_subgroups_dof[i]}\n'
            )
        f.write(f'\nModel Information:\n')
        f.write(f'Epochs: {epochs}\n')
        f.write(f'Training Batch Size: {training_batchsize}\n')
        f.write(f'Training Data: {data_folder}\n')
        f.write(
            f'Data Start/End Indices: {data_start_index} to {data_end_index}\n'
        )
        f.write(f'MinMax2d Scale Factor: {scale_data}\n')
        f.write(f'Saved Model: {path_to_save_model}\n')
        if warmstart_model:
            f.write(f'Warmstarted From: {warmstart_path}\n')
        if load_pretrained_model:
            f.write(f'Pretrained From: {pretrain_path}\n')

        f.write(f'\nValidation Results:\n')
        f.write(f'Total Validation Samples: {total}\n')
        f.write(f'F1 (macro): {f1_macro:.3f}\n')
        f.write(f'Top-1 Accuracy: {val_acc1:.2f}\n')
        f.write(f'Top-3 Accuracy: {val_acc3:.2f}\n')
        f.write(f'Top-5 Accuracy: {val_acc5:.2f}\n')

        f.write(
            f'\nValidation Top-k Accuracy: Prediction Rank (k), % Accuracy \n'
        )
        for i in range(len(validation_list)):
            f.write(f'Top {validation_list[i][0]} accuracy, {validation_list[i][1]} \n')

        f.write(f'\nRuntime: {(t1-t0)/60} minutes\n')
        f.write(f'Number of GPUs Used: {len(logical_gpus)}\n')
        f.write(f'\nModel Structure:\n {str(model)}\n')
        for item in study.best_params.items():
            f.write(f'{item}\n')

# === Simple shape-similarity check for top-k classes ===
with torch.no_grad():
    expt_vec = X_expt[0].cpu().numpy()  # (L,)

# L2-normalize
expt_vec = expt_vec - expt_vec.mean()
expt_vec = expt_vec / (np.linalg.norm(expt_vec) + 1e-8)

top_k = 25
shape_sims = []

### FIX: use 1D top_indices list directly
for idx in top_indices[:top_k]:
    sg = label_mapper[idx]
    fpath = None
    for f in data_files:
        if sg in f:
            fpath = f
            break
    if fpath is None:
        continue

    train_cls = torch.load(fpath)                    # (N, L_raw)
    train_cls = train_cls[:, data_start_index:data_end_index]
    train_cls = zscore_per_pattern(train_cls)
    train_cls = normalize_per_pattern(train_cls)

    mean_cls = train_cls.mean(dim=0).cpu().numpy()   # (L,)
    mean_cls = mean_cls - mean_cls.mean()
    mean_cls = mean_cls / (np.linalg.norm(mean_cls) + 1e-8)

    cos_sim = float((expt_vec * mean_cls).sum())
    shape_sims.append((sg, cos_sim))

print("\nCosine similarity of experimental pattern to class means (top-k net classes):")
for sg, sim in shape_sims:
    print(f"{sg}: {sim:.4f}")

### FIX: robust chi2 helper that crops to common length
def chi2_with_scale(obs, ref, eps=1e-8):
    """
    Chi2 with a free scale factor: obs ≈ a * ref.
    If lengths differ, crop both to the common minimum length.
    """
    obs = np.asarray(obs)
    ref = np.asarray(ref)

    L = min(obs.shape[-1], ref.shape[-1])
    obs = obs[..., :L]
    ref = ref[..., :L]

    # find scalar a that best fits obs ≈ a * ref in least squares sense
    num = (obs * ref).sum()
    den = (ref * ref).sum() + eps
    a = num / den
    resid = obs - a * ref
    return float((resid * resid).sum())

with torch.no_grad():
    expt_raw = expt_data_unscaled[0].cpu().numpy()  # BEFORE zscore/minmax
    expt_raw = expt_raw[data_start_index:data_end_index]

k = 25
ranking = []

### FIX: use 1D top_indices list directly here as well
for idx in top_indices[:k]:
    sg = label_mapper[idx]
    fpath = None
    for f in data_files:
        if sg in f:
            fpath = f
            break
    if fpath is None:
        continue
    cls_raw = torch.load(fpath).cpu().numpy()        # (N, L_raw)
    cls_raw = cls_raw[:, data_start_index:data_end_index]
    mean_raw = cls_raw.mean(axis=0)
    chi2 = chi2_with_scale(expt_raw, mean_raw)
    ranking.append((sg, idx, chi2))

ranking.sort(key=lambda x: x[2])  # smallest chi2 first
best_sg, best_idx, _ = ranking[0]
print("\nPhysics-aware re-ranking among Top-k:")
for sg, idx, chi2 in ranking:
    print(f"idx={idx}, sg={sg}, chi2={chi2:.3e}")
print(f"\nFinal choice after re-ranking: {best_sg}")

#^============================================================================================
#* Create CAM (Class Activation Map)
#^============================================================================================
if create_cam:

    # Make sure we're in eval mode
    model.eval()

    # Select the correct target layer for GradCAM
    target_layer = model.module.conv2 if isinstance(model, torch.nn.DataParallel) else model.conv2
    grad_cam = xsym.GradCAM(model, target_layer=target_layer)

    # predicted_subgroups is already defined above from experimental classification
    for tsg in predicted_subgroups:
        # tsg is a subgroup name like '0009', '0019', ...

        #----------------------------------------------------------------------
        # Find corresponding training data file for this subgroup
        #----------------------------------------------------------------------
        cam_data_path = ''
        for fpath in data_files:
            if tsg in fpath:
                cam_data_path = fpath
                break

        if cam_data_path == '':
            raise ValueError(
                f"Subgroup {tsg} not found in data files. "
                f"Please check the subgroup name or data files."
            )

        #----------------------------------------------------------------------
        # Load and slice CAM input data
        #----------------------------------------------------------------------
        cam_data = torch.load(cam_data_path)          # shape: (N,) or (N, L)
        cam_data = cam_data.unsqueeze(0) if cam_data.ndim == 1 else cam_data
        cam_data = cam_data[:, data_start_index:data_end_index]  # region of interest

        print(f'Scaling CAM data: Test Max is max of {tsg} training data\n')

        # funcs.MinMaxScale_2d usually returns numpy arrays
        X_train_dummy, cam_data_scaled, X_expt_dummy = funcs.MinMaxScale_2d(
            X_train_unscaled, cam_data, expt_data_unscaled, scale=scale_data
        )

        # Convert to tensor on the right device and enable gradients
        cam_data_scaled = torch.as_tensor(cam_data_scaled, dtype=torch.float32, device=device)
        input_batch = cam_data_scaled
        input_batch.requires_grad_(True)

        #----------------------------------------------------------------------
        # Map subgroup string -> class index
        #----------------------------------------------------------------------
        if tsg not in subgroup_to_label:
            raise KeyError(
                f"Subgroup {tsg} not found in subgroup_to_label mapping. "
                f"Check that label_mapper and subgroup_to_label are consistent."
            )

        target_class = subgroup_to_label[tsg]  # integer class index
        cam_save_name = f'{cam_save_basename}_{tsg}.png'

        #----------------------------------------------------------------------
        # Compute average CAM for this subgroup
        #----------------------------------------------------------------------
        avg_cam = grad_cam.generate_average_cam(input_batch, target_class)

        #----------------------------------------------------------------------
        # Plot over experimental pattern
        #----------------------------------------------------------------------
        xsym.plot_avg_cam_with_expt(
            xrd_expt=X_expt.cpu().numpy(),
            avg_cam=avg_cam,
            x_axis=expt_x_axis,
            x_units=cam_x_units_label,
            save_name=cam_save_name,
            legend_label=cam_legend_label
        )
