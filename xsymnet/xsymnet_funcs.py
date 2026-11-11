import os
import glob
import json
import time
import re
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.preprocessing import MaxAbsScaler
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ^ ------------------------------------------------------------------------------------------------------
# * Diffraction Data Conversion Functions
# ^ ------------------------------------------------------------------------------------------------------

def tof2d(tof, tzero, difa, difc):
    if difa > 0:
        d = (-difc/(2*difa) + np.sqrt((difc/(2*difa))**2 + tof/difa - tzero/difa))
    elif difa < 0:
        d = (-difc/(2*difa) - np.sqrt((difc/(2*difa))**2 + tof/difa - tzero/difa))
    elif difa == 0:
        d = (tof - tzero) / difc
    return d


def d2tof(d, tzero, difa, difc):
    tof = (difa * d**2) + (difc * d) + tzero
    return tof


def twoTheta_to_d(twoTheta, wavelength):
    d = wavelength / (2 * np.sin(np.radians(twoTheta/2)))
    return d


def find_closest_tof(xy_file, desired_tof, snip_name = '_snipped', create_snipped_data = False):
    # read in xy file, find closest x to desired tof, return index
    #remove comments and blank lines from xy file
    data_check = ('-','0','1','2','3','4','5','6','7','8','9')
    with open(xy_file, 'r') as f:    
        lines = f.readlines() 
        if not lines[0].startswith(data_check):
            lines = [line for line in lines if line.strip() and not line.startswith(data_check)]
            new_xy_file = (os.path.splitext(os.path.basename(xy_file))[0] + '_dataonly' + os.path.splitext(os.path.basename(xy_file))[1])
            new_xy_file = os.path.join(os.path.dirname(xy_file), new_xy_file)
            xy_file = new_xy_file
            with open(new_xy_file, 'w') as f:
                f.writelines(lines)
            print(f'Wrote new xy file without comments or blank lines: {new_xy_file}')

    data = np.loadtxt(xy_file)
    x = data[:,0]
    y = data[:,1]
    index = (np.abs(x - desired_tof)).argmin()
    
    # remove values before index
    if create_snipped_data:
        snip_x = x[index:]
        snip_y = y[index:]
        snip_xy_file = new_xy_file.replace('_dataonly', snip_name)
        with open(snip_xy_file, 'w') as f:
            for i in range(len(snip_x)):
                f.write(f'{snip_x[i]:.3f} {snip_y[i]:.6f}\n')
        print(f'Wrote snipped xy file without comments or blank lines: {snip_xy_file}')

    return index, x[index], y[index]

# ^ ------------------------------------------------------------------------------------------------------
# * Preprocessing and Data Handling Functions
# ^ ------------------------------------------------------------------------------------------------------

def load_xy_data(filepath, x_col=None, y_col=None, return_type='torch'):
    """
    Load x-y data from a file and return as torch tensors.

    Automatically detects:
    - Comment lines (lines starting with #, !, %, or other non-numeric characters)
    - Delimiter (comma, space, tab, or multi-space)
    - Single column vs. two column data

    Args:
        filepath (str): Path to the data file
        x_col (int, optional): Column index for x data (0-based). If None, uses
            column 0 for multi-column files or generates indices for single-column.
        y_col (int, optional): Column index for y data (0-based). If None, uses
            column 1 for multi-column files or column 0 for single-column.
        return_type (str): 'torch' (default) or 'numpy' for output format

    Returns:
        tuple: (x, y) as torch tensors or numpy arrays depending on return_type
            - If file has 2 columns: returns (x_data, y_data)
            - If file has 1 column: returns (range(len(y_data)), y_data)

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is empty or has invalid format

    Examples:
        # Default behavior (columns 0 and 1):
        x, y = load_xy_data('data.txt')

        # Read x from column 0, y from column 2 (e.g., background):
        x, bkg = load_xy_data('data.txt', x_col=0, y_col=2)

        # Read only y from column 3, generate x as indices:
        x, y_minus_bkg = load_xy_data('data.txt', x_col=None, y_col=3)
    """
    
    # Read all lines from file
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    if not lines:
        raise ValueError(f"File is empty: {filepath}")
    
    # Filter out comment lines and empty lines
    data_lines = []
    comment_chars = ['#', '!', '%', ';', '*']
    
    for line in lines:
        stripped = line.strip()
        
        # Skip empty lines
        if not stripped:
            continue
        
        # Skip lines starting with common comment characters
        if any(stripped.startswith(char) for char in comment_chars):
            continue
        
        # Try to detect if line starts with a number (data line)
        # This catches lines that might not start with standard comment chars
        try:
            # Check if first non-whitespace character could be start of a number
            first_char = stripped[0]
            if first_char.isdigit() or first_char in ['-', '+', '.']:
                data_lines.append(stripped)
            # Otherwise, assume it's a comment/header line
        except (IndexError, ValueError):
            continue
    
    if not data_lines:
        raise ValueError(f"No data lines found in file: {filepath}")
    
    # Determine delimiter by examining the first data line
    first_line = data_lines[0]
    
    # Check for comma
    if ',' in first_line:
        delimiter = ','
    # Check for tab
    elif '\t' in first_line:
        delimiter = '\t'
    # Otherwise assume whitespace (spaces)
    else:
        delimiter = None  # Will use split() which handles multiple spaces
    
    # Parse all data lines
    x_data = []
    y_data = []

    for line in data_lines:
        try:
            if delimiter:
                parts = line.split(delimiter)
            else:
                # split() with no argument splits on any whitespace and removes empty strings
                parts = line.split()

            # Remove empty strings
            parts = [p.strip() for p in parts if p.strip()]

            if not parts:
                # Empty line after splitting - skip
                continue

            # Determine which columns to use
            if x_col is not None and y_col is not None:
                # Both columns explicitly specified
                max_col = max(x_col, y_col)
                if len(parts) <= max_col:
                    print(f"Warning: Skipping line with insufficient columns (need {max_col+1}, got {len(parts)}): {line.strip()}")
                    continue
                x_data.append(float(parts[x_col]))
                y_data.append(float(parts[y_col]))
            elif y_col is not None:
                # Only y_col specified, x will be generated as indices
                if len(parts) <= y_col:
                    print(f"Warning: Skipping line with insufficient columns (need {y_col+1}, got {len(parts)}): {line.strip()}")
                    continue
                y_data.append(float(parts[y_col]))
            elif x_col is not None:
                # Only x_col specified, use default y_col=1
                eff_y_col = 1
                max_col = max(x_col, eff_y_col)
                if len(parts) <= max_col:
                    print(f"Warning: Skipping line with insufficient columns (need {max_col+1}, got {len(parts)}): {line.strip()}")
                    continue
                x_data.append(float(parts[x_col]))
                y_data.append(float(parts[eff_y_col]))
            else:
                # Default behavior (original logic)
                if len(parts) == 1:
                    # Single column - y data only
                    y_data.append(float(parts[0]))
                elif len(parts) >= 2:
                    # Two or more columns - take first two as x and y
                    x_data.append(float(parts[0]))
                    y_data.append(float(parts[1]))

        except ValueError as e:
            # Skip lines that can't be converted to float
            print(f"Warning: Skipping line that couldn't be parsed: {line.strip()}")
            continue
    
    if not y_data:
        raise ValueError(f"No valid numeric data found in file: {filepath}")
    
    # Convert to numpy arrays first for easier handling
    y_array = np.array(y_data, dtype=np.float32)
    
    # Handle x data
    if x_data:
        # We have x data from the file
        if len(x_data) != len(y_data):
            raise ValueError(
                f"Mismatch in data columns: {len(x_data)} x values but {len(y_data)} y values"
            )
        x_array = np.array(x_data, dtype=np.float32)
    else:
        # Generate x data as indices
        x_array = np.arange(len(y_data), dtype=np.float32)
    
    if return_type == 'numpy':
        return x_array, y_array
    elif return_type == 'torch':
        # Convert to torch tensors
        x_tensor = torch.from_numpy(x_array)
        y_tensor = torch.from_numpy(y_array)
        return x_tensor, y_tensor
    else:
        raise ValueError(f"Invalid return_type: {return_type}. Must be 'torch' or 'numpy'.")


def set_plot_style():
    rcParams.update({
    "text.usetex": False,  # Use True for LaTeX rendering if desired
    "font.family": "serif",
    "font.size": 16,
    "axes.labelsize": 16,
    "axes.titlesize": 14,
    "legend.fontsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "xtick.major.size": 10,
    "ytick.major.size": 10,
    "figure.dpi": 300,
    "savefig.bbox": "tight"
    })


def save_kernel_weights(model, epoch, save_file="kernel_weights.txt"):
    """Appends convolutional layer weights to a text file, grouped by kernel with epochs appended."""
    if epoch % 50 != 0:
        return  # Only save every 50 epochs
    
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    
    conv1_weights = model.conv1.weight.detach().cpu().numpy()
    conv2_weights = model.conv2.weight.detach().cpu().numpy()
    
    existing_data = {}
    
    # Read existing data if the file already exists
    try:
        with open(save_file, "r") as f:
            lines = f.readlines()
            kernel_name = None
            for line in lines:
                if line.startswith("conv"):
                    kernel_name = line.strip()
                    existing_data[kernel_name] = []
                elif kernel_name and line.strip():
                    existing_data[kernel_name].append(line.strip())
    except FileNotFoundError:
        pass  # No previous data exists, so we start fresh
    
    with open(save_file, "w") as f:
        for layer_name, weights in zip(["conv1", "conv2"], [conv1_weights, conv2_weights]):
            for i, kernel in enumerate(weights):
                kernel_key = f"{layer_name} kernel {i+1}"
                if kernel_key not in existing_data:
                    f.write(f"{kernel_key}\n")
                
                weights_str = ", ".join(map(str, kernel.flatten()))
                existing_data.setdefault(kernel_key, []).append(f"epoch {epoch} {weights_str}")
        
        # Write back grouped data
        for kernel_key, records in existing_data.items():
            f.write(f"{kernel_key}\n")
            f.write("\n".join(records) + "\n\n")
    
    print(f"Kernel weights appended for epoch {epoch} in {save_file}")
    return 0


def train_test_split(path2braggdata, n_randomstates, n_testing):
    
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

    print(f'train_test_split:')
    print(f'Train_MAX: {torch.max(xtrain):.2f}\nTrain_MIN: {torch.min(xtrain):.2f}')
    print(f'Test_MAX: {torch.max(xtest):.2f}\nTest_MIN: {torch.min(xtest):.2f}\n')


    return xtrain, xtest, ytrain, ytest


def train_test_split_number(path2braggdata, n_randomstates, n_training, n_testing):

    assert(n_training + n_testing <= n_randomstates), "In train_test_split_number: \nn_training + n_testing must be less than or equal to n_randomstates"
    print(f'\nIn train_test_split_number:')
    print(f'n_randomstates = {n_randomstates}, n_training = {n_training}, n_testing = {n_testing}\n')

    filepaths = sorted(glob.glob(os.path.join(path2braggdata, '*.pt')))

    xtrain_list = []
    xtest_list = []
    ytrain_list = []
    ytest_list = []

    for i in range(len(filepaths)):
        temp = torch.load(filepaths[i])

        # xtest_list.append(temp[0:n_testing, :])
        # ytest_list.extend([i]*n_testing)
        # xtrain_list.append(temp[n_testing:(n_testing + n_training), :])
        # ytrain_list.extend([i]*n_training)

        for j in range(n_testing):
            xtest_list.append(temp[j])
            ytest_list.append(i)
        for j in range(n_testing, n_testing + n_training):
            xtrain_list.append(temp[j])
            ytrain_list.append(i)

    
    xtrain = torch.vstack(xtrain_list)
    xtest = torch.vstack(xtest_list)

    ytrain = torch.tensor(ytrain_list)
    ytest = torch.tensor(ytest_list)

    
    print(f'Train_MAX: {torch.max(xtrain):.2f}\nTrain_MIN: {torch.min(xtrain):.2f}')
    print(f'Test_MAX: {torch.max(xtest):.2f}\nTest_MIN: {torch.min(xtest):.2f}\n')
    print(f'Num training samples: {xtrain.shape} with {xtrain.shape[0] / len(filepaths)} per class')
    print(f'Num testing samples: {xtest.shape}with {xtest.shape[0] / len(filepaths)} per class')
    print(f'Num used / total available {xtrain.shape[0] + xtest.shape[0]} / {n_randomstates}\n\n')

    return xtrain, xtest, ytrain, ytest


def MinMaxScale_2d(train, test, expt=None, scale=1):
   
    min_val = torch.min(train)
    max_val = torch.max(train)

    train_scaled = ((train-min_val) / (max_val - min_val))*scale
    test_scaled = ((test - min_val) / (max_val - min_val))*scale
    
    if expt != None:
        expt_scaled = ((expt - min_val) / (max_val - min_val))*scale 
    else:
        expt_scaled = None

    return train_scaled, test_scaled, expt_scaled


def MaxAbsScale(X_train_load, X_test_load, X_expt_load):
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
    expt_local = X_expt_load.numpy(force=True)
    print(f'Pre-MAXABS SCALE \n Expt MAX: {np.max(expt_local):.2f}\nExpt MIN: {np.min(expt_local):.2f}\n')

    transformer = MaxAbsScaler().fit(training_local)
    training_scaled = transformer.transform(training_local)
    testing_scaled = transformer.transform(testing_local)
    expt_scaled = transformer.transform(expt_local)
    print(f'Expt MAX: {np.max(expt_scaled):.2f}\nExpt MIN: {np.min(expt_scaled):.2f}\n')

    X_train = torch.tensor(training_scaled)
    X_test = torch.tensor(testing_scaled)
    X_expt = torch.tensor(expt_scaled)
       
    print(f'Train_MAX: {torch.max(X_train):.2f}\nTrain_MIN: {torch.min(X_train):.2f}\n')
    print(f'Test_MAX: {torch.max(X_test):.2f}\nTest_MIN: {torch.min(X_test):.2f}\n')
    print(f'Expt MAX: {torch.max(X_expt):.2f}\nExpt MIN: {torch.min(X_expt):.2f}\n')

    return X_train, X_test, X_expt


def sqrt_scale_diffraction(train, test, expt, quantile: float = 1.0, eps: float = 1e-12):
    """
    Apply square-root scaling to diffraction data using statistics
    computed from the training set only.

    Steps:
      1. Clip negative values to zero
      2. Compute sqrt-scaling constant from training data
      3. Apply identical transform to train, test, and experimental data

    Args:
        train (torch.Tensor): Training data tensor
        test (torch.Tensor): Testing/validation data tensor
        expt (torch.Tensor): Experimental/inference data tensor
        quantile (float): Quantile for robust max (1.0 = true max)
        eps (float): Small constant for numerical stability

    Returns:
        scaled_train (torch.Tensor)
        scaled_test (torch.Tensor)
        scaled_expt (torch.Tensor)
    """

    # --- Clip negatives (Poisson-safe) ---
    train_clipped = torch.clamp(train, min=0.0)
    test_clipped  = torch.clamp(test,  min=0.0)
    expt_clipped  = torch.clamp(expt,  min=0.0)

    # --- Compute scaling constant from TRAINING data only ---
    if quantile < 1.0:
        # Robust max (recommended for diffraction)
        scale_intensity = torch.quantile(
            train_clipped.flatten(),
            quantile
        )
    else:
        scale_intensity = train_clipped.max()

    scale = torch.sqrt(scale_intensity + eps)

    # --- Apply square-root scaling ---
    scaled_train = (torch.sqrt(train_clipped) / scale) *100
    scaled_test  = (torch.sqrt(test_clipped)  / scale) *100
    scaled_expt  = (torch.sqrt(expt_clipped)  / scale) *100

    return scaled_train, scaled_test, scaled_expt


def MinMaxScale(tensor):
    scaled_tensor = torch.zeros_like(tensor)
    for i in range(len(tensor)):
        min_val = torch.min(tensor[i])
        max_val = torch.max(tensor[i])
        temp_tensor = (tensor[i]-min_val) / (max_val - min_val)
        scaled_tensor[i] = temp_tensor
    return scaled_tensor



#^ ------------------------------------------------------------------------------------------------------
# * Multi-Channel Input Functions
#^ ------------------------------------------------------------------------------------------------------

def compute_log_channel(x, eps=1e-8):
    """Compute log-scaled channel from intensity data.

    Uses log1p (log(1+x)) to handle zero values safely.
    Shifts data to positive range before applying log.

    Args:
        x: (batch, seq_len) tensor of intensity data
        eps: Small value to prevent log(0)

    Returns:
        Log-scaled tensor of same shape
    """
    x_shifted = x - x.min() + eps  # Ensure all positive
    return torch.log1p(x_shifted)


def _gaussian_smooth_1d(x, sigma=1.0):
    """Apply 1D Gaussian smoothing using conv1d.

    Args:
        x: (batch, seq_len) tensor
        sigma: Gaussian kernel standard deviation

    Returns:
        Smoothed tensor of same shape
    """
    kernel_size = int(6 * sigma) | 1  # Ensure odd (covers ~99.7% of Gaussian)
    half_k = kernel_size // 2

    # Create Gaussian kernel
    positions = torch.arange(-half_k, half_k + 1, dtype=torch.float32)
    kernel = torch.exp(-positions**2 / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, -1).to(x.device)

    # Apply convolution with padding
    x = x.unsqueeze(1)  # (batch, 1, seq_len)
    x = F.pad(x, (half_k, half_k), mode='replicate')
    x = F.conv1d(x, kernel)
    return x.squeeze(1)


def compute_curvature_channel(x, apply_smoothing=False, sigma=1.0):
    """Compute second derivative (curvature) via finite differences.

    Uses the central difference formula: f''[i] = f[i+1] - 2*f[i] + f[i-1]

    Args:
        x: (batch, seq_len) tensor
        apply_smoothing: If True, apply Gaussian smoothing first (default: False)
        sigma: Gaussian kernel sigma if smoothing enabled

    Returns:
        Curvature tensor of same shape
    """
    if apply_smoothing:
        x = _gaussian_smooth_1d(x, sigma)

    # Second derivative: f''[i] = f[i+1] - 2*f[i] + f[i-1]
    curvature = x[:, 2:] - 2*x[:, 1:-1] + x[:, :-2]

    # Pad edges to restore original length
    curvature = F.pad(curvature, (1, 1), mode='replicate')

    return curvature


def create_multichannel_input(X, apply_curvature_smoothing=False, sigma=1.0):
    """Create 3-channel input tensor from 1D intensity data.

    Channels:
        1. Intensity (as-is)
        2. Log-scaled intensity (reveals weak peaks)
        3. Curvature / second derivative (highlights peak boundaries)

    Args:
        X: (batch, seq_len) MinMax-scaled intensity data
        apply_curvature_smoothing: Enable Gaussian smoothing before curvature
        sigma: Gaussian sigma if smoothing enabled

    Returns:
        (batch, 3, seq_len) tensor with channels [intensity, log, curvature]
    """
    ch1 = X  # Intensity as-is
    ch2 = compute_log_channel(X)
    ch3 = compute_curvature_channel(X, apply_curvature_smoothing, sigma)

    return torch.stack([ch1, ch2, ch3], dim=1)


def plot_multichannel_input(filepath, x_units='TOF (μs)', apply_smoothing=False, sigma=1.0,
                            scale=100, save_path=None, figsize=(14, 10)):
    """Plot all 3 channels of multi-channel input for a diffraction pattern.

    Loads an xy file, applies MinMax scaling, computes the 3 channels,
    and creates a visualization with subplots.

    Args:
        filepath: Path to xy data file (2-column: x, intensity)
        x_units: Label for x-axis (default: 'TOF (μs)')
        apply_smoothing: Apply Gaussian smoothing before curvature (default: False)
        sigma: Gaussian sigma if smoothing enabled
        scale: MinMax scaling factor (default: 100)
        save_path: If provided, save figure to this path
        figsize: Figure size tuple (default: (14, 10))

    Returns:
        fig, axes: matplotlib figure and axes objects
    """
    # Set up nice plotting defaults
    rcParams.update({
        "font.family": "serif",
        "font.size": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 150,
    })

    # Load the data
    x_axis, intensity = load_xy_data(filepath)

    # Convert to 2D tensor for processing (batch size 1)
    intensity_2d = intensity.unsqueeze(0)

    # Apply MinMax scaling
    min_val = intensity_2d.min()
    max_val = intensity_2d.max()
    intensity_scaled = ((intensity_2d - min_val) / (max_val - min_val)) * scale

    # Compute all 3 channels
    X_multi = create_multichannel_input(intensity_scaled, apply_smoothing, sigma)

    # Extract channels (remove batch dimension)
    ch1 = X_multi[0, 0, :].numpy()  # Intensity
    ch2 = X_multi[0, 1, :].numpy()  # Log-scaled
    ch3 = X_multi[0, 2, :].numpy()  # Curvature
    x = x_axis.numpy()

    # Create figure with 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)

    # Channel 1: Intensity
    axes[0].plot(x, ch1, 'b-', linewidth=0.8)
    axes[0].set_ylabel('Intensity (scaled)')
    axes[0].set_title('Channel 1: MinMax-Scaled Intensity')
    axes[0].grid(True, alpha=0.3)

    # Channel 2: Log-scaled
    axes[1].plot(x, ch2, 'g-', linewidth=0.8)
    axes[1].set_ylabel('Log Intensity')
    axes[1].set_title('Channel 2: Log-Scaled Intensity (log1p)')
    axes[1].grid(True, alpha=0.3)

    # Channel 3: Curvature
    axes[2].plot(x, ch3, 'r-', linewidth=0.8)
    axes[2].axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
    axes[2].set_ylabel('Curvature')
    smoothing_note = f' (σ={sigma})' if apply_smoothing else ''
    axes[2].set_title(f'Channel 3: Curvature (2nd Derivative){smoothing_note}')
    axes[2].set_xlabel(x_units)
    axes[2].grid(True, alpha=0.3)

    # Add filename to figure title
    filename = os.path.basename(filepath)
    fig.suptitle(f'Multi-Channel Input: {filename}', fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Figure saved to: {save_path}')

    return fig, axes



#^ ------------------------------------------------------------------------------------------------------
# * Output and Analysis Functions
#^ ------------------------------------------------------------------------------------------------------

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



def get_str_info(str_path, str_type):
    '''
    Parameters
    ----------
    str_path: str
        Full path to the topas_str file generated by isodistort
    str_type: string
        accepted inputs: 
            'regular' for a regular isodistort topas str file
            'strain' for an isodistort topas str file with strain modes
        
    Returns:
    ----------
    a,b,c,alpha,beta,gamma: list of floats
    sg_o = spacegroup & origin & HM_name: list of ints and string
    '''
    lines = []
    with open(str_path, 'r') as f:
        lines = f.readlines()

    # extract the spacegroup, origin choice, and Herman-Maugin Spacegroup name    
    oline = lines[4].strip()
    sg_name = oline.split(':')[0].strip("'")

    if ':' in oline:
        origin = int(oline.split(':')[1])
    else:
        origin = 1
    
    sg = int(lines[5].strip().split()[1])


    # grabs the standard lattice parameters
    if str_type == 'regular':
        for i in range(len(lines)):
            if 'space_group' in lines[i]:
                alat = float(lines[i+1].strip().split()[1])
                blat = float(lines[i+2].strip().split()[1])
                clat = float(lines[i+3].strip().split()[1])
                alpha = float(lines[i+4].strip().split()[1])
                beta = float(lines[i+5].strip().split()[1])
                gamma = float(lines[i+6].strip().split()[1])
                break
        

    # grabs the strain mode defined lattice parameters
    elif str_type == 'strain':
        for i in range(len(lines)):
            if 'celltransform(' in lines[i]:
                alat = float(lines[i+2].strip().split(':')[1].strip())
                blat = float(lines[i+3].strip().split(':')[1].strip())
                clat = float(lines[i+4].strip().split(':')[1].strip())
                alpha = float(lines[i+5].strip().split(':')[1].strip())
                beta = float(lines[i+6].strip().split(':')[1].strip())
                gamma = float(lines[i+7].strip().split(':')[1].strip())
                break
    
    else:
        raise ValueError("Invalid str_type. It must be either 'regular' or 'strain'.")

    latparams = [alat, blat, clat, alpha, beta, gamma]
    sg_o = [sg, origin, sg_name]
    return latparams, sg_o 


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


def analyze_prediction_stability(seeded_predictions, top_n, str_files):
    """
    Analyzes the stability of top-N predictions across multiple runs with different seeds.

    Calculates:
    1. Appearance Frequency: How many runs a class appeared in the top-N.
    2. Average Rank: The average position of the class when it appeared in the top-N.
    3. Std Dev of Rank: The standard deviation of the class's position in the top-N.

    Args:
        seeded_predictions: A list of lists/arrays, where each inner list/array 
                            contains the *ranked class indices* (from 1st to 10th 
                            prediction) from one seeded run.
        top_n: The number of top predictions to consider (default is 10).

    Returns:
        A pandas DataFrame showing the stability metrics for each unique class 
        that appeared in the top-N.
    """
    
    # 1. Prepare and Combine Data
    all_runs_data = []
    
    # Create (class_id, rank) tuples for every class in every run
    for run_index, top_n_classes in enumerate(seeded_predictions):
        # We only care about the top_n predictions
        for rank, class_id in enumerate(top_n_classes[:top_n]):
            # rank starts at 0, so position/rank_display is rank + 1
            all_runs_data.append({
                'run_id': run_index,
                'class_id': class_id,
                'rank': rank + 1  # 1-based rank (1st, 2nd, 3rd, ...)
            })

    if not all_runs_data:
        return pd.DataFrame(columns=['Class ID', 'Appearance Frequency', 'Average Rank', 'Std Dev of Rank'])

    df = pd.DataFrame(all_runs_data)

    # 2. Calculate Stability Metrics
    
    # Group by the class ID
    stability_metrics = df.groupby('class_id')['rank'].agg(
        # Appearance Frequency: Count of how many runs the class appeared in the top N
        appearance_frequency='count',
        # Average Rank: Mean position of the class when it appeared
        average_rank='mean',
        # Std Dev of Rank: Standard deviation of its positions
        std_dev_of_rank='std'
    ).reset_index()

    # Rename columns for clarity
    stability_metrics.columns = [
        'Class ID', 
        'Appearance Frequency', 
        'Average Rank', 
        'Std Dev of Rank'
    ]
    
    # Convert frequency to a percentage of total runs
    num_runs = len(seeded_predictions)
    stability_metrics['Frequency (%)'] = (
        stability_metrics['Appearance Frequency'] / num_runs * 100
    ).round(2)
    
    # Handle NaN for Std Dev (occurs if a class only appeared once)
    stability_metrics['Std Dev of Rank'] = stability_metrics['Std Dev of Rank'].fillna(0).round(2)
    stability_metrics['Average Rank'] = stability_metrics['Average Rank'].round(2)
    
    # Sort for easy interpretation: most frequent/stable classes first
    stability_metrics = stability_metrics.sort_values(
        by=['Appearance Frequency', 'Average Rank'], 
        ascending=[False, True]
    )

    spacegroup_origin = []
    hm_name = []
    for id in stability_metrics['Class ID']:
        for i in range(len(str_files)):
            if id in str_files[i]:
                latparams, sgo = get_str_info(str_files[i], str_type='strain')
                spacegroup_origin.append(f'{sgo[0]}:{sgo[1]}')
                hm_name.append(sgo[2])


    stability_metrics.insert(1, 'HM Name', hm_name)
    stability_metrics.insert(1, 'Spacegroup:Origin', spacegroup_origin)
    print(stability_metrics)
    return stability_metrics


def calculate_prediction_stability(seed_outputs, penalty_rank=11):
    """
    Calculates the standard deviation of ranks for classes across multiple seeded runs.
    
    Args:
        seed_outputs (list of lists): List where each entry is the top 10 predictions (labels) for a seed.
        penalty_rank (int): Rank assigned to a class if it doesn't appear in the top 10 of a run.
        
    Returns:
        pd.DataFrame: A dataframe containing the mean rank and std dev for each class.
    """
    # 1. Identify all unique classes that appeared in any top 10 list
    all_seen_classes = set(item for sublist in seed_outputs for item in sublist)
    
    stability_data = {}

    for cls in all_seen_classes:
        ranks = []
        for run in seed_outputs:
            if cls in run:
                # Adding 1 because list index is 0-based, but rank starts at 1
                ranks.append(run.index(cls) + 1)
            else:
                # Class fell out of top 10
                ranks.append(penalty_rank)
        
        stability_data[cls] = ranks

    # 2. Calculate statistics
    results = []
    for cls, ranks in stability_data.items():
        results.append({
            'class': cls,
            'mean_rank': np.mean(ranks),
            'std_dev': np.std(ranks),
            'occurrences_in_top_10': sum(1 for r in ranks if r <= 10)
        })

    # 3. Create DataFrame and sort by best mean rank
    df = pd.DataFrame(results).sort_values(by='mean_rank')
    return df


def get_top10_output(filepath_list):
    '''
    Parameters
    ----------
    filepath_list: list of str
        List of filepaths to the top10 output files from different seeds

    Returns:
    ----------
    seed_outputs: list of lists
        Each inner list contains the top 10 predicted class labels for a seed
    '''
    seed_outputs = []
    for filepath in filepath_list:
        with open(filepath, 'r') as f:
            lines = f.readlines()
            top10 = []
            for i in range(len(lines)):
                if 'Top Predicted Subgroups' in lines[i]:
                    for j in range(1, len(lines)):
                        line = lines[i + j].strip()
                        if line == '':
                            break
                        parts = line.split()
                        class_label = parts[0].strip(',')  # class label is the first element
                        top10.append(class_label)
                    break
            seed_outputs.append(top10)
    return seed_outputs


def get_validation_metrics(filepath_list):
    '''
    Extracts validation metrics (Top-1/3/5 Accuracy and F1 macro) from output files.

    Parameters
    ----------
    filepath_list: list of str
        List of filepaths to the output files from different seeds

    Returns:
    ----------
    metrics_dict: dict
        Dictionary containing lists of metrics from each file:
        - 'top1_accuracy': list of Top-1 Accuracy values
        - 'top3_accuracy': list of Top-3 Accuracy values
        - 'top5_accuracy': list of Top-5 Accuracy values
        - 'f1_macro': list of F1 (macro) values
    '''
    metrics_dict = {
        'top1_accuracy': [],
        'top3_accuracy': [],
        'top5_accuracy': [],
        'f1_macro': []
    }

    for filepath in filepath_list:
        with open(filepath, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if line.startswith('Top-1 Accuracy:'):
                    val = float(line.split(':')[1].strip())
                    metrics_dict['top1_accuracy'].append(val)
                elif line.startswith('Top-3 Accuracy:'):
                    val = float(line.split(':')[1].strip())
                    metrics_dict['top3_accuracy'].append(val)
                elif line.startswith('Top-5 Accuracy:'):
                    val = float(line.split(':')[1].strip())
                    metrics_dict['top5_accuracy'].append(val)
                elif line.startswith('F1 (macro):'):
                    val = float(line.split(':')[1].strip())
                    metrics_dict['f1_macro'].append(val)

    return metrics_dict


def summarize_validation_metrics(filepath_list):
    '''
    Extracts and summarizes validation metrics across multiple seed runs.

    Parameters
    ----------
    filepath_list: list of str
        List of filepaths to the output files from different seeds

    Returns:
    ----------
    summary_df: pandas DataFrame
        DataFrame with columns: Metric, Mean, Std Dev, Min, Max, N
    '''
    metrics = get_validation_metrics(filepath_list)

    summary_data = []
    metric_names = {
        'top1_accuracy': 'Top-1 Accuracy',
        'top3_accuracy': 'Top-3 Accuracy',
        'top5_accuracy': 'Top-5 Accuracy',
        'f1_macro': 'F1 (macro)'
    }

    for key, display_name in metric_names.items():
        values = metrics[key]
        if len(values) > 0:
            summary_data.append({
                'Metric': display_name,
                'Mean': np.mean(values),
                'Std Dev': np.std(values),
                'Min': np.min(values),
                'Max': np.max(values),
                'N': len(values)
            })

    summary_df = pd.DataFrame(summary_data)
    summary_df['Mean'] = summary_df['Mean'].round(4)
    summary_df['Std Dev'] = summary_df['Std Dev'].round(4)
    summary_df['Min'] = summary_df['Min'].round(4)
    summary_df['Max'] = summary_df['Max'].round(4)

    return summary_df


