import os
import glob
import json
import re
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MaxAbsScaler


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

    print(f'Configure Bragg:')
    print(f'Train_MAX: {torch.max(xtrain):.2f}\nTrain_MIN: {torch.min(xtrain):.2f}')
    print(f'Test_MAX: {torch.max(xtest):.2f}\nTest_MIN: {torch.min(xtest):.2f}\n')


    return xtrain, xtest, ytrain, ytest


def configure_bragg_data_trunc(path2braggdata, n_randomstates, n_testing, scaling_dict={}, start_index:int = 0, end_index:int = -1):
    '''depreciated'''
    filepaths = sorted(glob.glob(os.path.join(path2braggdata, '*.pt')))

    xtrain_list = []
    xtest_list = []
    ytrain_list = []
    ytest_list = []

    split_index = n_randomstates - n_testing

    keys = list(scaling_dict.keys())

    # expt_min = torch.min(expt)
    # expt_max = torch.max(expt)
    # print(f'\n\nConfigure Bragg PreScaling:')
    # print(f'Expt MAX:{expt_max:.2f}\nExpt MIN:{expt_min:.2f}\n')


    if len(keys) > 0:
        print(('Applying scaling dict to data:\n'))
        
    for i in range(len(filepaths)):
        temp = torch.load(filepaths[i])
        #! Turn ManualScale on/off here
        # temp = ManualScale(temp, expt_max, i)
        # temp = ManualScaleConstant(temp, scalefactor=0.001)
        #^ here we apply the optimal scale factor for each subgroup
        # if len(keys) > 0:
        #     temp_max = torch.max(temp)
        #     temp = temp*scaling_dict[keys[i]]
        #     scaled_max = torch.max(temp)
        #     print(f'SGT {os.path.splitext(os.path.basename(filepaths[i]))[0]} max: {temp_max:.2f}   scaled: {scaled_max:.2f}')

        count = 0
        for j in range(temp.shape[0]):
            if count < split_index:
                # xtrain_list.append(temp[j,start_index:end_index])
                xtrain_list.append(temp[j])
                ytrain_list.append(i)
            else:
                # xtest_list.append(temp[j,start_index:end_index])
                xtest_list.append(temp[j])
                ytest_list.append(i)
            count+=1

    xtrain = torch.vstack(xtrain_list)
    xtest = torch.vstack(xtest_list)
    ytrain = torch.tensor(ytrain_list)
    ytest = torch.tensor(ytest_list)

    print(f'Post Configure Bragg:')
    # print(f'Expt MAX:{expt_max:.2f}\nExpt MIN:{expt_min:.2f}')
    print(f'Train_MAX: {torch.max(xtrain):.2f}\nTrain_MIN: {torch.min(xtrain):.2f}')
    print(f'Test_MAX: {torch.max(xtest):.2f}\nTest_MIN: {torch.min(xtest):.2f}\n')


    # train_max = torch.max(xtrain)
    # test_max = torch.max(xtest)

    # whole_max = max(train_max, test_max)
    # # Compute the scaling factor to match the order of magnitude
    # order_of_mag_simulated = 10 ** (len(str(int(whole_max))) - 1)
    # order_of_mag_experimental = 10 ** (len(str(int(expt_max))) - 1)
    # scaling_factor = order_of_mag_experimental / order_of_mag_simulated 

    # scaled_xtrain = xtrain*scaling_factor
    # scaled_xtest = xtest*scaling_factor

    # print(f'scaled_xtrain max: {torch.max(scaled_xtrain)}')
    # print(f'scaled_xtest max: {torch.max(scaled_xtest)}')

    return xtrain, xtest, ytrain, ytest


def load_xy_data(filepath):
    """
    Load x-y data from a file and return as torch tensors.
    
    Automatically detects:
    - Comment lines (lines starting with #, !, %, or other non-numeric characters)
    - Delimiter (comma, space, tab, or multi-space)
    - Single column vs. two column data
    
    Args:
        filepath (str): Path to the data file
        
    Returns:
        tuple: (expt_x_axis, bragg_tensor) as torch tensors
            - If file has 2 columns: returns (x_data, y_data)
            - If file has 1 column: returns (range(len(y_data)), y_data)
            
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is empty or has invalid format
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
            
            if len(parts) == 1:
                # Single column - y data only
                y_data.append(float(parts[0]))
            elif len(parts) >= 2:
                # Two or more columns - take first two as x and y
                x_data.append(float(parts[0]))
                y_data.append(float(parts[1]))
            else:
                # Empty line after splitting - skip
                continue
                
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
    
    # Convert to torch tensors
    expt_x_axis = torch.from_numpy(x_array)
    bragg_tensor = torch.from_numpy(y_array)
    
    return expt_x_axis, bragg_tensor


def MinMaxScale_2d(train, test, expt=None, scale=1):

    expt_min = 0
    expt_max = 0
   
    min_val = torch.min(train)
    max_val = torch.max(train)
    
    if expt != None:
        # expt_scale = 1/0.000295037729
        # expt = expt*expt_scale
        expt_min = torch.min(expt)
        expt_max = torch.max(expt)
        # print(f'Expt MAX:{expt_max:.2f}\nExpt MIN:{expt_min:.2f}\n')

    # print(f'Normalization with MinMaxScale2d:')

    train_scaled = ((train-min_val) / (max_val - min_val))*scale
    test_scaled = ((test - min_val) / (max_val - min_val))*scale
    
    if expt != None:
        expt_scaled = ((expt - min_val) / (max_val - min_val))*scale 
        #^ make sure that after scaling, the values don't go negative
        # print('MIN clamping EXPT after scaling to prevent negative values\n')
        # expt_scaled = torch.clamp(expt_scaled, min=0)
        #^ debug clamp and check output
        # expt_scaled_write = expt_scaled.t().numpy()
        # np.savetxt('./TESTINGEXPTTENSOR.txt', expt_scaled_write, newline='\n')
    else:
        expt_scaled = None
        
    # if expt != None:
    #     print(f'Expt MAX:{torch.max(expt_scaled):.2f}\nExpt MIN:{torch.min(expt_scaled):.2f}\n')
    # print(f'Train_MAX:{torch.max(train_scaled):.2f}\nTrain_MIN:{torch.min(train_scaled):.2f}\n')
    # print(f'Test_MAX:{torch.max(test_scaled):.2f}\nTest_MIN:{torch.min(test_scaled):.2f}\n')
    return train_scaled, test_scaled, expt_scaled


def MaxAbsScale(X_train_load, X_test_load, X_expt_load=None):
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
    if X_expt_load is not None:
        expt_local = X_expt_load.numpy(force=True)
        print(f'Pre-MAXABS SCALE \n Expt MAX: {np.max(expt_local):.2f}\nExpt MIN: {np.min(expt_local):.2f}\n')
    else:
        expt_local = None

    transformer = MaxAbsScaler().fit(training_local)
    training_scaled = transformer.transform(training_local)
    testing_scaled = transformer.transform(testing_local)
    if expt_local is not None:
        expt_scaled = transformer.transform(expt_local)
        print(f'Expt MAX: {np.max(expt_scaled):.2f}\nExpt MIN: {np.min(expt_scaled):.2f}\n')
    else:
        expt_scaled = None

    X_train = torch.tensor(training_scaled)
    X_test = torch.tensor(testing_scaled)

    if expt_scaled == None:
        print(f'Train_MAX: {torch.max(X_train):.2f}\nTrain_MIN: {torch.min(X_train):.2f}\n')
        print(f'Test_MAX: {torch.max(X_test):.2f}\nTest_MIN: {torch.min(X_test):.2f}\n')
        X_expt = None

    else:
        X_expt = torch.tensor(expt_scaled)
        print(f'Train_MAX: {torch.max(X_train):.2f}\nTrain_MIN: {torch.min(X_train):.2f}\n')
        print(f'Test_MAX: {torch.max(X_test):.2f}\nTest_MIN: {torch.min(X_test):.2f}\n')
        print(f'Expt MAX: {torch.max(X_expt):.2f}\nExpt MIN: {torch.min(X_expt):.2f}\n')

    return X_train, X_test, X_expt


def MinMaxScale(tensor):
    scaled_tensor = torch.zeros_like(tensor)
    for i in range(len(tensor)):
        min_val = torch.min(tensor[i])
        max_val = torch.max(tensor[i])
        temp_tensor = (tensor[i]-min_val) / (max_val - min_val)
        scaled_tensor[i] = temp_tensor
    return scaled_tensor


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


def configure_bragg_data(path2braggdata, n_randomstates, n_testing):
    
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

    return xtrain, xtest, ytrain, ytest


def ManualScale(data, expt_max, i):

    temp_max = torch.max(data)
    scaling = expt_max/temp_max
    scaled_data = data*scaling
    
    print(f'subgroup {i} max: {temp_max}, {torch.max(scaled_data)}, {scaling}')

    return scaled_data


def ManualScaleConstant(data, scalefactor):

    scaled_data = data*scalefactor
    
    return scaled_data


def configure_hkl_data_trunc(path2braggdata, trunc_index, n_randomstates, n_testing, maplocation):
    
    filepaths = sorted(glob.glob(os.path.join(path2braggdata, '*.pt')))

    xtrain_list = []
    xtest_list = []
    ytrain_list = []
    ytest_list = []

    split_index = n_randomstates - n_testing
    for i in range(len(filepaths)):
        temp = torch.load(filepaths[i], map_location=maplocation)
        count = 0
        for j in range(temp.shape[0]):
            if count < split_index:
                xtrain_list.append(temp[j,trunc_index:])
                ytrain_list.append(i)
            else:
                xtest_list.append(temp[j,trunc_index:])
                ytest_list.append(i)
            count+=1

    xtrain = torch.vstack(xtrain_list)
    xtest = torch.vstack(xtest_list)
    ytrain = torch.tensor(ytrain_list)
    ytest = torch.tensor(ytest_list)

    return xtrain, xtest, ytrain, ytest


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


