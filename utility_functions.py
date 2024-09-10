'''
Author: Dr. Dayton  Kizzire
Last Updated: 02/08/2024
Email: kizziredg@ornl.gov
'''

import os
import glob
import json
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from zipfile import ZipFile
import tarfile

def determine_gpu_or_cpu(gpu_index):
    whichgpu = f'cuda:{gpu_index}'

    device = (
        "cuda" if torch.cuda.is_available()
        else "cpu" )  

    if device == 'cuda':
        print(f"Using {device} device")
        cuda0 = torch.device(whichgpu)
        print(f'CUDA device properties: \n', torch.cuda.get_device_properties(cuda0))
        print(f'Pytorch CUDA state initialized : {torch.cuda.is_initialized()}')
        torch.cuda.set_device(cuda0)
        maplocation = cuda0

    elif device == 'cpu':
        maplocation = 'cpu'

    return maplocation


def convert_hkl_list_to_tensors(filepath, tensor_writepath):
    '''
    '''

    # read in hkl lists from topas output and sort the hkl lists by ascending d_spacing
    hkl = pd.read_csv(filepath, sep='\s+', header = None, names = ['d','i'], dtype=np.float32)
    sorted_hkl = hkl.sort_values(by=['d'], ascending=True)
    hkl_tensor = torch.tensor(sorted_hkl.values, dtype=torch.float)

    torch.save(hkl_tensor, tensor_writepath)

    return None


def convert_bragg_to_tensors(filepath):
    '''
    '''

    # read in hkl lists from topas output and sort the hkl lists by ascending d_spacing
    bragg = pd.read_csv(filepath, sep='\s+', header = None, names = ['tof','i'], usecols=['i']).T
    bragg_tensor = torch.tensor(bragg.values, dtype=torch.float)

    return bragg_tensor


def get_d_bounds_for_binning(hkl_files_paths, experimental_hkl_path=None):
    '''
    '''
    lowest_d_value = 100.0
    largest_d_value = 0.0
    for i in range(len(hkl_files_paths)):
        temp_tensor = torch.load(hkl_files_paths[i])
        if temp_tensor[0,0].item() < lowest_d_value:
            lowest_d_value = temp_tensor[0,0].item()
        if temp_tensor[-1,0].item() > largest_d_value:
            largest_d_value = temp_tensor[-1,0].item()
                
    if experimental_hkl_path != None:
        temp_tensor = torch.load(experimental_hkl_path[0])
        if temp_tensor[0,0].item() < lowest_d_value:
            lowest_d_value = temp_tensor[0,0].item()
        if temp_tensor[-1,0].item() > largest_d_value:
            largest_d_value = temp_tensor[-1,0].item()


    return lowest_d_value, largest_d_value


def create_bin_edges(phase_name, c, lowest_d_value, largest_d_value):
    cwd = os.getcwd()
    # choice of 1st bin center : b0
    # c = delta_d / d = resolution factor; instrument and frame dependent
    bc = lowest_d_value
    cstring_temp = str(c).replace('.','p')
    cstring = f'C{cstring_temp}'

    # calculate the starting edge of the first bin
    bs = bc - ((bc*c)/2)

    # create lists of bin centers and edges for appending to 
    bin_centers = []
    bin_centers.append(bc)
    bin_edges = []
    bin_edges.append(bs)

    # constant for next bin calculation formula : (constant)
    denom_factor = 1 / (1 - (c/2))

    # initialize 1st bin center as variable bpc: previous bin center
    bpc = bc
    # loop to calculate bin centers and edges
    while bc < largest_d_value:

        # calculate the bin center and append  note:: bs = starting edge of previous bin
        bc = (bs + (bpc*c))*denom_factor
        bin_centers.append(bc)

        # calculate the bin starting edge and append
        bs = bc - ((bc*c)/2)
        bin_edges.append(bs)

        # update the value of previous bin center to the bin center we just calculated
        bpc = bc 

    # calculate the ending edge of the final bin & append to bin_edges
    bfedge = bc + ((bc*c)/2)
    bin_edges.append(bfedge)

    # write bin centers and bin edges to files
    save_dir = os.path.join(cwd, 'ml_data', phase_name)
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    with open(os.path.join(save_dir, f'{phase_name}_bin_centers_{cstring}.txt', 'w')) as f:
        for i in range(len(bin_centers)):
            f.write(f'{bin_centers[i]}\n')

    with open(os.path.join(save_dir, f'{phase_name}_bin_edges_{cstring}.txt', 'w')) as f:
        for i in range(len(bin_edges)):
            f.write(f'{bin_edges[i]}\n')


    return bin_edges


def pyunzip(zip_path, extract_path):
    with ZipFile(zip_path, 'r') as zobj:
        zobj.extractall(path=extract_path)


def make_tarfile(tar_name, source_dir):
    '''
    tar_name = "/path/to/your/output/archive.tar"
    source_directory = "/path/to/your/source/directory"
   
    make_tarfile(tar_name, source_directory)

    using basename ensures files will be extracted relative to their directory
    '''
    with tarfile.open(tar_name, "w") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))


def count_refinable_parameters_in_topas_strs(topas_str_path):

     # define ranges of bravais lattices
    dof6_range = np.arange(1,3).tolist()  # triclinic
    dof4_range = np.arange(3,16).tolist() # monoclinic
    dof3_range = np.arange(16,75).tolist() # orthorhombic
    dof2_range = np.arange(75,195).tolist() # tetragonal, trigonal, hexagonal
    dof1_range = np.arange(195,231).tolist() # cubic

    count = 0
    with open(topas_str_path, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if 'space_group' in lines[i]:
                sline = lines[i].strip().split()
                spacegroup = int(sline[1])
                if spacegroup in dof6_range:
                    count = count + 6
                if spacegroup in dof4_range:
                    count = count + 4
                if spacegroup in dof3_range:
                    count = count + 3
                if spacegroup in dof2_range:
                    count = count + 2
                if spacegroup in dof1_range:
                    count = count + 1
            
            if 'mode definitions' in lines[i]:
                for j in range(i+1, len(lines)):
                    if 'prm' in lines[j]:
                        count = count+1
                    if "}}}" in lines[j]:
                        i = j
                        break
            
            if 'mode-dependent sites' in lines[i]:
                for j in range(i+1, len(lines)):
                    if 'site' in lines[j]:
                        count = count+1
                    if "}}}" in lines[j]:                        
                        break   
                break
    
    return count


def save_dict_to_json(dictionary, filename):
    with open(filename, 'w') as f:
        json.dump(dictionary, f, indent=4)

