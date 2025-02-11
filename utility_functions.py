'''
Author: Dr. Dayton  Kizzire
Last Updated: 02/08/2024
Email: kizziredg@ornl.gov
'''

import os
import glob
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from zipfile import ZipFile
import tarfile
import json
import math 

def determine_gpu_or_cpu(gpu_index):
    whichgpu = f'cuda:{gpu_index}'

    determine = (
        "cuda" if torch.cuda.is_available()
        else "cpu" )  

    if determine == 'cuda':
        print(f"Using {determine} device")
        device = torch.device(whichgpu)
        print(f'CUDA device properties: \n', torch.cuda.get_device_properties(device))
        print(f'Pytorch CUDA state initialized : {torch.cuda.is_initialized()}')


    elif determine == 'cpu':
        device = 'cpu'

    return device


def convert_hkl_list_to_tensors(filepath, tensor_writepath):
    '''
    '''

    # read in hkl lists from topas output and sort the hkl lists by ascending d_spacing
    hkl = pd.read_csv(filepath, sep='\s+', header = None, names = ['d','i'], dtype=np.float64)
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
    with open(os.path.join(save_dir, f'{phase_name}_bin_centers_{cstring}.txt'), 'w') as f:
        for i in range(len(bin_centers)):
            f.write(f'{bin_centers[i]}\n')

    with open(os.path.join(save_dir, f'{phase_name}_bin_edges_{cstring}.txt'), 'w') as f:
        for i in range(len(bin_edges)):
            f.write(f'{bin_edges[i]}\n')

    print(f'Bin centers and bin edges files were written to {save_dir}')

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


def save_dict_to_json(dictionary, filename):
    with open(filename, 'w') as f:
        json.dump(dictionary, f, indent=4)


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


def count_dofs_topas_strs_dir(topas_str_dir, dof_count_save_path):
    strs = sorted(glob.glob(os.path.join(topas_str_dir, '*')))
    names = [os.path.basename(x).strip('topas').strip('.str') for x in strs]
    dof_dict = {}
    for i in range(len(strs)):
        dof = count_refinable_parameters_in_topas_strs(strs[i])
        dof_dict[names[i]] = dof
    
    save_dict_to_json(dof_dict, dof_count_save_path)
    

def straincell(pa0, pb0, pc0, pal0, pbe0, pga0):
    radian = 57.2957795
    palpha0 = pal0/radian
    pbeta0  = pbe0/radian
    pgamma0 = pga0/radian
    # parent-lattice basis vectors in cartesian coords
    zzz  = (math.cos(palpha0)- math.cos(pbeta0)*math.cos(pgamma0))/math.sin(pgamma0)
    a0x = pa0
    b0x = pb0*math.cos(pgamma0)
    b0y = pb0*math.sin(pgamma0)
    c0x = pc0*math.cos(pbeta0)
    c0y = pc0*zzz
    c0z = pc0*math.sqrt(1 - math.cos(pbeta0)**2 - zzz**2)

    bv = [a0x, b0x, b0y, c0x, c0y, c0z]

    return bv


def celltransform(s_mat, p_mat, bv, tm):

    pe1, pe2, pe3, pe4, pe5, pe6 = p_mat[0], p_mat[1], p_mat[2], p_mat[3], p_mat[4], p_mat[5]

    a0x, b0x, b0y, c0x, c0y, c0z = bv[0], bv[1], bv[2], bv[3], bv[4], bv[5]

    t11, t12, t13, t21, t22, t23, t31, t32, t33 = tm[0], tm[1], tm[2], tm[3], tm[4], tm[5], tm[6], tm[7], tm[8]  

    radian = 57.2957795
    #parent-lattice basis vectors in cartesian coords
    pv11 = a0x*(pe1 + 1)
    pv21 = a0x*(pe6/2)
    pv31 = a0x*(pe5/2)
    pv12 = b0x*(pe1 + 1) + b0y*(pe6/2)
    pv22 = b0x*(pe6/2)   + b0y*(pe2 + 1)
    pv32 = b0x*(pe5/2)   + b0y*(pe4/2)
    pv13 = c0x*(pe1 + 1) + c0y*(pe6/2) + c0z*(pe5/2)
    pv23 = c0x*(pe6/2)+ c0y*(pe2 + 1) + c0z*(pe4/2)
    pv33 = c0x*(pe5/2)   + c0y*(pe4/2) + c0z*(pe3 + 1)
    #supercell basis vectors in cartesian coords
    sv11 = (t11)*pv11 + (t12)*pv12 + (t13)*pv13
    sv21 = (t11)*pv21 + (t12)*pv22 + (t13)*pv23
    sv31 = (t11)*pv31 + (t12)*pv32 + (t13)*pv33
    sv12 = (t21)*pv11 + (t22)*pv12 + (t23)*pv13
    sv22 = (t21)*pv21 + (t22)*pv22 + (t23)*pv23
    sv32 = (t21)*pv31 + (t22)*pv32 + (t23)*pv33
    sv13 = (t31)*pv11 + (t32)*pv12 + (t33)*pv13
    sv23 = (t31)*pv21 + (t32)*pv22 + (t33)*pv23
    sv33 = (t31)*pv31 + (t32)*pv32 + (t33)*pv33
    #distorted supercell parameters
    sa = math.sqrt(sv11**2+sv21**2+sv31**2)
    sb = math.sqrt(sv12**2+sv22**2+sv32**2)
    sc = math.sqrt(sv13**2+sv23**2+sv33**2)
    salpha = math.acos((sv12*sv13+sv22*sv23+sv32*sv33)/(sb*sc))
    sbeta  = math.acos((sv11*sv13+sv21*sv23+sv31*sv33)/(sa*sc))
    sgamma = math.acos((sv11*sv12+sv21*sv22+sv31*sv32)/(sa*sb))
    sal = salpha*radian
    sbe = sbeta*radian
    sga = sgamma*radian

    # print()
    # print(f's_vars = [{s1:.4f}, {s2:.4f}, {s3:.4f}, {s4:.4f}, {s5:.4f}, {s6:.4f}]')
    # print(f'a = {sa}')
    # print(f'b = {sb}')
    # print(f'c = {sc}')
    # print(f'al = {sal}')
    # print(f'be = {sbe}')
    # print(f'ga = {sga}')
    # print()

    return sa, sb, sc, sal, sbe, sga