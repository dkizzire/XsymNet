import os
import glob
import pandas as pd
import torch
import tarfile
import utility_functions as util

cwd = os.getcwd()

do_hkl_intensities = True
do_diffraction_pattern = False

phase_name = 'beta'
experimental_hkl_filename = 'exp_hkl.dat'
experimental_bragg_filename = 'exp_xy.xy'
ml_randomstates_dir_name = 'randomstates_strain'

hkl_tarname = f'{phase_name}_hkl_tensors_strain_notbinned.tar'
bragg_tarname = f'{phase_name}_bragg_strain_tensors_unscaled.tar'

transfer_directory = os.path.join(cwd, 'ml_data', phase_name)
if not os.path.exists(transfer_directory): os.makedirs(transfer_directory)

if do_hkl_intensities == True:

    data_suffix = f'*hkl.dat'  # either *hkl.dat (for unscaled hkl intensities)   or  *hkl_scaled.dat  (for topas scaled hkl intensities)  
    # directory to store tensors   either hkl_tensors_notbinned_unscaled    or     hkl_tensors_notbinned_topas_scaled
    dir_to_store_data = 'hkl_tensors_strain_notbinned_unscaled'

    # path and filename to store tar file of all hkl tensors (to transfer to GPU for binning procedure)
    tar_filepath = os.path.join(transfer_directory,  f'{phase_name}_hkl_tensors_notbinned.tar')
    save_tensor_dir = os.path.join(transfer_directory, dir_to_store_data)
    if not os.path.exists(save_tensor_dir): os.makedirs(save_tensor_dir)

    
    # grab hkl lists from all randomstates
    hkl_lists = sorted(glob.glob(os.path.join(cwd, 'TOPAS', phase_name, 'ml_data_generation', ml_randomstates_dir_name,'*', data_suffix)))
    # prepare filenames for writing tensors
    hkl_lists_basenames = [os.path.basename(x) for x in hkl_lists]
    hkl_lists_basenames_noexts = [os.path.splitext(x)[0] for x in hkl_lists_basenames]
    hkl_tensor_writepaths = [f'{save_tensor_dir}/{hkl_lists_basenames_noexts[x]}.pt' for x in range(len(hkl_lists_basenames_noexts))]

    # do the same for experimental hkl list
    experimental_hkl = glob.glob(os.path.join(cwd, 'TOPAS', phase_name, 'experimental_refinement', experimental_hkl_filename))
    tensor_expt_writepath = os.path.join(transfer_directory, f'{phase_name}_hkl_expt_notbinned.pt')

    # convert hkl lists to tensors
    for i in range(len(hkl_lists)):
        tensor_filename = util.convert_hkl_list_to_tensors(hkl_lists[i], hkl_tensor_writepaths[i])

    # convert experimental hkl list to tensor
    util.convert_hkl_list_to_tensors(experimental_hkl[0], tensor_expt_writepath)
    tar_name = os.path.join(transfer_directory, hkl_tarname)
    print('Finished converting *hkl.dat files to tensors')
    print(f'Files stored at {save_tensor_dir}')
    print('Making Tarball of unbinned hkl Tensors')
    util.make_tarfile(tar_name, save_tensor_dir)
    print(f'Tarball of unbinned hkl tensor files to transfer to GPU for binning stored at {tar_name}')


if do_diffraction_pattern == True:

    data_suffix = f'*.xy'
    dir_to_store_data = 'bragg_strain_randomstates_tensors_unscaled'
    save_tensor_dir = os.path.join(transfer_directory, dir_to_store_data)
    if not os.path.exists(save_tensor_dir): os.makedirs(save_tensor_dir)

    cif_dirs = sorted(glob.glob(os.path.join(cwd, 'TOPAS', phase_name, 'ml_data_generation', ml_randomstates_dir_name,'*')))
    dirs_basenames = [os.path.basename(x) for x in cif_dirs]
    
    tensor_list_percif = []
    for i in range(len(cif_dirs)):
        print(f'Working on {dirs_basenames[i]}')
        bragg_files = sorted(glob.glob(os.path.join(cif_dirs[i], '*.xy')))
        for j in range(len(bragg_files)):
            tensor_list_percif.append(util.convert_bragg_to_tensors(bragg_files[j]))
        
        stacked_tensor = torch.vstack(tensor_list_percif)
        torch.save(stacked_tensor, os.path.join(save_tensor_dir, f'{dirs_basenames[i]}_bragg.pt'))
        tensor_list_percif = []
    
    print('Finished creating bragg tensors of randomstates for every cif\n')
    print('Making tar file of Bragg tensors\n')

    tar_name = os.path.join(transfer_directory, bragg_tarname)
    util.make_tarfile(tar_name, save_tensor_dir)
    
    # make tensor from experimental data
    path2exp = os.path.join(cwd, 'TOPAS', phase_name, 'experimental_refinement', experimental_bragg_filename)
    tempexp = util.convert_bragg_to_tensors(path2exp)
    torch.save(tempexp, os.path.join(transfer_directory, f'{phase_name}_bragg_expt_unscaled.pt'))

