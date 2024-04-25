'''
Author: Dr. Dayton  Kizzire
Last Updated: 02/08/2024
Email: kizziredg@ornl.gov
'''

import os
import glob
import numpy as np
import pandas as pd
import random
from pathlib import Path, PureWindowsPath
import write_TOPAS_funcs as funcs


def main():
    cwd = os.getcwd()
    # ------------------------------------------------------------------------------------------------------
    # START OF USER INPUT
    # ------------------------------------------------------------------------------------------------------
    phase_name = 'alpha'
    sub_folder_1 = 'ml_data_generation/randomstates_sectioned'
    number_of_random_states = 250
    topas_version = 'TOPAS6'
    # important: get the refined background line from the topas out from the pawley refinement of the base of the subgroup tree and paste it here below as a string
    background_line = 'bkg  @  1.29418438` -0.155958019`  0.266659299` -0.187983295`  0.0124036031` -0.148950118`  0.283286922` -0.177585255`  0.163836522` -0.124518031`  0.165087545`  0.0483678157`'
    peak_shape_line = 'tof_sample_peakshape(lor, 0.292805187`,dsp, 23.88727`,dspsq, 20.34070`)'
    refine_beq = False
    beq_dict = {
        'Bi1': [0.0, 1.5],
        'Sn1': [0.0, 1.5],
        'O1': [0.0, 2.5], 
        'O2': [0.0, 2.5]
    }
    # beq_dict = {
    #     'Bi1': [0.0],
    #     'Sn1': [0.0],
    #     'O1':  [0.0], 
    #     'O2':  [0.0]
    # }

    '''
    beq control!
    refine_beq: bool
        if True: 'beq' in mode_dependent_sites section will change to 'beq bval_{atom_type}'
        if False: 'beq' in mode_dependent_sites section will change to 'beq !bval_{atom_type}'  -> '!' tells topas to not refine this parameter
    beq_dict: dict
        USE:
        {'atom_type': [0, 2.5]} -> will use np.random.uniform to pull a random number from the range [0.0, 2.5)
        {'atom_type': [3.2]}    -> will set the beq value to 3.2 for that atom type
        Note: atom_type can be found in the topas_str file under mode_dependent_sites section
        EX: site Bi1_1 ....  --> atom_type = Bi1
    '''

    if phase_name == 'alpha':
        experimental_data_file = f'hrp27188_bs_alpha_neutron.xye'
        TOF_x_axis_calibration = 'TOF_x_axis_calibration(t0_hrpd_bs,-49.37818`,!difc_hrpd_bs, 48255.09936_0.61297, difa_hrpd_bs,  15.23076`)'
    if phase_name == 'beta':
        experimental_data_file = f'Hrp27190_bs.xye'
        TOF_x_axis_calibration = 'TOF_x_axis_calibration(t0_hrpd_bs, 2.00496`_0.0846189708,!difc_hrpd_bs, 48255.09936_0.61297, difa_hrpd_bs, -3.22266`_0.0415199177)'
   
    experimental_data_path = os.path.join(cwd, 'TOPAS', phase_name, experimental_data_file)
    xdd_winpath = PureWindowsPath(Path(experimental_data_path).absolute())

    start_finish_X = [35000, 121000] # TOF start & finish position in microseconds

    # ------------------------------------------------------------------------------------------------------
    # END OF USER INPUT
    # ------------------------------------------------------------------------------------------------------

    cif_files = sorted(glob.glob(os.path.join(cwd, 'TOPAS', phase_name, 'subgroup_cifs_sectioned', '*')))
    cif_filenames = [os.path.basename(x) for x in cif_files]
    cif_noext = [os.path.splitext(x)[0] for x in cif_filenames]
    subgroup_nums = [x[3:] for x in cif_noext]

    str_files = sorted(glob.glob(os.path.join(cwd, 'TOPAS', phase_name, 'topas_strs_sectioned', '*' )))

    # ------------------------------------------------------------------------------------------------------
    # batch cifs for running in topas in groups of 100 cifs
    # ------------------------------------------------------------------------------------------------------
    number_of_cifs_in_a_batch = 50
    number_of_batchfiles = int(len(cif_files)/number_of_cifs_in_a_batch)
    batch_count_list = [number_of_cifs_in_a_batch]*number_of_batchfiles
    batch_remainder = len(cif_files) % number_of_cifs_in_a_batch
    if batch_remainder != 0:
        number_of_batchfiles += 1
        batch_count_list.append(batch_remainder)


    # ------------------------------------------------------------------------------------------------------
    # make directories for files to be written to if they don't already exist
    # ------------------------------------------------------------------------------------------------------
    parent_dir = os.path.join(cwd, 'TOPAS', phase_name, sub_folder_1)
    random_state = np.arange(1,number_of_random_states+1).tolist()
    random_state = [str(x).zfill(4) for x in random_state]
    random_state = [f'rs_{x}' for x in random_state]
    batch_dir = os.path.join(cwd, 'TOPAS', phase_name, 'batchfiles_topas')
    if not os.path.exists(batch_dir): os.makedirs(batch_dir)

    # ------------------------------------------------------------------------------------------------------
    # main loop for writting topas inputs for random states of each subgroup tree member 
    # ------------------------------------------------------------------------------------------------------

    input_file_list = []
    count = 0
    batch_count = 0
    for i in range(len(cif_noext)):  # for each member of the subgroup tree
        print(f'Writing random states for {cif_noext[i]}') 
        count += 1
        # get origin choice from subgroup cif file
        cif_info = funcs.get_cif_info(cif_files[i])
        # create random numbers for perturbation of strain modes (effects lattice parameters)
        random_array = np.random.uniform(low=-0.10, high=0.10, size=(number_of_random_states, 6))

        random_latts = np.random.uniform(low=-0.01, high=0.01, size=(number_of_random_states, 3))
        random_angles = np.random.uniform(low=-0.10, high=0.10, size=(number_of_random_states, 3))
        
        # for each random state of subgroup member i
        for j in range(len(random_state)): 
            
            # create random states folder for subgroup member i
            input_path = os.path.join(cwd, 'TOPAS', phase_name, sub_folder_1, f'{cif_noext[i]}_randomstates')
            if not os.path.exists(input_path): os.makedirs(input_path)

            # create windows path for hkl and xy output files
            topas_input_file = os.path.join(input_path, f'{cif_noext[i]}_{random_state[j]}.inp')
            input_file_list.append(topas_input_file)

            hkl_out_file = PureWindowsPath(Path(os.path.join(input_path, f'{cif_noext[i]}_{random_state[j]}_hkl.dat')).absolute())
            hkl_scaled_out_file = PureWindowsPath(Path(os.path.join(input_path, f'{cif_noext[i]}_{random_state[j]}_hkl_scaled.dat')).absolute())
            xy_out_file = PureWindowsPath(Path(os.path.join(input_path, f'{cif_noext[i]}_{random_state[j]}_xy.xy')).absolute())

            # get the mode amplitude lines from topas_strs (an output from isodistort)
            # topas_str_file = os.path.join(cwd, 'TOPAS', 'topas_strs', f'topas{subgroup_nums[i]}.str')
            cif_spacegroup_and_origin = [cif_info[0], cif_info[7]]
            modes_lines = funcs.generate_strain_and_symmetry_mode_lines_from_topas_str(str_files[i], cif_spacegroup_and_origin, peak_shape_line, random_array[j], refine_beq=refine_beq, beq_dict=beq_dict, use_vc_ma_macro=False)
            
            # write the topas input file for cif i random_state j
            funcs.write_inp_for_ML_with_strain_modes(topas_input_file, 
                                    modes_lines,
                                    background_line, 
                                    TOF_x_axis_calibration,
                                    start_finish_X, 
                                    xdd_winpath, 
                                    hkl_out_file, 
                                    hkl_scaled_out_file, 
                                    xy_out_file)

        # write a windows batch file to run 100 cifs of random states of input files at a time
        if count == batch_count_list[batch_count]:
            # reset count and increment batch_count by 1
            count = 0
            batch_count += 1
            inp_winpath = [PureWindowsPath(Path(x).absolute()) for x in input_file_list]

            batchfile_write = os.path.join(batch_dir, f'ml_randomstates_sectioned_batch_{str(batch_count).zfill(4)}.bat')
            with open(batchfile_write, 'w') as f:
                f.write(f'cd C:\{topas_version} \n')
                for i in range(len(inp_winpath)):
                    f.write(f'tc "{inp_winpath[i]}" \n')
            
            input_file_list = []


if __name__ == "__main__":
    main()
































