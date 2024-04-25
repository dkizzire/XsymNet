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
    sub_folder_1 = 'ml_data_generation/randomstates_strain'
    number_of_random_states = 200
    topas_version = 'TOPAS6'
    random_strain_bounds = [-0.01, 0.01]

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
    beq[0] = True : beq is set as a refinable parameter in topas  //  False : beq is not refined
    len(beq) = 2 : beq is set as input value for all sites/symmetry modes
    len(beq) = 3 : beq is pulled from a guassian distribution where beq[1] is the center and beq[2] is the standard deviation 
    '''

    if phase_name == 'alpha':
        experimental_data_file = f'hrp27188_bs_alpha_neutron.xye'
        TOF_x_axis_calibration = 'TOF_x_axis_calibration(t0_hrpd_bs,-48.35479`,!difc_hrpd_bs, 48255.09936_0.61297, difa_hrpd_bs,  16.7979`)'
        TOF_sample_peakshape = 'tof_sample_peakshape(lor, 0.200007268`,dsp, 1.00000`,dspsq, 30.43510`)'
        background_line = 'bkg  @  1.23166856` -0.0124816464`  0.209502581` -0.119357101` -0.0364802546` -0.0805659581`  0.280382189` -0.180514454`  0.157958349` -0.129704358`  0.215427676`  0.0027121254`'
    
    if phase_name == 'beta':
        experimental_data_file = f'Hrp27190_bs.xye'
        TOF_x_axis_calibration = 'TOF_x_axis_calibration(t0_hrpd_bs, 24.28584`,!difc_hrpd_bs, 48255.09936_0.61297, difa_hrpd_bs, -25.44401`)'
        TOF_sample_peakshape = 'tof_sample_peakshape(lor, 0.143711169`, dsp, 11.64083`, dspsq, 11.71737`)'
        background_line = 'bkg  @  1.67370339`  0.0515350436`  0.00972952795` -0.12759364`  0.20547493` -0.177201821`  0.116513928` -0.0589855793`  0.0192231893`  0.0184925627` -0.000494961725` -0.000104051593`'

    experimental_data_path = os.path.join(cwd, 'TOPAS', phase_name, experimental_data_file)
    xdd_winpath = PureWindowsPath(Path(experimental_data_path).absolute())

    start_finish_X = [35000, 121000] # TOF start & finish position in microseconds

    # ------------------------------------------------------------------------------------------------------
    # END OF USER INPUT
    # ------------------------------------------------------------------------------------------------------

    str_files = sorted(glob.glob(os.path.join(cwd, 'TOPAS', f'topas_strs_alpha', '*.str')))

    cif_files = sorted(glob.glob(os.path.join(cwd, 'subgroup_cifs_alpha_new', '*.cif')))
    cif_filenames = [os.path.basename(x) for x in cif_files]
    cif_noext = [os.path.splitext(x)[0] for x in cif_filenames]
    subgroup_nums = [x[3:] for x in cif_noext]

    # ------------------------------------------------------------------------------------------------------
    # batch cifs for running in topas in groups of 100 cifs
    # ------------------------------------------------------------------------------------------------------
    number_of_cifs_in_a_batch = 100
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
    for i in range(len(str_files)):  # for each member of the subgroup tree
        print(f'Writing random states for {cif_noext[i]}') 
        count += 1
        # get origin choice from subgroup cif file
        latparams, spacegroup_origin = funcs.get_str_info(str_files[i], 'strain')

        # create random numbers for perturbation of lattice parameters
        random_strain = np.random.uniform(low=random_strain_bounds[0], high=random_strain_bounds[1], size=(number_of_random_states, 6))
        
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
            # mode_amplitude_lines = funcs.generate_symmetry_mode_lines_from_topas_str(str_files[i], refine_beq=refine_beq, beq_dict=beq_dict, use_vc_ma_macro=False )
            mode_lines = funcs.generate_strain_and_symmetry_mode_lines_from_topas_str(str_files[i], 
                                                                                      spacegroup_origin, 
                                                                                      TOF_sample_peakshape, 
                                                                                      random_strain[j], 
                                                                                      refine_beq = refine_beq, 
                                                                                      beq_dict = beq_dict)
            
            # write the topas input file for cif i random_state j
            funcs.write_inp_for_ML_with_strain_modes(topas_input_file, 
                                                     mode_lines, 
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

            batchfile_write = os.path.join(batch_dir, f'ml_strain_randomstates_batch_{str(batch_count).zfill(4)}.bat')
            with open(batchfile_write, 'w') as f:
                f.write(f'cd C:\{topas_version} \n')
                for i in range(len(inp_winpath)):
                    f.write(f'tc "{inp_winpath[i]}" \n')
            
            input_file_list = []


if __name__ == "__main__":
    main()
































