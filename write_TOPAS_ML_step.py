'''
Author: Dr. Dayton  Kizzire
Last Updated: 10/7/2024
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
    phase_name = 'bso'
    instrument_file = 'hrpd.txt'
    peakshape_file = 'hrpd_peak_type.txt'
    sub_folder_1 = 'randomstates_alpha' # for multiple datasets of the same phase, change to randomstates_#
    number_of_random_states = 250
    topas_version = 'TOPAS6'
    diffraction_type = 'neutron'   #* choices: neutron or xray
    use_strain_modes = True
    random_strain_bounds = [-0.01, 0.01]
    random_perturb_bounds = [-0.1, 0.1, -5.0, 5.0] # bounds for perturbing custom lattice parameters [abc_lo(angstrom), abc_hi(angstrom), angle_lo(degrees), angle_hi(degrees)] 
    use_custom_latparams = False
    overwrite_input_file_with_output_file = True

    # for multiphase use ------------------
    make_multiphase_classes = False
    topas_scale_value_multiphase = 20.0

    # because of how TOPAS is coded, we specify the weight % range of the second phase, if there are more than 2 phases, the remaining wt% is split between the other phases 
    multiphase_dict =  {0: {"subgroups":['0009', '0028'], 
                              'weight_range':[10.0, 40.0]},
                        1: {"subgroups":['0009','0028', '0029'], 
                              'weight_range':[15.0, 85.0]}}
    refine_beq = False

    # beq_dict = {
    #     'V': [0.0, 1.0],
    #     'P': [0.0, 1.0],
    #     'O': [0.0, 1.0], 
    #     'F': [0.0, 1.0],
    #     'Na': [0.0, 1.0]
    # }
    # beq_dict = {
    #     'Pb1': [0.5, 4.5],
    #     'Zr1': [0.5, 1.5],
    #     'Ti1': [0.5, 2.5], 
    #     'O1': [0.5, 3.0]
    # }
    beq_dict = {
        'Bi1': [0.0, 1.5],
        'Sn1': [0.0, 1.5],
        'O1': [0.0, 2.5], 
        'O2': [0.0, 2.5]
    }


    experimental_data_file = f'hrp27188_bs_alpha_neutron.xye'
    experimental_data_path = os.path.join(cwd, phase_name, experimental_data_file)
    xdd_winpath = PureWindowsPath(Path(experimental_data_path).absolute())

    # ------------------------------------------------------------------------------------------------------
    # END OF USER INPUT
    # ------------------------------------------------------------------------------------------------------

    str_files = sorted(glob.glob(os.path.join(cwd, phase_name, f'topas_strs_{phase_name}', '*.str')))

    cif_files = sorted(glob.glob(os.path.join(cwd, phase_name, f'subgroup_cifs_{phase_name}', '*.cif')))
    cif_filenames = [os.path.basename(x) for x in cif_files]
    cif_noext = [os.path.splitext(x)[0] for x in cif_filenames]
    subgroup_nums = [x[3:] for x in cif_noext]

    # topas output files from preliminary pawley refinement of subgroup tree lattice parameters
    if use_custom_latparams == True:
        easy_grab_files = sorted(glob.glob(os.path.join(cwd, phase_name, 'SGT_pawley_refinement', '*', '*.out')))
    # ------------------------------------------------------------------------------------------------------
    # batch cifs for running in topas in groups of 100 cifs
    # ------------------------------------------------------------------------------------------------------
    number_of_cifs_in_a_batch = 100
    number_of_batchfiles = int(len(cif_files)/number_of_cifs_in_a_batch)
    if make_multiphase_classes == True:
        number_of_batchfiles = int((len(cif_files)+len(multiphase_dict))/number_of_cifs_in_a_batch)
    batch_count_list = [number_of_cifs_in_a_batch]*number_of_batchfiles
    batch_remainder = len(cif_files) % number_of_cifs_in_a_batch
    if make_multiphase_classes == True:
        batch_remainder = (len(cif_files)+len(multiphase_dict)) % number_of_cifs_in_a_batch
    if batch_remainder != 0:
        number_of_batchfiles += 1
        batch_count_list.append(batch_remainder)

    topas_input_file_list = []
    count = 0
    batch_count = 0

    # ------------------------------------------------------------------------------------------------------
    # make directories for files to be written to if they don't already exist
    # ------------------------------------------------------------------------------------------------------
    parent_dir = os.path.join(cwd, phase_name, 'ml_data_generation', sub_folder_1)
    random_state = np.arange(1,number_of_random_states+1).tolist()
    random_state = [str(x).zfill(4) for x in random_state]
    random_state = [f'rs_{x}' for x in random_state]
    batch_dir = os.path.join(cwd, phase_name, 'batchfiles_topas')
    if not os.path.exists(batch_dir): os.makedirs(batch_dir)

    # ------------------------------------------------------------------------------------------------------
    # Generate lines of topas input file that are the same for every randomstate
    # ------------------------------------------------------------------------------------------------------
    
    refinement_lines = funcs.refinement_options(iters=0, continue_on_convergence=True, suppress_out_file=False)
    global_lines = funcs.global_parameters()
    xdd_lines = funcs.xdd(xdd_winpath)
    # ----- generate instrument specific lines -----
    instrument_lines = funcs.instrument_parameters(os.path.join(cwd, 'instrument_parameters', instrument_file))
    peakshape_lines = funcs.instrument_parameters(os.path.join(cwd, 'instrument_parameters', peakshape_file))

    
    phase_lines = f'''
    str
        scale 1
        r_bragg 0.0
        {peakshape_lines}
    '''

    # ------------------------------------------------------------------------------------------------------
    # main loop for writting topas inputs for random states of each subgroup tree member 
    # ------------------------------------------------------------------------------------------------------
    
    for i in range(len(str_files)):  # for each member of the subgroup tree
        print(f'Writing random states for {cif_noext[i]}') 
        count += 1

        # setup for using custom lattice parameters (ex: pawley refined latparams of subgroup tree)
        # else grabs latparams from TOPAS str file from isodistort
        if use_custom_latparams == True:
            _, spacegroup_origin = funcs.get_str_info(str_files[i], 'strain')
            latparams = funcs.get_pawley_latparams_easy_grab(easy_grab_files[i])

        else: 
            latparams, spacegroup_origin = funcs.get_str_info(str_files[i], 'strain')

        # create random numbers for perturbation of lattice parameters   
        random_perturb_lat = np.random.uniform(low=random_perturb_bounds[0], high=random_perturb_bounds[1], size=(number_of_random_states, 3))
        random_perturb_angle = np.random.uniform(low=random_perturb_bounds[2], high=random_perturb_bounds[3], size=(number_of_random_states, 3))
        random_strain = np.random.uniform(low=random_strain_bounds[0], high=random_strain_bounds[1], size=(number_of_random_states, 6))
        
        # create random states folder for subgroup member i
        input_path = os.path.join(cwd, phase_name, 'ml_data_generation', sub_folder_1, f'{cif_noext[i]}_randomstates')
        if not os.path.exists(input_path): os.makedirs(input_path)
        
        # for each random state of subgroup member i
        for j in range(len(random_state)): 

            # create windows path for hkl and xy output files
            topas_input_file = os.path.join(input_path, f'{cif_noext[i]}_{random_state[j]}.inp')
            topas_input_file_list.append(topas_input_file)

            xy_out_file = PureWindowsPath(Path(os.path.join(input_path, f'{cif_noext[i]}_{random_state[j]}_xy.xy')).absolute())
            dy_out_file = PureWindowsPath(Path(os.path.join(input_path, f'{cif_noext[i]}_{random_state[j]}_dy.xy')).absolute())
            
            xy_out_lines = f'\nOut_X_Ycalc("{xy_out_file}")\n'
            if diffraction_type == 'neutron':
                dy_out_lines = f'\nOut_d_from_tof_bragg_dayton("{dy_out_file}", d0_1, difc_1, difa_1)\n'
            if diffraction_type == 'xray':
                dy_out_lines = f'\nOut_2th_TO_dspacing_Ycalc("{dy_out_file}")\n'
            
            if use_strain_modes == True:
                mode_lines = funcs.generate_strain_and_symmetry_mode_lines_from_topas_str_v2(str_files[i], spacegroup_origin, random_strain[j], beq_dict=beq_dict)
                
                mode_lines.insert(4, f'\t\t{peakshape_lines}\n')
                mode_lines.insert(4, f'\t\tr_bragg 0.0\n')
                mode_lines.insert(4, f'\t\tscale @ 1.0\n')

            else:
                perturbed_latparams = funcs.perturb_latparams(latparams, spacegroup_origin, random_perturb_lat[j], random_perturb_angle[j])
                topas_latparam_lines = funcs.generate_latparam_lines_ml(perturbed_latparams, spacegroup_origin)
                mode_lines = funcs.generate_symmetry_mode_lines_from_topas_str(str_files[i], beq_dict=beq_dict)
            
            with open(topas_input_file, 'w') as f:

                if overwrite_input_file_with_output_file == True:
                    f.write(f'out_file="{topas_input_file}";\n')
                    f.write(f'\'WARNING WARNING WARNING: Above line only works with command line version of TOPAS. If using GUI interface, comment out the above out_file line!\n')
                    f.write(f'\'If you accept "update input" option with GUI, then INP file will be deleted!\n')
                f.write(refinement_lines)
                f.write(global_lines)
                f.write(xdd_lines)
                f.write(instrument_lines)
                
                if use_strain_modes == False:
                    f.write(phase_lines)
                    f.write(topas_latparam_lines)

                for k in range(len(mode_lines)):
                    f.write(mode_lines[k])
                f.write(xy_out_lines)
                #f.write(dy_out_lines)

            # mode_lines = funcs.generate_strain_and_symmetry_mode_lines_from_topas_str(str_files[i], 
            #                                                                           spacegroup_origin, 
            #                                                                           peakshape_lines, 
            #                                                                           random_strain[j], 
            #                                                                           custom_latparams=perturbed_latparams, 
            #                                                                           refine_beq = refine_beq, 
            #                                                                           beq_dict = beq_dict)
            
            # write the topas input file for cif i random_state j
            
            # funcs.write_inp_for_ML_with_strain_modes(beamline_choice,
            #                                          topas_input_file, 
            #                                          mode_lines, 
            #                                          start_finish_X, 
            #                                          xdd_winpath, 
            #                                          hkl_out_file=hkl_out_file, 
            #                                          xy_out_file=xy_out_file,
            #                                          dy_out_file=dy_out_file)
            
        # write a windows batch file to run 100 cifs of random states of input files at a time
        if count == batch_count_list[batch_count]:
            # reset count and increment batch_count by 1
            count = 0
            batch_count += 1
            inp_winpath = [PureWindowsPath(Path(x).absolute()) for x in topas_input_file_list]

            batchfile_write = os.path.join(batch_dir, f'{sub_folder_1}_randomstates_{str(batch_count).zfill(4)}.bat')
            with open(batchfile_write, 'w') as f:
                f.write(f'cd C:\{topas_version} \n')
                for i in range(len(inp_winpath)):
                    f.write(f'tc "{inp_winpath[i]}" \n')
            
            topas_input_file_list = []

    # if make_multiphase_classes == True:
        
    #     for i in range(len(multiphase_dict)):  # for each mixed phase folder
    #         print(f'Writing random states for mixed phase of {multiphase_dict[i]["subgroups"]}')
    #         count+=1
    #         num_phases = len(multiphase_dict[i]["subgroups"])

    #         # create name of folder and files for mixed phase states
    #         if num_phases == 2:
    #             mixed_name = f'mixed_{multiphase_dict[i]["subgroups"][0]}_{multiphase_dict[i]["subgroups"][1]}'
    #         if num_phases == 3:
    #             mixed_name = f'mixed_{multiphase_dict[i]["subgroups"][0]}_{multiphase_dict[i]["subgroups"][1]}_{multiphase_dict[i]["subgroups"][2]}'
    #         if num_phases >= 4:
    #             raise Exception(f"Error: Max number of phases in a single sample exceeds 3, offending index in multiphase_dict = {multiphase_dict[i]}")
            
    #         # build list of paths of str files to use to create mixed phases - above random states loop since these don't change for each random state
    #         str_files = []
    #         latparams = []
    #         spacegroup_origin = []
    #         for j in range(num_phases):
    #             str_files.append(glob.glob(os.path.join(cwd, phase_name, f'topas_strs_{phase_name}', f'topas{multiphase_dict[i]["subgroups"][j]}.str'))[0])
    #             if use_custom_latparams == True:
    #                 lat_params_pp = funcs.get_pawley_latparams_easy_grab(glob.glob(os.path.join(cwd, phase_name, 'SGT_pawley_refinement', f'cif{multiphase_dict[i]["subgroups"][j]}', '*.out'))[0])
    #                 _, spacegroup_origin_pp = funcs.get_str_info(str_files[j], 'strain')
    #             else:
    #                 lat_params_pp, spacegroup_origin_pp = funcs.get_str_info(str_files[j], 'strain')
    #             latparams.append(lat_params_pp)
    #             spacegroup_origin.append(spacegroup_origin_pp)

    #         # create random phase weights in bounds
    #         random_weights = np.random.uniform(low=multiphase_dict[i]['weight_range'][0], high=multiphase_dict[i]['weight_range'][1], size=(number_of_random_states, 1))
            
    #         # create random states folder for subgroup member i
    #         write_folder = os.path.join(cwd, phase_name, 'ml_data_generation', sub_folder_1, f'{mixed_name}_randomstates')
    #         if not os.path.exists(write_folder): os.makedirs(write_folder)
            
    #         for k in range(len(random_state)):
    #             topas_input_file = os.path.join(write_folder, f'{mixed_name}_{random_state[k]}.inp')
    #             topas_input_file_list.append(topas_input_file)

    #             hkl_out_file = PureWindowsPath(Path(os.path.join(write_folder, f'{mixed_name}_{random_state[k]}_hkl.dat')).absolute())
    #             xy_out_file = PureWindowsPath(Path(os.path.join(write_folder, f'{mixed_name}_{random_state[k]}_xy.xy')).absolute())
    #             dy_out_file = PureWindowsPath(Path(os.path.join(write_folder, f'{mixed_name}_{random_state[k]}_dy.xy')).absolute())

    #             xdd_wt_percent_lines = funcs.generate_multiphase_wt_percent_lines(num_phases, topas_scale_value_multiphase, random_weights[k])
    #             if num_phases == 2: 
    #                 wt_dets_lines = ['\t\twt_dets(1, 2, 0)\n', '\t\twt_dets(2, 2, 0)\n']
    #             if num_phases == 3:
    #                 wt_dets_lines = ['\t\twt_dets(1, 3, 0)\n', '\t\twt_dets(2, 3, 0)\n', '\t\twt_dets(3, 3, 0)\n']
                

    #             strain_and_distortion_modes = []
    #             phase_counter = 1
    #             for l in range(num_phases):
    #                 if use_custom_latparams == True:
    #                     random_perturb_lat = np.random.uniform(low=random_perturb_bounds[0], high=random_perturb_bounds[1], size=(3))
    #                     random_perturb_angle = np.random.uniform(low=random_perturb_bounds[0], high=random_perturb_bounds[1], size=(3))
    #                     perturbed_latparams = funcs.perturb_latparams(latparams[l], spacegroup_origin[l], random_perturb_lat, random_perturb_angle)
                    
    #                 random_strain = np.random.uniform(low=random_strain_bounds[0], high=random_strain_bounds[1], size=(1, 6))
                    
    #                 tmp_mode_lines =  funcs.generate_strain_and_symmetry_mode_lines_from_topas_str(str_files[l], 
    #                                                                                                 spacegroup_origin[l], 
    #                                                                                                 peakshape_lines, 
    #                                                                                                 random_strain, 
    #                                                                                                 custom_latparams=perturbed_latparams,
    #                                                                                                 refine_beq = refine_beq, 
    #                                                                                                 beq_dict = beq_dict,
    #                                                                                                 use_vc_ma_macro=False,
    #                                                                                                 multiphase=True,
    #                                                                                                 wt_dets_macro = wt_dets_lines[l],
    #                                                                                                 phase_count = phase_counter)
    #                 # use extend instead of append to make sure strain_and_distortion_modes stays 1d 
    #                 strain_and_distortion_modes.extend(tmp_mode_lines)
    #                 phase_counter += 1
                
    #             funcs.write_inp_for_multiphase_ML_with_strain_modes(beamline_choice, 
    #                                                                 topas_input_file, 
    #                                                                 strain_and_distortion_modes, 
    #                                                                 start_finish_X, 
    #                                                                 xdd_winpath, 
    #                                                                 xdd_wt_percent_lines = xdd_wt_percent_lines,
    #                                                                 xy_out_file = xy_out_file, 
    #                                                                 dy_out_file=dy_out_file,
    #                                                                 hkl_out_file = hkl_out_file)

    #         # write a windows batch file to run 100 cifs of random states of input files at a time
    #         if count == batch_count_list[batch_count]:
    #             # reset count and increment batch_count by 1
    #             count = 0
    #             batch_count += 1
    #             inp_winpath = [PureWindowsPath(Path(x).absolute()) for x in topas_input_file_list]

    #             batchfile_write = os.path.join(batch_dir, f'ml_strain_randomstates_batch_{str(batch_count).zfill(4)}.bat')
    #             with open(batchfile_write, 'w') as f:
    #                 f.write(f'cd C:\{topas_version} \n')
    #                 for i in range(len(inp_winpath)):
    #                     f.write(f'tc "{inp_winpath[i]}" \n')
                
    #             topas_input_file_list = []



if __name__ == "__main__":
    main()



