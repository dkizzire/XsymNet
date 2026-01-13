import os
import glob
import numpy as np
from pathlib import Path, PureWindowsPath
import write_TOPAS_funcs as funcs


def main():
    cwd = os.getcwd()
    # ------------------------------------------------------------------------------------------------------
    # START OF USER INPUT
    # ------------------------------------------------------------------------------------------------------
    material_name = 'bso'
    diffraction_type = 'neutron'   # choices: neutron or xray
    instrument_file = 'hrpd.txt'
    peakshape_file = 'hrpd_peak_type.txt'
    sub_folder_1 = 'Rietveld_top_5'
    phase_name = 'alpha'
    
    use_vc_macro = False
    path_to_topas_directory = r'C:\TOPAS6'           #^ windows path to your TOPAS directory; should be 'C:\TOPAS#' 

    # use selected subgroups
    use_only_selected_subgroups = True
    if use_only_selected_subgroups:
        selected_subgroups = ['0093', '0031', '0029', '0088', '0089']
    else:
        selected_subgroups = ['*']

    # strain mode or custom latparam refinement
    use_strain_modes = True
    random_strain_bounds = [-0.002, 0.002]          #^ USED if using strain modes default -> [-0.001, 0.001]  ::NOTE:: tiny changes here make a BIG difference in lattice parameter
    # when not using strain modes, use the custom_latparam_bounds below
    custom_latparam_bounds = [[-0.1, -0.05], [-0.1, 0.1], [0.1, 0.2], [-1, 1], [-1, 1], [-1, 1]] # custom lats for PZT90


    #^ EXPERIMENTAL XY FILE to use for xdd section of TOPAS when creating randomstates
    experimental_data_file = f'Hrp27190_bs.xye'
    experimental_data_path = os.path.join(cwd, material_name, experimental_data_file)
    assert os.path.exists(experimental_data_path), f'Error: {experimental_data_path} does not exist. Please check the path to your experimental data file.'
    xdd_winpath = PureWindowsPath(Path(experimental_data_path).absolute())

    #^ CUSTOM MACROS FILE (appended to bottom of input files)
    use_custom_macros = True
    if use_custom_macros:
        custom_macros_file = 'hrpd_macros.txt'
        custom_macro_path = os.path.join(cwd, 'instrument_parameters', custom_macros_file) 
        assert os.path.exists(custom_macro_path), f'Error: {custom_macro_path} does not exist. Please check the path to your custom macros file.'
        custom_macros = funcs.instrument_parameters(custom_macro_path)
    else:
        custom_macros = '\n'

    #^ CUSTOM PRE-XDD FILE (custom macros/text inserted before xdd section)
    use_pre_xdd = False
    if use_pre_xdd:
        pre_xdd_file = 'nomad_bank_5_pre_xdd.txt'
        pre_xdd_file_path = os.path.join(cwd, 'instrument_parameters', pre_xdd_file)
        assert os.path.exists(pre_xdd_file_path), f'Error: {pre_xdd_file_path} does not exist. Please check the path to your pre xdd file.'
        pre_xdd_lines = funcs.instrument_parameters(os.path.join(cwd, 'instrument_parameters', pre_xdd_file))
    else:
        pre_xdd_lines = '\n'

    beq_dict = {
        'Bi1': [0.0, 1.0],
        'Sn1': [0.0, 1.0],
        'O1': [0.0, 1.0], 
        'O2': [0.0, 1.0]
    }

    ''' 
    vc macro: val on continue macro in topas
    vc_matrix[[v1, v2, min, max], ....] --> fill in all 4 values for lattice parameters a, b, c, alpha, beta, gamma
    demo use:
    macro vc { val_on_continue = Val*Rand(v1,v2); }     --> where the current value of lattice parameter will be multiplied by a random value between v1 & v2
    a lpa 10.725 vc min 7.0 max 25.0                    --> where 7.0 & 25.0 are the lower & upper bounds of the lattice parameter refinement
    '''
    vc_matrix =[[0.9, 1.1, 7.0, 25.0],
                [0.9, 1.1, 7.0, 25.0],
                [0.9, 1.1, 7.0, 25.0],
                [0.95, 1.05, 70.0, 130.0],
                [0.95, 1.05, 70.0, 130.0],
                [0.95, 1.05, 70.0, 130.0],
                ]

    # create batchfile for running all pawley refinments sequentially
    batchfile_folder = os.path.join(cwd, material_name, 'batchfiles_topas')
    if not os.path.exists(batchfile_folder): os.makedirs(batchfile_folder)
    batchfile_path = os.path.join(batchfile_folder, f'rietveld_refinement_{material_name}.bat')

    # ------------------------------------------------------------------------------------------------------
    # END OF USER INPUT
    # ------------------------------------------------------------------------------------------------------

    str_path = os.path.join(cwd, material_name, f'topas_strs_{material_name}')
    if use_only_selected_subgroups:
        str_files = [os.path.join(str_path, f'topas{x}.str') for x in selected_subgroups]
    else:
        str_files = sorted(glob.glob(os.path.join(str_path, '*.str')))
    assert len(str_files) > 0, f'Error: No .str files found in {os.path.join(cwd, material_name, f"topas_strs_{material_name}")}. Please check the path to your .str files.'

    str_filenames = [os.path.basename(x) for x in str_files]
    str_noext = [os.path.splitext(x)[0] for x in str_filenames]
    subgroup_nums = [x[-4:] for x in str_noext]

    # cif_files = sorted(glob.glob(os.path.join(cwd, material_name, f'subgroup_cifs_{material_name}', '*')))
    # assert len(cif_files) > 0, f'Error: No CIF files found in {os.path.join(cwd, material_name, f"subgroup_cifs_{material_name}")}. Please check the path to your CIF files.'
    # cif_filenames = [os.path.basename(x) for x in cif_files]
    # cif_noext = [os.path.splitext(x)[0] for x in cif_filenames]
    # subgroup_nums = [x[3:] for x in cif_noext]

    

    refinement_lines = funcs.refinement_options(iters=10000, continue_on_convergence=False, suppress_out_file=False)
    global_lines = funcs.global_parameters()
    xdd_lines = funcs.xdd(xdd_winpath)
    # ----- generate instrument specific lines -----
    instrument_file_path = os.path.join(cwd, 'instrument_parameters', instrument_file)
    assert os.path.exists(instrument_file_path), f'Error: {instrument_file_path} does not exist. Please check the path to your instrument file.'
    peakshape_file_path = os.path.join(cwd, 'instrument_parameters', peakshape_file)
    assert os.path.exists(peakshape_file_path), f'Error: {peakshape_file_path} does not exist. Please check the path to your peakshape file.'
    instrument_lines = funcs.instrument_parameters(os.path.join(cwd, 'instrument_parameters', instrument_file))
    peakshape_lines = funcs.instrument_parameters(os.path.join(cwd, 'instrument_parameters', peakshape_file))
    
    phase_lines = f'''
    str
    \tscale @ 1
    \tr_bragg 0\n'''

    input_dict = {'refinement_lines' :refinement_lines,
                'global_lines' :global_lines,
                'pre_xdd_lines' :pre_xdd_lines,
                'xdd_lines' :xdd_lines,
                'phase_lines' :phase_lines,
                'instrument_lines' :instrument_lines,
                'peakshape_lines' :peakshape_lines,
                'custom_macros' :custom_macros}
    
    input_winpath_list = []
    for i in range(len(str_files)):
        str_file = str_files[i]
        str_name = os.path.splitext(os.path.basename(str_file))[0]

        # grab lattice parameters and spacegroup from str file
        latparams, spacegroup_origin = funcs.get_str_info(str_file, 'strain')
        #^ line above is actually redundant with the new implementation of generate mode lines in write_TOPAS_funcs

        # create random numbers for augmenting strain values or latparam values directly
        if use_strain_modes:
            random_strain = np.random.uniform(low=random_strain_bounds[0], high=random_strain_bounds[1], size=6)
        if not use_strain_modes:
            random_strain = np.column_stack([np.round(np.random.uniform(low=low, high=high, size=1),6) for low, high in custom_latparam_bounds])
            random_strain = random_strain.flatten()

        # input file to write
        topas_input_path = os.path.join(cwd, material_name, sub_folder_1, phase_name) 
        if not os.path.exists(topas_input_path): os.makedirs(topas_input_path)
        topas_input_file = os.path.join(topas_input_path, f'{material_name}_{subgroup_nums[i]}.inp')
        topas_input_winpath = PureWindowsPath(Path(topas_input_file).absolute())
        input_winpath_list.append(topas_input_winpath)

        # windows path for xy and dy out files (so topas can write files)
        xy_out_file = PureWindowsPath(Path(os.path.join(topas_input_path, f'rietveld_{subgroup_nums[i]}_xy.xy')).absolute())
        dy_out_file = PureWindowsPath(Path(topas_input_path, f'rietveld_{subgroup_nums[i]}_dy.xy').absolute())
        cif_out_file = PureWindowsPath(Path(os.path.join(topas_input_path, f'rietveld_str_{subgroup_nums[i]}.cif')).absolute())

        xy_out_lines = f'\nOut_X_Ycalc("{xy_out_file}")\n'
        cif_out_lines = f'\nOut_CIF_STR("{cif_out_file}")\n'
        if diffraction_type == 'neutron':
            dy_out_lines = f'\nOut_d_from_tof_bragg_dayton("{dy_out_file}", d0_1, difc_1, difa_1)\n'
        if diffraction_type == 'xray':
            dy_out_lines = f'\nOut_2th_TO_dspacing_Ycalc("{dy_out_file}")\n'

        if use_strain_modes:
            # create strain and symmetry mode lines from str_file with random values
            mode_lines = funcs.generate_mode_lines(str_file, spacegroup_origin, random_strain, use_strain_modes=True, refine_strain=True, beq_dict=beq_dict, refine_beq=True, refine_distortion=True, distortion_limits_off=True)

        if not use_strain_modes:
            mode_lines = funcs.generate_mode_lines(str_file, spacegroup_origin, random_strain, use_strain_modes=False, beq_dict=beq_dict, refine_beq=True, refine_distortion=True, distortion_limits_off=True)

        # insert peakshape, r_bragg, and scale into mode lines
        mode_lines.insert(4, f'\t\tphase_name {str_name}\n')
        mode_lines.insert(4, f'{input_dict["peakshape_lines"]}\n')
        mode_lines.insert(4, f'\t\tr_bragg 0.0\n')
        mode_lines.insert(4, f'\t\tscale @ 100.0\n')

        
        # write TOPAS input file for this random state
        with open(topas_input_file, 'w') as f:
            f.write(input_dict['refinement_lines'])
            f.write(input_dict['global_lines'])
            f.write(input_dict['pre_xdd_lines'])
            f.write(input_dict['xdd_lines'])
            f.write(input_dict['instrument_lines'])
            
            # if use_strain_modes == False:
            #     f.write(phase_lines)
            #     f.write(topas_latparam_lines)

            f.write('\n')
            for k in range(len(mode_lines)):
                f.write(mode_lines[k])
            f.write(xy_out_lines)
            f.write(cif_out_lines)
            f.write(input_dict['custom_macros'])

        
    # write file for batch running topas pawley refinements of subgrouptree
    with open(batchfile_path, 'w') as f:
        f.write(f'cd {path_to_topas_directory} \n')
        for i in range(len(input_winpath_list)):
            f.write(f'tc {input_winpath_list[i]} \n')

    print(f'\nWrote {len(str_files)} TOPAS Rietveld refinement input files to {os.path.join(cwd, material_name, sub_folder_1, phase_name)}\n')
    print(f'\nWrote batchfile to run all refinements to {batchfile_path}\n')

if __name__ == "__main__":
    main()