import os
from glob import glob
from pathlib import Path, PureWindowsPath
from data_prep import write_TOPAS_funcs as funcs


def main():
    cwd = os.getcwd()
    # ------------------------------------------------------------------------------------------------------
    # START OF USER INPUT
    # ------------------------------------------------------------------------------------------------------
    material_name = 'fe2o3'
    data_folder = 'data_bank_5_truncated'
    write_refinement_folder = 'pawley_P1'

    diffraction_type = 'neutron'   # choices: neutron or xray
    instrument_file = 'nomad_bank_5.txt'  
    peakshape_file = 'nomad_bank_5_peak_type.txt' 
    
    use_subgroups = ['0022']
    str_files = [os.path.join(cwd, material_name, f'topas_strs_{material_name}', f'topas{x}.str') for x in use_subgroups]

    use_strain_modes = True
    use_vc_macro = False
    # set convergence criteria
    conv_criteria = 0.0001

    #^ choose single experimental data file or entire folder
    use_single_experimental_data = False
    if use_single_experimental_data:
        # single experimental data file
        experimental_data_file = 'SNAP063403_bank-0.xye'
        experimental_data_path = sorted(glob(os.path.join(cwd, material_name, data_folder, experimental_data_file)))
    else:
        # entire folder of experimental data files
        experimental_data_path = sorted(glob(os.path.join(cwd, material_name, data_folder, '*')))
        experiment_names = [os.path.splitext(os.path.basename(x))[0] for x in experimental_data_path]
    # convert to windows path for TOPAS
    # xdd_winpath = PureWindowsPath(Path(experimental_data_path[0]).absolute())
    xdd_winpath = [PureWindowsPath(Path(x).absolute()) for x in experimental_data_path]
    # xdd_winpath = [PureWindowsPath(Path(experimental_data_path[i]).absolute()) for i in range(len(experimental_data_path))]


    #^ custom macros file (appended to bottom of input files)
    use_custom_macros = True
    if use_custom_macros:
        custom_macros_file = 'nomad_bank_5_macros.txt'
        custom_macros = funcs.instrument_parameters(os.path.join(cwd, 'data_prep', 'instrument_parameters', custom_macros_file))
    else:
        custom_macros = '\n'

    #^ custom lines to add before xdd section
    use_pre_xdd = True
    if use_pre_xdd:
        pre_xdd_file = 'nomad_bank_5_pre_xdd.txt'
        pre_xdd_lines = funcs.instrument_parameters(os.path.join(cwd, 'data_prep', 'instrument_parameters', pre_xdd_file))
    else:
        pre_xdd_lines = '\n'

    # create batchfile for running all pawley refinments sequentially
    batchfile_folder = os.path.join(cwd, material_name, 'batchfiles_topas')
    if not os.path.exists(batchfile_folder): os.makedirs(batchfile_folder)
    batchfile_path = os.path.join(batchfile_folder, f'pawley_refinement_of_experiment.bat')
        
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

    # ------------------------------------------------------------------------------------------------------
    # END OF USER INPUT
    # ------------------------------------------------------------------------------------------------------

    # cif_files = sorted(glob(os.path.join(cwd, material_name, f'subgroup_cifs_{material_name}', f'*.txt')))
    # cif_filenames = [os.path.basename(x) for x in cif_files]
    # cif_noext = [os.path.splitext(x)[0] for x in cif_filenames]
    # subgroup_nums = [x[3:] for x in cif_noext]

    

    #! Add try/except assert here to make sure file it globbed correctly 

    refinement_lines = funcs.refinement_options(iters=1000, continue_on_convergence=False)
    global_lines = funcs.global_parameters()
    
    # ----- generate instrument specific lines -----
    instrument_lines = funcs.instrument_parameters(os.path.join(cwd, 'data_prep', 'instrument_parameters', instrument_file))
    peakshape_lines = funcs.instrument_parameters(os.path.join(cwd, 'data_prep', 'instrument_parameters', peakshape_file))


    # phase_lines = f'''
    # str
    # \tscale @ 1
    # \tr_bragg 0\n'''

    phase_lines = f'''
    hkl_Is
    \tscale @ 1
    \tr_bragg 0\n'''
    
    input_winpath_list = []
    topas_input_path = os.path.join(cwd, material_name, write_refinement_folder) 
    if not os.path.exists(topas_input_path): os.makedirs(topas_input_path)

    for i in range(len(xdd_winpath)):
        xdd_lines = funcs.xdd(xdd_winpath[i])
        for j in range(len(use_subgroups)):
            str_file = str_files[j]
            # input file to write
            topas_input_file = os.path.join(topas_input_path, f'{experiment_names[i]}_sgt{use_subgroups[j]}_pawley.inp')

            # windows path to input file for batch script
            topas_input_winpath = PureWindowsPath(Path(topas_input_file).absolute())
            input_winpath_list.append(topas_input_winpath)
            # windows path for hkl and xy out files (so topas can write files)
            hkl_out_file = PureWindowsPath(Path(os.path.join(topas_input_path, f'{experiment_names[i]}_sgt{use_subgroups[j]}_hkl.dat')).absolute())
            hkl_scaled_out_file = PureWindowsPath(Path(os.path.join(topas_input_path, f'{experiment_names[i]}_sgt{use_subgroups[j]}_hkl_scaled.dat')).absolute())
            xy_out_file = PureWindowsPath(Path(os.path.join(topas_input_path, f'{experiment_names[i]}_sgt{use_subgroups[j]}_xy.xy')).absolute())
            dy_out_file = PureWindowsPath(Path(topas_input_path, f'{experiment_names[i]}_sgt{use_subgroups[j]}_dy.xy').absolute())


            # generate pawley lattice parameter refinement lines - with or without vc macro
            latparams, spacegroup_origin = funcs.get_str_info(str_file, 'strain')
            topas_latparam_lines = funcs.generate_pawley_lattice_lines_v2(latparams, spacegroup_origin, use_vc = False)

            # generate lines for writing TOPAS output
            if diffraction_type == 'neutron':
                xy_out_lines = f'\nOut_X_Ycalc("{xy_out_file}")\n'
                dy_out_lines = f'\nOut_d_from_tof_bragg_dayton("{dy_out_file}", d0_1, difc_1, difa_1)\n'
            if diffraction_type == 'xray':
                xy_out_lines = f'\nOut_X_Ycalc("{xy_out_file}")\n'
                dy_out_lines = f'\nOut_2th_TO_dspacing_Ycalc("{dy_out_file}")\n'

            # write input file
            with open(topas_input_file, 'w') as f:
                f.write(refinement_lines)
                f.write(global_lines)
                f.write(pre_xdd_lines)
                f.write(xdd_lines)
                f.write(instrument_lines)
                f.write(phase_lines)
                f.write(peakshape_lines)
                f.write(topas_latparam_lines)
                f.write(xy_out_lines)
                # f.write(dy_out_lines)
                f.write(custom_macros)

            
    # write file for batch running topas pawley refinements of subgrouptree
    with open(batchfile_path, 'w') as f:
        f.write('cd C:\TOPAS6 \n')
        for i in range(len(input_winpath_list)):
            f.write(f'tc {input_winpath_list[i]} \n')


if __name__ == "__main__":
    main()