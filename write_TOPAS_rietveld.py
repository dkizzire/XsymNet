import os
import glob
from pathlib import Path, PureWindowsPath
import write_TOPAS_funcs as funcs


def main():
    cwd = os.getcwd()
    # ------------------------------------------------------------------------------------------------------
    # START OF USER INPUT
    # ------------------------------------------------------------------------------------------------------
    phase_name = 'wo3'
    diffraction_type = 'xray'   # choices: neutron or xray
    instrument_file = 'wo3_init.txt'
    peakshape_file = 'wo3_peak_type.txt'
    sub_folder_1 = 'experimental_refinement'
    primitive_base_structure = '0131'
    use_vc_macro = True
    use_strain_modes = True
    # set convergence criteria
    conv_criteria = 0.0001

    experimental_data_file = f'd8_03901_t1205.xy'
    experimental_data_path = os.path.join(cwd, phase_name, experimental_data_file)
    xdd_winpath = PureWindowsPath(Path(experimental_data_path).absolute())

    
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
    batchfile_folder = os.path.join(cwd, phase_name, 'batchfiles_topas')
    if not os.path.exists(batchfile_folder): os.makedirs(batchfile_folder)
    batchfile_path = os.path.join(batchfile_folder, f'pawley_refinement_of_experiment.bat')

    # ------------------------------------------------------------------------------------------------------
    # END OF USER INPUT
    # ------------------------------------------------------------------------------------------------------

    cif_files = sorted(glob.glob(os.path.join(cwd, phase_name, f'subgroup_cifs_{phase_name}', f'cif{primitive_base_structure}.txt')))
    cif_filenames = [os.path.basename(x) for x in cif_files]
    cif_noext = [os.path.splitext(x)[0] for x in cif_filenames]
    subgroup_nums = [x[3:] for x in cif_noext]

    str_files = sorted(glob.glob(os.path.join(cwd, phase_name, f'topas_strs_{phase_name}', f'topas{primitive_base_structure}.str')))

    #! Add try/except assert here to make sure file it globbed correctly 

    refinement_lines = funcs.refinement_options(iters=1000, continue_on_convergence=True)
    global_lines = funcs.global_parameters()
    xdd_lines = funcs.xdd(xdd_winpath)
    # ----- generate instrument specific lines -----
    instrument_lines = funcs.instrument_parameters(os.path.join(cwd, 'instrument_parameters', instrument_file))
    peakshape_lines = funcs.instrument_parameters(os.path.join(cwd, 'instrument_parameters', peakshape_file))

    phase_lines = f'''
    str
    \tscale @ 1
    \tr_bragg 0\n'''
    
    input_winpath_list = []
    for i in range(len(cif_files)):

        # input file to write
        topas_input_path = os.path.join(cwd, phase_name, sub_folder_1) 
        if not os.path.exists(topas_input_path): os.makedirs(topas_input_path)
        topas_input_file = os.path.join(topas_input_path, f'experimental_{phase_name}_pawley.inp')

        # windows path to input file for batch script
        topas_input_winpath = PureWindowsPath(Path(topas_input_file).absolute())
        input_winpath_list.append(topas_input_winpath)
        # windows path for hkl and xy out files (so topas can write files)
        hkl_out_file = PureWindowsPath(Path(os.path.join(topas_input_path, f'exp_str{subgroup_nums[i]}_hkl.dat')).absolute())
        hkl_scaled_out_file = PureWindowsPath(Path(os.path.join(topas_input_path, f'exp_str{subgroup_nums[i]}_hkl_scaled.dat')).absolute())
        xy_out_file = PureWindowsPath(Path(os.path.join(topas_input_path, f'exp_str{subgroup_nums[i]}_xy.xy')).absolute())
        dy_out_file = PureWindowsPath(Path(topas_input_path, f'exp_str{subgroup_nums[i]}_dy.xy').absolute())


        # generate pawley lattice parameter refinement lines - with or without vc macro
        latparams, spacegroup_origin = funcs.get_str_info(str_files[i], 'strain')
        topas_latparam_lines = funcs.generate_pawley_lattice_lines_v2(latparams, spacegroup_origin, use_vc = True)

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
            f.write(xdd_lines)
            f.write(instrument_lines)
            f.write(phase_lines)
            f.write(peakshape_lines)
            f.write(topas_latparam_lines)
            f.write(xy_out_lines)
            f.write(dy_out_lines)


        
    # write file for batch running topas pawley refinements of subgrouptree
    with open(batchfile_path, 'w') as f:
        f.write('cd C:\TOPAS6 \n')
        for i in range(len(input_winpath_list)):
            f.write(f'tc {input_winpath_list[i]} \n')


if __name__ == "__main__":
    main()