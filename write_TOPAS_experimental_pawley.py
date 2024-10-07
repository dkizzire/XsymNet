import os
import glob
from pathlib import Path, PureWindowsPath
import write_TOPAS_funcs as funcs


def main():
    cwd = os.getcwd()
    # ------------------------------------------------------------------------------------------------------
    # START OF USER INPUT
    # ------------------------------------------------------------------------------------------------------
    phase_name = 'nvpf'
    diffraction_type = 'xray'   # choices: neutron or xray
    instrument_file = 'mspd.txt'
    peakshape_file = 'mspd_peak_type.txt'
    sub_folder_1 = 'experimental_refinement'
    primitive_base_structure = 'cif0134'
    use_vc_macro = True
    use_strain_modes = True
    # set convergence criteria
    conv_criteria = 0.0001

    experimental_data_file = f'NVPF_100K_Pattern.xye'
    experimental_data_path = os.path.join(cwd, phase_name, experimental_data_file)
    xdd_winpath = PureWindowsPath(Path(experimental_data_path).absolute())
    # TOF_x_axis_calibration = 'TOF_x_axis_calibration(t0_hrpd_bs, 24.28584`,!difc_hrpd_bs, 48255.09936_0.61297, difa_hrpd_bs, -25.44401`)'
    # TOF_sample_peakshape = 'tof_sample_peakshape(lor, 0.143711169`, dsp, 11.64083`, dspsq, 11.71737`)'
    # background_line = 'bkg  @  1.67370339`  0.0515350436`  0.00972952795` -0.12759364`  0.20547493` -0.177201821`  0.116513928` -0.0589855793`  0.0192231893`  0.0184925627` -0.000494961725` -0.000104051593`'
    
    # if phase_name == 'alpha':
    #     experimental_data_file = f'hrp27188_bs_alpha_neutron.xye'
    #     TOF_x_axis_calibration = 'TOF_x_axis_calibration(t0_hrpd_bs,-22.0105`_0.19111746,!difc_hrpd_bs, 48283.52311_1.67763, difa_hrpd_bs, -14.00371`_0.0876098944)'
    #     TOF_sample_peakshape = 'tof_sample_peakshape(lor, 0.168698361, dsp, 34.96508, dspsq, 2.83011)'
    # if phase_name == 'beta':
    #     experimental_data_file = f'Hrp27190_bs.xye'
    #     TOF_x_axis_calibration = 'TOF_x_axis_calibration(t0_hrpd_bs, 2.00496`_0.0846189708,!difc_hrpd_bs, 48255.09936_0.61297, difa_hrpd_bs, -3.22266`_0.0415199177)'
    #     TOF_sample_peakshape = 'tof_sample_peakshape(lor, 0.168698361, dsp, 34.96508, dspsq, 2.83011)'
    


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


    # TOF start & finish position in microseconds
    # start_finish_x = [25000, 96000] 

    # number of iterations for pawley refinement
    # refinement_iterations = 1000


    # create batchfile for running all pawley refinments sequentially
    batchfile_folder = os.path.join(cwd, phase_name, 'batchfiles_topas')
    if not os.path.exists(batchfile_folder): os.makedirs(batchfile_folder)
    batchfile_path = os.path.join(batchfile_folder, f'pawley_refinement_of_experiment.bat')

    # ------------------------------------------------------------------------------------------------------
    # END OF USER INPUT
    # ------------------------------------------------------------------------------------------------------

    cif_files = sorted(glob.glob(os.path.join(cwd, phase_name, f'subgroup_cifs_{phase_name}', f'{primitive_base_structure}.cif')))
    cif_filenames = [os.path.basename(x) for x in cif_files]
    cif_noext = [os.path.splitext(x)[0] for x in cif_filenames]
    subgroup_nums = [x[3:] for x in cif_noext]

    refinement_lines = funcs.refinement_options(iters=1000, continue_on_convergence=True)
    global_lines = funcs.global_parameters()
    xdd_lines = funcs.xdd(xdd_winpath)
    # ----- generate instrument specific lines -----
    instrument_lines = funcs.instrument_parameters(os.path.join(cwd, 'instrument_parameters', instrument_file))
    peakshape_lines = funcs.instrument_parameters(os.path.join(cwd, 'instrument_parameters', peakshape_file))

    phase_lines = f'''
    hkl_Is
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
        hkl_out_file = PureWindowsPath(Path(os.path.join(topas_input_path, f'{cif_noext[i]}_hkl.dat')).absolute())
        hkl_scaled_out_file = PureWindowsPath(Path(os.path.join(topas_input_path, f'{cif_noext[i]}_hkl_scaled.dat')).absolute())
        xy_out_file = PureWindowsPath(Path(os.path.join(topas_input_path, f'{cif_noext[i]}_xy.xy')).absolute())
        dy_out_file = PureWindowsPath(Path(f'{topas_input_path}/{cif_noext[i]}_dy.xy').absolute())

        # get lattice information from cifs, use info to generate lines for hkl_Is section of topas input file
        # latparams, spacegroup_origin = funcs.get_cif_info(cif_files[i])
        # str_file = os.path.join(cwd, 'TOPAS', f'topas_strs_{phase_name}', f'topas{subgroup_nums[i]}.str')
        
        # generate pawley lattice parameter refinement lines - with or without vc macro
        # if use_vc_macro == True:
        #     if use_strain_modes == True:
        #         topas_latparam_lines,macro_lines = funcs.generate_latparam_lines_with_strain_modes_V2(str_file, spacegroup_origin, TOF_sample_peakshape)
        #     else:
        #         topas_latparam_lines = funcs.generate_lattice_parameter_lines_with_vc_macro(latparams, vc_matrix)
        #         macro_lines = []
        # else: 
        #     topas_latparam_lines = funcs.generate_pawley_lattice_lines_v2(latparams, spacegroup_origin)
        #     macro_lines = []

        # generate pawley lattice parameter refinement lines - with or without vc macro
        latparams, spacegroup_origin = funcs.get_cif_info(cif_files[i])
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

        # funcs.write_exp_refinement_strain_modes_stage_1(beamline_choice,
        #                                                 topas_input_file, 
        #                                                 use_vc_macro, 
        #                                                 TOF_x_axis_calibration, 
        #                                                 topas_latparam_lines,
        #                                                 macro_lines,
        #                                                 xdd_winpath, 
        #                                                 refinement_iterations, 
        #                                                 conv_criteria, 
        #                                                 start_finish_x,
        #                                                 background_line, 
        #                                                 xy_out_file=xy_out_file)
        
    # write file for batch running topas pawley refinements of subgrouptree
    with open(batchfile_path, 'w') as f:
        f.write('cd C:\TOPAS6 \n')
        for i in range(len(input_winpath_list)):
            f.write(f'tc {input_winpath_list[i]} \n')


if __name__ == "__main__":
    main()