'''
Author: Dr. Dayton  Kizzire
Last Updated: 02/08/2024
Email: kizziredg@ornl.gov
'''

import os
import glob
from pathlib import Path, PureWindowsPath
import write_TOPAS_funcs as funcs


def main():
    cwd = os.getcwd()
    # ------------------------------------------------------------------------------------------------------
    # START OF USER INPUT  -- check all entries until 'END OF USER IMPUT'
    # ------------------------------------------------------------------------------------------------------
    
    phase_name = 'alpha'
    # directory structure : ./TOPAS/phase_name/sub_folder_1/sub_folder_2/
    sub_folder_1 = 'rietveld_cif0548'
    sub_folder_2 = 'modes_no_beq'
    # subgroup_tree cif to get structure information
    cif_to_refine = 'cif0548'


    # bool control flags
    use_custom_lattice_parameters = False  # define custom lattice parameters instead of reading them from a cif
    use_symmetry_mode_analysis = True # define structure with isodistort symmetry modes instead of traditional site information 
    use_vc_lp_macro = True  # randomly augment lattice parameter values (with symmetry constraints) after each topas convergence (vc = vary on continue)
    use_vc_ma_macro = True  # randomly augment symmetry mode amplitudes after each convergence (isodistort symmetry modes)
    scale_parameter = [True, 1.0] # [bool, set_value] -> True: scale is refined;  False: scale is set as set_value
    refine_BEQ = False
    BEQ_dict = {
        'Bi1': [0.0, 1.5],
        'Sn1': [0.0, 1.5],
        'O1': [0.0, 2.5], 
        'O2': [0.0, 2.5]
    }

    # input variables for .inp file
    start_finish_x = [35000, 121000]  # TOF start & finish position in microseconds
    refinement_iterations = 1000  # number of iterations for pawley refinement
    conv_criteria = 0.0001  # set convergence criteria

    # experimental xy file & TOF_x_axis_calibration line for TOPAS (instrument specific)
    if phase_name == 'alpha':
        experimental_data_file = f'hrp27188_bs_alpha_neutron.xye'
        TOF_x_axis_calibration = 'TOF_x_axis_calibration(t0_hrpd_bs,-22.0105`_0.19111746,!difc_hrpd_bs, 48283.52311_1.67763, difa_hrpd_bs, -14.00371`_0.0876098944)'
    if phase_name == 'beta':
        experimental_data_file = f'Hrp27190_bs.xye'
        TOF_x_axis_calibration = 'TOF_x_axis_calibration(t0_hrpd_bs, 2.00496`_0.0846189708,!difc_hrpd_bs, 48255.09936_0.61297, difa_hrpd_bs, -3.22266`_0.0415199177)'
    
    experimental_data_path = os.path.join(cwd, 'TOPAS', phase_name, experimental_data_file)
    xdd_winpath = PureWindowsPath(Path(experimental_data_path).absolute())

    # ------------------------------------------------------------------------------------------------------
    # custom lattice parameters section
    custom_spacegroup = 1
    custom_a = 14.910898
    custom_b = 14.988320
    custom_c = 21.748003
    custom_alpha = 90.26411
    custom_beta = 90.05252
    custom_gamma = 90.21629
    custom_originchoice = 1  # either 1 or 2
    custom_lattice_parameters = [custom_spacegroup, custom_a, custom_b, custom_c, custom_alpha, custom_beta, custom_gamma, custom_originchoice]
    # ------------------------------------------------------------------------------------------------------

    ''' 
    vc macro: val on continue macro in topas
    vc_matrix[[v1, v2, min_bound, max_bound], ....] --> fill in all 4 values for lattice parameters a, b, c, alpha, beta, gamma
    demo use:
    macro vc { val_on_continue = Val*Rand(v1,v2); }     --> where the current value of lattice parameter will be multiplied by a random value between v1 & v2
    a lpa 10.725 vc min 7.0 max 25.0                    --> where 7.0 & 25.0 are the lower & upper bounds of the lattice parameter refinement
    '''
    vc_lp_matrix = [[0.9, 1.1, 7.0, 25.0],
                    [0.9, 1.1, 7.0, 25.0],
                    [0.9, 1.1, 7.0, 25.0],
                    [0.95, 1.05, 70.0, 130.0],
                    [0.95, 1.05, 70.0, 130.0],
                    [0.95, 1.05, 70.0, 130.0],
                    ]
    
    # topas str already contains lower & upper bounds for symmetry modes, here we only give the [v1,v2] for random multiplication with vary on convergence
    vc_ma_matrix = [0.9, 1.1]

    # create batchfile for running all pawley refinments sequentially   
    # batchfile_folder = os.path.join(cwd, 'TOPAS', phase_name, 'batchfiles_topas')
    batchfile_folder = os.path.join(cwd, 'TOPAS', phase_name, sub_folder_1, sub_folder_2)
    if not os.path.exists(batchfile_folder): os.makedirs(batchfile_folder)
    batchfile_path = os.path.join(batchfile_folder, f'rietveld_refinement_{phase_name}_{sub_folder_2}.bat')

    # which cif files to use
    cif_files = sorted(glob.glob(os.path.join(cwd, 'subgroup_cifs', f'{cif_to_refine}.cif')))
    cif_filenames = [os.path.basename(x) for x in cif_files]
    cif_noext = [os.path.splitext(x)[0] for x in cif_filenames]
    subgroup_nums = [x[3:] for x in cif_noext]

    # ------------------------------------------------------------------------------------------------------
    # END OF USER INPUT
    # ------------------------------------------------------------------------------------------------------

    input_winpath_list = []
    for i in range(len(cif_files)):

        # input file to write
        topas_input_path = os.path.join(cwd, 'TOPAS', phase_name, sub_folder_1, sub_folder_2) 
        if not os.path.exists(topas_input_path): os.makedirs(topas_input_path)
        topas_input_file = os.path.join(topas_input_path, f'rietveld_{phase_name}_{cif_to_refine}_{sub_folder_2}.inp')

        # windows path to input file for batch script
        topas_input_winpath = PureWindowsPath(Path(topas_input_file).absolute())
        input_winpath_list.append(topas_input_winpath)
        # windows path for hkl and xy out files (so topas can write files)
        hkl_out_file = PureWindowsPath(Path(os.path.join(topas_input_path, f'{cif_noext[i]}_hkl.dat')).absolute())
        hkl_scaled_out_file = PureWindowsPath(Path(os.path.join(topas_input_path, f'{cif_noext[i]}_hkl_scaled.dat')).absolute())
        xy_out_file = PureWindowsPath(Path(os.path.join(topas_input_path, f'{cif_noext[i]}_xy.xy')).absolute())

        # get lattice information from cifs or from custom_lattice_parameters
        latparams, site_info = funcs.get_cif_info_with_site_info(cif_files[i])

        # overwrite latparams if using custom lattice parameters
        if use_custom_lattice_parameters == True:
            latparams = custom_lattice_parameters

        # generate site lines for topas from cif site info
        site_lines = funcs.generate_site_lines(site_info, refine_beq=refine_BEQ, beq_dict=BEQ_dict )
        
        # get symmetry mode lines from isodistort generated topasXXXX.str file & make changes as necessary
        symmetry_mode_file = os.path.join(cwd, 'TOPAS', 'topas_strs', f'topas{subgroup_nums[i]}.str')
        symmetry_mode_lines = funcs.generate_symmetry_mode_lines_from_topas_str(symmetry_mode_file, refine_beq=refine_BEQ, beq_dict=BEQ_dict, use_vc_ma_macro=use_vc_ma_macro)

        # generate pawley lattice parameter refinement lines - with or without vc macro
        if use_vc_lp_macro == True:
            topas_latparam_lines = funcs.generate_lattice_parameter_lines_with_vc_macro(latparams, vc_lp_matrix)
        else: 
            topas_latparam_lines = funcs.generate_pawley_lattice_lines(latparams)

        # write input file
        funcs.write_inp_for_rietveld_with_vc_macro(topas_input_file,
                                                    use_symmetry_mode_analysis,
                                                    use_vc_lp_macro,
                                                    use_vc_ma_macro, 
                                                    vc_lp_matrix,
                                                    vc_ma_matrix,
                                                    TOF_x_axis_calibration,
                                                    cif_noext[i],
                                                    topas_latparam_lines,
                                                    site_lines,
                                                    symmetry_mode_lines,
                                                    xdd_winpath, 
                                                    refinement_iterations, 
                                                    conv_criteria, 
                                                    start_finish_x,  
                                                    hkl_out_file,
                                                    hkl_scaled_out_file,  
                                                    xy_out_file)

    # write file for batch running topas pawley refinements of subgrouptree
    with open(batchfile_path, 'w') as f:
        f.write('cd C:\TOPAS6 \n')
        for i in range(len(input_winpath_list)):
            f.write(f'tc {input_winpath_list[i]} \n')


if __name__ == "__main__":
    main()