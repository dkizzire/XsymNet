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
    sub_folder_1 = 'primitive_refinements'
    sub_folder_2 = 'pawley'
    topas_version = 'TOPAS6'
    use_vc_macro = True
    # set convergence criteria
    conv_criteria = 0.0001

    if phase_name == 'alpha':
        experimental_data_file = f'hrp27188_bs_alpha_neutron.xye'
        TOF_x_axis_calibration = 'TOF_x_axis_calibration(t0_hrpd_bs,-49.37818`,!difc_hrpd_bs, 48255.09936_0.61297, difa_hrpd_bs,  15.23076`)'
    if phase_name == 'beta':
        experimental_data_file = f'Hrp27190_bs.xye'
        TOF_x_axis_calibration = 'TOF_x_axis_calibration(t0_hrpd_bs, 2.00496`_0.0846189708,!difc_hrpd_bs, 48255.09936_0.61297, difa_hrpd_bs, -3.22266`_0.0415199177)'
    
    experimental_data_path = os.path.join(cwd, 'TOPAS', phase_name, experimental_data_file)
    xdd_winpath = PureWindowsPath(Path(experimental_data_path).absolute())

    ''' 
    vc macro: val on continue macro in topas
    vc_matrix[[v1, v2, min, max], ....] --> fill in all 4 values for lattice parameters a, b, c, alpha, beta, gamma
    demo use:
    macro vc { val_on_continue = Val*Rand(v1,v2); }     --> where the current value of lattice parameter will be multiplied by a random value between v1 & v2
    a lpa 10.725 vc min 7.0 max 25.0                    --> where 7.0 & 25.0 are the lower & upper bounds of the lattice parameter refinement
    '''
    vc_matrix =[[0.80, 1.20, 6.0, 30.0],
                [0.80, 1.20, 6.0, 30.0],
                [0.80, 1.20, 6.0, 30.0],
                [0.95, 1.05, 70.0, 130.0],
                [0.95, 1.05, 70.0, 130.0],
                [0.95, 1.05, 70.0, 130.0],
                ]


    # TOF start & finish position in microseconds
    start_finish_x = [35000, 121000] 

    # number of iterations for pawley refinement
    refinement_iterations = 10000

    # set scale parameter & if refined : list: [bool, value] True: scale is refined;  False: scale is set at value
    scale_parameter = [True, 1.0] 

    # ------------------------------------------------------------------------------------------------------
    # END OF USER INPUT
    # ------------------------------------------------------------------------------------------------------

    subgroup_cifs_directory = os.path.join(cwd, 'subgroup_cifs')
    primitive_paths = funcs.find_p1_sgt_members(subgroup_cifs_directory)

    cif_filenames = [os.path.basename(x) for x in primitive_paths]
    cif_noext = [os.path.splitext(x)[0] for x in cif_filenames]
    subgroup_nums = [x[3:] for x in cif_noext]
    print(f'Primitive Subgroup members : ')
    print(subgroup_nums)

    # ------------------------------------------------------------------------------------------------------
    # batch cifs for running in topas in groups of 100 cifs
    # ------------------------------------------------------------------------------------------------------
    number_of_cifs_in_a_batch = 1
    number_of_batchfiles = int(len(primitive_paths)/number_of_cifs_in_a_batch)
    batch_count_list = [number_of_cifs_in_a_batch]*number_of_batchfiles
    batch_remainder = len(primitive_paths) % number_of_cifs_in_a_batch
    if batch_remainder != 0:
        number_of_batchfiles += 1
        batch_count_list.append(batch_remainder)

    batch_dir = os.path.join(cwd, 'TOPAS', phase_name, 'batchfiles_topas')
    if not os.path.exists(batch_dir): os.makedirs(batch_dir)

    # ------------------------------------------------------------------------------------------------------
    # Main Loop for writing topas input files for Pawley refinement of Primitive P1 structures in the SGT
    # ------------------------------------------------------------------------------------------------------

    for i in range(len(primitive_paths)):

        # input file to write
        topas_input_path = os.path.join(cwd, 'TOPAS', phase_name, sub_folder_1) 
        if not os.path.exists(topas_input_path): os.makedirs(topas_input_path)
        topas_input_file = os.path.join(topas_input_path, f'{phase_name}_P1_{subgroup_nums[i]}_pawley.inp')

        # windows path to input file for batch script
        topas_input_winpath = PureWindowsPath(Path(topas_input_file).absolute())
        # windows path for hkl and xy out files (so topas can write files)
        hkl_out_file = PureWindowsPath(Path(os.path.join(topas_input_path, f'{cif_noext[i]}_hkl.dat')).absolute())
        hkl_scaled_out_file = PureWindowsPath(Path(os.path.join(topas_input_path, f'{cif_noext[i]}_hkl_scaled.dat')).absolute())
        xy_out_file = PureWindowsPath(Path(os.path.join(topas_input_path, f'{cif_noext[i]}_xy.xy')).absolute())

        # get lattice information from cifs, use info to generate lines for hkl_Is section of topas input file
        latparams = funcs.get_cif_info(primitive_paths[i])

        # generate pawley lattice parameter refinement lines - with or without vc macro
        if use_vc_macro == True:
            topas_latparam_lines = funcs.generate_lattice_parameter_lines_with_vc_macro(latparams, vc_matrix)
        else: 
            topas_latparam_lines = funcs.generate_pawley_lattice_lines(latparams)

        # write input file
        funcs.write_inp_for_pawley_with_vc_macro(topas_input_file,
                                                    use_vc_macro,
                                                    TOF_x_axis_calibration,
                                                    cif_noext[i],
                                                    topas_latparam_lines,
                                                    vc_matrix,
                                                    xdd_winpath, 
                                                    refinement_iterations, 
                                                    conv_criteria, 
                                                    start_finish_x,  
                                                    hkl_out_file,
                                                    hkl_scaled_out_file,  
                                                    xy_out_file)

        batchfile_write = os.path.join(batch_dir, f'P1_pawlwy_vc_{subgroup_nums[i]}.bat')
        # write file for batch running topas pawley refinements of subgrouptree
        with open(batchfile_write, 'w') as f:
            f.write(f'cd C:\{topas_version} \n')
            f.write(f'tc "{topas_input_winpath}" \n')




if __name__ == "__main__":
    main()