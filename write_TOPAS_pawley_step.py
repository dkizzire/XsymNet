import os
import glob
from pathlib import Path, PureWindowsPath
import write_TOPAS_funcs as funcs


def main():
    cwd = os.getcwd()
    # ------------------------------------------------------------------------------------------------------
    # START OF USER INPUT
    # ------------------------------------------------------------------------------------------------------
    phase_name = 'beta'
    sub_folder_1 = 'subgrouptree_refinement'
    topas_version = 'TOPAS6'
    use_strain_modes = True

    if phase_name == 'alpha':
        experimental_data_file = f'hrp27188_bs_alpha_neutron.xye'
        TOF_x_axis_calibration = 'TOF_x_axis_calibration(t0_hrpd_bs,-49.37818`,!difc_hrpd_bs, 48255.09936_0.61297, difa_hrpd_bs,  15.23076`)'
        TOF_sample_peakshape = 'tof_sample_peakshape(lor, 0.200007268`,dsp, 1.00000`,dspsq, 30.43510`)'
        bkg_line = 'bkg  @  1.23166856` -0.0124816464`  0.209502581` -0.119357101` -0.0364802546` -0.0805659581`  0.280382189` -0.180514454`  0.157958349` -0.129704358`  0.215427676`  0.0027121254`'
    
    if phase_name == 'beta':
        experimental_data_file = f'Hrp27190_bs.xye'
        TOF_x_axis_calibration = 'TOF_x_axis_calibration(t0_hrpd_bs, 24.28584`,!difc_hrpd_bs, 48255.09936_0.61297, difa_hrpd_bs, -25.44401`) '
        TOF_sample_peakshape = 'tof_sample_peakshape(lor, 0.143711169`, dsp, 11.64083`, dspsq, 11.71737`)'
        bkg_line = 'bkg  @  1.67370339`  0.0515350436`  0.00972952795` -0.12759364`  0.20547493` -0.177201821`  0.116513928` -0.0589855793`  0.0192231893`  0.0184925627` -0.000494961725` -0.000104051593`'
    
    experimental_data_path = os.path.join(cwd, 'TOPAS', phase_name, experimental_data_file)
    xdd_winpath = PureWindowsPath(Path(experimental_data_path).absolute())

    start_finish_x = [35000, 121000] # TOF start & finish position in microseconds

    # number of iterations for pawley refinement
    refinement_iterations = 1000

    # set convergence criteria
    conv_criteria = 0.00001

    # paths to all cif and topas_str files
    cif_files = sorted(glob.glob(f'./subgroup_cifs_beta/*.cif'))
    str_files = sorted(glob.glob(os.path.join(cwd, 'TOPAS', 'topas_strs_beta', '*.str')))

    # create batchfiles for running all pawley refinments
    number_of_cifs_in_a_batch = 100
    number_of_batchfiles = int(len(cif_files)/number_of_cifs_in_a_batch)
    batch_count_list = [number_of_cifs_in_a_batch]*number_of_batchfiles
    batch_remainder = len(cif_files) % number_of_cifs_in_a_batch
    if batch_remainder != 0:
        number_of_batchfiles += 1
        batch_count_list.append(batch_remainder)

    batch_dir = f'./TOPAS/{phase_name}/batchfiles_topas'
    if not os.path.exists(batch_dir): os.makedirs(batch_dir)


    # ------------------------------------------------------------------------------------------------------
    # END OF USER INPUT
    # ------------------------------------------------------------------------------------------------------
    
    cif_filenames = [os.path.basename(x) for x in cif_files]
    cif_noext = [os.path.splitext(x)[0] for x in cif_filenames]
    subgroup_nums = [x[3:] for x in cif_noext]

    input_winpath_list = []
    batch_count = 0
    count = 0
    for i in range(len(cif_files)):
        print(f'Writing pawley refinement for {cif_noext[i]}') 
        count += 1
        # input file to write
        topas_input_folder = f'./TOPAS/{phase_name}/{sub_folder_1}/{cif_noext[i]}'
        if not os.path.exists(topas_input_folder): os.makedirs(topas_input_folder)
        topas_input_file = f'{topas_input_folder}/{cif_noext[i]}_pawley.inp'

        # windows path to input file for batch script
        topas_input_winpath = PureWindowsPath(Path(topas_input_file).absolute())
        input_winpath_list.append(topas_input_winpath)
        # windows path for hkl and xy out files (so topas can write files)
        hkl_out_file = PureWindowsPath(Path(f'{topas_input_folder}/{cif_noext[i]}_hkl.dat').absolute())
        xy_out_file = PureWindowsPath(Path(f'{topas_input_folder}/{cif_noext[i]}_xy.xy').absolute())

        # get lattice information from cifs, use info to generate lines for hkl_Is section of topas input file
        latparams, spacegroup_origin = funcs.get_str_info(str_files[i], 'strain')
        if use_strain_modes == True:
            topas_latparam_lines, macro_lines = funcs.generate_latparam_lines_with_strain_modes_V2(str_files[i], spacegroup_origin, TOF_sample_peakshape)
        else:
            topas_latparam_lines = funcs.generate_pawley_lattice_lines(latparams)
            macro_lines = []
        # write input file
        funcs.write_inp_for_pawley_on_subgroup_tree(use_strain_modes, 
                                                    topas_input_file, 
                                                    cif_noext[i], 
                                                    topas_latparam_lines, 
                                                    macro_lines, 
                                                    TOF_x_axis_calibration, 
                                                    bkg_line, 
                                                    TOF_sample_peakshape, 
                                                    xdd_winpath, 
                                                    refinement_iterations, 
                                                    conv_criteria, 
                                                    start_finish_x, 
                                                    hkl_out_file, 
                                                    xy_out_file)

        # write file for batch running topas pawley refinements of subgrouptree
        # write a windows batch file to run 100 cifs of random states of input files at a time
        if count == batch_count_list[batch_count]:
            # reset count and increment batch_count by 1
            count = 0
            batch_count += 1
            batchfile_write = os.path.join(batch_dir, f'pawley_on_SGtree_batch_{str(batch_count).zfill(4)}.bat')
            with open(batchfile_write, 'w') as f:
                f.write(f'cd C:\{topas_version} \n')
                for i in range(len(input_winpath_list)):
                    f.write(f'tc "{input_winpath_list[i]}" \n')
                
            input_winpath_list = []


if __name__ == "__main__":
    main()