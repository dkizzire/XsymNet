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
    phase_name = 'wo3'
    instrument_file = 'wo3.txt'
    peakshape_file = 'wo3_peak_type.txt'
    refinement_type = 'rietveld' #choices: rietveld or pawley
    sub_folder_1 = f'SGT_{refinement_type}'
    topas_version = 'TOPAS6'
    use_strain_modes = True
    diffraction_type = 'xray'   # choices: neutron or xray
    
    experimental_data_file = f'd8_03901_t1103.xy'
    experimental_data_path = os.path.join(cwd, phase_name, experimental_data_file)
    xdd_winpath = PureWindowsPath(Path(experimental_data_path).absolute())

    # ----- generate TOPAS Launch lines -----
    refinement_lines = funcs.refinement_options(iters=1000, continue_on_convergence=False)
    global_lines = funcs.global_parameters()
    xdd_lines = funcs.xdd(xdd_winpath)

    # ----- generate instrument specific lines -----
    instrument_lines = funcs.instrument_parameters(os.path.join(cwd, 'instrument_parameters', instrument_file))
    peakshape_lines = funcs.instrument_parameters(os.path.join(cwd, 'instrument_parameters', peakshape_file))

    # paths to all cif and topas_str files
    cif_files = sorted(glob.glob(os.path.join(cwd, phase_name, f'subgroup_cifs_{phase_name}', '*')))
    str_files = sorted(glob.glob(os.path.join(cwd, phase_name, f'topas_strs_{phase_name}', '*.str')))

    # create batchfiles for running all pawley refinments
    number_of_cifs_in_a_batch = 40
    number_of_batchfiles = int(len(cif_files)/number_of_cifs_in_a_batch)
    batch_count_list = [number_of_cifs_in_a_batch]*number_of_batchfiles
    batch_remainder = len(cif_files) % number_of_cifs_in_a_batch
    if batch_remainder != 0:
        number_of_batchfiles += 1
        batch_count_list.append(batch_remainder)

    batch_dir = os.path.join(cwd, phase_name, 'batchfiles_topas')
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
    data  = {}
    
    if refinement_type == 'pawley':
        phase_lines = f'''
    hkl_Is
    \tscale @ 1
    \tr_bragg 0\n'''
    elif refinement_type == 'rietveld':
        phase_lines = f'''
    str
    \tscale @ 1
    \tr_bragg 0\n'''
    else:
        print('Value Error: refinement_type should be either pawley or rietveld\n')

    data['refinement_lines'] = refinement_lines
    data['global_lines'] = global_lines
    data['xdd_lines'] = xdd_lines
    data['instrument_lines'] = instrument_lines
    data['peakshape_ilnes'] = peakshape_lines
    data['phase_lines'] = phase_lines

    for i in range(len(cif_files)):
        print(f'Writing pawley refinement for {cif_noext[i]}') 
        count += 1
        # input file to write
        topas_input_path = os.path.join(cwd, phase_name, sub_folder_1)
        if not os.path.exists(topas_input_path): os.makedirs(topas_input_path)
        topas_input_file = f'{topas_input_path}/{cif_noext[i]}_{refinement_type}.inp'

        # windows path to input file for batch script ran with TOPAS terminal version
        topas_input_winpath = PureWindowsPath(Path(topas_input_file).absolute())
        input_winpath_list.append(topas_input_winpath)

        # windows path for hkl and xy out files (so topas can write files)
        hkl_out_file = PureWindowsPath(Path(f'{topas_input_path}/{cif_noext[i]}_hkl.dat').absolute())
        xy_out_file = PureWindowsPath(Path(f'{topas_input_path}/{cif_noext[i]}_xy.xy').absolute())
        dy_out_file = PureWindowsPath(Path(f'{topas_input_path}/{cif_noext[i]}_dy.xy').absolute())

        xy_out_lines = f'\nOut_X_Ycalc("{xy_out_file}")\n'
        if diffraction_type == 'neutron':
            dy_out_lines = f'\nOut_d_from_tof_bragg_dayton("{dy_out_file}", d0_1, difc_1, difa_1)\n'
        if diffraction_type == 'xray':
            dy_out_lines = f'\nOut_2th_TO_dspacing_Ycalc("{dy_out_file}")\n'


        if refinement_type == 'pawley':
            # get lattice information from strs_files, use info to generate lines for hkl_Is section of topas input file
            latparams, spacegroup_origin = funcs.get_str_info(str_files[i], 'strain')
            if use_strain_modes == True:
                lattice_lines, macro_lines = funcs.generate_pawley_latparam_lines_with_strain_modes(str_files[i], spacegroup_origin, peakshape_lines)
            else:
                lattice_lines = funcs.generate_pawley_lattice_lines_v2(latparams, spacegroup_origin)
                macro_lines = []
            

        elif refinement_type == 'rietveld':
            latparams, spacegroup_origin = funcs.get_str_info(str_files[i], 'strain')
            random_strain = np.random.uniform(low=-0.001, high=0.001, size=(6))
            lattice_lines = funcs.generate_strain_and_symmetry_mode_lines_from_topas_str_v2(str_files[i], spacegroup_origin, random_strain)

        else:
            print('Value Error: refinement_type should be either pawley or rietveld\n')

        # write topas input file
        with open(topas_input_file, 'w') as f:
            f.write(refinement_lines)
            f.write(global_lines)
            f.write(xdd_lines)
            f.write(instrument_lines)
            f.write(phase_lines)
            f.write(peakshape_lines)
            f.write(lattice_lines)
            f.write(xy_out_lines)
            # f.write(dy_out_lines)
            

        # write file for batch running topas pawley refinements of subgrouptree
        # write a windows batch file to run 100 cifs of random states of input files at a time
        if count == batch_count_list[batch_count]:
            # reset count and increment batch_count by 1
            count = 0
            batch_count += 1
            batchfile_write = os.path.join(batch_dir, f'SGT_{refinement_type}_batch_{str(batch_count).zfill(4)}.bat')
            with open(batchfile_write, 'w') as f:
                f.write(f'cd C:\{topas_version} \n')
                for i in range(len(input_winpath_list)):
                    f.write(f'tc "{input_winpath_list[i]}" \n')
                
            input_winpath_list = []


if __name__ == "__main__":
    main()