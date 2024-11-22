'''
Author: Dr. Dayton  Kizzire
Last Updated: 11/7/2024
Email: kizziredg@ornl.gov
'''

import os
import shutil
import glob
import numpy as np
import pandas as pd
import torch
import random
from pathlib import Path, PureWindowsPath
import write_TOPAS_funcs as funcs

import threading
import queue
import random
import time
from datetime import timedelta
import subprocess




def main():
    cwd = os.getcwd()
    t0 = time.time()
    # ------------------------------------------------------------------------------------------------------
    # START OF USER INPUT
    # ------------------------------------------------------------------------------------------------------
    NUM_THREADS = 5
    material_name = 'bso'
    phase_name = 'alpha_2'
    instrument_file = 'hrpd.txt'
    peakshape_file = 'hrpd_peak_type.txt'
    number_of_random_states = 1
    path_to_topas_directory = 'C:\TOPAS6'
    diffraction_type = 'neutron'   #* choices: neutron or xray
    use_strain_modes = True
    random_strain_bounds = [-0.005, 0.005]
    random_perturb_bounds = [-0.1, 0.1, -5.0, 5.0] # bounds for perturbing custom lattice parameters [abc_lo(angstrom), abc_hi(angstrom), angle_lo(degrees), angle_hi(degrees)] 
    use_custom_latparams = False
    overwrite_input_file_with_output_file = True

    experimental_data_file = f'hrp27188_bs_alpha_neutron.xye'
    experimental_data_path = os.path.join(cwd, material_name, experimental_data_file)
    xdd_winpath = PureWindowsPath(Path(experimental_data_path).absolute())

    beq_dict = {
        'Bi1': [0.0, 1.5],
        'Sn1': [0.0, 1.5],
        'O1': [0.0, 2.5], 
        'O2': [0.0, 2.5]
    }

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


    # ------------------------------------------------------------------------------------------------------
    # END OF USER INPUT
    # ------------------------------------------------------------------------------------------------------

    str_files = sorted(glob.glob(os.path.join(cwd, material_name, f'topas_strs_{material_name}', '*.str')))
    

    cif_files = sorted(glob.glob(os.path.join(cwd, material_name, f'subgroup_cifs_{material_name}', '*')))
    cif_filenames = [os.path.basename(x) for x in cif_files]
    cif_noext = [os.path.splitext(x)[0] for x in cif_filenames]
    subgroup_nums = [x[3:] for x in cif_noext]

    # topas output files from preliminary pawley refinement of subgroup tree lattice parameters
    if use_custom_latparams == True:
        easy_grab_files = sorted(glob.glob(os.path.join(cwd, material_name, 'SGT_pawley_refinement', '*', '*.out')))

    # ------------------------------------------------------------------------------------------------------
    # Initiate threading task queue
    # ------------------------------------------------------------------------------------------------------
    # Number of tasks and threads
    NUM_TASKS = len(str_files)

    # Create a queue and populate it with tasks (in this case, None as placeholders)
    task_queue = queue.Queue()
    for i in range(NUM_TASKS):
        task_queue.put(str_files[i])  # We can use None since the tasks are identical


    # ------------------------------------------------------------------------------------------------------
    # generate names for random state files
    # ------------------------------------------------------------------------------------------------------

    random_state = np.arange(1,number_of_random_states+1).tolist()
    random_state = [str(x).zfill(4) for x in random_state]
    random_state = [f'rs_{x}' for x in random_state]

    # ------------------------------------------------------------------------------------------------------
    # Generate lines of topas input file that are the same for every randomstate
    # ------------------------------------------------------------------------------------------------------
    
    refinement_lines = funcs.refinement_options(iters=0, continue_on_convergence=False, suppress_out_file=False)
    global_lines = funcs.global_parameters()
    xdd_lines = funcs.xdd(xdd_winpath)
    # ----- generate instrument specific lines -----
    instrument_lines = funcs.instrument_parameters(os.path.join(cwd, 'instrument_parameters', instrument_file))
    peakshape_lines = funcs.instrument_parameters(os.path.join(cwd, 'instrument_parameters', peakshape_file))

    # phase_lines only used if use_strain_modes=False
    phase_lines = f'''
    str
        scale 1
        r_bragg 0.0
        {peakshape_lines}
    '''
    input_dict = {'refinement_lines':refinement_lines,
                  'global_lines':global_lines,
                  'xdd_lines':xdd_lines,
                  'instrument_lines':instrument_lines,
                  'peakshape_lines':peakshape_lines}
    

    # implement threading with thread_task
    def thread_task(cwd, material_name, phase_name, str_file, random_strain_bounds, random_state, diffraction_type, input_dict, beq_dict, path_to_topas_directory):

        thread_id = threading.get_ident()
        str_name = os.path.splitext(os.path.basename(str_file))[0]
        print(f'Thread {thread_id} Writing random states for {str_name}')

        # make temporary folder to store input/output data per thread
        input_path = os.path.join(cwd, material_name, 'ml_data_generation', phase_name, f'temp_{thread_id}')
        if not os.path.exists(input_path): os.makedirs(input_path)
        rs_saved_input_folder = os.path.join(cwd, material_name, 'ml_data_generation', phase_name, 'RS_SGT_INP')
        if not os.path.exists(rs_saved_input_folder): os.makedirs(rs_saved_input_folder)

        # grab lattice parameters and spacegroup from str file
        latparams, spacegroup_origin = funcs.get_str_info(str_file, 'strain')

        # create random numbers for augmenting strain values
        random_strain = np.random.uniform(low=random_strain_bounds[0], high=random_strain_bounds[1], size=(len(random_state), 6))
        
        # create randomly perturbed TOPAS input files
        topas_input_file_list = []
        for i in range(len(random_state)):

            topas_input_file = os.path.join(input_path, f'{random_state[i]}.inp')
            topas_input_file_list.append(topas_input_file)
            
            xy_out_file = PureWindowsPath(Path(os.path.join(input_path, f'{random_state[i]}_xy.xy')).absolute())
            dy_out_file = PureWindowsPath(Path(os.path.join(input_path, f'{random_state[i]}_dy.xy')).absolute())

            xy_out_lines = f'\nOut_X_Ycalc("{xy_out_file}")\n'
            if diffraction_type == 'neutron':
                dy_out_lines = f'\nOut_d_from_tof_bragg_dayton("{dy_out_file}", d0_1, difc_1, difa_1)\n'
            if diffraction_type == 'xray':
                dy_out_lines = f'\nOut_2th_TO_dspacing_Ycalc("{dy_out_file}")\n'

            # create strain and symmetry mode lines from str_file with random values
            mode_lines = funcs.generate_strain_and_symmetry_mode_lines_from_topas_str_v2(str_file, spacegroup_origin, random_strain[i], beq_dict=beq_dict)

            # insert peakshape, r_bragg, and scale into mode lines
            mode_lines.insert(4, f'\t\tphase_name {str_name}\n')
            mode_lines.insert(4, f'\t\t{input_dict["peakshape_lines"]}\n')
            mode_lines.insert(4, f'\t\tr_bragg 0.0\n')
            mode_lines.insert(4, f'\t\tscale  1.0\n')

            # write TOPAS input file for this random state
            with open(topas_input_file, 'w') as f:

                f.write(f'out_file="";\n')
                f.write(f'\'WARNING WARNING WARNING: Above line suppresses output file, comment out if you want a .out file\n')
                f.write(input_dict['refinement_lines'])
                f.write(input_dict['global_lines'])
                f.write(input_dict['xdd_lines'])
                f.write(input_dict['instrument_lines'])
                
                # if use_strain_modes == False:
                #     f.write(phase_lines)
                #     f.write(topas_latparam_lines)

                for k in range(len(mode_lines)):
                    f.write(mode_lines[k])
                f.write(xy_out_lines)
            # Save the first random state file created for each subgroup for manual running/debugging
            if i == 0:
                # write TOPAS input file for this random state
                rs_saved_input_file = os.path.join(rs_saved_input_folder, f'{str_name}.inp')
                with open(rs_saved_input_file, 'w') as f:
                    f.write(input_dict['refinement_lines'])
                    f.write(input_dict['global_lines'])
                    f.write(input_dict['xdd_lines'])
                    f.write(input_dict['instrument_lines'])
                    # if use_strain_modes == False:
                    #     f.write(phase_lines)
                    #     f.write(topas_latparam_lines)
                    for k in range(len(mode_lines)):
                        f.write(mode_lines[k])
                    f.write(xy_out_lines)


        # Convert topas input path to windows so TOPAS can run it (TOPAS is Windows only)
        inp_winpath = [PureWindowsPath(Path(x).absolute()) for x in topas_input_file_list]

        # write a windows batchfile to run all random_states with tc.exe (topas command line)
        batchfile_path = os.path.join(input_path, 'batch.bat')
        with open(batchfile_path, 'w') as f:
            f.write(f'cd {path_to_topas_directory} \n')
            for i in range(len(inp_winpath)):
                f.write(f'tc "{inp_winpath[i]}" \n')
        
        batchfile_path = PureWindowsPath(Path(batchfile_path).absolute())

        # Run the batch file and capture output and errors
        result = subprocess.run(batchfile_path, shell=False, capture_output=True, text=True)

        # Print the output and error messages
        # print("Output:", result.stdout)
        # print("Errors:", result.stderr)

        # Collect the xy files, strip out the x column, save all random_states y columns into a pytorch tensor and save
        data_suffix = f'*xy.xy'
        dir_to_store_data = 'ml_data'
        save_tensor_dir = os.path.join(cwd, material_name, dir_to_store_data, phase_name)
        if not os.path.exists(save_tensor_dir): os.makedirs(save_tensor_dir)

        xy_files = sorted(glob.glob(os.path.join(input_path, data_suffix)))
        tensor_list = []
        for i in range(len(xy_files)):
            tensor_list.append(funcs.convert_bragg_to_tensors(xy_files[i]))
            
        stacked_tensor = torch.vstack(tensor_list)
        torch.save(stacked_tensor, os.path.join(save_tensor_dir, f'{str_name}_{material_name}_{phase_name}_rs.pt'))


    def worker():
        while True:
            try:
                # Retrieve a task without blocking
                task = task_queue.get_nowait()
                str_file = task
                thread_task(cwd, material_name, phase_name, str_file, random_strain_bounds, random_state, diffraction_type, input_dict, beq_dict, path_to_topas_directory)
                task_queue.task_done()
            except queue.Empty:
                # No more tasks are available
                break
    
    # Create and start NUM_THREADS threads
    threads = []
    for i in range(NUM_THREADS):
        t = threading.Thread(target=worker, name=f'Worker-{i}')
        t.start()
        threads.append(t)

    # Wait for all tasks to be processed
    task_queue.join()

    # Wait for all threads to finish
    for t in threads:
        t.join()

    
    t1 = time.time()
    trun = (t1 - t0)
    timed = timedelta(seconds=trun)
    # print('Time in hh:mm:ss:', td)
    
    print(f'ML Data Generation Has Finished\nRuntime: {timed}')

    temp_folders = sorted(glob.glob(os.path.join(cwd, material_name, 'ml_data_generation', phase_name, 'temp*')))
    for i in range(len(temp_folders)):
        shutil.rmtree(temp_folders[i])
        

if __name__ == "__main__":
    main()















