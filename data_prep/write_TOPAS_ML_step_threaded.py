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
from data_prep import write_TOPAS_funcs as funcs

import threading
import queue
import random
import time
from datetime import timedelta
import subprocess




def main():
    cwd = os.getcwd()
    t0 = time.time()
    #* ------------------------------------------------------------------------------------------------------
    #* START OF USER INPUT
    #* ------------------------------------------------------------------------------------------------------
    NUM_THREADS = 5                                 #^ default = 5; number of threads to launch to optimize scale or create randomstates
    material_name = 'pzt90'                           #^ name of your material
    phase_name = 'beta_8'                       
    instrument_file = 'hrpd_beta.txt'                     #^ located in ./data_prep/instrument_parameters
    peakshape_file = 'hrpd_peak_type.txt'            #^ located in ./data_prep/instrument_parameters
    number_of_random_states = 250                   #^ = 1 : Rietveld of whole subgroup tree;  = ~250-500 create randomstates for ML training 
    path_to_topas_directory = 'C:\TOPAS6'           #^ windows path to your TOPAS directory; should be 'C:\TOPAS#' 
    diffraction_type = 'neutron'                       #^ choices: neutron or xray
    optimize_scale = False                           #^ Generates better scale values for each subgroup
    flag_create_randomstates = True                 #^ True or False

    use_strain_modes = False                         #^ only change if you are using custom latparams
    random_strain_bounds = [-0.002, 0.002]          #^ default -> [-0.001, 0.001]  ::NOTE:: tiny changes here make a BIG difference in lattice parameter

    #^ Bounds for a,b,c,alpha,beta,gamma random perturbation (used if use_strain_modes == False)
    use_custom_latparams = False                    #^ If you have a structure with crazy strain
    if use_custom_latparams:
        use_strain_modes = False
        
    # custom_latparam_bounds = [[-0.02, 0.02], [-0.02, 0.02], [-0.02, 0.02], [-1, 1], [-1, 1], [-1, 1]] # custom lats for alpha BiSn2O7
    custom_latparam_bounds = [[-0.1, 0.1], [-0.1, 0.1], [-0.01, 0.2], [-1, 1], [-1, 1], [-1, 1]] # custom lats for PZT90
    
    
    #^ experimental xy file to use for xdd section of TOPAS when creating randomstates
    experimental_data_file = f'Hrp27190_bs.xye'
    experimental_data_path = os.path.join(cwd, material_name, experimental_data_file)
    assert os.path.exists(experimental_data_path), f'Error: {experimental_data_path} does not exist. Please check the path to your experimental data file.'
    xdd_winpath = PureWindowsPath(Path(experimental_data_path).absolute())

    #^ custom macros file (appended to bottom of input files)
    use_custom_macros = False
    if use_custom_macros:
        custom_macros_file = 'powgen_macros.txt'
        custom_macro_path = os.path.join(cwd, 'data_prep', 'instrument_parameters', custom_macros_file) 
        assert os.path.exists(custom_macro_path), f'Error: {custom_macro_path} does not exist. Please check the path to your custom macros file.'
        custom_macros = funcs.instrument_parameters(custom_macro_path)
    else:
        custom_macros = '\n'

    #^ custom lines to add before xdd section
    use_pre_xdd = False
    if use_pre_xdd:
        pre_xdd_file = 'nomad_bank_5_pre_xdd.txt'
        pre_xdd_file_path = os.path.join(cwd, 'data_prep', 'instrument_parameters', pre_xdd_file)
        assert os.path.exists(pre_xdd_file_path), f'Error: {pre_xdd_file_path} does not exist. Please check the path to your pre xdd file.'
        pre_xdd_lines = funcs.instrument_parameters(os.path.join(cwd, 'data_prep', 'instrument_parameters', pre_xdd_file))
    else:
        pre_xdd_lines = '\n'

    # beq_dict = {
    #     'W': [0.0, 1.5],
    #     'O': [0.0, 1.5]
    # }
    
    beq_dict = {
        'Bi1': [0.0, 1.0],
        'Sn1': [0.0, 1.0],
        'O1': [0.0, 1.0], 
        'O2': [0.0, 1.0]
    }
    # beq_dict = {
    #     'V': [0.0, 1.0],
    #     'P': [0.0, 1.0],
    #     'O': [0.0, 1.0], 
    #     'F': [0.0, 1.0],
    #     'Na': [0.0, 1.0]
    # }
    # beq_dict = {
    #     'Pb1': [0.0, 1.0],
    #     'Zr1': [0.0, 1.0],
    #     'Ti1': [0.0, 1.0], 
    #     'O1':  [0.0, 1.0]
    # }

    # beq_dict = {
    #     'Mo1': [0.0, 1.0],
    #     'O1': [0.0, 1.0],
    #     'N1': [0.0, 1.0], 
    #     'O2': [0.0, 1.0],
    #     'C1': [0.0, 1.0],
    #     'C3': [0.0, 1.0],
    #     'H3': [0.0, 1.0],
    #     'C2': [0.0, 1.0],
    #     'H2': [0.0, 1.0]
    # }

    # beq_dict = {
    #     'Fe': [0.0, 1.0],
    #     'O': [0.0, 1.0]
    # }

    #* ------------------------------------------------------------------------------------------------------
    #* END OF USER INPUT
    #* ------------------------------------------------------------------------------------------------------


    str_files = sorted(glob.glob(os.path.join(cwd, material_name, f'topas_strs_{material_name}', '*.str')))
    assert len(str_files) > 0, f'Error: No .str files found in {os.path.join(cwd, material_name, f"topas_strs_{material_name}")}. Please check the path to your .str files.'
    # str_files = str_files[25:30]

    cif_files = sorted(glob.glob(os.path.join(cwd, material_name, f'subgroup_cifs_{material_name}', '*')))
    assert len(cif_files) > 0, f'Error: No CIF files found in {os.path.join(cwd, material_name, f"subgroup_cifs_{material_name}")}. Please check the path to your CIF files.'
    cif_filenames = [os.path.basename(x) for x in cif_files]
    cif_noext = [os.path.splitext(x)[0] for x in cif_filenames]
    subgroup_nums = [x[3:] for x in cif_noext]


    # topas output files from preliminary pawley refinement of subgroup tree lattice parameters
    if use_custom_latparams == True:
        easy_grab_files = sorted(glob.glob(os.path.join(cwd, material_name, 'SGT_pawley_refinement', '*', '*.out')))
        assert len(easy_grab_files) > 0, f'Error: No .out files found in {os.path.join(cwd, material_name, "SGT_pawley_refinement")}. Please check the path to your .out files.'

    #* ------------------------------------------------------------------------------------------------------
    #* Initiate threading task queue
    #* ------------------------------------------------------------------------------------------------------
    # Number of tasks and threads
    NUM_TASKS = len(str_files)

    # Create a queue and populate it with tasks (in this case, None as placeholders)
    task_queue = queue.Queue()
    for i in range(NUM_TASKS):
        task_queue.put(str_files[i])  


    #* ------------------------------------------------------------------------------------------------------
    #* generate names for random state files
    #* ------------------------------------------------------------------------------------------------------

    random_state = np.arange(1,number_of_random_states+1).tolist()
    random_state = [str(x).zfill(4) for x in random_state]
    random_state = [f'rs_{x}' for x in random_state]

    #* ------------------------------------------------------------------------------------------------------
    #* Generate lines of topas input file that are the same for every randomstate
    #* ------------------------------------------------------------------------------------------------------
    
    # scale_refinement_lines = funcs.refinement_options(iters=1000, continue_on_convergence=False, suppress_out_file=False)
    refinement_lines = funcs.refinement_options(iters=0, continue_on_convergence=False, suppress_out_file=False)
    global_lines = funcs.global_parameters()
    xdd_lines = funcs.xdd(xdd_winpath)
    # ----- generate instrument specific lines -----
    instrument_file_path = os.path.join(cwd, 'data_prep', 'instrument_parameters', instrument_file)
    assert os.path.exists(instrument_file_path), f'Error: {instrument_file_path} does not exist. Please check the path to your instrument file.'
    peakshape_file_path = os.path.join(cwd, 'data_prep', 'instrument_parameters', peakshape_file)
    assert os.path.exists(peakshape_file_path), f'Error: {peakshape_file_path} does not exist. Please check the path to your peakshape file.'
    instrument_lines = funcs.instrument_parameters(os.path.join(cwd, 'data_prep', 'instrument_parameters', instrument_file))
    peakshape_lines = funcs.instrument_parameters(os.path.join(cwd, 'data_prep', 'instrument_parameters', peakshape_file))

    

    # phase_lines only used if use_strain_modes=False
    phase_lines = f'''
    str
        scale 1
        r_bragg 0.0
        {peakshape_lines}
    '''
    input_dict = {'refinement_lines':refinement_lines,
                  'global_lines':global_lines,
                  'pre_xdd_lines':pre_xdd_lines,
                  'xdd_lines':xdd_lines,
                  'phase_lines':phase_lines,
                  'instrument_lines':instrument_lines,
                  'peakshape_lines':peakshape_lines,
                  'custom_macros':custom_macros}
    
    #*----------------------------------------------------------------------------------------------------------------------------
    #* create save locations for scale optimiztion file, saved first random state file, and ml data tensors
    #*----------------------------------------------------------------------------------------------------------------------------

    # scale_file_destination = os.path.join(cwd, material_name, 'ml_data_generation', phase_name, 'scale_optimization')
    # if not os.path.exists(scale_file_destination): os.makedirs(scale_file_destination)
    
    save_topas_input_dirpath = os.path.join(cwd, material_name, 'ml_data_generation', phase_name, 'RS_SGT_INP')
    if not os.path.exists(save_topas_input_dirpath): os.makedirs(save_topas_input_dirpath)

    dir_to_store_data = 'ml_data'
    save_tensor_dirpath = os.path.join(cwd, material_name, dir_to_store_data, phase_name)
    if not os.path.exists(save_tensor_dirpath): os.makedirs(save_tensor_dirpath)

    #*----------------------------------------------------------------------------------------------------------------------------
    #* Start threaded tasks
    #*----------------------------------------------------------------------------------------------------------------------------

    def thread_task(cwd, material_name, phase_name, str_file, random_strain_bounds, random_state, diffraction_type, input_dict, beq_dict, path_to_topas_directory, save_topas_input_dirpath, save_tensor_dirpath=None ):

        thread_id = threading.get_ident()
        str_name = os.path.splitext(os.path.basename(str_file))[0]
        print(f'Thread {thread_id} Writing random states for {str_name}')

        # make temporary folder to store input/output data per thread
        input_path = os.path.join(cwd, material_name, 'ml_data_generation', phase_name, f'temp_{thread_id}')
        if not os.path.exists(input_path): os.makedirs(input_path)

        # grab lattice parameters and spacegroup from str file
        latparams, spacegroup_origin = funcs.get_str_info(str_file, 'strain')
        #^ line above is actually redundant with the new implementation of generate mode lines in write_TOPAS_funcs

        # create random numbers for augmenting strain values or latparam values directly
        if use_strain_modes:
            random_strain = np.random.uniform(low=random_strain_bounds[0], high=random_strain_bounds[1], size=(number_of_random_states, 6))
        if not use_strain_modes:
            random_strain = np.column_stack([np.round(np.random.uniform(low=low, high=high, size=number_of_random_states), 6) for low, high in custom_latparam_bounds])

        # create randomly perturbed TOPAS input files
        topas_input_file_list = []

        #*----------------------------------------------------------------------------------------------------------------------------
        #* Create Random States for training data
        #*----------------------------------------------------------------------------------------------------------------------------
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

            if use_strain_modes:
                # create strain and symmetry mode lines from str_file with random values
                mode_lines = funcs.generate_mode_lines(str_file, spacegroup_origin, random_strain[i], use_strain_modes=True, beq_dict=beq_dict)

            if not use_strain_modes:
                mode_lines = funcs.generate_mode_lines(str_file, spacegroup_origin, random_strain[i], use_strain_modes=False, beq_dict=beq_dict)

            # insert peakshape, r_bragg, and scale into mode lines
            mode_lines.insert(4, f'\t\tphase_name {str_name}\n')
            mode_lines.insert(4, f'{input_dict["peakshape_lines"]}\n')
            mode_lines.insert(4, f'\t\tr_bragg 0.0\n')
            mode_lines.insert(4, f'\t\tscale 1.0\n')

            
            # write TOPAS input file for this random state
            with open(topas_input_file, 'w') as f:

                f.write(f'out_file="";\n')
                f.write(f'\'WARNING WARNING WARNING: Above line suppresses output file, comment out if you want a .out file\n')
                f.write(input_dict['refinement_lines'])
                f.write(input_dict['global_lines'])
                f.write(input_dict['pre_xdd_lines'])
                f.write(input_dict['xdd_lines'])
                f.write(input_dict['instrument_lines'])
                
                # if use_strain_modes == False:
                #     f.write(phase_lines)
                #     f.write(topas_latparam_lines)

                for k in range(len(mode_lines)):
                    f.write(mode_lines[k])
                f.write(xy_out_lines)
                f.write(input_dict['custom_macros'])
                
            #*----------------------------------------------------------------------------------------------------------------------------
            #* Save the first random state file created for each subgroup for manual running/debugging
            #*----------------------------------------------------------------------------------------------------------------------------

            if i == 0:
                # write TOPAS input file for this random state
                saved_xy_out_file = PureWindowsPath(Path(os.path.join(save_topas_input_dirpath, f'{str_name}_xy.xy')).absolute())
                saved_xy_out_lines = f'\nOut_X_Ycalc("{saved_xy_out_file}")\n'
                rs_saved_input_file = os.path.join(save_topas_input_dirpath, f'{str_name}.inp')
                with open(rs_saved_input_file, 'w') as f:
                    f.write(input_dict['refinement_lines'])
                    f.write(input_dict['global_lines'])
                    f.write(input_dict['pre_xdd_lines'])
                    f.write(input_dict['xdd_lines'])
                    f.write(input_dict['instrument_lines'])
                    # if use_strain_modes == False:
                    #     f.write(phase_lines)
                    #     f.write(topas_latparam_lines)
                    for k in range(len(mode_lines)):
                        f.write(mode_lines[k])
                    f.write(saved_xy_out_lines)
                    f.write(input_dict['custom_macros'])


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

        xy_files = sorted(glob.glob(os.path.join(input_path, data_suffix)))
        tensor_list = []
        for i in range(len(xy_files)):
            tensor_list.append(funcs.convert_bragg_to_tensors(xy_files[i]))
            
        stacked_tensor = torch.vstack(tensor_list)
        torch.save(stacked_tensor, os.path.join(save_tensor_dirpath, f'{str_name}_{material_name}_{phase_name}_rs.pt'))

        #! Temporary code for exploring effect of refining scale 
        # save_output = True
        # out_suffix = f'*.out'
        # if save_output == True:
        #     out_files = sorted(glob.glob(os.path.join(input_path, out_suffix)))
        #     shutil.move(out_files[0], os.path.join(save_topas_input_dirpath, f'{str_name}.out'))
        #     shutil.move(xy_files[0], os.path.join(save_topas_input_dirpath, f'{str_name}_xy.xy'))

    #*----------------------------------------------------------------------------------------------------------------------------
    #* Use worker function to call thread_task and execute through the task_queue
    #*----------------------------------------------------------------------------------------------------------------------------
    def worker():
        while True:
            try:
                # Retrieve a task without blocking
                task = task_queue.get_nowait()
                str_file = task
                thread_task(cwd, material_name, phase_name, str_file, random_strain_bounds, random_state, diffraction_type, input_dict, beq_dict, path_to_topas_directory, save_topas_input_dirpath, save_tensor_dirpath)
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

    #*----------------------------------------------------------------------------------------------------------------------------
    #* Collect run time and delete temp folders created by thread_task
    #*----------------------------------------------------------------------------------------------------------------------------
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















