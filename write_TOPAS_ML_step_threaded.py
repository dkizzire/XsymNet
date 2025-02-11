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
    #* ------------------------------------------------------------------------------------------------------
    #* START OF USER INPUT
    #* ------------------------------------------------------------------------------------------------------
    NUM_THREADS = 5                                 #^ default = 5; number of threads to launch to optimize scale or create randomstates
    material_name = 'wo3'                           #^ name of your material
    phase_name = '1085K'                       
    instrument_file = 'wo3.txt'                     #^ located in ./instrument_parameters
    peakshape_file = 'wo3_peak_type.txt'            #^ located in ./instrument_parameters
    number_of_random_states = 250                   #^ = 1 : Rietveld of whole subgroup tree;  = ~250-500 create randomstates for ML training 
    path_to_topas_directory = 'C:\TOPAS6'           #^ windows path to your TOPAS directory; should be 'C:\TOPAS#' 
    diffraction_type = 'xray'                       #^ choices: neutron or xray
    optimize_scale = True                           #^ Generates better scale values for each subgroup
    flag_create_randomstates = True                 #^ True or False
    use_custom_latparams = False                    #^ If you have a structure with crazy strain
    use_strain_modes = True                         #^ only change if you are using custom latparams
    random_strain_bounds = [-0.001, 0.001]          #^ default -> [-0.002, 0.002]  ::NOTE:: tiny changes here make a BIG difference in lattice parameter
    random_perturb_bounds = [-0.1, 0.1, -5.0, 5.0]  #^ bounds for perturbing custom latparams [abc_lo(angstrom), abc_hi(angstrom), angle_lo(degrees), angle_hi(degrees)] 
    
    #^ experimental xy file to use for xdd section of TOPAS when creating randomstates
    experimental_data_file = f'd8_03901_t1085.xy'
    experimental_data_path = os.path.join(cwd, material_name, experimental_data_file)
    xdd_winpath = PureWindowsPath(Path(experimental_data_path).absolute())

    beq_dict = {
        'W': [0.0, 1.5],
        'O': [0.0, 1.5]
    }
    
    # beq_dict = {
    #     'Bi1': [0.0, 1.5],
    #     'Sn1': [0.0, 1.5],
    #     'O1': [0.0, 2.5], 
    #     'O2': [0.0, 2.5]
    # }
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


    #* ------------------------------------------------------------------------------------------------------
    #* END OF USER INPUT
    #* ------------------------------------------------------------------------------------------------------

    str_files = sorted(glob.glob(os.path.join(cwd, material_name, f'topas_strs_{material_name}', '*.str')))
    # str_files = str_files[25:30]

    cif_files = sorted(glob.glob(os.path.join(cwd, material_name, f'subgroup_cifs_{material_name}', '*')))
    cif_filenames = [os.path.basename(x) for x in cif_files]
    cif_noext = [os.path.splitext(x)[0] for x in cif_filenames]
    subgroup_nums = [x[3:] for x in cif_noext]


    # topas output files from preliminary pawley refinement of subgroup tree lattice parameters
    if use_custom_latparams == True:
        easy_grab_files = sorted(glob.glob(os.path.join(cwd, material_name, 'SGT_pawley_refinement', '*', '*.out')))

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

        # create random numbers for augmenting strain values
        random_strain = np.random.uniform(low=random_strain_bounds[0], high=random_strain_bounds[1], size=(len(random_state), 6))
        
        # create randomly perturbed TOPAS input files
        topas_input_file_list = []

        #*----------------------------------------------------------------------------------------------------------------------------
        #* Get optimized scale
        #*----------------------------------------------------------------------------------------------------------------------------

        # scale_input_file = os.path.join(input_path, f'scale_{str_name}.inp')
        # scale_output_file = os.path.join(input_path, f'scale_{str_name}.out')

        # mode_lines = funcs.generate_strain_and_symmetry_mode_lines_from_topas_str_v2(str_file, spacegroup_origin, random_strain[0], beq_dict=beq_dict, refine_strain=True, refine_distortion=True)
        # # insert peakshape, r_bragg, and scale into mode lines
        # mode_lines.insert(4, f'\t\tphase_name {str_name}\n')
        # mode_lines.insert(4, f'{input_dict["peakshape_lines"]}\n')
        # mode_lines.insert(4, f'\t\tr_bragg 0.0\n')
        # mode_lines.insert(4, f'\t\tscale @ 1.0\n')

        # # write scale optimization input file
        # with open(scale_input_file, 'w') as f:
        #     f.write(input_dict['scale_refinement_lines'])
        #     f.write(input_dict['global_lines'])
        #     f.write(input_dict['xdd_lines'])
        #     f.write(input_dict['instrument_lines'])
        #     for k in range(len(mode_lines)):
        #         f.write(mode_lines[k])

        # # Convert topas input path to windows so TOPAS can run it (TOPAS is Windows only)
        # scaling_winpath = PureWindowsPath(Path(scale_input_file).absolute()) 

        # # write a windows batchfile to run all random_states with tc.exe (topas command line)
        # batchfile_path = os.path.join(input_path, 'scaling.bat')
        # with open(batchfile_path, 'w') as f:
        #     f.write(f'cd {path_to_topas_directory} \n')
        #     f.write(f'tc "{scaling_winpath}" \n')
        # batchfile_path = PureWindowsPath(Path(batchfile_path).absolute())

        # # Run the batch file and capture output and errors
        # result = subprocess.run(batchfile_path, shell=False, capture_output=True, text=True)

        # # grab the optimized scale from output file
        # scale = 1.0
        # with open(scale_output_file, 'r') as f:
        #     lines = f.readlines()
        #     for i in range(len(lines)):
        #         if 'scale @ ' in lines[i]:
        #             scale = float(lines[i].strip().split('@')[1].strip().strip('`'))
        #             break 
                
        # print(f'{str_name} scale = {scale}\n')
        # shutil.copy(scale_input_file, os.path.join(scale_file_destination, os.path.basename(scale_input_file)))
        # shutil.copy(scale_output_file, os.path.join(scale_file_destination, os.path.basename(scale_output_file)))

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

            # create strain and symmetry mode lines from str_file with random values
            mode_lines = funcs.generate_strain_and_symmetry_mode_lines_from_topas_str_v2(str_file, spacegroup_origin, random_strain[i], beq_dict=beq_dict)

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
                f.write(input_dict['xdd_lines'])
                f.write(input_dict['instrument_lines'])
                
                # if use_strain_modes == False:
                #     f.write(phase_lines)
                #     f.write(topas_latparam_lines)

                for k in range(len(mode_lines)):
                    f.write(mode_lines[k])
                f.write(xy_out_lines)
                
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
                    f.write(input_dict['xdd_lines'])
                    f.write(input_dict['instrument_lines'])
                    # if use_strain_modes == False:
                    #     f.write(phase_lines)
                    #     f.write(topas_latparam_lines)
                    for k in range(len(mode_lines)):
                        f.write(mode_lines[k])
                    f.write(saved_xy_out_lines)


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















