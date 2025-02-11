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
    #^ ------------------------------------------------------------------------------------------------------
    #^ START OF USER INPUT
    #^ ------------------------------------------------------------------------------------------------------
    NUM_THREADS = 5                                 #^ default = 5; number of threads to launch to optimize scale or create randomstates
    material_name = 'wo3'                           #^ name of your material
    instrument_file = 'wo3.txt'                     #^ located in ./instrument_parameters
    peakshape_file = 'wo3_peak_type.txt'            #^ located in ./instrument_parameters
    path_to_topas_directory = 'C:\TOPAS6'           #^ windows path to your TOPAS directory; should be 'C:\TOPAS#' 
    diffraction_type = 'xray'                       #^ choices: neutron or xray
    optimize_scale = True                           #^ Generates better scale values for each subgroup
    use_custom_latparams = False                    #^ If you have a structure with crazy strain
    use_strain_modes = True                         #^ only change if you are using custom latparams
    random_strain_bounds = [-0.003, 0.003]          #^ default -> [-0.005, 0.005]  ::NOTE:: tiny changes here make a BIG difference in lattice parameter
    random_perturb_bounds = [-0.1, 0.1, -5.0, 5.0]  #^ bounds for perturbing custom latparams [abc_lo(angstrom), abc_hi(angstrom), angle_lo(degrees), angle_hi(degrees)] 
    
    #^ range for randomly sampling beq values per atom type; examples commented below
    beq_dict = {
        'W': [0.0, 1.5],
        'O': [0.0, 1.5]
    }

    #^ ------------------------------------------------------------------------------------------------------
    #^ Info for all experimental data to find scaling for
    #^ ------------------------------------------------------------------------------------------------------
    
    #^ list of experimental xy files; used to generate better scale values for each subgroup memeber for each experimental data
    # experimental_data_filenames = ['d8_03901_t1325.xy', 'd8_03901_t1243.xy', 'd8_03901_t1103.xy', 'd8_03901_t1043.xy']
    # experimental_data_paths = [os.path.join(cwd, material_name, x) for x in experimental_data_filenames]

    experimental_data_filenames = ['bkg_sub_P1_wo3_1325K_xy.xy', 'bkg_sub_P1_wo3_1243K_xy.xy', 'bkg_sub_P1_wo3_1103K_xy.xy', 'bkg_sub_P1_wo3_1043K_xy.xy']
    experimental_data_paths = [os.path.join(cwd, material_name, 'experimental_refinement', 'pawley', x) for x in experimental_data_filenames]
    #^ id associated with each experimental data file above
    phase_names = ['1325K_scale_bkgsub', '1243K_scale_bkgsub', '1103K_scale_bkgsub', '1043K_scale_bkgsub']                      

    #^ Pawley P1 refinement output files for each experiment - not sure if i need this 
    p1_filenames = ['pawley_P1_wo3_1325K.out', 'pawley_P1_wo3_1243K.out', 'pawley_P1_wo3_1103K.out', 'pawley_P1_wo3_1043K.out']


    

    # zip paths and phase_names together to be passed into the thread queue together
    expt = list(zip(experimental_data_paths, phase_names))

    #^ ------------------------------------------------------------------------------------------------------
    #^ END OF USER INPUT
    #^ ------------------------------------------------------------------------------------------------------
    funcs.assert_positive_integer(NUM_THREADS)


    str_files = sorted(glob.glob(os.path.join(cwd, material_name, f'topas_strs_{material_name}', '*.str')))
    # str_files = str_files[25:30]

    # topas output files from preliminary pawley refinement of subgroup tree lattice parameters
    if use_custom_latparams == True:
        easy_grab_files = sorted(glob.glob(os.path.join(cwd, material_name, 'SGT_pawley_refinement', '*', '*.out')))

    #* ------------------------------------------------------------------------------------------------------
    #* Initiate threading task queue
    #* ------------------------------------------------------------------------------------------------------
    # Number of tasks and threads
    NUM_TASKS = len(expt)

    # # if there are more threads than tasks, 
    # if NUM_THREADS > NUM_TASKS:
    #     NUM_THREADS = NUM_TASKS

    # Create a queue and populate it with tasks (in this case, None as placeholders)
    task_queue = queue.Queue()
    for i in range(NUM_TASKS):
        task_queue.put(expt[i])  



    #* ------------------------------------------------------------------------------------------------------
    #* Generate lines of topas input file that are the same for every randomstate
    #* ------------------------------------------------------------------------------------------------------
    
    refinement_lines = funcs.refinement_options(iters=1000, continue_on_convergence=False, suppress_out_file=False)
    global_lines = funcs.global_parameters()
    #! xdd_lines = funcs.xdd(xdd_winpath)
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
                  'instrument_lines':instrument_lines,
                  'peakshape_lines':peakshape_lines}
    

    #^----------------------------------------------------------------------------------------------------------------------------
    #^ Threaded Methods Below
    #^----------------------------------------------------------------------------------------------------------------------------

    def thread_task(cwd, material_name, expt_path, phase_name, str_files, random_strain_bounds, input_dict, beq_dict, path_to_topas_directory):

        thread_id = threading.get_ident()

        print(f'Thread {thread_id} Optimizing Subgroup Tree Scaling for {os.path.basename(expt_path)}')

        # make temporary folder to store input/output data per thread
        input_path = os.path.join(cwd, material_name, 'ml_data_generation', phase_name, f'tempthread_{thread_id}')
        if not os.path.exists(input_path): os.makedirs(input_path)
        
        xdd_winpath = PureWindowsPath(Path(expt_path).absolute())
        xdd_lines = funcs.xdd(xdd_winpath)

        input_dict['xdd_lines'] = xdd_lines


        #*----------------------------------------------------------------------------------------------------------------------------
        #* Get optimized scale
        #*----------------------------------------------------------------------------------------------------------------------------

        scaling_dict_after = optimize_scale(cwd, material_name, phase_name, str_files, input_dict, random_strain_bounds, beq_dict, path_to_topas_directory, input_path)


    #*----------------------------------------------------------------------------------------------------------------------------
    #* - Method - Optimize the scale for each subgroup using TOPAS
    #*----------------------------------------------------------------------------------------------------------------------------   
    def optimize_scale(cwd, material_name, phase_name, str_files, input_dict, random_strain_bounds, beq_dict, path_to_topas_directory, input_path):
        
        scale_file_destination = os.path.join(cwd, material_name, 'ml_data_generation', phase_name, 'scale_optimization')
        if not os.path.exists(scale_file_destination): os.makedirs(scale_file_destination)
        # create random numbers for augmenting strain values
        random_strain = np.random.uniform(low=random_strain_bounds[0], high=random_strain_bounds[1], size=(len(str_files), 6))

        scaling_dict = {}

        for k in range(len(str_files)):

            str_file = str_files[k]
            str_name = os.path.splitext(os.path.basename(str_file))[0]

            # grab lattice parameters and spacegroup from str file
            latparams, spacegroup_origin = funcs.get_str_info(str_file, 'strain')

            # names for topas input/output files
            scale_input_file = os.path.join(input_path, f'scale_{str_name}.inp')
            scale_output_file = os.path.join(input_path, f'scale_{str_name}.out')

            mode_lines = funcs.generate_strain_and_symmetry_mode_lines_from_topas_str_v2(str_file, spacegroup_origin, random_strain[0], beq_dict=beq_dict, refine_strain=True, refine_distortion=True, refine_beq=True)
            # insert peakshape, r_bragg, and scale into mode lines
            mode_lines.insert(4, f'\t\tphase_name {str_name}\n')
            mode_lines.insert(4, f'{input_dict["peakshape_lines"]}\n')
            mode_lines.insert(4, f'\t\tr_bragg 0.0\n')
            mode_lines.insert(4, f'\t\tscale @ 1.0\n')

            # write scale optimization input file
            with open(scale_input_file, 'w') as f:
                # f.write(input_dict['scale_refinement_lines'])
                f.write(input_dict['global_lines'])
                f.write(input_dict['xdd_lines'])
                f.write(input_dict['instrument_lines'])
                for k in range(len(mode_lines)):
                    f.write(mode_lines[k])

            # Convert topas input path to windows so TOPAS can run it (TOPAS is Windows only)
            scaling_winpath = PureWindowsPath(Path(scale_input_file).absolute()) 

            # write a windows batchfile to run all random_states with tc.exe (topas command line)
            batchfile_path = os.path.join(input_path, 'scaling.bat')
            with open(batchfile_path, 'w') as f:
                f.write(f'cd {path_to_topas_directory} \n')
                f.write(f'tc "{scaling_winpath}" \n')
            batchfile_path = PureWindowsPath(Path(batchfile_path).absolute())

            # Run the batch file and capture output and errors
            result = subprocess.run(batchfile_path, shell=False, capture_output=True, text=True)

            # grab the optimized scale from output file
            scale = 1.0
            with open(scale_output_file, 'r') as f:
                lines = f.readlines()
                for i in range(len(lines)):
                    if 'scale @ ' in lines[i]:
                        scale = float(lines[i].strip().split('@')[1].strip().strip('`'))
                        break

            print(f'{str_name} scale = {scale}\n')
            shutil.copy(scale_input_file, os.path.join(scale_file_destination, os.path.basename(scale_input_file)))
            shutil.copy(scale_output_file, os.path.join(scale_file_destination, os.path.basename(scale_output_file)))
            
            scaling_dict[str_name] = scale
        
        funcs.save_dict_to_json(scaling_dict, os.path.join(scale_file_destination, f'{phase_name}_scale_dict.json'))
        return scaling_dict



    #*----------------------------------------------------------------------------------------------------------------------------
    #* - Method - Use worker function to call thread_task and execute through the task_queue
    #*----------------------------------------------------------------------------------------------------------------------------
    def worker():
        while True:
            try:
                # Retrieve a task without blocking
                task = task_queue.get_nowait()
                expt_path = task[0]
                phase_name = task[1]
                thread_task(cwd, material_name, expt_path, phase_name, str_files, random_strain_bounds, input_dict, beq_dict, path_to_topas_directory)
                task_queue.task_done()
            except queue.Empty:
                # No more tasks are available
                break
        

    #*----------------------------------------------------------------------------------------------------------------------------
    #* create save locations for scale optimiztion file, saved first random state file, and ml data tensors
    #*----------------------------------------------------------------------------------------------------------------------------
    
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
    
    print(f'Scaling Optimization Has Finished\nRuntime: {timed}')

    for x in range(len(phase_names)):
        temp_folders = sorted(glob.glob(os.path.join(cwd, material_name, 'ml_data_generation', phase_names[x], 'tempthread_*')))
        for y in range(len(temp_folders)):
            shutil.rmtree(temp_folders[y])




if __name__ == "__main__":
    main()















