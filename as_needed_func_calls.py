# import os 
# import glob
# import json
# import write_TOPAS_funcs as funcs

# cwd = os.getcwd()

# material_name = 'wo3'
# phase_name = 'scale_base_1243K'

# out_files = sorted(glob.glob(os.path.join(cwd, material_name, 'ml_data_generation', phase_name, 'scale_optimization', '*.out')))

# str_files = sorted(glob.glob(os.path.join(cwd, material_name, f'topas_strs_{material_name}', '*.str')))
# str_names = [os.path.splitext(os.path.basename(x))[0] for x in str_files]

# scale_dict = {}

# for i in range(len(out_files)):
#     with open(out_files[i], 'r') as f:
#         lines = f.readlines()
#         for j in range(len(lines)):
#             if 'scale @ ' in lines[j]:
#                 scale = float(lines[j].strip().split('@')[1].strip().strip('`'))
#                 break
    
#     scale_dict[str_names[i]] = scale

# dict_path = os.path.join(cwd, material_name, 'ml_data', phase_name, f'{phase_name}_scales.json')
# funcs.save_dict_to_json(scale_dict, dict_path)
# print(f'\nscaling dict for {phase_name} was written to {dict_path}')

# test = {}
# test1 = list(test.keys())
# print(test1)
# print(len(test1))

import os
import shutil
import glob
import numpy as np
import pandas as pd
import torch
import random
from pathlib import Path, PureWindowsPath
import write_TOPAS_funcs as funcs
import utility_functions as util

import threading
import queue
import random
import time
from datetime import timedelta
import subprocess





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
random_strain_bounds = [-0.005, 0.005]          #^ default -> [-0.005, 0.005]  ::NOTE:: tiny changes here make a BIG difference in lattice parameter
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
experimental_data_filenames = ['d8_03901_t1325.xy', 'd8_03901_t1243.xy', 'd8_03901_t1103.xy', 'd8_03901_t1043.xy']

#^ id associated with each experimental data file above
phase_names = ['1325K_scale', '1243K_scale', '1103K_scale', '1043K_scale']                      

#^ Pawley P1 refinement output files for each experiment - not sure if i need this 
p1_filenames = ['pawley_P1_wo3_1325K.out', 'pawley_P1_wo3_1243K.out', 'pawley_P1_wo3_1103K.out', 'pawley_P1_wo3_1043K.out']


# experimental_data_paths = [os.path.join(cwd, material_name, x) for x in experimental_data_filenames]
# xdd_windpaths = [PureWindowsPath(Path(x).absolute()) for x in experimental_data_paths]
# expt = list(zip(experimental_data_paths, phase_names))


scaleopt = os.path.join(cwd, material_name, 'intensity_test', '*scaleopt_xy.xy')
scale1 = os.path.join(cwd, material_name, 'intensity_test', '*scale1_xy.xy')

scale_data1 = pd.read_csv(scale1, sep='\s+', header=None)

scale


scale1_data_scaled = scale_data1*0.001

