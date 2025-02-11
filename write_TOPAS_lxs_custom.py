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
    material_name = 'lxs'
    phase_name = '6040'
    instrument_file = 'nomad.txt'
    peakshape_file = 'nomad_peak_type.txt'
    number_of_random_states = 1
    path_to_topas_directory = 'C:\TOPAS6'
    diffraction_type = 'neutron'   #* choices: neutron or xray
    use_strain_modes = True
    random_strain_bounds = [-0.001, 0.001]
    random_perturb_bounds = [-0.1, 0.1, -5.0, 5.0] # bounds for perturbing custom lattice parameters [abc_lo(angstrom), abc_hi(angstrom), angle_lo(degrees), angle_hi(degrees)] 
    use_custom_latparams = False
    overwrite_input_file_with_output_file = True

    experimental_data_file = f'd8_03901_t1043.xy'
    experimental_data_path = os.path.join(cwd, material_name, experimental_data_file)
    xdd_winpath = PureWindowsPath(Path(experimental_data_path).absolute())

    beq_dict = {
        'Li1': [0, 3],
        'Li2': [0,3],
        'Mn1': [0,3],
        'Mn2': [0,3],
        'Ni1': [0,3],
        'Ni2': [0,3],
        'O1': [0,3]
    }

    # beq_dict = {
    #     'W': [0.0, 1.5],
    #     'O': [0.0, 1.5]
    # }
    
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


    # ------------------------------------------------------------------------------------------------------
    # END OF USER INPUT
    # ------------------------------------------------------------------------------------------------------

    str_files = sorted(glob.glob(os.path.join(cwd, material_name, f'topas_strs_{material_name}', '*.str')))
    # str_files = str_files[25:30]

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
    
    # refinement_lines = funcs.refinement_options(iters=1000, continue_on_convergence=False, suppress_out_file=False)
    refinement_lines = f'''
macro file_name {{replace_file_name_here}}  
 
'----------------------------------------------------------------------------------------### refinement methods and global residual ### 
iters 1000 
chi2_convergence_criteria 0.00001                                                  
'do_errors 
no_LIMIT_warnings 
'continue_after_convergence 
r_wp  7.68664147 r_exp  0.695994204 r_p  6.10824139 r_wp_dash  19.6701663 r_p_dash  29.6182211 r_exp_dash  1.78105376 weighted_Durbin_Watson  0.0138587159 gof  11.0441171
'---------------------------------------------------------------------------------------- 

'-----------------------------------------------------------------### tof-to-d conversion, tof = t0 + difc*d + difa * d^2 ### 
prm !difc1  1420.29010_3.67526 
prm !difc2  2853.24519_2.01870 
prm !difc3  5630.18705_3.66965 
prm !difc4  9047.31525_3.62748 
prm !difc5  9886.86429_3.46229 
 
prm !t01  -2.70883_3.08892     min -40 max 40' for NAC 
prm !t02 -2.11955_1.78070    min -40 max 40 
prm !t03  -20.60111_2.32190    min -40 max 40 
prm !t04 -11.93783_2.01543     min -40 max 40 
prm !t05    0.95264_1.76093  min -40 max 40 

'------------------------ can change below, not above -----------------
prm difa1  0.07711` min -40 max 40 
prm difa2  27.26042` min -40 max 40 
prm difa3  1.53081` min -40 max 40 
prm difa4  14.28534` min -40 max 40 
prm difa5   2.55741` min -40 max 40 
 
prm difa1_LaB6  1.49989` min -40 max 40 
prm difa2_LaB6  5.32037` min -40 max 40 
prm difa3_LaB6 -1.97340` min -40 max 40 
prm difa4_LaB6 -5.06288` min -40 max 40 
prm difa5_LaB6  0.22977` min -40 max 40 
 
 
prm t01_long  40.00000` min -40 max 40 'only with large d-spacing peaks 
prm t02_long  39.84904` min -40 max 40  
prm t03_long  40.00000` min -40 max 40  
prm t04_long -40.00000` min -40 max 40  
prm difa11  0.32735` min -40 max 40  
prm difa22  11.43291` min -40 max 40  
prm difa33 -16.95156` min -40 max 40  
prm difa44 -40.00000` min -40 max 40  
 
 
prm t01_long_LaB6  13.70728` min -40 max 40 
prm t02_long_LaB6 -1.54995` min -40 max 40 
prm t03_long_LaB6  6.55850` min -40 max 40 
prm t04_long_LaB6 -13.57898` min -40 max 40 
prm difa11_LaB6   0.79323` min -40 max 40 
prm difa22_LaB6   5.19396` min -40 max 40 
prm difa33_LaB6  -4.59939` min -40 max 40 
prm difa44_LaB6  -4.71493` min -40 max 40 
 
 
prm difa1_Si  1.95195`    min -40 max 40 
prm difa2_Si  3.67861`    min -40 max 40 
prm difa3_Si -14.70950` min -40 max 40 
prm difa4_Si -20.21344`   min -40 max 40 
prm difa5_Si  0.07494`    min -40 max 40 
'----------------------------------------------------------------- 
 
 
 
'------------------------------------------------------------------ ##structure refinement related parameters##### 
prm scaleLFP  139888765.00000` min 0.0001 
prm !scaleLTO  0 min 0.0001 
prm !scaleCu  0 min 0.0001 
prm !scaleAl  0 min 0.0001 
prm !scaleNi  0 min 0.0001 
prm scalebank2  0.10000` 'min 0.1 max 10 
prm scalebank3  0.10000` 'min 0.1 max 10 
prm scalebank4  0.10000` 'min 0.1 max 10 
prm scalebank5  0.15863` 'min 0.1 max 10 
 
'--------------------------------strain broadening for Si------------------------------------ 
prm  gama0_sample_bank1     15.39478` min 0   max 2000  ' see GSAS manual for details, it is found that PV function works better than pure Lorentzian function   
prm  gama1_strain_bank1     9.79722` min 0   max 2000 
prm  gama2_size_bank1       20.03883` min 0   max 2000 
prm  sample_lor_bank1       0.36189` min 0.0001 max 1 
 
prm  gama0_sample_bank2   56.85549` min 0   max 2000 
prm  gama1_strain_bank2   2.36223` min 0   max 2000 
prm  gama2_size_bank2     26.51735` min 0   max 2000 
prm  sample_lor_bank2     0.00012` min 0.0001 max 1 
 
 
prm  gama0_sample_bank3   33.27300` min 0   max 2000 
prm  gama1_strain_bank3   114.44641` min 0   max 2000 
prm  gama2_size_bank3     0.06033` min 0   max 2000 
prm  sample_lor_bank3     0.39504` min 0.0001 max 1 
 
prm  gama0_sample_bank4   163.06666` min 0   max 200 
prm  gama1_strain_bank4   21.52294` min 0   max 200 
prm  gama2_size_bank4     3.35242` min 0   max 200 
prm  sample_lor_bank4     0.71233` min 0.0001 max 1 
 
prm gama0_sample_bank5   147.30413` min 0     max 200 
prm gama1_strain_bank5  31.63998` min 0    max 200 
prm gama2_size_bank5     2.53294` min 0    max 200 
prm sample_lor_bank5     0.63708` min 0.0001 max 1 
	 
prm !cystallite_size = 0.94* difc5/gama2_size_bank5;:  3669.11669` ' use the highest resoluiton bank to extract crystallite size and microstrain information.  
prm !sample_microstrain = gama1_strain_bank5/difc5;:  0.00320`
'------------------------------------------------------------------- 
 
 
 
'---------------------------------------------# automatically change file names for multiple data sets 
   
macro NAC_file_name  {{NOM_S2_Mn0p6}} 
  
'----------------------------------------------- 
 
'------------------------------------------------------------------------------------------ ### Defining data sets ### 
 
'#define NAC_NOMAD_bank1    
#define NAC_NOMAD_bank2          
' #define NAC_NOMAD_bank3    
' #define NAC_NOMAD_bank4    
' #define NAC_NOMAD_bank5 
       
'------------------------------------------------------------------------------------------ ### Defining data sets ### 
 
'------------------------------------------------------------------------------------------ ### Main refinement block ###  
 
 
 
#ifdef NAC_NOMAD_bank1  
 
xdd NAC_file_name-0.xye  'replace this file  
r_wp  4.57109769 r_exp  1.74748109 r_p  8.11540692 r_wp_dash  14.8995592 r_p_dash  41.0620515 r_exp_dash  5.69593998 weighted_Durbin_Watson  1.19483526 gof  2.61582097	 
	NeutronDiffraction_NOMAD                                                                   ' TOF neutron diffraction data 
	NOMAD_bank1	                                                                               ' bank1 center two theta, L2 distance and absorption refinement  
	local mu  0.108055748 min 0 max 1 
	bkg @  0.0837208462 -0.01506836  0.00839472999  0.00107103095  0.000897053034 -0.00235502749  0.00497430779 -0.00327479405  0.0028434576  0.000191142819  0.00018998119  0.00101965134	 
    'TOF_x_axis_calibration(,t01,,difc1,,difa1)                                                 'use this if no Bragg peak with d_spacing larger than 4 angstrom  
	TOF_x_axis_calibration_front_scattering_banks(,t01,,t01_long,,difc1,,difa1,,difa11)                                                                            
     
 
	str                                                                                          ' primary phase  
	phase_name "bank_15_LFP" 
    scale @  0.00318485094
    NOMAD_bank1_ipf                                                                              ' instrument broadening parameter  
  	sample_broadening(, gama0_sample_bank1,, gama1_strain_bank1,, gama2_size_bank1, , sample_lor_bank1 ) 
	 
 
 #endif 
 
'###########################################################################

' filename -1 is for bank 2 since -0 in the filename corresponds to bank 1
#ifdef NAC_NOMAD_bank2  
	 
xdd NAC_file_name-1.xye  'replace this file  
r_wp  7.68664147 r_exp  0.695994204 r_p  6.10824139 r_wp_dash  19.6701663 r_p_dash  29.6182211 r_exp_dash  1.78105376 weighted_Durbin_Watson  0.0138587159 gof  11.0441171	 
    NeutronDiffraction_NOMAD                      
    NOMAD_bank2 
    local mu  0.26861` min 0 max 1 
    bkg @  0.0801287816` -0.0177692347` -0.00244895846` -0.00403402166` -0.00276243637` '-0.0052893548 -0.00460314863 -0.00358328544 -0.00493986309 -0.000956875459
    'TOF_x_axis_calibration(,t02 , ,difc2 , ,  difa2)         ' TOF = zero + difc*d + difa*d^2 
    TOF_x_axis_calibration_front_scattering_banks(,t02,,t02_long,,difc2,,difa2,,difa22)      
    start_X 5000
	finish_X 17000
    local !NAC 1 
	local !CaF2 1 
	str 
		' scattering angle = bank_32_NAC  ... data usually stored in phase_name
		phase_name "lxs_trial_1" 
		scale @  0.000424708713`
		NOMAD_bank2_ipf 
		sample_broadening(, gama0_sample_bank2,, gama1_strain_bank2,, gama2_size_bank2, , sample_lor_bank2 ) 
	 
 #endif 
 
'###########################################################################

#ifdef NAC_NOMAD_bank3  
 
xdd NAC_file_name-2.xye 'replace this file  
r_wp  8.39158765 r_exp  0.519733345 r_p  6.02530302 r_wp_dash  19.7466798 r_p_dash  20.348801 r_exp_dash  1.22301147 weighted_Durbin_Watson  0.0135912115 gof  16.1459482
	NeutronDiffraction_NOMAD  
	NOMAD_bank3 
	local mu  0.02303 min 0 max 1 
	bkg @  0.0751776507 -0.00366252174  0.00279569988 -0.00171958381 -0.00160063789  '0.00117295902  0.000200180172  0.000215916381  0.00337273146  0.00491198972  0.00397959813  0.00227057679  0.000823646636
	'TOF_x_axis_calibration(,t03, , difc3, ,difa3) 
	 TOF_x_axis_calibration_front_scattering_banks(,t03, ,t03_long,, difc3 , ,difa3,,difa33) 
     start_X 4000
	 finish_X 18000
	 
	local !NAC 1 
	local !CaF2 1 
	str 
	phase_name "bank_67_NAC" 
	scale @  0.00896796813
    NOMAD_bank3_ipf 
    	sample_broadening(, gama0_sample_bank3,, gama1_strain_bank3,, gama2_size_bank3, , sample_lor_bank3 ) 
	 
 
 
 #endif 
 
'###########################################################################

#ifdef NAC_NOMAD_bank4  
  
xdd NAC_file_name-3.xye 'replace this file  
r_wp  7.92311477 r_exp  0.566372203 r_p  5.54496274 r_wp_dash  22.2647453 r_p_dash  19.4749569 r_exp_dash  1.59156256 weighted_Durbin_Watson  0.0156759079 gof  13.9892366
	 
	NeutronDiffraction_NOMAD  
    NOMAD_bank4 
	local mu  0.00000 min 0 max 1		 
	bkg @  0.0695640411 -0.00876497448  0.00594363829 -9.51509746e-005  0.0019776187 -0.00264346009 '-0.00187188443  6.00083883e-005  0.00448938946 -0.00132614679 -0.00232015987 -0.00129304114  0.00309677376
	'TOF_x_axis_calibration(,t04 , ,difc4 , ,difa4) 
	TOF_x_axis_calibration_front_scattering_banks(,t04, ,t04_long,, difc4 , ,difa4,,difa44) 
    start_X 3000
	finish_X 16000
  	local !NAC 1 
	local !CaF2 1 
	str 
	phase_name "bank_122_NAC" 
	scale @  0.0131089516
    NOMAD_bank4_ipf 
 	sample_broadening(, gama0_sample_bank4,, gama1_strain_bank4,, gama2_size_bank4, , sample_lor_bank4 ) 
	 
 
 
 #endif 
 
'###########################################################################

#ifdef NAC_NOMAD_bank5 
 
xdd NAC_file_name-4.xye   'replace this file  
r_wp  5.48963199 r_exp  0.592695141 r_p  4.33067704 r_wp_dash  2.42375535 r_p_dash  1.8831438 r_exp_dash  0.261683847 weighted_Durbin_Watson  0.0626164813 gof  9.26215117
    NeutronDiffraction_NOMAD 
    NOMAD_bank5 
	local mu  0.11305 min 0 max 1 
    bkg @ -0.117214902  0.00406185715  0.00701513639 -0.0166959043  0.0182867641 -0.00947485079  0.0109846199 -0.0111934565  0.0133207335 '-0.00562908461  0.000376003634  0.000721322472  0.000533909438
	TOF_x_axis_calibration(,t05 , , difc5, ,difa5) 
	  
	 start_X 5000
	 finish_X 15000
	str 
	phase_name "bank_154_LFP" 
	scale @  0.00810107552
    NOMAD_bank5_ipf 
		sample_broadening(, gama0_sample_bank5,, gama1_strain_bank5,, gama2_size_bank5, , sample_lor_bank5 ) 
	 
 
	 
#endif 
for xdds {{ 
 
                                                                                      '!!!! For dual space refinement, remove the prime in front of #define PDF and add selxdd1 after xdds!!!!! 
 for strs 1 to 1 {{  
 
 '''
    
    global_lines = f'''
'''
    # global_lines = funcs.global_parameters()
    # xdd_lines = funcs.xdd(xdd_winpath)
    
    xdd_lines = f''' 
'''
    # ----- generate instrument specific lines -----
    instrument_lines = f'''
'''
    # instrument_lines = funcs.instrument_parameters(os.path.join(cwd, 'instrument_parameters', instrument_file))
    peakshape_lines = f'''
'''
    # peakshape_lines = funcs.instrument_parameters(os.path.join(cwd, 'instrument_parameters', peakshape_file))
    end_of_for_xdd_lines = f'''
}}
}}
'''
    end_of_file_lines= f''' 
prm !tspow   0.26012_0.46845 min 0.01 max 5.5  
prm !alph0   6.63435_1.90309 min 0.0001 
prm !alph1   0.01745_0.87312 min 0.0001 
prm !beta0   0.00010_0.01728 min 0.0001 
prm !beta1   0.77385_0.15070 min 0.0001 
prm !k_eff   278.64531_45.08282 min 0.0001 max 100000 
prm !hk      1.70815_0.51561 min 0.0001 
'----------------------------------------------------------------- 
prm  !sigma0_bank1  0.00000_190.77105 min 0           max 5000 
prm  !sigma1_bank1  1343.44659_190.59290 min 0        max 5000 
prm  !sigma2_bank1  320.00283_33.15729 min 0          max 5000 
prm  !sigma1_long_bank1  5000.00000_340.96188 min 0   max 5000 
prm  !sigma2_long_bank1  56.92977_21.22494 min 0       max 5000 
prm  !sigma1e_bank1 0  
prm  !sigma2e_bank1 0 
prm  !phitheta_bank1 90 
prm  !sigma0_bank2  359.19667_154.94016 min 0      max 5000 
prm  !sigma1_bank2  2203.08447_201.64639 min 0   max 5000 
prm  !sigma2_bank2  1091.79695_55.49866 min 0      max 5000 
prm  !sigma1_long_bank2  5000.00000_74.37218 min 0   max 5000 
prm  !sigma2_long_bank2  254.26726_8.18301 min 0       max 5000 
 
 
prm  !sigma1e_bank2 0  
prm  !sigma2e_bank2 0 
prm  !phitheta_bank2 90 
prm  !sigma0_bank3  766.59190_149.10677 min 0   max 5000 
prm  !sigma1_bank3  5000.00000_188.43404 min 0   max 5000 
prm  !sigma2_bank3  457.69250_48.34978 min 0   max 5000 
prm  !sigma1_long_bank3  3691.66279_177.09373 min 0   max 5000 
prm  !sigma2_long_bank3  648.06595_36.87080 min 0       max 5000 
prm  !sigma1e_bank3 0  
prm  !sigma2e_bank3 0 
prm  !phitheta_bank3 90 
prm  !sigma0_bank4  1161.33224_68.35456 min 0   max 5000 
prm  !sigma1_bank4  12.09941_106.34473 min 0     max 5000 
prm  !sigma2_bank4  931.95502_40.04382 min 0   max 5000 
prm  !sigma1_long_bank4  227.82377 min 0   max 5000 
prm  !sigma2_long_bank4  1088.93571 min 0       max 5000 
 
prm  !sigma1e_bank4   0 
prm  !sigma2e_bank4   0 
prm  !phitheta_bank4 90 
prm  !sigma0_bank5  121.33904_29.50975 min 0   max 5000 
prm  !sigma1_bank5  156.07635_54.26293 min 0   max 5000 
prm  !sigma2_bank5  135.35457_27.85021 min 0   max 5000 
prm  !sigma1e_bank5 0  
prm  !sigma2e_bank5 0 
prm  !phitheta_bank5 90 
prm !lor_bank1   0.15063_0.03683 min 0 max 1 
prm !lor_bank2   0.13897_0.01272 min 0 max 1 
prm !lor_bank3   0.41705_0.00772 min 0 max 1 
prm !lor_bank4   0.64374_0.00858 min 0 max 1 
prm !lor_bank5   0.37677_0.01364 min 0 max 1 

'''
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
                  'peakshape_lines':peakshape_lines,
                  'end_of_file_lines': end_of_file_lines,
                  'end_of_for_xdd_lines': end_of_for_xdd_lines}
    
    save_topas_input_dirpath = os.path.join(cwd, material_name, 'ml_data_generation', phase_name, 'RS_SGT_INP')
    if not os.path.exists(save_topas_input_dirpath): os.makedirs(save_topas_input_dirpath)

    dir_to_store_data = 'ml_data'
    save_tensor_dirpath = os.path.join(cwd, material_name, dir_to_store_data, phase_name)
    if not os.path.exists(save_tensor_dirpath): os.makedirs(save_tensor_dirpath)

    # implement threading with thread_task
    def thread_task(cwd, material_name, phase_name, str_file, random_strain_bounds, random_state, diffraction_type, input_dict, beq_dict, path_to_topas_directory, save_topas_input_dirpath, save_tensor_dirpath ):

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
            mode_lines = funcs.generate_strain_and_symmetry_mode_lines_from_topas_str_v2(str_file, spacegroup_origin, random_strain[i], beq_dict=beq_dict, refine_beq=True, refine_strain=True, refine_distortion=True)
            #! remove first 3 lines of topas.str so we don't double up on str entries 
            mode_lines = mode_lines[4:]

            # insert peakshape, r_bragg, and scale into mode lines
            # mode_lines.insert(4, f'\t\tphase_name {str_name}\n')
            # mode_lines.insert(4, f'{input_dict["peakshape_lines"]}\n')
            # mode_lines.insert(4, f'\t\tr_bragg 0.0\n')
            # ! mode_lines.insert(4, f'\t\tscale  1.0\n')
            # mode_lines.insert(4, f'\t\tscale 0.0002\n')

            # write TOPAS input file for this random state
            with open(topas_input_file, 'w') as f:

                # f.write(f'out_file="";\n')
                f.write(f'\'WARNING WARNING WARNING: Above line suppresses output file, comment out if you want a .out file\n')
                f.write(input_dict['refinement_lines'])
                # f.write(input_dict['global_lines'])
                # f.write(input_dict['xdd_lines'])
                # f.write(input_dict['instrument_lines'])
                
                # if use_strain_modes == False:
                #     f.write(phase_lines)
                #     f.write(topas_latparam_lines)

                for k in range(len(mode_lines)):
                    f.write(mode_lines[k])
                # f.write(xy_out_lines)
                f.write(input_dict['end_of_for_xdd_lines'])

                #! more custom stuff
                f.write(input_dict['end_of_file_lines'])

            # Save the first random state file created for each subgroup for manual running/debugging
            if i == 0:
                # write TOPAS input file for this random state
                saved_xy_out_file = PureWindowsPath(Path(os.path.join(save_topas_input_dirpath, f'{str_name}_xy.xy')).absolute())
                saved_xy_out_lines = f'\nOut_X_Ycalc("{saved_xy_out_file}")\n'
                rs_saved_input_file = os.path.join(save_topas_input_dirpath, f'{str_name}.inp')
                with open(rs_saved_input_file, 'w') as f:
                    f.write(input_dict['refinement_lines'])
                    # f.write(input_dict['global_lines'])
                    # f.write(input_dict['xdd_lines'])
                    # f.write(input_dict['instrument_lines'])
                    # if use_strain_modes == False:
                    #     f.write(phase_lines)
                    #     f.write(topas_latparam_lines)
                    for k in range(len(mode_lines)):
                        f.write(mode_lines[k])
                    # f.write(saved_xy_out_lines)
                    f.write(input_dict['end_of_for_xdd_lines'])

                    #! more custom stuff
                    f.write(input_dict['end_of_file_lines'])

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

        # xy_files = sorted(glob.glob(os.path.join(input_path, data_suffix)))
        # tensor_list = []
        # for i in range(len(xy_files)):
        #     tensor_list.append(funcs.convert_bragg_to_tensors(xy_files[i]))
            
        # stacked_tensor = torch.vstack(tensor_list)
        # torch.save(stacked_tensor, os.path.join(save_tensor_dirpath, f'{str_name}_{material_name}_{phase_name}_rs.pt'))

        #! Temporary code for exploring effect of refining scale 
        # save_output = True
        # out_suffix = f'*.out'
        # if save_output == True:
        #     out_files = sorted(glob.glob(os.path.join(input_path, out_suffix)))
        #     shutil.move(out_files[0], os.path.join(save_topas_input_dirpath, f'{str_name}.out'))
        #     # shutil.move(xy_files[0], os.path.join(save_topas_input_dirpath, f'{str_name}_xy.xy'))

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