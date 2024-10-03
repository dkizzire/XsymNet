'''
Author: Dr. Dayton  Kizzire
Last Updated: 03/05/2024
Email: kizziredg@ornl.gov
'''

import os
import json
import glob
import math
import numpy as np
import pandas as pd
from pathlib import Path, PureWindowsPath
import re
from fractions import Fraction

# ------ Refinement Options  ------
def refinement_options(iters:int = 0, converge:float = 0.0001, continue_on_convergence:bool = False, anneal:bool = False, anneal_param:float = 0.00001, do_errors:bool = False, limit_warn:bool = False, suppress_out_file:bool = False):
    
    continue_line = f"'continue_after_convergence"
    if continue_on_convergence:
        continue_line = continue_line[1:]

    anneal_line = f"'Auto_T({anneal_param})"
    if anneal:
        anneal_line = anneal_line[1:]

    error_line = f"'do_errors"
    if do_errors:
        error_line = error_line[1:]
    
    limit_warn_line = f"'no_LIMIT_warnings"
    if limit_warn:
        limit_warn_line = limit_warn_line[1:]

    suppress_out_line = f" "
    if suppress_out_file: 
        suppress_out_line = f'out_file="";'
    
    
    refine_lines = f"""\'----------------------------------------### Refinement methods ###----------------------------------------
iters  {iters}
chi2_convergence_criteria {converge}
{continue_line}
{anneal_line}
{error_line}
{limit_warn_line}
{suppress_out_line}

r_wp  0.0 r_exp  0.0 r_p  0.0 r_wp_dash  0.0 r_p_dash  0.0 r_exp_dash  0.0 weighted_Durbin_Watson  0.0 gof  0.0

"""
    return refine_lines



# ------ Global Parameters ------
def global_parameters():
    global_lines = f"""\'----------------------------------------### Global parameters ###----------------------------------------
prm !pow  4
"""
    return global_lines

def xdd(data_winpath:str):

    xdd_line = f'''xdd "{data_winpath}" \n'''
    return xdd_line



# ------ Instrument specific info for TOPAS INP files ------
def instrument_parameters(instrument_file:str):
    with open(instrument_file, 'r') as f:
        lines = f.readlines()
    instrument_lines = ''.join(lines)
    return instrument_lines


def powgen_inp_lines(xdd_winpath, start_finish_X, refine_instrument_parameters=False):
    '''
    Parameters
    ----------
    xdd_winpath: str
        Full Windows path to the data file for xdd
    start_finish_X: list of floats
        start and finish of TOF range for Topas
    refine_instrument_parameters: bool: False
        writes ! in front of instrument parameters to prevent them from refining
    '''


    if refine_instrument_parameters == False:
        param_lines = """    local    !d0_1    9.32399	    min -20  max 20 
	local    !difc_1  22571.96264   min 22500 max 22650
	local    !difa_1 -0.01750		   	    min -10  max 10	
	local    !difb_1 -5.83330		        min -10   max 5
	TOF_x_axis_calibration_POWGEN(@,=d0_1;,
								  @,=difc_1;,
								  @,=difa_1;,
								  @,=difb_1;)
	
    POWGEN_instrument_2						   '2019B values in below
    local   !sig0 -36.48037 min -100 max 200   'sig0 -42.50381_0.30318 min -100 max 200 
    local   !sig1  44.86277 min -100 max 200   'sig1  47.53093_0.18553 min -100 max 200 
    local   !sig2 -0.22180 min -100 max 200    'sig2 -1.25106_0.02154 min -100 max 200  
    local   !sig3  1.32642 min -100 max 200    'sig3  7.17003_0.11297 min -100 max 200  
    local   !gam0 -1.45221 min -20 max 20      'gam0 -11.17295_0.33002 min -20 max 20   
    local   !gam1  14.40400 min -20 max 20     'gam1  15.25484_0.66956 min -20 max 20   
    local   !gam2 -1.76409 min -100 max 200    'gam2  3.02682_0.30385 min -100 max 200  
            
    GSAS_btb
    local   !alpha0  152.73005 min 0 max 250    'alpha0 -95.16910_1.07001 min -150 max 100
    local   !alpha1 -31.59878 min -100 max 100  'alpha1  41.69600_0.78367 min -100 max 100
    local   !alpha2 -14.38509 min -100 max 100  'alpha1  41.69600_0.78367 min -100 max 100
    local   !beta0 -62.66207 min -100 max 0     'beta0  73.08348_0.29513 min -100 max 100 
    local   !beta1  16.38983 min -100 max 100   'beta1 -2.68722_0.05497 min -100 max 100  

"""

    else:
        param_lines = """	 local    d0_1    9.32610	  	min -20  max 20 
	local    difc_1  22571.96329	min 22500 max 22650
	local    difa_1  3.29809		min -10  max 10	
	local    difb_1 -1.40727		min -5 max 5 
	TOF_x_axis_calibration_POWGEN(@,=d0_1;,
										   @,=difc_1;,
										   @,=difa_1;,
										   @,=difb_1;)
	

    POWGEN_instrument_2						  '2019B values in below
    local   sig0 -36.48037 min -100 max 200   'sig0 -42.50381_0.30318 min -100 max 200 
    local   sig1  44.86277 min -100 max 200   'sig1  47.53093_0.18553 min -100 max 200 
    local   sig2 -0.22180 min -100 max 200    'sig2 -1.25106_0.02154 min -100 max 200  
    local   sig3  1.32642 min -100 max 200    'sig3  7.17003_0.11297 min -100 max 200  
    local   gam0 -1.45221 min -20 max 20      'gam0 -11.17295_0.33002 min -20 max 20   
    local   gam1  14.40400 min -20 max 20     'gam1  15.25484_0.66956 min -20 max 20   
    local   gam2 -1.76409 min -100 max 200    'gam2  3.02682_0.30385 min -100 max 200  
            
    GSAS_btb
    local   alpha0  152.73005 min 0 max 250   'alpha0 -95.16910_1.07001 min -150 max 100
    local   alpha1 -31.59878 min -100 max 100 'alpha1  41.69600_0.78367 min -100 max 100
    local   alpha2 -14.38509 min -100 max 100 'alpha1  41.69600_0.78367 min -100 max 100
    local   beta0 -62.66207 min -100 max 0    'beta0  73.08348_0.29513 min -100 max 100 
    local   beta1  16.38983 min -100 max 100  'beta1 -2.68722_0.05497 min -100 max 100  

"""
    POWGEN_lines =  f"""\'----------------------------------------### Defining data sets ###----------------------------------------
#define POWGEN_bank2_HR     #define PZT

\'------------------------------------ ### Main refinement block ### ------------------------------------
\'------------------------------------ ### Bank 2 Refinement ### ----------------------------------------

\'bank 2 CWL = 1.5 A
\'Auto_T(0.00001)
#ifdef POWGEN_bank2_HR  \' CW 1.5
xdd "{xdd_winpath}"
    r_wp  0.00 r_exp  0.00 r_p  0.00 r_wp_dash  0.00 r_p_dash  0.00 r_exp_dash  0.00 weighted_Durbin_Watson  0.00 gof  0.00
	NeutronDiffraction   \' TOF neutron diffraction data         
	scale_pks = D_spacing^pow;  \' Scale peak intensity as
    bkg   @  14.048056` -6.61727545`  2.9504521` -0.5453947`  0.965912817` -0.365159705`  0.114676395`  0.0222332242` -0.0315144216` -0.256784311` -0.146714196`  0.142224656` -0.299421417`  0.031692733`  0.147907249`
    start_X {start_finish_X[0]}
    finish_X {start_finish_X[1]}
	
{param_lines}

\'------------------------------------ ### STR BLOCK ### ----------------------------------------

"""
    return POWGEN_lines


def hrpd_inp_lines(xdd_winpath, start_finish_X):
    HRPD_lines = f"""\'----------------------------------------### Defining data sets ###----------------------------------------
#define use_data 
#ifdef use_data
#define USE_BS    macro filename_bs    {{"{xdd_winpath}"}}
#endif

\'----------------------------------------### Main refinement block ###----------------------------------------
#ifdef USE_BS
xdd filename_bs xye_format
    start_X {start_finish_X[0]}
    finish_X {start_finish_X[1]}
    local !neutron 1 local !xray 0 
    neutron_data
    r_wp 0.0
    TOF_x_axis_calibration(t0_hrpd_bs,-49.37818`,!difc_hrpd_bs, 48255.09936_0.61297, difa_hrpd_bs,  15.23076`)
    local !bank_two_theta 168.567 
    local !mu  0.07076 
    local !fixb  0.0 min -10 max 10 
    bkg  @  1.23166856` -0.0124816464`  0.209502581` -0.119357101` -0.0364802546` -0.0805659581`  0.280382189` -0.180514454`  0.157958349` -0.129704358`  0.215427676`  0.0027121254`
    
    scale_pks = (Exp(-2*fixb/(2 D_spacing)^2)); \'fixb correction local to each bank
    weighting = If(SigmaYobs < 1e-9, 1, 1/SigmaYobs^2);
        x_calculation_step = Yobs_dx_at(Xo); convolution_step 4
        TOF_LAM(0.001)
        scale_pks = D_spacing^4;
        scale_pks = Exp(-mu*2 D_spacing Sin(0.00872664626 two_theta));   \'absorption correction local to each bank

        #define peak1
        #ifdef peak1
            wifd_pkshape(!two_theta, =bank_two_theta;, !aprm, 2.50000, !tauf_2, 9.80744, !tauf_1, 0.43977, !tauf_0, 0.000, !taus_0, 25.27420, !taus_1, 2.66797, !t_eff, 144.41296, !hhh, 3.58595, !hhh_0, 11.61348)
        #endif

"""
    return HRPD_lines


def mspd_inp_lines(xdd_winpath, start_finish_X, refine_instrument_parameters=False):
    '''
    Parameters
    ----------
    xdd_winpath: str
        Full Windows path to the data file for xdd
    start_finish_X: list of floats
        start and finish of TOF range for Topas
    refine_instrument_parameters: bool: False
        writes ! in front of instrument parameters to prevent them from refining
    '''
    if refine_instrument_parameters == False:
        param_lines = """    local    !d0_1    9.32399	    min -20  max 20 
	local    !difc_1  22571.96264   min 22500 max 22650
	local    !difa_1 -0.01750		   	    min -10  max 10	
	local    !difb_1 -5.83330		        min -10   max 5
	TOF_x_axis_calibration_POWGEN(@,=d0_1;,
								  @,=difc_1;,
								  @,=difa_1;,
								  @,=difb_1;)
	
    POWGEN_instrument_2						   '2019B values in below
    local   !sig0 -36.48037 min -100 max 200   'sig0 -42.50381_0.30318 min -100 max 200 
    local   !sig1  44.86277 min -100 max 200   'sig1  47.53093_0.18553 min -100 max 200 
    local   !sig2 -0.22180 min -100 max 200    'sig2 -1.25106_0.02154 min -100 max 200  
    local   !sig3  1.32642 min -100 max 200    'sig3  7.17003_0.11297 min -100 max 200  
    local   !gam0 -1.45221 min -20 max 20      'gam0 -11.17295_0.33002 min -20 max 20   
    local   !gam1  14.40400 min -20 max 20     'gam1  15.25484_0.66956 min -20 max 20   
    local   !gam2 -1.76409 min -100 max 200    'gam2  3.02682_0.30385 min -100 max 200  
            
    GSAS_btb
    local   !alpha0  152.73005 min 0 max 250    'alpha0 -95.16910_1.07001 min -150 max 100
    local   !alpha1 -31.59878 min -100 max 100  'alpha1  41.69600_0.78367 min -100 max 100
    local   !alpha2 -14.38509 min -100 max 100  'alpha1  41.69600_0.78367 min -100 max 100
    local   !beta0 -62.66207 min -100 max 0     'beta0  73.08348_0.29513 min -100 max 100 
    local   !beta1  16.38983 min -100 max 100   'beta1 -2.68722_0.05497 min -100 max 100  

"""




# ------ Get info from cif and str files ------
def save_dict_to_json(dictionary, filename):
    with open(filename, 'w') as f:
        json.dump(dictionary, f, indent=4)


def get_cif_info(cif_path):
    '''
    Parameters
    ----------
    cif_path: str
        Full path to the cif file
        
    Returns:
        List containing the spacegroup and the lattice parameters
        [spacegroup_int, a, b, c, alpha, beta, gamma, origin_choice]

    '''
    with open(cif_path, 'r') as f:
        for line in f:
            if '_cell_length_a' in line:
                a = float(line.split()[-1])
            if '_cell_length_b' in line:
                b = float(line.split()[-1])
            if '_cell_length_c' in line:
                c = float(line.split()[-1])
            if '_cell_angle_alpha' in line:
                alpha = float(line.split()[-1])
            if '_cell_angle_beta' in line:
                beta = float(line.split()[-1])
            if '_cell_angle_gamma' in line:
                gamma = float(line.split()[-1])
            if '# Subgroup ' in line:
                sg_name = line.split(': ')[1].split()[1]
            if '_symmetry_Int_Tables_number' in line: 
                spacegroup_int = int(line.split()[-1])
            if '_symmetry_space_group_name_H-M' in line:
                sline = line.split('"') 
                if '(origin choice' in sline[1]:
                    tline = sline[1].split(' (origin choice ') 
                    origin_choice = int(tline[1][0])
                    # print(f'{cif_path} : \t has ({tline[1]}')
                else: origin_choice = 1

        latparams = [a, b, c, alpha, beta, gamma]
        sg_o = [spacegroup_int, origin_choice, sg_name]

    return latparams, sg_o 


def get_str_info(str_path, str_type):
    '''
    Parameters
    ----------
    str_path: str
        Full path to the topas_str file generated by isodistort
    str_type: string
        accepted inputs: 
            'regular' for a regular isodistort topas str file
            'strain' for an isodistort topas str file with strain modes
        
    Returns:
    ----------
    a,b,c,alpha,beta,gamma: list of floats
    sg_o = spacegroup & origin & HM_name: list of ints and string
    '''
    lines = []
    with open(str_path, 'r') as f:
        lines = f.readlines()

    if str_type == 'regular':
        oline = lines[4].strip()
        sg_name = oline.split(':')[0].strip("'")
        if ':' in oline:
            origin = int(oline.split(':')[1])
        else:
            origin = 1
        
        sg = int(lines[5].strip().split()[1])
        alat = float(lines[6].strip().split()[1])
        blat = float(lines[7].strip().split()[1])
        clat = float(lines[8].strip().split()[1])
        alpha = float(lines[9].strip().split()[1])
        beta = float(lines[10].strip().split()[1])
        gamma = float(lines[11].strip().split()[1])


    if str_type == 'strain':
        f = iter(lines)
        for i in range(len(lines)):
            if '\tstr\n' in lines[i]:
                oline = lines[i+1].strip()
                sg_name = oline.split(':')[0].strip("'")
                if ':' in oline:
                    origin = int(oline.split(':')[1])
                else:
                    origin = 1
            
            if 'space_group' in lines[i]:
                sg = int(lines[i].strip().split()[1])
            
            if '\t\t\ta ' in lines[i]:
                aline = lines[i].strip()
                if ':' in aline:
                    alat = float(aline.split(':')[1].strip())
                    blat = float(lines[i+1].strip().split(':')[1].strip())
                    clat = float(lines[i+2].strip().split(':')[1].strip())
                    alpha = float(lines[i+3].strip().split(':')[1].strip())
                    beta = float(lines[i+4].strip().split(':')[1].strip())
                    gamma = float(lines[i+5].strip().split(':')[1].strip())
                    break

    latparams = [alat, blat, clat, alpha, beta, gamma]
    sg_o = [sg, origin, sg_name]
    return latparams, sg_o 


def get_pawley_latparams_easy_grab(file):
    '''
    Parameters
    ----------
    file: str
        path to the topas out file
        
    Returns:
        list of lattice parameters: [a,b,c,alpha,beta,gamma]

    Purpose:
        This function is used to retrieve lattice parameters refined by topas

    Important Note:
        only works for my easy grab reporting setup
    '''
    in_lines = []
    with open(file,'r') as f:
        in_lines = f.readlines()
        
    print(f'read lattice parameters from : {file}')
    
    easy_grab_index = 0
    for i in range(len(in_lines)):
        if "Easy grab" in in_lines[i]:
            easy_grab_index = i
            break
    

    a = float(in_lines[easy_grab_index+1].strip().strip('`').split(':')[-1])
    b = float(in_lines[easy_grab_index+2].strip().strip('`').split(':')[-1])
    c = float(in_lines[easy_grab_index+3].strip().strip('`').split(':')[-1])
    alpha = float(in_lines[easy_grab_index+4].strip().strip('`').split(':')[-1])
    beta = float(in_lines[easy_grab_index+5].strip().strip('`').split(':')[-1])
    gamma = float(in_lines[easy_grab_index+6].strip().strip('`').split(':')[-1])

    refined_latparams = [a, b, c, alpha, beta, gamma]

    return refined_latparams


def count_refinable_parameters_in_topas_strs(topas_str_path):

     # define ranges of bravais lattices
    dof6_range = np.arange(1,3).tolist()  # triclinic
    dof4_range = np.arange(3,16).tolist() # monoclinic
    dof3_range = np.arange(16,75).tolist() # orthorhombic
    dof2_range = np.arange(75,195).tolist() # tetragonal, trigonal, hexagonal
    dof1_range = np.arange(195,231).tolist() # cubic

    f = []
    count = 0
    with open(topas_str_path, 'r') as file:
        f = file.readlines()

    for i in range(len(f)):
        if 'space_group' in f[i]:
            sline = f[i].strip().split()
            spacegroup = int(sline[1])
            if spacegroup in dof6_range:
                count = count + 6
            if spacegroup in dof4_range:
                count = count + 4
            if spacegroup in dof3_range:
                count = count + 3
            if spacegroup in dof2_range:
                count = count + 2
            if spacegroup in dof1_range:
                count = count + 1
        
        if 'mode definitions' in f[i]:
            for j in range(i+1, len(f)):
                if 'prm' in f[j]:
                    count = count+1
                if len(f[j].strip()) == 0:
                    i = j
                    break
        
        if 'mode-dependent sites' in f[i]:
            for j in range(i+1, len(f)):
                if 'site' in f[j]:
                    count = count+1
                if len(f[j].strip()) == 0:                        
                    break   
            break
    
    return count


def get_cif_info_with_site_info(cif_path):
    '''
    Parameters
    ----------
    cif_path: str
        Full path to the cif file
        
    Returns:
    a,b,c,alpha,beta,gamma: float
    spacegroup_int: int
    site_lines: list of str
    '''
    site_info = []
    with open(cif_path, 'r') as f:
        for line in f:
            if '_cell_length_a' in line:
                a = float(line.split()[-1])
            if '_cell_length_b' in line:
                b = float(line.split()[-1])
            if '_cell_length_c' in line:
                c = float(line.split()[-1])
            if '_cell_angle_alpha' in line:
                alpha = float(line.split()[-1])
            if '_cell_angle_beta' in line:
                beta = float(line.split()[-1])
            if '_cell_angle_gamma' in line:
                gamma = float(line.split()[-1])
            if '_symmetry_Int_Tables_number' in line: 
                spacegroup_int = int(line.split()[-1])
            if '_symmetry_space_group_name_H-M' in line:
                sline = line.split('"') 
                if '(origin choice' in sline[1]:
                    tline = sline[1].split(' (origin choice ') 
                    origin_choice = int(tline[1][0])
                    print(f'{cif_path} : \t has ({tline[1]}') 
                else: origin_choice = 1 

            #if '_symmetry_space_group_name_H-M' in line:
            #    origin_name = ':1' 
            #    sline = line.split('"')  
            #    if '(' in sline[1]:
            #        tline = sline[1].split(' (')
            #        sline[1]=tline[0]
            #        origin_name = tline[1].split(')')[0]

            #    longname = sline[1]
            #    shortname = sgd.get(sline[1])
            #    sg_long_names.append(longname)
            #    sg_short_names.append(shortname)
            #    sg_origin.append(origin_name)

            if '_atom_site_fract_symmform' in line:
                for line in f:
                    if line == '\n':
                        break
                    site_info.append(line)

    lp_list = [spacegroup_int, a,b,c,alpha,beta,gamma, origin_choice ]

    return lp_list, site_info


def get_cif_spacegroup_and_origin(cif_path):
    '''
    Parameters
    ----------
    cif_path: str
        Full path to the cif file
        
    Returns: list of 2 ints
        
        integer: 1 or 2 for origin choice
    '''
    with open(cif_path, 'r') as f:
        for line in f:
            if '_symmetry_Int_Tables_number' in line: 
                spacegroup_int = int(line.split()[-1])
            if '_symmetry_space_group_name_H-M' in line:
                sline = line.split('"') 
                if '(origin choice' in sline[1]:
                    tline = sline[1].split(' (origin choice ')
                    origin_choice = int(tline[1][0])
                    # print(f'{cif_path} : \t has ({tline[1]}') 

                else: origin_choice = 1

    space_group_and_origin = [spacegroup_int, origin_choice]
    return space_group_and_origin


def find_p1_sgt_members(subgroup_cifs_path):
    '''
    Parameters
    ----------
    subgroup_cifs_path: str
        Full path to the subgroup_cifs_directory
        
    Returns:
        List containing the paths to cifs which have P1 symmetry (primitive structures)

    '''
    cif_paths = sorted(glob.glob(os.path.join(subgroup_cifs_path, '*.cif')))
    primitive_paths = []
    for i in range(len(cif_paths)):
        with open(cif_paths[i], 'r') as f:
            for line in f:

                if '_symmetry_Int_Tables_number' in line: 
                    spacegroup_int = int(line.split()[-1])
                    break
        
        if spacegroup_int == 1:
            primitive_paths.append(cif_paths[i])
    
    return primitive_paths





# ------ Generate latparam and site lines ------
def generate_pawley_lattice_lines(latparams, spacegroup_origin, peakshape_lines):
    '''
    Parameters
    ----------
    latparams: list
        [a, b, c, alpha, beta, gamma]
    spacegroup_origin: list
        [spacegroup: int, origin_choice: int]
    peakshape_lines: str
        a string defining the peakshape (has to be included in this section)

    Returns:
    ----------
    topas_latparam_lines: list of strings that can be input into a topas input file for pawley lattice parameter refinement
    '''
    sg = spacegroup_origin[0]
    # initialize topas_latparam_lines
    latparam_lines = ['a', 'b', 'c','alpha','beta','gamma', 'sg']

    # define ranges of bravais lattices
    triclinic_range = np.arange(1,3).tolist()
    monoclinic_range = np.arange(3,16).tolist()
    orthorhombic_range = np.arange(16,75).tolist()
    tetragonal_range = np.arange(75,143).tolist()
    # in our subgroup tree from isodistort, every trigonal cif is in trigonal_hexagonal axes format (even if they rhombohedral space groups) 
    # --> if needed for a different project, can turn these back on and comment out trigonal_range
    # trigonal_hex_range = [143, 144, 145, 147, 149, 150, 151, 152, 153, 154, 156, 157, 158, 159, 162, 163, 164, 165]
    # trigonal_rhomb_range = [146, 148, 155, 160, 161, 166, 167]
    trigonal_range = np.arange(143, 168).tolist()
    hexagonal_range = np.arange(168,195).tolist()
    cubic_range = np.arange(195,231).tolist()

    # set spacegroup:origin_choice
    latparam_lines[6] = f'space_group {spacegroup_origin[0]}:{spacegroup_origin[1]}'

    # set pawley refinement lines based on spacegroup symmetry
    if sg in triclinic_range:
        latparam_lines[0] = f'a lpa {latparams[0]} vc'
        latparam_lines[1] = f'b lpb {latparams[1]} vc'
        latparam_lines[2] = f'c lpc {latparams[2]} vc'
        latparam_lines[3] = f'al alph {latparams[3]} vc'
        latparam_lines[4] = f'be bet {latparams[4]} vc'
        latparam_lines[5] = f'ga gam {latparams[5]} vc'

    if sg in monoclinic_range:
        latparam_lines[0] = f'a lpa {latparams[0]} vc'
        latparam_lines[1] = f'b lpb {latparams[1]} vc'
        latparam_lines[2] = f'c lpc {latparams[2]} vc'
        latparam_lines[3] = f'al 90.0'
        latparam_lines[4] = f'be bet {latparams[4]} vc'
        latparam_lines[5] = f'ga 90.0'

    if sg in orthorhombic_range:
        latparam_lines[0] = f'a lpa {latparams[0]} vc'
        latparam_lines[1] = f'b lpb {latparams[1]} vc'
        latparam_lines[2] = f'c lpc {latparams[2]} vc'
        latparam_lines[3] = f'al 90.0'
        latparam_lines[4] = f'be 90.0'
        latparam_lines[5] = f'ga 90.0'

    if sg in tetragonal_range:
        latparam_lines[0] = f'a lpa {latparams[0]} vc'
        latparam_lines[1] = f'b = lpa; : 0'
        latparam_lines[2] = f'c lpc {latparams[2]} vc'
        latparam_lines[3] = f'al 90.0'
        latparam_lines[4] = f'be 90.0'
        latparam_lines[5] = f'ga 90.0'

    if sg in trigonal_range:
        latparam_lines[0] = f'a lpa {latparams[0]} vc'
        latparam_lines[1] = f'b = lpa; : 0'
        latparam_lines[2] = f'c lpc {latparams[2]} vc'
        latparam_lines[3] = f'al 90.0'
        latparam_lines[4] = f'be 90.0'
        latparam_lines[5] = f'ga 120.0'

    if sg in hexagonal_range:
        latparam_lines[0] = f'a lpa {latparams[0]} vc'
        latparam_lines[1] = f'b = lpa; : 0'
        latparam_lines[2] = f'c lpc {latparams[2]} vc'
        latparam_lines[3] = f'al 90.0'
        latparam_lines[4] = f'be 90.0'
        latparam_lines[5] = f'ga 120.0'

    if sg in cubic_range:
        latparam_lines[0] = f'a lpa {latparams[0]} vc'
        latparam_lines[1] = f'b = lpa; : 0'
        latparam_lines[2] = f'c = lpa; : 0'
        latparam_lines[3] = f'al 90.0'
        latparam_lines[4] = f'be 90.0'
        latparam_lines[5] = f'ga 90.0'

    hkl_is_lines = f"""
    hkl_Is
    {peakshape_lines} 
        scale @ 1
        r_bragg  0
        
        {latparam_lines[0]}
        {latparam_lines[1]}
        {latparam_lines[2]}
        {latparam_lines[3]}
        {latparam_lines[4]}
        {latparam_lines[5]}
        macro vc {{ val_on_continue = Val+Val*Rand(-.1,.1); }}
        'Easy grab reporting
        local !latparam_a = Get(a); : 0
        local !latparam_b = Get(b); : 0
        local !latparam_c = Get(c); : 0
        local !latparam_al = Get(al); : 0
        local !latparam_be = Get(be); : 0
        local !latparam_ga = Get(ga); : 0

        {latparam_lines[6]}
"""
    return hkl_is_lines


def generate_pawley_lattice_lines_v2(latparams, spacegroup_origin, use_vc:bool = False):
    '''
    Parameters
    ----------
    latparams: list
        [a, b, c, alpha, beta, gamma]
    spacegroup_origin: list
        [spacegroup: int, origin_choice: int]
    use_vc: bool
        if true, eill put "vc" after each optimizable lattice parameter, which will acitivate the vc macro

    Returns:
    ----------
    topas_latparam_lines: list of strings that can be input into a topas input file for pawley lattice parameter refinement
    '''
    if use_vc == True:
        vc = 'vc'
    else:
        vc = ''

    sg = spacegroup_origin[0]
    # initialize topas_latparam_lines
    latparam_lines = ['a', 'b', 'c','alpha','beta','gamma', 'sg']

    # define ranges of bravais lattices
    triclinic_range = np.arange(1,3).tolist()
    monoclinic_range = np.arange(3,16).tolist()
    orthorhombic_range = np.arange(16,75).tolist()
    tetragonal_range = np.arange(75,143).tolist()
    # in our subgroup tree from isodistort, every trigonal cif is in trigonal_hexagonal axes format (even if they rhombohedral space groups) 
    # --> if needed for a different project, can turn these back on and comment out trigonal_range
    # trigonal_hex_range = [143, 144, 145, 147, 149, 150, 151, 152, 153, 154, 156, 157, 158, 159, 162, 163, 164, 165]
    # trigonal_rhomb_range = [146, 148, 155, 160, 161, 166, 167]
    trigonal_range = np.arange(143, 168).tolist()
    hexagonal_range = np.arange(168,195).tolist()
    cubic_range = np.arange(195,231).tolist()

    # set spacegroup:origin_choice
    latparam_lines[6] = f'space_group {spacegroup_origin[0]}:{spacegroup_origin[1]}'

    # set pawley refinement lines based on spacegroup symmetry
    if sg in triclinic_range:
        latparam_lines[0] = f'a lpa {latparams[0]} {vc}'
        latparam_lines[1] = f'b lpb {latparams[1]} {vc}'
        latparam_lines[2] = f'c lpc {latparams[2]} {vc}'
        latparam_lines[3] = f'al alph {latparams[3]} {vc}'
        latparam_lines[4] = f'be bet {latparams[4]} {vc}'
        latparam_lines[5] = f'ga gam {latparams[5]} {vc}'

    if sg in monoclinic_range:
        latparam_lines[0] = f'a lpa {latparams[0]} {vc}'
        latparam_lines[1] = f'b lpb {latparams[1]} {vc}'
        latparam_lines[2] = f'c lpc {latparams[2]} {vc}'
        latparam_lines[3] = f'al 90.0'
        latparam_lines[4] = f'be bet {latparams[4]} {vc}'
        latparam_lines[5] = f'ga 90.0'

    if sg in orthorhombic_range:
        latparam_lines[0] = f'a lpa {latparams[0]} {vc}'
        latparam_lines[1] = f'b lpb {latparams[1]} {vc}'
        latparam_lines[2] = f'c lpc {latparams[2]} {vc}'
        latparam_lines[3] = f'al 90.0'
        latparam_lines[4] = f'be 90.0'
        latparam_lines[5] = f'ga 90.0'

    if sg in tetragonal_range:
        latparam_lines[0] = f'a lpa {latparams[0]} {vc}'
        latparam_lines[1] = f'b = lpa; : 0'
        latparam_lines[2] = f'c lpc {latparams[2]} {vc}'
        latparam_lines[3] = f'al 90.0'
        latparam_lines[4] = f'be 90.0'
        latparam_lines[5] = f'ga 90.0'

    if sg in trigonal_range:
        latparam_lines[0] = f'a lpa {latparams[0]} {vc}'
        latparam_lines[1] = f'b = lpa; : 0'
        latparam_lines[2] = f'c lpc {latparams[2]} {vc}'
        latparam_lines[3] = f'al 90.0'
        latparam_lines[4] = f'be 90.0'
        latparam_lines[5] = f'ga 120.0'

    if sg in hexagonal_range:
        latparam_lines[0] = f'a lpa {latparams[0]} {vc}'
        latparam_lines[1] = f'b = lpa; : 0'
        latparam_lines[2] = f'c lpc {latparams[2]} {vc}'
        latparam_lines[3] = f'al 90.0'
        latparam_lines[4] = f'be 90.0'
        latparam_lines[5] = f'ga 120.0'

    if sg in cubic_range:
        latparam_lines[0] = f'a lpa {latparams[0]} {vc}'
        latparam_lines[1] = f'b = lpa; : 0'
        latparam_lines[2] = f'c = lpa; : 0'
        latparam_lines[3] = f'al 90.0'
        latparam_lines[4] = f'be 90.0'
        latparam_lines[5] = f'ga 90.0'

    hkl_is_lines = f"""


        {latparam_lines[0]}
        {latparam_lines[1]}
        {latparam_lines[2]}
        {latparam_lines[3]}
        {latparam_lines[4]}
        {latparam_lines[5]}
        macro vc {{ val_on_continue = Val+Val*Rand(-.1,.1); }}

        'Easy grab reporting
        local !latparam_a = Get(a); : 0
        local !latparam_b = Get(b); : 0
        local !latparam_c = Get(c); : 0
        local !latparam_al = Get(al); : 0
        local !latparam_be = Get(be); : 0
        local !latparam_ga = Get(ga); : 0

        space_group {spacegroup_origin[0]}:{spacegroup_origin[1]}
"""
    return hkl_is_lines


def generate_pawley_latparam_lines_with_strain_modes(isodistort_str_file, spacegroup_origin, peakshape_lines):
    file_lines = []
    with open(isodistort_str_file, 'r') as f:
        file_lines = f.readlines()
    str_lines = []
    for i in range(len(file_lines)):
        if '\tstr\n' in file_lines[i]:
            file_lines[i] = '\thkl_Is\n'

        if 'space_group' in file_lines[i]:
            file_lines[i] = f'\t\t{peakshape_lines}\n'

        if 'prm  !s' in file_lines[i]:
            sline = file_lines[i]
            sline = sline.replace('!s', 's')
            sline = sline.replace('min','vc min')
            file_lines[i] = sline

        if 'unitless parent cell strains' in file_lines[i]:
            str_lines.append('\n\t\tmacro vc { val_on_continue = Val+Val*Rand(-.1,.1); }\n\n')

        # BREAKS LOOP WHEN ENCOUNTERING END OF LATTICE SECTION
        if '#endif' in file_lines[i]:
            str_lines.append('\t\t#endif\n\n')
            str_lines.append('\t\tscale @ 1\n')
            str_lines.append(f'\t\tspace_group {spacegroup_origin[0]}:{spacegroup_origin[1]}\n')
            break

        str_lines.append(file_lines[i])
    
    macro_lines = []
    for i in range(len(file_lines)):
        if '{{{macros ' in file_lines[i]:
            for j in range(i,len(file_lines)):
                macro_lines.append(file_lines[j])

    return str_lines, macro_lines


def generate_lattice_parameter_lines_with_vc_macro(latparams, spacegroup_origin, vc_lp_matrix):
    '''
    Parameters
    ----------
    latparams: list
        [a, b, c, alpha, beta, gamma]
    spacegroup_origin: list
        [spacegroup: int, origin_choice: int]
    
    vc_lp_matrix: 2d list of float
            vc macro: val on continue macro in topas
            vc_lp_matrix[[v1, v2, min, max], ....] --> fill in all 4 values for lattice parameters a, b, c, alpha, beta, gamma
            demo use:
            macro vc { val_on_continue = Val*Rand(v1,v2); }     --> where the current value of lattice parameter will be multiplied by a random value between v1 & v2
            a lpa 10.725 vc min 7.0 max 25.0                    --> where 7.0 & 25.0 are the lower & upper bounds of the lattice parameter refinement

    Returns:
    topas_latparam_lines: list of strings that can be input into a topas input file for pawley lattice parameter refinement using val on continue
    '''
    sg = spacegroup_origin[0]
    # initialize topas_latparam_lines
    latparam_lines = ['a', 'b', 'c','alpha','beta','gamma', 'sg']

    # define ranges of bravais lattices
    triclinic_range = np.arange(1,3).tolist()
    monoclinic_range = np.arange(3,16).tolist()
    orthorhombic_range = np.arange(16,75).tolist()
    tetragonal_range = np.arange(75,143).tolist()
    # in our subgroup tree from isodistort, every trigonal cif is in trigonal_hexagonal axes format (even if they rhombohedral space groups) 
    # --> if needed for a different project, can turn these back on and comment out trigonal_range
    # trigonal_hex_range = [143, 144, 145, 147, 149, 150, 151, 152, 153, 154, 156, 157, 158, 159, 162, 163, 164, 165]
    # trigonal_rhomb_range = [146, 148, 155, 160, 161, 166, 167]
    trigonal_range = np.arange(143, 168).tolist()
    hexagonal_range = np.arange(168,195).tolist()
    cubic_range = np.arange(195,231).tolist()

    # set spacegroup:origin_choice
    latparam_lines[6] = f'space_group {spacegroup_origin[0]}:{spacegroup_origin[1]}'

    # set pawley refinement lines based on spacegroup symmetry
    if sg in triclinic_range:
        latparam_lines[0] = f'a lpa {latparams[1]} vca min {vc_lp_matrix[0][2]} max {vc_lp_matrix[0][3]}'
        latparam_lines[1] = f'b lpb {latparams[2]} vcb min {vc_lp_matrix[1][2]} max {vc_lp_matrix[1][3]}'
        latparam_lines[2] = f'c lpc {latparams[3]} vcc min {vc_lp_matrix[2][2]} max {vc_lp_matrix[2][3]}'
        latparam_lines[3] = f'al alph {latparams[4]} vcalpha min {vc_lp_matrix[3][2]} max {vc_lp_matrix[3][3]}'
        latparam_lines[4] = f'be bet {latparams[5]} vcbeta min {vc_lp_matrix[4][2]} max {vc_lp_matrix[4][3]}'
        latparam_lines[5] = f'ga gam {latparams[6]} vcgamma min {vc_lp_matrix[5][2]} max {vc_lp_matrix[5][3]}'

    if sg in monoclinic_range:
        latparam_lines[0] = f'a lpa {latparams[1]} vca min {vc_lp_matrix[0][2]} max {vc_lp_matrix[0][3]}'
        latparam_lines[1] = f'b lpb {latparams[2]} vcb min {vc_lp_matrix[1][2]} max {vc_lp_matrix[1][3]}'
        latparam_lines[2] = f'c lpc {latparams[3]} vcc min {vc_lp_matrix[2][2]} max {vc_lp_matrix[2][3]}'
        latparam_lines[3] = f'al 90.0'
        latparam_lines[4] = f'be bet {latparams[5]} vcbeta min {vc_lp_matrix[4][2]} max {vc_lp_matrix[4][3]}'
        latparam_lines[5] = f'ga 90.0'

    if sg in orthorhombic_range:
        latparam_lines[0] = f'a lpa {latparams[1]} vca min {vc_lp_matrix[0][2]} max {vc_lp_matrix[0][3]}'
        latparam_lines[1] = f'b lpb {latparams[2]} vcb min {vc_lp_matrix[1][2]} max {vc_lp_matrix[1][3]}'
        latparam_lines[2] = f'c lpc {latparams[3]} vcc min {vc_lp_matrix[2][2]} max {vc_lp_matrix[2][3]}'
        latparam_lines[3] = f'al 90.0'
        latparam_lines[4] = f'be 90.0'
        latparam_lines[5] = f'ga 90.0'

    if sg in tetragonal_range:
        latparam_lines[0] = f'a lpa {latparams[1]} vca min {vc_lp_matrix[0][2]} max {vc_lp_matrix[0][3]}'
        latparam_lines[1] = f'b = lpa; : 0'
        latparam_lines[2] = f'c lpc {latparams[3]} vcc min {vc_lp_matrix[2][2]} max {vc_lp_matrix[2][3]}'
        latparam_lines[3] = f'al 90.0'
        latparam_lines[4] = f'be 90.0'
        latparam_lines[5] = f'ga 90.0'

    if sg in trigonal_range:
        latparam_lines[0] = f'a lpa {latparams[1]} vca min {vc_lp_matrix[0][2]} max {vc_lp_matrix[0][3]}'
        latparam_lines[1] = f'b = lpa; : 0'
        latparam_lines[2] = f'c lpc {latparams[3]} vcc min {vc_lp_matrix[2][2]} max {vc_lp_matrix[2][3]}'
        latparam_lines[3] = f'al 90.0'
        latparam_lines[4] = f'be 90.0'
        latparam_lines[5] = f'ga 120.0'

    if sg in hexagonal_range:
        latparam_lines[0] = f'a lpa {latparams[1]} vca min {vc_lp_matrix[0][2]} max {vc_lp_matrix[0][3]}'
        latparam_lines[1] = f'b = lpa; : 0'
        latparam_lines[2] = f'c lpc {latparams[3]} vcc min {vc_lp_matrix[2][2]} max {vc_lp_matrix[2][3]}'
        latparam_lines[3] = f'al 90.0'
        latparam_lines[4] = f'be 90.0'
        latparam_lines[5] = f'ga 120.0'

    if sg in cubic_range:
        latparam_lines[0] = f'a lpa {latparams[1]} vca min {vc_lp_matrix[0][2]} max {vc_lp_matrix[0][3]}'
        latparam_lines[1] = f'b = lpa; : 0'
        latparam_lines[2] = f'c = lpa; : 0'
        latparam_lines[3] = f'al 90.0'
        latparam_lines[4] = f'be 90.0'
        latparam_lines[5] = f'ga 90.0'
    

    return latparam_lines


def generate_latparam_lines_ml(latparams, spacegroup_origin):
    latparam_lines = ['a', 'b', 'c','alpha','beta','gamma']
    latparam_lines[0] = f'a {latparams[0]} '
    latparam_lines[1] = f'b {latparams[1]} '
    latparam_lines[2] = f'c {latparams[2]} '
    latparam_lines[3] = f'al {latparams[3]} '
    latparam_lines[4] = f'be {latparams[4]} '
    latparam_lines[5] = f'ga {latparams[5]} '

    return_lines = f"""

        {latparam_lines[0]}
        {latparam_lines[1]}
        {latparam_lines[2]}
        {latparam_lines[3]}
        {latparam_lines[4]}
        {latparam_lines[5]}

        'Easy grab reporting
        local !latparam_a = Get(a); : 0
        local !latparam_b = Get(b); : 0
        local !latparam_c = Get(c); : 0
        local !latparam_al = Get(al); : 0
        local !latparam_be = Get(be); : 0
        local !latparam_ga = Get(ga); : 0

        space_group {spacegroup_origin[0]}:{spacegroup_origin[1]}


"""
    return return_lines


def generate_site_lines(site_info, refine_beq=False, beq_dict=None):
    '''
    Parameters
    ----------
    site_info: list of str
        List of the site info lines from the cif file
    beq: List[Bool, int_lo, optional:int_hi]
        beq[0] = True : beq is set as a refinable parameter in topas  //  False : beq is not refined
        len(beq) = 2 : beq is set as input value for all sites/symmetry modes
        len(beq) = 3 : beq is set as a random integer between values input for beq[1] and beq[2]  (ie beq_lo & beq_hi)
        
    Returns:
    topas_site_lines: list of str
        List of the site lines from the cif file formatted for topas input
    '''
    # Split mode_dependent_sites at spaces to modify certain values
    site_info_lines = [line.split() for line in site_info]

    # get atom types from topas str file
    all_atom_types_list = [site_info_lines[i][1].split('_')[0] for i in range(len(site_info_lines))]
    all_atom_types_list_unique = np.unique(all_atom_types_list).tolist()

    # generate a beq_value_dict from ranges in beq_matrix for each atom type
    beq_value_dict = {}
    if beq_dict !=None:
        for i in range(len(all_atom_types_list_unique)):
            if all_atom_types_list_unique[i] == 'origin':
                continue
            beq_range = beq_dict.get(all_atom_types_list_unique[i])
            
            if len(beq_range) == 2:
                beq_value = np.random.uniform(beq_range[0], beq_range[1])
            elif len(beq_range) == 1:
                beq_value = beq_range[0]
            else: print('beq_matrix is configured incorrectly, please fix and try again')
            beq_value_dict[all_atom_types_list_unique[i]] = beq_value 

    topas_site_lines = []
    for i in range(len(site_info)):

        # break iteration if blank line appears
        sline = site_info[i].split()
        if len(sline) == 0:
            break
        
        atom_type = sline[0].split('_')[0]    

        if refine_beq == False:
                templine = f'\t    site {sline[0]} num_posns {sline[2]} x {sline[4]} y {sline[5]} z {sline[6]} occ {sline[1]} 1 beq !bval_{atom_type} {str(beq_value_dict.get(atom_type))}\n'

        if refine_beq == True:
                templine = f'\t    site {sline[0]} num_posns {sline[2]} x {sline[4]} y {sline[5]} z {sline[6]} occ {sline[1]} 1 beq bval_{atom_type} {str(beq_value_dict.get(atom_type))}\n'
        
        topas_site_lines.append(templine)

    return topas_site_lines


def get_latparams_from_topas_out(file):
    '''
    Parameters
    ----------
    file: str
        path to the topas out file
        
    Returns:
        list of lattice parameters: [spacegroup, a,b,c,alpha,beta,gamma, origin]

    Purpose:
        This function is used to retrieve lattice parameters refined by topas

    Important Note:
        In the str or hkl_Is section, you must have the following lines in order: 
        phase_name
        a
        b
        c
        al
        be
        ga
        space_group
    '''
    in_lines = []
    with open(file,'r') as f:
        in_lines = f.readlines()
        
    print(f'read lattice parameters from : {file}')
    
    rwp_index = 0
    phase_index = 0
    for i in range(len(in_lines)):
        if "r_wp" in in_lines[i]:
            rwp_index = i
        if "phase_name" in in_lines[i]:
            phase_index = i
        if "load hkl_m_d_th2 I" in in_lines[i]:
            break
    
    rwp = float(in_lines[rwp_index].strip().strip('`').split()[-1])
    phase_line = in_lines[phase_index].strip().replace('"','').split()
    a = float(in_lines[phase_index+1].strip().strip('`').split()[-1])
    b = float(in_lines[phase_index+2].strip().strip('`').split()[-1])
    c = float(in_lines[phase_index+3].strip().strip('`').split()[-1])
    alpha = float(in_lines[phase_index+4].strip().strip('`').split()[-1])
    beta = float(in_lines[phase_index+5].strip().strip('`').split()[-1])
    gamma = float(in_lines[phase_index+6].strip().strip('`').split()[-1])
    space_group_line = (in_lines[phase_index+7].strip().split()[-1].strip('"').split(':'))
    space_group = int(space_group_line[0])
    origin = int(space_group_line[1])

    refined_latparams = [space_group, a, b, c, alpha, beta, gamma, origin]

    return refined_latparams


def get_strainmode_latparams_from_topas_out(file):
    '''
    Parameters
    ----------
    file: str
        path to the topas out file
        
    Returns:
        list of lattice parameters: [spacegroup, a,b,c,alpha,beta,gamma, origin]

    Purpose:
        This function is used to retrieve lattice parameters refined by topas

    Important Note:
        In the str or hkl_Is section, you must have the following lines in order: 
        phase_name
        a
        b
        c
        al
        be
        ga
        space_group
    '''
    in_lines = []
    with open(file,'r') as f:
        in_lines = f.readlines()
        
    print(f'read lattice parameters from : {file}')
    
    rwp_index = 0
    phase_index = 0
    sg_index = 0
    for i in range(len(in_lines)):
        if "r_wp" in in_lines[i]:
            rwp_index = i
        if "unitless parent cell strains" in in_lines[i]:
            phase_index = i
        if "space_group " in in_lines[i]:
            sg_index = i
        if "load hkl_m_d_th2 I" in in_lines[i]:
            break
    
    rwp = float(in_lines[rwp_index].strip().strip('`').split()[-1])
    phase_line = in_lines[phase_index].strip().replace('"','').split()
    a = float(in_lines[phase_index+11].strip().strip('`').split()[-1])
    b = float(in_lines[phase_index+12].strip().strip('`').split()[-1])
    c = float(in_lines[phase_index+13].strip().strip('`').split()[-1])
    alpha = float(in_lines[phase_index+14].strip().strip('`').split()[-1])
    beta = float(in_lines[phase_index+15].strip().strip('`').split()[-1])
    gamma = float(in_lines[phase_index+16].strip().strip('`').split()[-1])
    space_group_line = (in_lines[sg_index].strip().split()[-1].strip('"').split(':'))
    space_group = int(space_group_line[0])
    origin = int(space_group_line[1])

    refined_latparams = [ a, b, c, alpha, beta, gamma]
    sg_o = [space_group, origin]

    return refined_latparams, sg_o


def get_rwp_latparams_from_primitive_refinements(file):
    '''
    Parameters
    ----------
    file: str
        path to the topas out file
        
    Returns:
        list of lattice parameters: [a,b,c,alpha,beta,gamma]
        space_group: int
        origin: int
        rwp: float (goodness of fit from TOPAS)

    Purpose:
        This function is used to retrieve lattice parameters refined by topas

    Important Note:
        In the str or hkl_Is section, you must have the following lines in order: 
        phase_name
        a
        b
        c
        al
        be
        ga
        space_group
    '''
    in_lines = []
    with open(file,'r') as f:
        in_lines = f.readlines()
        
    
    rwp_index = 0
    phase_index = 0
    for i in range(len(in_lines)):
        if "r_wp" in in_lines[i]:
            rwp_index = i
        if "phase_name" in in_lines[i]:
            phase_index = i
        if "load hkl_m_d_th2 I" in in_lines[i]:
            break
    
    rwp = float(in_lines[rwp_index].strip().strip('`').split()[-1])
    phase_line = in_lines[phase_index].strip().replace('"','').split()
    a = float(in_lines[phase_index+1].strip().split()[2].strip('`'))
    b = float(in_lines[phase_index+2].strip().split()[2].strip('`'))
    c = float(in_lines[phase_index+3].strip().split()[2].strip('`'))
    alpha = float(in_lines[phase_index+4].strip().split()[2].strip('`'))
    beta = float(in_lines[phase_index+5].strip().split()[2].strip('`'))
    gamma = float(in_lines[phase_index+6].strip().split()[2].strip('`'))
    space_group_line = (in_lines[phase_index+7].strip().split()[-1].strip('"').split(':'))
    space_group = int(space_group_line[0])
    origin = int(space_group_line[1])

    refined_latparams = [a, b, c, alpha, beta, gamma]

    return refined_latparams, space_group, origin, rwp


def latparams2vectormatrix(latparams):
    a = latparams[0]
    b = latparams[1]
    c = latparams[2]
    al = math.radians(latparams[3])
    be = math.radians(latparams[4])
    ga = math.radians(latparams[5])

    cell = np.zeros((3,3))

    cell[2][0] = a*math.cos(be)
    cell[0][1] = 0.0
    cell[1][1] = b*math.sin(al)
    cell[2][1] = b*math.cos(al)
    cell[0][2] = 0.0
    cell[1][2] = 0.0
    cell[2][2] = c
    cell[1][0] = (a*b*math.cos(ga) - cell[2][0]*cell[2][1])/cell[1][1]
    cell[0][0] = np.sqrt(a**2 - cell[1][0]**2 - cell[2][0]**2)

    # print(cell, '\n')
    return cell


def get_basis_transformation_matirx(cif_path):
    '''
    Parameters
    ----------
    cif_path: str
        Full path to the cif file generated by isodistort
        
    Returns:
        basis_cols : Basis Transformation vectors in numpy matrix form
        origin : Origin column vector in numpy matrix form

    '''
    with open(cif_path, 'r') as f:
        for line in f:
            # get the line with basis 
            if 'basis={' in line:
                bline = line.split(',', maxsplit=1)[1]
                bline = bline.strip()
                bline = bline.split('=')

                # remove all the text junk
                basis = bline[1].strip("'").strip('{(').strip(')}, origin').split('),(')
                basis = [basis[i].split(',') for i in range(len(basis))]

                # change string fractions to floats while preserving sign
                for i in range(len(basis)):
                    for x in range(len(basis[i])):
                        basis[i][x] = float(Fraction(basis[i][x]))

                # the basis as printed from isodistort : Ex: {(0,0,0),(0,0,0),(0,0,0)} is actually 3 column vectors, so transpose the list using numpy
                # We transpose since python does math row majored
                basis_cols = np.array(basis).T
                # print(basis_cols)

                # grab the origin and remove text junk
                origin = bline[2].strip('(').strip('), s').split(',')
                origin = [float(Fraction(origin[i])) for i in range(len(origin))]
                origin = np.array(origin).reshape(3,1)
                # print(origin)

                break

    return basis_cols, origin





# ------ perturb latparms ------

def perturb_latparams(latparams, spacegroup_origin, ran_array_lat, ran_array_angle):
    # takes three random numbers between 0.999 & 1.001 to multiply lattice parameters a, b, c 
    # produces a small purturbation of the lattice parameters
    spacegroup = spacegroup_origin[0]

    triclinic_range = np.arange(1,3).tolist()
    monoclinic_range = np.arange(3,16).tolist()
    orthorhombic_range = np.arange(16,75).tolist()
    tetragonal_range = np.arange(75,143).tolist()
    # in our subgroup tree from isodistort, every trigonal cif is in trigonal_hexagonal axes format (even if they rhombohedral space groups) - if needed for a different project, can turn these back on
    # trigonal_hex_range = [143, 144, 145, 147, 149, 150, 151, 152, 153, 154, 156, 157, 158, 159, 162, 163, 164, 165]
    # trigonal_rhomb_range = [146, 148, 155, 160, 161, 166, 167]
    trigonal_range = np.arange(143, 168).tolist()
    hexagonal_range = np.arange(168,195).tolist()
    cubic_range = np.arange(195,231).tolist()
    
    if spacegroup in triclinic_range:
        perturbed_a = latparams[0]+ran_array_lat[0]
        perturbed_b = latparams[1]+ran_array_lat[1]
        perturbed_c = latparams[2]+ran_array_lat[2]
        perturbed_alpha = latparams[3]+ran_array_angle[0]
        perturbed_beta =  latparams[4]+ran_array_angle[1]
        perturbed_gamma = latparams[5]+ran_array_angle[2]

    if spacegroup in monoclinic_range:
        perturbed_a = latparams[0]+ran_array_lat[0]
        perturbed_b = latparams[1]+ran_array_lat[1]
        perturbed_c = latparams[2]+ran_array_lat[2]
        perturbed_alpha = latparams[3]
        perturbed_beta = latparams[4]+ran_array_angle[1]
        perturbed_gamma = latparams[5]

    if spacegroup in orthorhombic_range:
        perturbed_a = latparams[0]+ran_array_lat[0]
        perturbed_b = latparams[1]+ran_array_lat[1]
        perturbed_c = latparams[2]+ran_array_lat[2]
        perturbed_alpha = latparams[3]
        perturbed_beta = latparams[4]
        perturbed_gamma = latparams[5]

    if spacegroup in tetragonal_range:
        perturbed_a = latparams[0]+ran_array_lat[0]
        perturbed_b = perturbed_a
        perturbed_c = latparams[2]+ran_array_lat[2]
        perturbed_alpha = latparams[3]
        perturbed_beta = latparams[4]
        perturbed_gamma = latparams[5]

    if spacegroup in trigonal_range:
        perturbed_a = latparams[0]+ran_array_lat[0]
        perturbed_b = perturbed_a
        perturbed_c = latparams[2]+ran_array_lat[2]
        perturbed_alpha = latparams[3]
        perturbed_beta = latparams[4]
        perturbed_gamma = latparams[5]

    if spacegroup in hexagonal_range:
        perturbed_a = latparams[0]+ran_array_lat[0]
        perturbed_b = perturbed_a
        perturbed_c = latparams[2]+ran_array_lat[2]
        perturbed_alpha = latparams[3]
        perturbed_beta = latparams[4]
        perturbed_gamma = latparams[5]

    if spacegroup in cubic_range:
        perturbed_a = latparams[0]+ran_array_lat[0]
        perturbed_b = perturbed_a
        perturbed_c = perturbed_a
        perturbed_alpha = latparams[3]
        perturbed_beta = latparams[4]
        perturbed_gamma = latparams[5]

    perturbed_params = [perturbed_a, perturbed_b, perturbed_c, perturbed_alpha, perturbed_beta, perturbed_gamma]
    return perturbed_params


def perturb_latparams_strainmodes(str_file, ran_array):
    lines = []

    with open(str_file, 'r') as f:
        lines = f.readlines()
    
    for i in range(len(lines)):
        sline = lines[i].strip()
        if '#ifdef refine_strain_modes' in sline:
            random_count = 0
            for j in range(i+1, len(lines)):
                split = lines[j].split()
                print(split)
                if len(split) == 0:
                    break
                split = lines[j].split('0.00000')
                print(split)
                newline = f'{split[0]}{ran_array[random_count]} {split[1]}'
                random_count += 1
                print(newline)
                lines[j] = newline
                print(lines[j])
            
            break
    
    return lines 





# ------ Generate modes lines ------
def get_mode_amplitudes(isodistort_str_file, use_vc_ma_macro=False):
    '''
    Parameters
    ----------
    isodistort_str_file: str
        Path to str (structure) file generated by isodistort
        
    Returns:
        List: Lines for mode amplitudes for writing into a topas input file
        Following returns are currently commented out, uncomment if needed
        int: Number of refineable parameters in the distorted_parameters section
        int: Total number of refineable parameters in the distorted parameters section
        - the two numbers can be used to create a weighting vector in the ML algorithm 

    Purpose:
        This function is used to modify the mode amplitudes in the topas str file and return them
        in a form for writing to a topas input file
    '''
    lines = []
    with open(isodistort_str_file, 'r') as f:
        lines = f.readlines()

    mode_definition_str = '\'{{{mode definitions\n'
    mode_amplitude_str = '\'{{{mode-amplitude to delta transformation\n'
    distorted_parameters_str = '\'{{{distorted parameters\n'
    end_index_str = '\'}}}\n'
    start_index = lines.index(mode_definition_str)
    end_index = lines.index(end_index_str)
    modedef_start = start_index+1
    modedef_end = end_index-1

    modedef_lines = lines[modedef_start:modedef_end]
    rest_of_lines = lines[end_index-1:]

    modedef_lines = [line.split() for line in modedef_lines]
    modedef_df = pd.DataFrame(modedef_lines, columns=None)

    # generate random intial values for mode amplitudes
    rand_values = np.random.normal(0, .33, len(modedef_df))
    modedef_df[2] = rand_values

    # augment mode amplitude lines to include vc macro
    vc_column = ['vc min']*len(modedef_df)
    if use_vc_ma_macro == True:
        modedef_df[3] = vc_column

    modedef_string = modedef_df.to_string(header=False, index=False)
    modedef_string_split = modedef_string.split(sep='\n')
    modedef_string_split = [f'\t\t{modedef_string_split[i]}\n' for i in range(len(modedef_string_split))]


    updated_mode_lines = [mode_definition_str] + modedef_string_split + rest_of_lines

    # with open('./TOPAS/debug.txt', 'w') as f:
    #     for i in range(len(updated_mode_lines)):
    #         f.write(updated_mode_lines[i])

    # lines below this point are for calculating the total number of distorted parameters lines
    # and counting how many of them are allowed to refine by default (ie if variable name is not preceded by '!')
    # Will be used for weighting results of ML program if necessary
    '''
    start_index_mode_amplitude = lines.index(mode_amplitude_str)
    end_index_mode_amplitude = lines.index(end_index_str, end_index+1)

    start_index_distorted_parameters = lines.index(distorted_parameters_str)
    sub_lines = lines[start_index_distorted_parameters:]
    regex = '.._occ'
    temp_regex = re.compile(regex)
    search_result = [temp_regex.search(line_to_search) for line_to_search in sub_lines]
    not_none = [i for i in range(len(search_result)) if search_result[i] !=None]
    end_index_distorted_parameters = start_index_distorted_parameters + not_none[0]

    distorted_parameters_lines = lines[start_index_distorted_parameters+1:end_index_distorted_parameters-1]
    distorted_parameters_lines = [line.split() for line in distorted_parameters_lines]
    distorted_df = pd.DataFrame(distorted_parameters_lines, columns=None)
    unrefined_bool = distorted_df[1].str.startswith('!')
    count_unrefined = unrefined_bool.sum()
    numberof_refineable_parameters = len(distorted_df) - count_unrefined
    total_number_of_parameters = len(distorted_df)
    '''

    return updated_mode_lines


def generate_symmetry_mode_lines_from_topas_str(isodistort_str_file, refine_beq = False, beq_dict = None, use_vc_ma_macro=False ):
    '''
    Parameters
    ----------
    isodistort_str_file: str
        Path to str (structure) file generated by isodistort
    refine_beq: bool
        if True: 'beq' in mode_dependent_sites section will change to 'beq bval_{atom_type}'
        if False: 'beq' in mode_dependent_sites section will change to 'beq !bval_{atom_type}'  -> '!' tells topas to not refine this parameter
    beq_dict: dict
        USE:
        {'atom_type': [0, 2.5]} -> will use np.random.uniform to pull a random number from the range [0.0, 2.5]
        {'atom_type': [3.2]}    -> will set the beq value to 3.2 for that atom type
        Note: atom_type can be found in the topas_str file under mode_dependent_sites section
        EX: site Bi1_1 ....  --> atom_type = Bi1
    use_vc_ma_macro: Bool
        Whether to apply vc_ma macro for continue on convergence to augment the mode_amplitude value after every convergence
        Also removes '!' before every prm aX in mode definitions to allow them to refine
    Returns:
        List of strings: for symmetry mode analysis for writing into a topas input file

    Purpose:
        This function is used to modify the mode amplitudes and mode_dependent_sites sections in the topas str file and return them
        in a form for writing to a topas input file
    '''

    file_lines = []
    with open(isodistort_str_file, 'r') as f:
        file_lines = f.readlines()

    mode_definition_str = '\'{{{mode definitions'
    mode_amplitude_str = '\'{{{mode-amplitude to delta transformation'
    distorted_parameters_str = '\'{{{distorted parameters'
    mode_dependent_sites_str = '\'{{{mode-dependent sites'
    difference_restrainsts_str = '\'{{{difference restraints for interconnected rigid bodies'
    end_str = '\'}}}'

    mode_definitions = []
    other_mode_definitions = []
    mode_amplitudes = []
    distorted_parameters = []
    mode_dependent_sites = []
    difference_restrainsts = []

    # parse topas str file into sections
    for (i, line) in enumerate(file_lines):
        if mode_definition_str in line:
            for append_lines in file_lines[(i+1):]: # use i+1 as start of slice so we don't include title string in appended list
                if end_str in append_lines: # break condition comes before appending lines so we don't append end_str to appended list 
                    break
                if len(append_lines.split()) < 2:
                    continue
                #* quick change here, to turn off occupancy modes for peter kalifah's data, to go back to normal, comment out if/else and uncomment append line 
                # mode_definitions.append(append_lines)
                if '!a' in append_lines:
                    mode_definitions.append(append_lines)
                else:
                    other_mode_definitions.append(append_lines)


        if mode_amplitude_str in line:
            for append_lines in file_lines[(i+1):]:
                if end_str in append_lines:
                    break
                if len(append_lines.split()) < 2:
                    continue
                mode_amplitudes.append(append_lines)

        if distorted_parameters_str in line:
            for append_lines in file_lines[(i+1):]:
                if end_str in append_lines:
                    break
                if len(append_lines.split()) < 2:
                    continue
                distorted_parameters.append(append_lines)

        if mode_dependent_sites_str in line:
            for append_lines in file_lines[(i+1):]:
                if end_str in append_lines:
                    break
                if len(append_lines.split()) < 2:
                    continue
                mode_dependent_sites.append(append_lines)

        if difference_restrainsts_str in line:
            for append_lines in file_lines[(i+1):]:
                if end_str in append_lines:
                    break
                if len(append_lines.split()) < 2:
                    continue
                difference_restrainsts.append(append_lines)

    # ----------------------------------------------------------------------------------------
    #* Augment mode amplitudes lines
    # ----------------------------------------------------------------------------------------

    # Transform mode_definitions_lines into dataframe to augment ititial values for mode amplitudes
    mode_definitions_lines = [line.split() for line in mode_definitions]

    # if statement catches special case where there are no mode definitions, prevents writing empty dataframe, which can cause issues 
    if len(mode_definitions_lines) != 0:
        mode_definitions_df = pd.DataFrame(mode_definitions_lines, columns=None)

        # generate random intial values for mode amplitudes
        rand_values = np.random.normal(0, .33, len(mode_definitions_df))
        mode_definitions_df[2] = rand_values

        # augment each line of mode amplitudes to include vc macro
        if use_vc_ma_macro == True:
            vc_column = ['vc min']*len(mode_definitions_df)
            mode_definitions_df[3] = vc_column

            # remove the ! from variable names in mode definitions ('!' tells topas to not refine this parameter)
            mode_var_names = mode_definitions_df[1].tolist()
            mode_var_names = [mode_var_names[i].strip('!') for i in range(len(mode_definitions_df))]
            mode_definitions_df[1] = mode_var_names


        # Transform mode amplitudes dataframe back to strings for writing
        mode_definitions_writing = mode_definitions_df.to_string(header=False, index=False)
        mode_definitions_writing_split = mode_definitions_writing.split(sep='\n')
        mode_definitions_writing = [f'\t\t{mode_definitions_writing_split[i]}\n' for i in range(len(mode_definitions_writing_split))]
        # in case other modes that are not wanted to be randomized are in topas str, will write them here
        if len(other_mode_definitions) != 0:
            mode_definitions_writing.append('\n')
            for x in range(len(other_mode_definitions)): 
                mode_definitions_writing.append(f'{other_mode_definitions[x]}')

    else:
        mode_definitions_writing = ['\n']

    
    # ----------------------------------------------------------------------------------------
    #* Augment mode dependent sites lines to have beq values
    # ----------------------------------------------------------------------------------------

    # Split mode_dependent_sites at spaces to modify certain values
    mode_dependent_sites_lines = [line.split() for line in mode_dependent_sites]

    # get atom types from topas str file
    all_atom_types_list = [mode_dependent_sites_lines[i][1].split('_')[0] for i in range(len(mode_dependent_sites_lines))]
    all_atom_types_list_unique = np.unique(all_atom_types_list).tolist()

    # generate a beq_value_dict from ranges in beq_matrix for each atom type
    beq_value_dict = {}
    if beq_dict !=None:
        for i in range(len(all_atom_types_list_unique)):
            if all_atom_types_list_unique[i] == 'origin':
                continue
            beq_range = beq_dict.get(all_atom_types_list_unique[i])
            
            if len(beq_range) == 2:
                beq_value = np.random.uniform(beq_range[0], beq_range[1])
            elif len(beq_range) == 1:
                beq_value = beq_range[0]
            else: print('beq_matrix is configured incorrectly, please fix and try again')
            beq_value_dict[all_atom_types_list_unique[i]] = beq_value 

    # use mode dependent sites lines to find atom types to apply beq
    for i in range(len(mode_dependent_sites_lines)):
        # if empty line OR 'origin' line in this section, continue past it
        if len(mode_dependent_sites_lines[i]) < 14:
            continue

        # get the atom type
        atom_type = mode_dependent_sites_lines[i][1].split('_')[0]
        if refine_beq == True:
            beq_var_name = f'beq bval_{atom_type}'
        else: 
            beq_var_name = f'beq !bval_{atom_type}'

        mode_dependent_sites_lines[i][17] = beq_var_name

        # apply beq value from beq_value_dict
        mode_dependent_sites_lines[i][18] = str(beq_value_dict.get(atom_type))
    

    # join list of strings back together
    mode_dependent_sites_lines = [' '.join(mode_dependent_sites_lines[i]) for i in range(len(mode_dependent_sites_lines))]
    mode_dependent_sites = [f'\t\t{mode_dependent_sites_lines[i]}\n' for i in range(len(mode_dependent_sites_lines))]
    # ----------------------------------------------------------------------------------------
    # Join all lines together back into list to be written into topas input file
    
    all_mode_lines = []
    # append mode definitions
    all_mode_lines.append(f'{mode_definition_str}\n')
    for i in range(len(mode_definitions_writing)):
        all_mode_lines.append(mode_definitions_writing[i])
    all_mode_lines.append(f'{end_str}\n\n')    

    # append mode amplitudes
    all_mode_lines.append(f'{mode_amplitude_str}\n')
    for i in range(len(mode_amplitudes)):
        all_mode_lines.append(mode_amplitudes[i])
    all_mode_lines.append(f'{end_str}\n\n')    

    # append distorted parameters
    all_mode_lines.append(f'{distorted_parameters_str}\n')
    for i in range(len(distorted_parameters)):
        all_mode_lines.append(distorted_parameters[i])
    all_mode_lines.append(f'{end_str}\n\n')   

    # append mode_dependent_sites
    all_mode_lines.append(f'{mode_dependent_sites_str}\n')
    for i in range(len(mode_dependent_sites)):
        all_mode_lines.append(mode_dependent_sites[i])
    all_mode_lines.append(f'{end_str}\n\n')   

    # append difference restraints
    all_mode_lines.append(f'{difference_restrainsts_str}\n')
    for i in range(len(difference_restrainsts)):
        all_mode_lines.append(difference_restrainsts[i])
    all_mode_lines.append(f'{end_str}\n\n')   


    return all_mode_lines



def generate_strain_and_symmetry_mode_lines_from_topas_str(isodistort_str_file, spacegroup_origin, peak_shape_line, random_strain, custom_latparams = None, refine_beq = False, beq_dict = None, use_vc_ma_macro=False, multiphase=False, wt_dets_macro='None', phase_count=None):
    '''
    Parameters
    ----------
    isodistort_str_file: str
        Path to str (structure) file generated by isodistort
    spacegroup_origin: list of ints
        the spacegroup and origin choice for the structure
    peak_shape_line: str
        Line to describe the peakshape in topas
    random_strain: np array 
        shape=[6], samples of np.random.uniform between min and max values for strain modes 
    custom_latparams: list of floats
        [a,b,c,alpha,beta,gamma] : use if wanting to supply custom lattice parameters other than what is found in the isodistort str file
    refine_beq: bool
        if True: 'beq' in mode_dependent_sites section will change to 'beq bval_{atom_type}'
        if False: 'beq' in mode_dependent_sites section will change to 'beq !bval_{atom_type}'  -> '!' tells topas to not refine this parameter
    beq_dict: dict
        USE:
        {'atom_type': [0, 2.5]} -> will use np.random.uniform to pull a random number from the range [0.0, 2.5)
        {'atom_type': [3.2]}    -> will set the beq value to 3.2 for that atom type
        Note: atom_type can be found in the topas_str file under mode_dependent_sites section
        EX: site Bi1_1 ....  --> atom_type = Bi1
    use_vc_ma_macro: Bool
        Whether to apply vc_ma macro for continue on convergence to augment the mode_amplitude value after every convergence
        Also removes '!' before every prm aX in mode definitions to allow them to refine
    mulitphase: Bool - Default: False
        if True: Changes prm to local to avoid variables cross contaminating, changes scaling behavior based on number of phases
    wt_dets_macro: str - Default: str(None)
        Line to control the wt% of each phase 
        Ex: for the first of 3 phases :: wt_dets(1,3,0) where the 0 in the third position just reports the wt% after TOPAS runs
    phase_count: int - Default: None
        an integer to append to beq variable names for multiphase inputs where atoms types are the same

    
    Returns:
    ----------
        List of strings: for structure perturbation using strain modes and symmetry mode analysis for writing into a topas input file

    Purpose:
    ----------
        This function is used to create perturbations of
        1) lattice parameters through strain mode amplitudes or custom lattice parameters, 
        2) atomic positions through mode amplitudes and mode_dependent_sites sections,
        3) thermal broadening of peaks in diffraction pattern through beq (topas isotropic thermal parameters)

    '''

    file_lines = []
    with open(isodistort_str_file, 'r') as f:
        file_lines = f.readlines()

    start_str = '\'Topas .str file generated by '
    mode_definition_str = '\'{{{mode definitions'
    mode_amplitude_str = '\'{{{mode-amplitude to delta transformation'
    distorted_parameters_str = '\'{{{distorted parameters'
    mode_dependent_sites_str = '\'{{{mode-dependent sites'
    difference_restrainsts_str = '\'{{{difference restraints for interconnected rigid bodies'
    macros_str = '\'{{{macros for calculations involving the lattice parameters'
    end_str = '\'}}}'

    str_strain = []
    mode_definitions = []
    mode_amplitudes = []
    distorted_parameters = []
    mode_dependent_sites = []
    difference_restrainsts = []
    strain_macros = []

    # ----------------------------------------------------------------------------------------
    #* parse topas str file into sections
    # ----------------------------------------------------------------------------------------

    for (i, line) in enumerate(file_lines):

        if start_str in line:
            for append_lines in file_lines:
                if 'scale' in append_lines:
                    if multiphase == True:
                        str_strain.append(f'\n')
                    else:
                        str_strain.append(f'\t\tscale @ 100.0\n')
                    break
                str_strain.append(append_lines)

        if mode_definition_str in line:
            for append_lines in file_lines[(i+1):]: # use i+1 as start of slice so we don't include title string in appended list
                if end_str in append_lines: # break condition comes before appending lines so we don't append end_str to appended list 
                    break
                if len(append_lines.split()) < 2:
                    continue
                mode_definitions.append(append_lines)

        if mode_amplitude_str in line:
            for append_lines in file_lines[(i+1):]:
                if end_str in append_lines:
                    break
                if len(append_lines.split()) < 2:
                    continue
                mode_amplitudes.append(append_lines)

        if distorted_parameters_str in line:
            for append_lines in file_lines[(i+1):]:
                if end_str in append_lines:
                    break
                if len(append_lines.split()) < 2:
                    continue
                distorted_parameters.append(append_lines)

        if mode_dependent_sites_str in line:
            for append_lines in file_lines[(i+1):]:
                if end_str in append_lines:
                    break
                if len(append_lines.split()) < 2:
                    continue
                mode_dependent_sites.append(append_lines)

        if difference_restrainsts_str in line:
            for append_lines in file_lines[(i+1):]:
                if end_str in append_lines:
                    break
                if len(append_lines.split()) < 2:
                    continue
                difference_restrainsts.append(append_lines)

        # if macros_str in line:
        #     for append_lines in file_lines[i:]:
        #         strain_macros.append(append_lines)
        #         if end_str in append_lines:
        #             break


    # ----------------------------------------------------------------------------------------
    #* Augment strain mode lines OR supply custom lattice parameters
    # ----------------------------------------------------------------------------------------
    # insert peak shape line in str_strain
    # str_strain.insert(4, f'\t\t{peak_shape_line}\n')
    str_strain.insert(4, f'\t\tpeak_type pv\n\t\tTOF_CS_L (@ , 581.69638, difc_1)\n')
    str_strain.insert(5, f'\t\tr_bragg 0.0\n')
    

    # section for using custom latparams 
    if custom_latparams != None:
        if multiphase == True:
            str_strain.insert(6, f'{wt_dets_macro}\n')
        for i in range(len(str_strain)):
            if 'space_group' in str_strain[i]:
                str_strain[i] = f'\t\tspace_group {spacegroup_origin[0]}:{spacegroup_origin[1]}\n'
            # comment out defining strain modes to use classic latparams method
            if '#define refine_strain_modes' in str_strain[i]:
                str_strain[i] =  "\t\t'#define refine_strain_modes 'comment out this line for a traditional supercell-parameter refinement."

            if '#else' in str_strain[i]:
                j = i+1
                latcount = 0
                for j in range(j, len(str_strain)):
                    if '#endif' in str_strain[j]:
                        break
                    split = str_strain[j].split()
                    str_strain[j] = f'\t\t\t{split[0]}    {custom_latparams[latcount]:.6f}\n' 
                    latcount +=1
    else:
        # section for using strain modes instead of classic latparams method 
        if multiphase==False:
            for i in range(len(str_strain)):
                if 'space_group' in str_strain[i]:
                    str_strain[i] = f'\t\tspace_group {spacegroup_origin[0]}:{spacegroup_origin[1]}\n'

                if '#ifdef refine_strain_modes' in str_strain[i]:
                    random_count = 0
                    for j in range(i+1, len(str_strain)):
                        split = str_strain[j].split()
                        # print(split)
                        if len(split) == 0:
                            break
                        split = str_strain[j].split('0.00000')
                        # print(split)
                        newline = f'{split[0]}{random_strain[random_count]} {split[1]}'
                        random_count += 1
                        # print(newline)
                        str_strain[j] = newline
                        # print(str_strain[j])
                    
                    break
        
        if multiphase == True:
            str_strain.insert(6, f'{wt_dets_macro}\n')
            for i in range(len(str_strain)):
                if 'space_group' in str_strain[i]:
                    str_strain[i] = f'\t\tspace_group {spacegroup_origin[0]}:{spacegroup_origin[1]}\n'

                if '#ifdef refine_strain_modes' in str_strain[i]:
                    random_count = 0
                    for j in range(i+1, len(str_strain)):
                        split = str_strain[j].split()
                        # print(split)
                        if len(split) == 0:
                            break
                        split = str_strain[j].split('0.00000')
                        # print(split)
                        local_split = split[0].replace('prm', 'local')
                        newline = f'{local_split}{random_strain[0][random_count]} {split[1]}'
                        random_count += 1
                        # print(newline)
                        str_strain[j] = newline
                        # print(str_strain[j])
                
                if 'unitless parent cell strains' in str_strain[i]:
                    for j in range(i+1, len(str_strain)):
                        if len(str_strain[j].split()) == 0:
                            break
                        newline = str_strain[j].replace('prm', 'local')
                        str_strain[j] = newline

                if 'straincell(' in str_strain[i]:
                    newline = str_strain[i].replace('straincell', 'straincell_local')
                    str_strain[i] = newline
                
                if 'celltransform(' in str_strain[i]:
                    newline = str_strain[i].replace('celltransform', 'celltransform_local')
                    str_strain[i] = newline


    # ----------------------------------------------------------------------------------------
    #* Augment mode amplitudes lines
    # ----------------------------------------------------------------------------------------

    # Transform mode_definitions_lines into dataframe to augment ititial values for mode amplitudes
    mode_definitions_lines = [line.split() for line in mode_definitions]
    mode_definitions_writing=[]
    # special case where there are no mode definitions, writes empty dataframe, if statement checking if len is 0 prevents this issue
    if len(mode_definitions_lines) != 0:
        mode_definitions_df = pd.DataFrame(mode_definitions_lines, columns=None)

        # generate random intial values for mode amplitudes
        rand_values = np.random.normal(0, .33, len(mode_definitions_df))
        mode_definitions_df[2] = rand_values

        # augment each line of mode amplitudes to include vc macro
        if use_vc_ma_macro == True:
            vc_column = ['vc min']*len(mode_definitions_df)
            mode_definitions_df[3] = vc_column

            # remove the ! from variable names in mode definitions ('!' tells topas to not refine this parameter)
            mode_var_names = mode_definitions_df[1].tolist()
            mode_var_names = [mode_var_names[i].strip('!') for i in range(len(mode_definitions_df))]
            mode_definitions_df[1] = mode_var_names


        # Transform mode amplitudes dataframe back to strings for writing
        mode_definitions_writing = mode_definitions_df.to_string(header=False, index=False)
        mode_definitions_writing_split = mode_definitions_writing.split(sep='\n')

        if multiphase == True:
            for i in range(len(mode_definitions_writing_split)):
                mode_definitions_writing_split[i] = mode_definitions_writing_split[i].replace('prm', 'local')

        mode_definitions_writing = [f'\t\t{mode_definitions_writing_split[i]}\n' for i in range(len(mode_definitions_writing_split))]

    # ----------------------------------------------------------------------------------------
    #* If multiphase - change prm to local in mode-amplitude-to-delta-transformation & distorted parameters sections
    # ----------------------------------------------------------------------------------------
    if multiphase == True:
        for i in range(len(mode_amplitudes)):
            newline = mode_amplitudes[i].replace('prm', 'local')
            mode_amplitudes[i] = newline

        for i in range(len(distorted_parameters)):
            newline = distorted_parameters[i].replace('prm', 'local')
            distorted_parameters[i] = newline

    # ----------------------------------------------------------------------------------------
    #* Augment mode dependent sites lines to have beq values
    # ----------------------------------------------------------------------------------------

    # Split mode_dependent_sites at spaces to modify certain values
    mode_dependent_sites_lines = [line.split() for line in mode_dependent_sites]

    # get atom types from topas str file
    all_atom_types_list = [mode_dependent_sites_lines[i][1].split('_')[0] for i in range(len(mode_dependent_sites_lines))]
    all_atom_types_list_unique = np.unique(all_atom_types_list).tolist()

    # generate a beq_value_dict from ranges in beq_matrix for each atom type
    beq_value_dict = {}
    if beq_dict !=None:
        for i in range(len(all_atom_types_list_unique)):
            if all_atom_types_list_unique[i] == 'origin':
                continue
            beq_range = beq_dict.get(all_atom_types_list_unique[i])
            
            if len(beq_range) == 2:
                beq_value = np.random.uniform(beq_range[0], beq_range[1])
            elif len(beq_range) == 1:
                beq_value = beq_range[0]
            else: print('beq_matrix is configured incorrectly, please fix and try again')
            beq_value_dict[all_atom_types_list_unique[i]] = beq_value 

    # use mode dependent sites lines to find atom types to apply beq
    for i in range(len(mode_dependent_sites_lines)):
        # if empty line OR 'origin' line in this section, continue past it
        if len(mode_dependent_sites_lines[i]) < 14:
            continue

        # get the atom type
        atom_type = mode_dependent_sites_lines[i][1].split('_')[0]
        if refine_beq == True:
            beq_var_name = f'beq bval_{atom_type}'
            if phase_count != None:
                beq_var_name = f'beq bval_{atom_type}_{phase_count}'
        else: 
            beq_var_name = f'beq !bval_{atom_type}'
            if phase_count != None:
                beq_var_name = f'beq !bval_{atom_type}_{phase_count}'

        mode_dependent_sites_lines[i][17] = beq_var_name

        # apply beq value from beq_value_dict
        mode_dependent_sites_lines[i][18] = str(beq_value_dict.get(atom_type))
    

    # join list of strings back together
    mode_dependent_sites_lines = [' '.join(mode_dependent_sites_lines[i]) for i in range(len(mode_dependent_sites_lines))]
    mode_dependent_sites = [f'\t\t{mode_dependent_sites_lines[i]}\n' for i in range(len(mode_dependent_sites_lines))]

    # ----------------------------------------------------------------------------------------
    #* Join all lines together back into list to be written into topas input file
    # ----------------------------------------------------------------------------------------

    all_mode_lines = []
    # append strain mode and str (structure definition) lines 
    for i in range(len(str_strain)):
        all_mode_lines.append(str_strain[i])
    all_mode_lines.append('\n')

    # append mode definitions
    all_mode_lines.append(f'{mode_definition_str}\n')
    for i in range(len(mode_definitions_writing)):
        all_mode_lines.append(mode_definitions_writing[i])
    all_mode_lines.append(f'{end_str}\n\n')    

    # append mode amplitudes
    all_mode_lines.append(f'{mode_amplitude_str}\n')
    for i in range(len(mode_amplitudes)):
        all_mode_lines.append(mode_amplitudes[i])
    all_mode_lines.append(f'{end_str}\n\n')    

    # append distorted parameters
    all_mode_lines.append(f'{distorted_parameters_str}\n')
    for i in range(len(distorted_parameters)):
        all_mode_lines.append(distorted_parameters[i])
    all_mode_lines.append(f'{end_str}\n\n')   

    # append mode_dependent_sites
    all_mode_lines.append(f'{mode_dependent_sites_str}\n')
    for i in range(len(mode_dependent_sites)):
        all_mode_lines.append(mode_dependent_sites[i])
    all_mode_lines.append(f'{end_str}\n\n')   

    # append difference restraints
    all_mode_lines.append(f'{difference_restrainsts_str}\n')
    for i in range(len(difference_restrainsts)):
        all_mode_lines.append(difference_restrainsts[i])
    all_mode_lines.append(f'{end_str}\n\n')   

    # # append macros
    # for i in range(len(strain_macros)):
    #     all_mode_lines.append(strain_macros[i])
    # all_mode_lines.append('\n\n')


    return all_mode_lines


def generate_strain_and_symmetry_mode_lines_from_topas_str_v2(isodistort_str_file, spacegroup_origin, random_strain, custom_latparams = None, refine_beq = False, beq_dict = None, use_vc_ma_macro=False, multiphase=False, wt_dets_macro='None', phase_count=None):
    '''
    Parameters
    ----------
    isodistort_str_file: str
        Path to str (structure) file generated by isodistort
    spacegroup_origin: list of ints
        the spacegroup and origin choice for the structure
    random_strain: np array 
        shape=[6], samples of np.random.uniform between min and max values for strain modes 
    custom_latparams: list of floats
        [a,b,c,alpha,beta,gamma] : use if wanting to supply custom lattice parameters other than what is found in the isodistort str file
    refine_beq: bool
        if True: 'beq' in mode_dependent_sites section will change to 'beq bval_{atom_type}'
        if False: 'beq' in mode_dependent_sites section will change to 'beq !bval_{atom_type}'  -> '!' tells topas to not refine this parameter
    beq_dict: dict
        USE:
        {'atom_type': [0, 2.5]} -> will use np.random.uniform to pull a random number from the range [0.0, 2.5)
        {'atom_type': [3.2]}    -> will set the beq value to 3.2 for that atom type
        Note: atom_type can be found in the topas_str file under mode_dependent_sites section
        EX: site Bi1_1 ....  --> atom_type = Bi1
    use_vc_ma_macro: Bool
        Whether to apply vc_ma macro for continue on convergence to augment the mode_amplitude value after every convergence
        Also removes '!' before every prm aX in mode definitions to allow them to refine
    mulitphase: Bool - Default: False
        if True: Changes prm to local to avoid variables cross contaminating, changes scaling behavior based on number of phases
    wt_dets_macro: str - Default: str(None)
        Line to control the wt% of each phase 
        Ex: for the first of 3 phases :: wt_dets(1,3,0) where the 0 in the third position just reports the wt% after TOPAS runs
    phase_count: int - Default: None
        an integer to append to beq variable names for multiphase inputs where atoms types are the same

    
    Returns:
    ----------
        List of strings: for structure perturbation using strain modes and symmetry mode analysis for writing into a topas input file

    Purpose:
    ----------
        This function is used to create perturbations of
        1) lattice parameters through strain mode amplitudes or custom lattice parameters, 
        2) atomic positions through mode amplitudes and mode_dependent_sites sections,
        3) thermal broadening of peaks in diffraction pattern through beq (topas isotropic thermal parameters)

    '''

    file_lines = []
    with open(isodistort_str_file, 'r') as f:
        file_lines = f.readlines()

    start_str = '\'Topas .str file generated by '
    mode_definition_str = '\'{{{mode definitions'
    mode_amplitude_str = '\'{{{mode-amplitude to delta transformation'
    distorted_parameters_str = '\'{{{distorted parameters'
    mode_dependent_sites_str = '\'{{{mode-dependent sites'
    difference_restrainsts_str = '\'{{{difference restraints for interconnected rigid bodies'
    macros_str = '\'{{{macros for calculations involving the lattice parameters'
    end_str = '\'}}}'

    str_strain = []
    mode_definitions = []
    mode_amplitudes = []
    distorted_parameters = []
    mode_dependent_sites = []
    difference_restrainsts = []
    strain_macros = []

    # ----------------------------------------------------------------------------------------
    #* parse topas str file into sections
    # ----------------------------------------------------------------------------------------

    for (i, line) in enumerate(file_lines):

        if start_str in line:
            for append_lines in file_lines:
                if 'scale' in append_lines:
                    str_strain.append(f'\n')
                    # if multiphase == True:
                    #     str_strain.append(f'\n')
                    # else:
                    #     str_strain.append(f'\t\tscale @ 100.0\n')
                    break
                str_strain.append(append_lines)

        if mode_definition_str in line:
            for append_lines in file_lines[(i+1):]: # use i+1 as start of slice so we don't include title string in appended list
                if end_str in append_lines: # break condition comes before appending lines so we don't append end_str to appended list 
                    break
                if len(append_lines.split()) < 2:
                    continue
                mode_definitions.append(append_lines)

        if mode_amplitude_str in line:
            for append_lines in file_lines[(i+1):]:
                if end_str in append_lines:
                    break
                if len(append_lines.split()) < 2:
                    continue
                mode_amplitudes.append(append_lines)

        if distorted_parameters_str in line:
            for append_lines in file_lines[(i+1):]:
                if end_str in append_lines:
                    break
                if len(append_lines.split()) < 2:
                    continue
                distorted_parameters.append(append_lines)

        if mode_dependent_sites_str in line:
            for append_lines in file_lines[(i+1):]:
                if end_str in append_lines:
                    break
                if len(append_lines.split()) < 2:
                    continue
                mode_dependent_sites.append(append_lines)

        if difference_restrainsts_str in line:
            for append_lines in file_lines[(i+1):]:
                if end_str in append_lines:
                    break
                if len(append_lines.split()) < 2:
                    continue
                difference_restrainsts.append(append_lines)

        # if macros_str in line:
        #     for append_lines in file_lines[i:]:
        #         strain_macros.append(append_lines)
        #         if end_str in append_lines:
        #             break


    # ----------------------------------------------------------------------------------------
    #* Augment strain mode lines OR supply custom lattice parameters
    # ----------------------------------------------------------------------------------------
    # insert peak shape line in str_strain
    # str_strain.insert(4, f'\t\t{peak_shape_line}\n')
    # str_strain.insert(4, f'\t\tpeak_type pv\n\t\tTOF_CS_L (@ , 581.69638, difc_1)\n')
    # str_strain.insert(5, f'\t\tr_bragg 0.0\n')
    

    # section for using custom latparams 
    if custom_latparams != None:
        if multiphase == True:
            str_strain.insert(6, f'{wt_dets_macro}\n')
        for i in range(len(str_strain)):
            if 'space_group' in str_strain[i]:
                str_strain[i] = f'\t\tspace_group {spacegroup_origin[0]}:{spacegroup_origin[1]}\n'
            # comment out defining strain modes to use classic latparams method
            if '#define refine_strain_modes' in str_strain[i]:
                str_strain[i] =  "\t\t'#define refine_strain_modes 'comment out this line for a traditional supercell-parameter refinement."

            if '#else' in str_strain[i]:
                j = i+1
                latcount = 0
                for j in range(j, len(str_strain)):
                    if '#endif' in str_strain[j]:
                        break
                    split = str_strain[j].split()
                    str_strain[j] = f'\t\t\t{split[0]}    {custom_latparams[latcount]:.6f}\n' 
                    latcount +=1
    else:
        # section for using strain modes instead of classic latparams method 
        if multiphase==False:
            for i in range(len(str_strain)):
                if 'space_group' in str_strain[i]:
                    str_strain[i] = f'\t\tspace_group {spacegroup_origin[0]}:{spacegroup_origin[1]}\n'

                if '#ifdef refine_strain_modes' in str_strain[i]:
                    random_count = 0
                    for j in range(i+1, len(str_strain)):
                        split = str_strain[j].split()
                        # print(split)
                        if len(split) == 0:
                            break
                        split = str_strain[j].split('0.00000')
                        # print(split)
                        newline = f'{split[0]}{random_strain[random_count]} {split[1]}'
                        random_count += 1
                        # print(newline)
                        str_strain[j] = newline
                        # print(str_strain[j])
                    
                    break
        
        if multiphase == True:
            str_strain.insert(6, f'{wt_dets_macro}\n')
            for i in range(len(str_strain)):
                if 'space_group' in str_strain[i]:
                    str_strain[i] = f'\t\tspace_group {spacegroup_origin[0]}:{spacegroup_origin[1]}\n'

                if '#ifdef refine_strain_modes' in str_strain[i]:
                    random_count = 0
                    for j in range(i+1, len(str_strain)):
                        split = str_strain[j].split()
                        # print(split)
                        if len(split) == 0:
                            break
                        split = str_strain[j].split('0.00000')
                        # print(split)
                        local_split = split[0].replace('prm', 'local')
                        newline = f'{local_split}{random_strain[0][random_count]} {split[1]}'
                        random_count += 1
                        # print(newline)
                        str_strain[j] = newline
                        # print(str_strain[j])
                
                if 'unitless parent cell strains' in str_strain[i]:
                    for j in range(i+1, len(str_strain)):
                        if len(str_strain[j].split()) == 0:
                            break
                        newline = str_strain[j].replace('prm', 'local')
                        str_strain[j] = newline

                if 'straincell(' in str_strain[i]:
                    newline = str_strain[i].replace('straincell', 'straincell_local')
                    str_strain[i] = newline
                
                if 'celltransform(' in str_strain[i]:
                    newline = str_strain[i].replace('celltransform', 'celltransform_local')
                    str_strain[i] = newline


    # ----------------------------------------------------------------------------------------
    #* Augment mode amplitudes lines
    # ----------------------------------------------------------------------------------------

    # Transform mode_definitions_lines into dataframe to augment ititial values for mode amplitudes
    mode_definitions_lines = [line.split() for line in mode_definitions]
    mode_definitions_writing=[]
    # special case where there are no mode definitions, writes empty dataframe, if statement checking if len is 0 prevents this issue
    if len(mode_definitions_lines) != 0:
        mode_definitions_df = pd.DataFrame(mode_definitions_lines, columns=None)

        # generate random intial values for mode amplitudes
        rand_values = np.random.normal(0, .33, len(mode_definitions_df))
        mode_definitions_df[2] = rand_values

        # augment each line of mode amplitudes to include vc macro
        if use_vc_ma_macro == True:
            vc_column = ['vc min']*len(mode_definitions_df)
            mode_definitions_df[3] = vc_column

            # remove the ! from variable names in mode definitions ('!' tells topas to not refine this parameter)
            mode_var_names = mode_definitions_df[1].tolist()
            mode_var_names = [mode_var_names[i].strip('!') for i in range(len(mode_definitions_df))]
            mode_definitions_df[1] = mode_var_names


        # Transform mode amplitudes dataframe back to strings for writing
        mode_definitions_writing = mode_definitions_df.to_string(header=False, index=False)
        mode_definitions_writing_split = mode_definitions_writing.split(sep='\n')

        if multiphase == True:
            for i in range(len(mode_definitions_writing_split)):
                mode_definitions_writing_split[i] = mode_definitions_writing_split[i].replace('prm', 'local')

        mode_definitions_writing = [f'\t\t{mode_definitions_writing_split[i]}\n' for i in range(len(mode_definitions_writing_split))]

    # ----------------------------------------------------------------------------------------
    #* If multiphase - change prm to local in mode-amplitude-to-delta-transformation & distorted parameters sections
    # ----------------------------------------------------------------------------------------
    if multiphase == True:
        for i in range(len(mode_amplitudes)):
            newline = mode_amplitudes[i].replace('prm', 'local')
            mode_amplitudes[i] = newline

        for i in range(len(distorted_parameters)):
            newline = distorted_parameters[i].replace('prm', 'local')
            distorted_parameters[i] = newline

    # ----------------------------------------------------------------------------------------
    #* Augment mode dependent sites lines to have beq values
    # ----------------------------------------------------------------------------------------

    # Split mode_dependent_sites at spaces to modify certain values
    mode_dependent_sites_lines = [line.split() for line in mode_dependent_sites]

    # get atom types from topas str file
    all_atom_types_list = [mode_dependent_sites_lines[i][1].split('_')[0] for i in range(len(mode_dependent_sites_lines))]
    all_atom_types_list_unique = np.unique(all_atom_types_list).tolist()

    # generate a beq_value_dict from ranges in beq_matrix for each atom type
    beq_value_dict = {}
    if beq_dict !=None:
        for i in range(len(all_atom_types_list_unique)):
            if all_atom_types_list_unique[i] == 'origin':
                continue
            beq_range = beq_dict.get(all_atom_types_list_unique[i])
            
            if len(beq_range) == 2:
                beq_value = np.random.uniform(beq_range[0], beq_range[1])
            elif len(beq_range) == 1:
                beq_value = beq_range[0]
            else: print('beq_matrix is configured incorrectly, please fix and try again')
            beq_value_dict[all_atom_types_list_unique[i]] = beq_value 

    # use mode dependent sites lines to find atom types to apply beq
    for i in range(len(mode_dependent_sites_lines)):
        # if empty line OR 'origin' line in this section, continue past it
        if len(mode_dependent_sites_lines[i]) < 14:
            continue

        # get the atom type
        atom_type = mode_dependent_sites_lines[i][1].split('_')[0]
        if refine_beq == True:
            beq_var_name = f'beq bval_{atom_type}'
            if phase_count != None:
                beq_var_name = f'beq bval_{atom_type}_{phase_count}'
        else: 
            beq_var_name = f'beq !bval_{atom_type}'
            if phase_count != None:
                beq_var_name = f'beq !bval_{atom_type}_{phase_count}'

        mode_dependent_sites_lines[i][17] = beq_var_name

        # apply beq value from beq_value_dict
        mode_dependent_sites_lines[i][18] = str(beq_value_dict.get(atom_type))
    

    # join list of strings back together
    mode_dependent_sites_lines = [' '.join(mode_dependent_sites_lines[i]) for i in range(len(mode_dependent_sites_lines))]
    mode_dependent_sites = [f'\t\t{mode_dependent_sites_lines[i]}\n' for i in range(len(mode_dependent_sites_lines))]

    # ----------------------------------------------------------------------------------------
    #* Join all lines together back into list to be written into topas input file
    # ----------------------------------------------------------------------------------------

    all_mode_lines = []
    # append strain mode and str (structure definition) lines 
    for i in range(len(str_strain)):
        all_mode_lines.append(str_strain[i])
    all_mode_lines.append('\n')

    # append mode definitions
    all_mode_lines.append(f'{mode_definition_str}\n')
    for i in range(len(mode_definitions_writing)):
        all_mode_lines.append(mode_definitions_writing[i])
    all_mode_lines.append(f'{end_str}\n\n')    

    # append mode amplitudes
    all_mode_lines.append(f'{mode_amplitude_str}\n')
    for i in range(len(mode_amplitudes)):
        all_mode_lines.append(mode_amplitudes[i])
    all_mode_lines.append(f'{end_str}\n\n')    

    # append distorted parameters
    all_mode_lines.append(f'{distorted_parameters_str}\n')
    for i in range(len(distorted_parameters)):
        all_mode_lines.append(distorted_parameters[i])
    all_mode_lines.append(f'{end_str}\n\n')   

    # append mode_dependent_sites
    all_mode_lines.append(f'{mode_dependent_sites_str}\n')
    for i in range(len(mode_dependent_sites)):
        all_mode_lines.append(mode_dependent_sites[i])
    all_mode_lines.append(f'{end_str}\n\n')   

    # append difference restraints
    all_mode_lines.append(f'{difference_restrainsts_str}\n')
    for i in range(len(difference_restrainsts)):
        all_mode_lines.append(difference_restrainsts[i])
    all_mode_lines.append(f'{end_str}\n\n')   

    # # append macros
    # for i in range(len(strain_macros)):
    #     all_mode_lines.append(strain_macros[i])
    # all_mode_lines.append('\n\n')


    return all_mode_lines


def generate_multiphase_wt_percent_lines(num_phases, topas_scale_value, random_weight):

    if num_phases == 2:
        xdd_wt_percent_lines = f"""\tprm scale1 {topas_scale_value}
    prm w1 = 100 - w2; : 0.0
    prm w2 {random_weight[0]}

"""
    if num_phases == 3:
        w2 = random_weight
        if w2 >= 70.0:
            w3 = 0.5*(100.0 - w2)
        
        else:
            rw = 0.25*(100.0 - w2)
            random_remainder = np.random.uniform(low=-rw, high=rw)
            w3 = 0.5*(100.0 - w2) + random_remainder
        
        xdd_wt_percent_lines = f"""\tprm scale1 {topas_scale_value}
    prm w1 = 100 - w2 - w3; : 0.0
    prm w2 {random_weight[0]}
    prm w3 {w3[0]}

"""
    return xdd_wt_percent_lines





# ------ Write Topas INP's for ML ------
def write_inp_for_ML(topas_input_file, latparams, cif_spacegroup_and_origin, background_line, TOF_x_axis_calibration, TOF_sample_peakshape, mode_amplitude_lines, start_finish_X, xdd_winpath, cif_name, hkl_out_file, hkl_scaled_out_file, xy_out_file):
    '''
    Parameters
    ----------
    topas_input_file_name: str
        Name of the INP file to write
    latparams: list 
        [int(space_group), a, b, c, alpha, beta, gamma]
    cif_spacegroup_and_origin: list of 2 ints
        [0]: spacegroup number
        [1]: '1' or '2' for origin choice
    background_line: fstring
        the background line from a good pawley fit of the experiment (makes the background consistent for all simulated powder patterns) 
        (if the background is the same, the differences in the powder patterns are purely from differences of structure)
    TOF_x_axis_calibration: fstring
        TOF_x_axis_calibration line from a good pawley fit of the experiment (same reason as background_line) TOF_sample_peakshape: fstring
        tof_sample_peakshape line from a good pawley fit of the experiment 
    mode_amplitude_lines: list of strings
        output from get_mode_amplitudes 
    start_finish_X: list of floats
        start and finish of TOF range for Topas
    xdd_winpath: str
        Full Windows path to the data file for xdd
    hkl_out_file: str
        Path and Name of the output file for the hkl intensities
    hkl_scaled_out_file: str
        Path and name of the output file for topas scaled hkl intensities
    xy_out_file: str
        Path and Name of the output file for the calculated xy diffraction pattern

    Returns:
    ----------
        Returns nothing, but writes the TOPAS input file
    '''

    refine_lines = f"""\'----------------------------------------### Refinement methods ###----------------------------------------
    iters  0
    chi2_convergence_criteria 0.0001
    \'Auto_T(0.00001)  \' simulated annealing 
    \'do_errors
    no_LIMIT_warnings
    \'continue_after_convergence
    r_wp  0.0 r_exp  0.0 r_p  0.0 r_wp_dash  0.0 r_p_dash  0.0 r_exp_dash  0.0 weighted_Durbin_Watson  0.0 gof  0.0

"""
    global_lines = f"""\'----------------------------------------### Global parameters ###----------------------------------------

"""
    data_lines_1 = f"""\'----------------------------------------### Defining data sets ###----------------------------------------
    #define use_data 
    #ifdef use_data
    #define USE_BS    macro filename_bs    {{"{xdd_winpath}"}}
    #endif

\'----------------------------------------### Main refinement block ###----------------------------------------
    #ifdef USE_BS
    xdd filename_bs xye_format
        start_X {start_finish_X[0]}
        finish_X {start_finish_X[1]}
        local !neutron 1 local !xray 0 
        neutron_data
        r_wp 0.0
        {TOF_x_axis_calibration}
        local !bank_two_theta 168.567 
        local !mu  0.07076 
        local !fixb  0.0 min -10 max 10 
        {background_line}
        
        scale_pks = (Exp(-2*fixb/(2 D_spacing)^2)); 'fixb correction local to each bank
        weighting = If(SigmaYobs < 1e-9, 1, 1/SigmaYobs^2);
            x_calculation_step = Yobs_dx_at(Xo); convolution_step 4
            TOF_LAM(0.001)
            scale_pks = D_spacing^4;
            scale_pks = Exp(-mu*2 D_spacing Sin(0.00872664626 two_theta));   'absorption correction local to each bank

            #define peak1
            #ifdef peak1
                wifd_pkshape(!two_theta, =bank_two_theta;, !aprm, 2.50000, !tauf_2, 9.80744, !tauf_1, 0.43977, !tauf_0, 0.000, !taus_0, 25.27420, !taus_1, 2.66797, !t_eff, 144.41296, !hhh, 3.58595, !hhh_0, 11.61348)
            #endif

"""

    str_lines = f"""
        str 
            scale @ 1
            {TOF_sample_peakshape}
            r_bragg  0.0704403462

            phase_name \"{cif_name}\"	
            a  {latparams[0]}
            b  {latparams[1]}
            c  {latparams[2]}
            al {latparams[3]}
            be {latparams[4]}
            ga {latparams[5]}
            space_group {cif_spacegroup_and_origin[0]}:{cif_spacegroup_and_origin[1]}
"""
    # SYMMETRY MODE LINES ARE WRITTEN HERE DURING FILE WRITING 
    
    output_lines_XYcalc = f""" 
        Out_X_Ycalc(\"{xy_out_file}\")
"""
    output_lines_hkl = f""" 
        Create_d_Ip_file_dayton(\"{hkl_out_file}\")
        \'Create_d_IScaled_file_dayton(\"{hkl_scaled_out_file}\")
        
        
#endif
"""

    usr_macros = """\'----------------------------------------### User defined macros ###----------------------------------------

"""

    with open(topas_input_file, 'w') as f:
        f.write(refine_lines)
        f.write(global_lines)
        f.write(data_lines_1)
        f.write(str_lines)
        for i in range(len(mode_amplitude_lines)):
            f.write(mode_amplitude_lines[i])
        f.write(output_lines_XYcalc)
        f.write(output_lines_hkl)
        f.write(usr_macros)

    return None 


def write_inp_for_ML_with_strain_modes(beamline_choice:str, topas_input_file:str, strain_and_distortion_modes:list, start_finish_X:list, xdd_winpath:str, xy_out_file:str=None, dy_out_file:str=None, hkl_out_file:str =None, hkl_scaled_out_file:str=None):

    '''
    Parameters
    ----------
    beamline_choice: str
        Name of the beamline from experiment.\n
        Choose: POWGEN, HRPD, 
    topas_input_file_name: str
        Name of the INP file to write
    strain_and_distortion_modes: list of strings
        Defines the str section of topas input, the strain modes for lattice parameter perturbation,  the distortion modes definitions, and the macros for strain modes
    start_finish_X: list of floats
        start and finish of TOF range for Topas
    xdd_winpath: str
        Full Windows path to the data file for xdd
    hkl_out_file: str
        Path and Name of the output file for the hkl intensities
    hkl_scaled_out_file: str
        Path and name of the output file for topas scaled hkl intensities
    xy_out_file: str
        Path and Name of the output file for the calculated xy diffraction pattern with TOF x axis
    dy_out_file: str
        Path and Name of the output file for the calculated xy diffraction pattern with d-spacing x axis
        
    Returns:
    ----------
    Returns nothing, but writes the TOPAS input file
    '''

    refine_lines = f"""\'----------------------------------------### Refinement methods ###----------------------------------------
iters  0
chi2_convergence_criteria 0.0001
\'Auto_T(0.00001)  \' simulated annealing 
\'do_errors
no_LIMIT_warnings
\'continue_after_convergence
r_wp  0.0 r_exp  0.0 r_p  0.0 r_wp_dash  0.0 r_p_dash  0.0 r_exp_dash  0.0 weighted_Durbin_Watson  0.0 gof  0.0

"""
    global_lines = f"""\'----------------------------------------### Global parameters ###----------------------------------------
prm !pow  4
"""
    HRPD_lines = hrpd_inp_lines(xdd_winpath, start_finish_X)
    POWGEN_lines = powgen_inp_lines(xdd_winpath, start_finish_X)

    #* MULTI-PHASE WT% LINES GO HERE - UNDER XDD, BUT ABOVE STR SCOPE

    #* STR SECTION, STRAIN MODES, AND SYMMETRY MODE LINES ARE WRITTEN HERE DURING FILE WRITING

    if xy_out_file != None:
        output_lines_XYcalc = f""" 
        Out_X_Ycalc(\"{xy_out_file}\")
"""
    if dy_out_file != None:
        output_lines_DYcalc = f""" 
        Out_d_from_tof_bragg_dayton("{dy_out_file}", d0_1, difc_1, difa_1)
"""
    if hkl_out_file != None:
        output_lines_hkl = f""" 
        Create_d_Ip_file_dayton(\"{hkl_out_file}\")
        
"""
    if hkl_scaled_out_file != None:
        output_lines_hkl_scaled = f""" 
        Create_d_IScaled_file_dayton(\"{hkl_scaled_out_file}\")
        
"""
    endif_line = '\n#endif \n'
    usr_macros = """\'----------------------------------------### User defined macros ###----------------------------------------

"""

    with open(topas_input_file, 'w') as f:
        f.write(refine_lines)
        f.write(global_lines)
        if beamline_choice == "HRPD":
            f.write(HRPD_lines)
        elif beamline_choice == "POWGEN":
            f.write(POWGEN_lines)
        else:
            raise Exception("Incorrect choice of beamline. Choose from: \nPOWGEN\nHRPD\n")
        
        for i in range(len(strain_and_distortion_modes)):
            f.write(strain_and_distortion_modes[i])

        if xy_out_file != None:
            f.write(output_lines_XYcalc)
        if dy_out_file != None:
            f.write(output_lines_DYcalc)
        if hkl_out_file != None:
            f.write(output_lines_hkl)
        if hkl_scaled_out_file != None:
            f.write(output_lines_hkl_scaled)

        f.write(endif_line)
        f.write(usr_macros)

    return None


def write_inp_for_multiphase_ML_with_strain_modes(beamline_choice, topas_input_file, strain_and_distortion_modes, start_finish_X, xdd_winpath, xdd_wt_percent_lines = None, xy_out_file = None, dy_out_file=None, hkl_out_file=None, hkl_scaled_out_file=None):

    '''
    Parameters
    ----------
    beamline_choice: str
        Name of the beamline from experiment
        Choose: POWGEN, HRPD, 
    topas_input_file_name: str
        Name of the INP file to write
    strain_and_distortion_modes: list of strings
        Defines the str section of topas input, the strain modes for lattice parameter perturbation,  the distortion modes definitions, and the macros for strain modes
    start_finish_X: list of floats
        start and finish of TOF range for Topas
    xdd_winpath: str
        Full Windows path to the data file for xdd
    hkl_out_file: str
        Path and Name of the output file for the hkl intensities
    hkl_scaled_out_file: str
        Path and name of the output file for topas scaled hkl intensities
    xy_out_file: str
        Path and Name of the output file for the calculated xy diffraction pattern

    Returns:
    -----------
        Returns nothing, but writes the TOPAS input file
    '''

    refine_lines = f"""\'----------------------------------------### Refinement methods ###----------------------------------------
iters  0
chi2_convergence_criteria 0.0001
\'Auto_T(0.00001)  \' simulated annealing 
\'do_errors
no_LIMIT_warnings
\'continue_after_convergence
r_wp  0.0 r_exp  0.0 r_p  0.0 r_wp_dash  0.0 r_p_dash  0.0 r_exp_dash  0.0 weighted_Durbin_Watson  0.0 gof  0.0

"""
    global_lines = f"""\'----------------------------------------### Global parameters ###----------------------------------------
prm !pow  4
"""
    HRPD_lines = hrpd_inp_lines(xdd_winpath, start_finish_X)
    POWGEN_lines = powgen_inp_lines(xdd_winpath, start_finish_X)

    #* MULTI-PHASE WT% LINES GO HERE - UNDER XDD, BUT ABOVE STR SCOPE

    #* STR SECTION, STRAIN MODES, AND SYMMETRY MODE LINES ARE WRITTEN HERE DURING FILE WRITING
    
    if xy_out_file != None:
        output_lines_XYcalc = f""" 
        Out_X_Ycalc(\"{xy_out_file}\")

"""
    if dy_out_file != None:
        output_lines_DYcalc = f""" 
        Out_d_from_tof_bragg_dayton("{dy_out_file}", d0_1, difc_1, difa_1)
"""
    if hkl_out_file != None:
        output_lines_hkl = f""" 
        Create_d_Ip_file_dayton(\"{hkl_out_file}\")
        
"""
    if hkl_scaled_out_file != None:
        output_lines_hkl_scaled = f""" 
        Create_d_IScaled_file_dayton(\"{hkl_scaled_out_file}\")
        
"""

    endif_line = '\n#endif\n'
    usr_macros = """\'----------------------------------------### User defined macros ###----------------------------------------

"""

    with open(topas_input_file, 'w') as f:

        f.write(refine_lines)

        f.write(global_lines)

        if beamline_choice == "HRPD":
            f.write(HRPD_lines)
        elif beamline_choice == "POWGEN":
            f.write(POWGEN_lines)
        else:
            raise Exception("Incorrect choice of beamline. Choose from: \nPOWGEN\nHRPD\n")
        
        if xdd_wt_percent_lines != None:
            for i in range(len(xdd_wt_percent_lines)):
                f.write(xdd_wt_percent_lines[i])
        
        for i in range(len(strain_and_distortion_modes)):
            f.write(strain_and_distortion_modes[i])

        if xy_out_file != None:
            f.write(output_lines_XYcalc)
        if dy_out_file != None:
            f.write(output_lines_DYcalc)
        if hkl_out_file != None:
            f.write(output_lines_hkl)
        if hkl_scaled_out_file != None:
            f.write(output_lines_hkl_scaled)
            
        f.write(endif_line)
        f.write(usr_macros)

    return None




# ------ Write Topas INP's for Pawley Refinements ------
def write_inp_for_pawley_BISN2O7(inp_filename, cif_name, topas_latparam_lines, TOF_x_axis_calibration, xdd_path, refinement_iterations, conv_criteria, start_finish_x, hkl_out_file, xy_out_file):
    '''
    Parameters
    ----------
    inp_filename: str
        Name of the INP file to write
    cif_name: str
        Name passed into topas keyword 'phase_name'
    topas_latparam_lines: list of str; tells topas which lattice parameters to refine based on spacegroup symmetry constraints
        output of write_TOPAS_funcs.generate_pawley_lattice_lines
    xdd_path: str
        Full windows path to the data file for xdd
    refinement_iterations: int
        number for topas keyword 'iters'
    conv_criteria: int
        chi2_convergence_criteria: default = 0.0001
    start_finish_x: list of float
        Time of Flight start and finish points in microseconds for refinement
    hkl_file_out: str
        Name of the output file for the hkl intensities
    simulated_xy_out: str
        Name of the output file for the calculated xy diffraction pattern

    Returns:
    -----------
        Returns nothing, but writes the TOPAS input file

    '''


    refine_lines = f"""\'----------------------------------------### Refinement methods ###----------------------------------------
    iters  {refinement_iterations}
    chi2_convergence_criteria {conv_criteria}
    \'Auto_T(0.00001)  \' simulated annealing 
    \'do_errors
    no_LIMIT_warnings
    \'continue_after_convergence
    r_wp  0.0 r_exp  0.0 r_p  0.0 r_wp_dash  0.0 r_p_dash  0.0 r_exp_dash  0.0 weighted_Durbin_Watson  0.0 gof  0.0

"""

    global_lines = f"""\'----------------------------------------### Global parameters ###----------------------------------------

"""

    # take variables from bkg, POWGEN_instrument_2, & GSAS_btb from topas output file in 02_pawley_on_P1 and put them in data_lines_1 and data_lines_2
    # set these variables to not refine so we have consistent background and peakshape for all simulated xy files
    data_lines_1 = f"""\'----------------------------------------### Defining data sets ###----------------------------------------
    #define use_data 
    #ifdef use_data
    #define USE_BS    macro filename_bs    {{"{xdd_path}"}}
    #endif

\'----------------------------------------### Main refinement block ###----------------------------------------
    #ifdef USE_BS
    xdd filename_bs xye_format
        start_X {start_finish_x[0]}
        finish_X {start_finish_x[1]}
        local !neutron 1 local !xray 0 
        neutron_data
        r_wp  2.60439382
        {TOF_x_axis_calibration}
        local !bank_two_theta 168.567 
        local !mu  0.07076 
        local !fixb  0.0 min -10 max 10 
        bkg  @  1.82819531` -0.164089368`  0.0804921835` -0.064352384`  0.0885898809` -0.0846077489`  0.0900426227` -0.0977319011`  0.0878555271` -0.0448681901`  0.0326556039` -0.0128961147`
        scale_pks = (Exp(-2*fixb/(2 D_spacing)^2)); 'fixb correction local to each bank
        weighting = If(SigmaYobs < 1e-9, 1, 1/SigmaYobs^2);
            x_calculation_step = Yobs_dx_at(Xo); convolution_step 4
            TOF_LAM(0.001)
            scale_pks = D_spacing^4;
            scale_pks = Exp(-mu*2 D_spacing Sin(0.00872664626 two_theta));   'absorption correction local to each bank

            #define peak1
            #ifdef peak1
                wifd_pkshape(!two_theta, =bank_two_theta;, !aprm, 2.50000, !tauf_2, 9.80744, !tauf_1, 0.43977, !tauf_0, 0.000, !taus_0, 25.27420, !taus_1, 2.66797, !t_eff, 144.41296, !hhh, 3.58595, !hhh_0, 11.61348)
            #endif

"""
    
    hkl_is_lines = f"""
        hkl_Is
            scale @ 1
            tof_sample_peakshape(lor, 0.0157897846`,dsp, 70.03538`,dspsq, 20.92662`) 
            r_bragg  37.2339895
            'Out_CIF_STR("topas_generated.cif")
    
            phase_name \"{cif_name}\"	
            {topas_latparam_lines[1]}
            {topas_latparam_lines[2]}
            {topas_latparam_lines[3]}
            {topas_latparam_lines[4]}
            {topas_latparam_lines[5]}
            {topas_latparam_lines[6]}
            {topas_latparam_lines[0]}
"""

    output_lines_XYcalc = f""" 
        Out_X_Ycalc(\"{xy_out_file}\")
"""
    output_lines_hkl = f""" 
        Create_d_Ip_file_dayton(\"{hkl_out_file}\")
        
#endif
"""

    usr_macros = """\'----------------------------------------### User defined macros ###----------------------------------------

"""

    with open(inp_filename, 'w') as f:
        f.write(refine_lines)
        f.write(global_lines)
        f.write(data_lines_1)
        f.write(hkl_is_lines)
        f.write(output_lines_XYcalc)
        f.write(output_lines_hkl)
        f.write(usr_macros)

    return None


def write_inp_for_pawley_with_vc_macro(inp_filename, use_vc_macro, TOF_x_axis_calibration, cif_name, topas_latparam_lines, vc_lp_matrix, xdd_path, refinement_iterations, conv_criteria, start_finish_x, hkl_out_file, hkl_scaled_out_file, xy_out_file):
    '''
    Parameters
    ----------
    inp_filename: str
        Name of the INP file to write
    cif_name: str
        Name passed into topas keyword 'phase_name'
    use_vc_macro: bool
        Whether to continue refinement after convergence by applying vc macro to perturb values with vc matrix and start refinement again
    topas_latparam_lines: list of str; tells topas which lattice parameters to refine based on spacegroup symmetry constraints
        output of write_TOPAS_funcs.generate_pawley_lattice_lines
    vc_lp_matrix: 2d list of floats 
        vc macro: val on continue macro in topas
        vc_lp_matrix[[v1, v2, min, max], ....] --> fill in all 4 values for lattice parameters a, b, c, alpha, beta, gamma
        demo use:
        macro vc { val_on_continue = Val*Rand(v1,v2); }     --> where the current value of lattice parameter will be multiplied by a random value between v1 & v2
        a lpa 10.725 vc min 7.0 max 25.0                    --> where 7.0 & 25.0 are the lower & upper bounds of the lattice parameter refinement
    xdd_path: str
        Full windows path to the data file for xdd
    refinement_iterations: int
        number for topas keyword 'iters'
    start_finish_x: list of float
        Time of Flight start and finish points in microseconds for refinement
    hkl_file_out: str
        Name of the output file for the hkl intensities
    simulated_xy_out: str
        Name of the output file for the calculated xy diffraction pattern

    '''
    if use_vc_macro == True:
        convergence_line = 'continue_after_convergence'
    else: convergence_line = '\'continue_after_convergence'

    refine_lines = f"""\'----------------------------------------### Refinement methods ###----------------------------------------
    iters  {refinement_iterations}
    chi2_convergence_criteria {conv_criteria}
    \'Auto_T(0.00001)  \' simulated annealing 
    \'do_errors
    no_LIMIT_warnings
    {convergence_line}
    r_wp  0.0 r_exp  0.0 r_p  0.0 r_wp_dash  0.0 r_p_dash  0.0 r_exp_dash  0.0 weighted_Durbin_Watson  0.0 gof  0.0

"""

    global_lines = f"""\'----------------------------------------### Global parameters ###----------------------------------------

"""

    # take variables from bkg, POWGEN_instrument_2, & GSAS_btb from topas output file in 02_pawley_on_P1 and put them in data_lines_1 and data_lines_2
    # set these variables to not refine so we have consistent background and peakshape for all simulated xy files
    data_lines_1 = f"""\'----------------------------------------### Defining data sets ###----------------------------------------
    #define use_data 
    #ifdef use_data
    #define USE_BS    macro filename_bs    {{"{xdd_path}"}}
    #endif

\'----------------------------------------### Main refinement block ###----------------------------------------
    #ifdef USE_BS
    xdd filename_bs xye_format
        start_X {start_finish_x[0]}
        finish_X {start_finish_x[1]}
        local !neutron 1 local !xray 0 
        neutron_data
        r_wp  2.60439382
        {TOF_x_axis_calibration} 
        local !bank_two_theta 168.567 
        local !mu  0.07076 
        local !fixb  0.0 min -10 max 10 
        bkg  @  1.82819531` -0.164089368`  0.0804921835` -0.064352384`  0.0885898809` -0.0846077489`  0.0900426227` -0.0977319011`  0.0878555271` -0.0448681901`  0.0326556039` -0.0128961147`
        scale_pks = (Exp(-2*fixb/(2 D_spacing)^2)); 'fixb correction local to each bank
        weighting = If(SigmaYobs < 1e-9, 1, 1/SigmaYobs^2);
            x_calculation_step = Yobs_dx_at(Xo); convolution_step 4
            TOF_LAM(0.001)
            scale_pks = D_spacing^4;
            scale_pks = Exp(-mu*2 D_spacing Sin(0.00872664626 two_theta));   'absorption correction local to each bank

            #define peak1
            #ifdef peak1
                wifd_pkshape(!two_theta, =bank_two_theta;, !aprm, 2.50000, !tauf_2, 9.80744, !tauf_1, 0.43977, !tauf_0, 0.000, !taus_0, 25.27420, !taus_1, 2.66797, !t_eff, 144.41296, !hhh, 3.58595, !hhh_0, 11.61348)
            #endif

"""
    
    hkl_is_lines = f"""
        hkl_Is
            scale @ 1
            tof_sample_peakshape(lor, 0.168698361, dsp, 34.96508, dspsq, 2.83011) 
            r_bragg  37.2339895
            'Out_CIF_STR("topas_generated.cif")
    
            phase_name \"{cif_name}\"	
            {topas_latparam_lines[1]}
            {topas_latparam_lines[2]}
            {topas_latparam_lines[3]}
            {topas_latparam_lines[4]}
            {topas_latparam_lines[5]}
            {topas_latparam_lines[6]}
            {topas_latparam_lines[0]}
"""

    output_lines_XYcalc = f""" 
        Out_X_Ycalc(\"{xy_out_file}\")
"""
    output_lines_hkl = f""" 
        Create_d_Ip_file_dayton(\"{hkl_out_file}\")
        Create_d_IScaled_file_dayton(\"{hkl_scaled_out_file}\")

        
#endif
"""

    usr_macros = f"""\'----------------------------------------### User defined macros ###----------------------------------------
    macro vca {{ val_on_continue = Val*Rand({vc_lp_matrix[0][0]},{vc_lp_matrix[0][1]}); }}
    macro vcb {{ val_on_continue = Val*Rand({vc_lp_matrix[1][0]},{vc_lp_matrix[1][1]}); }}
    macro vcc {{ val_on_continue = Val*Rand({vc_lp_matrix[2][0]},{vc_lp_matrix[2][1]}); }}
    macro vcalpha {{ val_on_continue = Val*Rand({vc_lp_matrix[3][0]},{vc_lp_matrix[3][1]}); }}
    macro vcbeta {{ val_on_continue = Val*Rand({vc_lp_matrix[4][0]},{vc_lp_matrix[4][1]}); }}
    macro vcgamma {{ val_on_continue = Val*Rand({vc_lp_matrix[5][0]},{vc_lp_matrix[5][1]}); }}

"""

    with open(inp_filename, 'w') as f:
        f.write(refine_lines)
        f.write(global_lines)
        f.write(data_lines_1)
        f.write(hkl_is_lines)
        f.write(output_lines_XYcalc)
        f.write(output_lines_hkl)
        f.write(usr_macros)

    return None





# ------ Write Topas INP's for Reitveld Refinements ------
def write_inp_for_rietveld_with_vc_macro(inp_filename, use_symmetry_mode_analysis, use_vc_lp_macro, use_vc_ma_macro, vc_lp_matrix, vc_ma_matrix, TOF_x_axis_calibration, cif_name, topas_latparam_lines, site_lines, symmetry_mode_lines, xdd_path, refinement_iterations, conv_criteria, start_finish_x, hkl_out_file, hkl_scaled_out_file, xy_out_file):
    '''
    Parameters
    ----------
    inp_filename: str
        Name of the INP file to write
    use_symmetry_mode_analysis: bool
        Whether to use isodistort symmetry modes for structure (True) or use traditional site info from cif for structure (False)
    use_vc_lp_macro: bool
        Whether to continue refinement after convergence by applying vc_lp_macro to perturb lattice parameter values with vc_lp_matrix and start refinement again
    use_vc_ma_macro: bool
        Whether to continue refinement after convergence by applying vc_ma_macro to perturb mode amplitude values with vc_ma_matrix and start refinement again
    vc_lp_matrix: 2d list of floats 
        vc macro: val on continue macro in topas
        vc_lp_matrix[[v1, v2, min, max], ....] --> fill in all 4 values for lattice parameters a, b, c, alpha, beta, gamma
        demo use:
        macro vc { val_on_continue = Val*Rand(v1,v2); }     --> where the current value of lattice parameter will be multiplied by a random value between v1 & v2
        a lpa 10.725 vc min 7.0 max 25.0                    --> where 7.0 & 25.0 are the lower & upper bounds of the lattice parameter refinement
    vc_ma_matrix: 1d list of floats -> lower and upper bound for random float generation to apply to mode amplitude values on convergence
        default: vc_ma_matrix = [0.9, 1.1] 
    cif_name: str
        Name passed into topas keyword 'phase_name'
    topas_latparam_lines: list of str; tells topas which lattice parameters to refine based on spacegroup symmetry constraints
        output of write_TOPAS_funcs.generate_pawley_lattice_lines
    site_lines: list of str; 
        structure definition lines from a cif file: beq can modify these
        - prepared by write_TOPAS_funcs.generate_lattice_parameter_lines_with_vc_macro
    symmetry_mode_lines: list of str
        structure definition lines from isodistort 
        - prepared by write_TOPAS_funcs.generate_symmetry_mode_lines_from_topas_str (a file generated by isodistort)
    xdd_path: str
        Full windows path to the data file for xdd
    refinement_iterations: int
        number for topas keyword 'iters'
    start_finish_x: list of float
        Time of Flight start and finish points in microseconds for refinement
    hkl_file_out: str
        Name of the output file for the hkl intensities
    simulated_xy_out: str
        Name of the output file for the calculated xy diffraction pattern

    '''
    if use_vc_lp_macro == True:
        convergence_line = 'continue_after_convergence'
    elif use_vc_ma_macro == True:
        convergence_line = 'continue_after_convergence'
    else: convergence_line = '\'continue_after_convergence'

    refine_lines = f"""\'----------------------------------------### Refinement methods ###----------------------------------------
    iters  {refinement_iterations}
    chi2_convergence_criteria {conv_criteria}
    \'Auto_T(0.00001)  \' simulated annealing 
    \'do_errors
    no_LIMIT_warnings
    {convergence_line}
    r_wp  0.0 r_exp  0.0 r_p  0.0 r_wp_dash  0.0 r_p_dash  0.0 r_exp_dash  0.0 weighted_Durbin_Watson  0.0 gof  0.0

"""

    global_lines = f"""\'----------------------------------------### Global parameters ###----------------------------------------

"""
    # TOF_x_axis_calibration = 'TOF_x_axis_calibration(t0_hrpd_bs,-22.0105`_0.19111746,!difc_hrpd_bs, 48283.52311_1.67763, difa_hrpd_bs, -14.00371`_0.0876098944)'

    data_lines_1 = f"""\'----------------------------------------### Defining data sets ###----------------------------------------
    #define use_data 
    #ifdef use_data
    #define USE_BS    macro filename_bs    {{"{xdd_path}"}}
    #endif

\'----------------------------------------### Main refinement block ###----------------------------------------
    #ifdef USE_BS
    xdd filename_bs xye_format
        start_X {start_finish_x[0]}
        finish_X {start_finish_x[1]}
        local !neutron 1 local !xray 0 
        neutron_data
        r_wp  2.60439382
        {TOF_x_axis_calibration}  
        local !bank_two_theta 168.567 
        local !mu  0.07076 
        local !fixb  0.0 min -10 max 10 
        bkg  @  1.82819531` -0.164089368`  0.0804921835` -0.064352384`  0.0885898809` -0.0846077489`  0.0900426227` -0.0977319011`  0.0878555271` -0.0448681901`  0.0326556039` -0.0128961147`
        scale_pks = (Exp(-2*fixb/(2 D_spacing)^2)); 'fixb correction local to each bank
        weighting = If(SigmaYobs < 1e-9, 1, 1/SigmaYobs^2);
            x_calculation_step = Yobs_dx_at(Xo); convolution_step 4
            TOF_LAM(0.001)
            scale_pks = D_spacing^4;
            scale_pks = Exp(-mu*2 D_spacing Sin(0.00872664626 two_theta));   'absorption correction local to each bank

            #define peak1
            #ifdef peak1
                wifd_pkshape(!two_theta, =bank_two_theta;, !aprm, 2.50000, !tauf_2, 9.80744, !tauf_1, 0.43977, !tauf_0, 0.000, !taus_0, 25.27420, !taus_1, 2.66797, !t_eff, 144.41296, !hhh, 3.58595, !hhh_0, 11.61348)
            #endif

"""
    
    structure_lines = f"""
        str
            scale @ 1
            tof_sample_peakshape(lor, 0.0157897846`,dsp, 70.03538`,dspsq, 20.92662`) 
            r_bragg  37.2339895
            'Out_CIF_STR("topas_generated.cif")
    
            phase_name \"{cif_name}\"	
            {topas_latparam_lines[1]}
            {topas_latparam_lines[2]}
            {topas_latparam_lines[3]}
            {topas_latparam_lines[4]}
            {topas_latparam_lines[5]}
            {topas_latparam_lines[6]}
            {topas_latparam_lines[0]}

"""
    # if use_symmetry_mode_analysis == True , then symmetry mode lines will appear here in topas input file when written

    # if use_symmetry_mode_analysis == False, then site lines will appear hear in topas input file when written

    output_lines_XYcalc = f""" 
        Out_X_Ycalc(\"{xy_out_file}\")
"""
    output_lines_hkl = f""" 
        Create_d_Ip_file_dayton(\"{hkl_out_file}\")
        Create_d_IScaled_file_dayton(\"{hkl_scaled_out_file}\")

        
#endif
"""

    usr_macros = f"""\'----------------------------------------### User defined macros ###----------------------------------------
    macro vc {{ val_on_continue = Val*Rand({vc_ma_matrix[0]},{vc_ma_matrix[1]}); }}
    macro vca {{ val_on_continue = Val*Rand({vc_lp_matrix[0][0]},{vc_lp_matrix[0][1]}); }}
    macro vcb {{ val_on_continue = Val*Rand({vc_lp_matrix[1][0]},{vc_lp_matrix[1][1]}); }}
    macro vcc {{ val_on_continue = Val*Rand({vc_lp_matrix[2][0]},{vc_lp_matrix[2][1]}); }}
    macro vcalpha {{ val_on_continue = Val*Rand({vc_lp_matrix[3][0]},{vc_lp_matrix[3][1]}); }}
    macro vcbeta {{ val_on_continue = Val*Rand({vc_lp_matrix[4][0]},{vc_lp_matrix[4][1]}); }}
    macro vcgamma {{ val_on_continue = Val*Rand({vc_lp_matrix[5][0]},{vc_lp_matrix[5][1]}); }}

"""

    with open(inp_filename, 'w') as f:
        f.write(refine_lines)
        f.write(global_lines)
        f.write(data_lines_1)
        f.write(structure_lines)

        if use_symmetry_mode_analysis == True:
            for i in range(len(symmetry_mode_lines)):
                f.write(symmetry_mode_lines[i])
        if use_symmetry_mode_analysis == False:
            for i in range(len(site_lines)):
                f.write(site_lines[i])

        f.write(output_lines_XYcalc)
        f.write(output_lines_hkl)
        f.write(usr_macros)

    return None


def write_exp_refinement_strain_modes_stage_1(beamline_choice, inp_filename, use_vc_macro, TOF_x_axis_calibration, str_lines, macro_lines, xdd_winpath, refinement_iterations, conv_criteria, start_finish_X, background_line, hkl_out_file=None, hkl_scaled_out_file=None, xy_out_file=None):
    '''
    Parameters
    ----------
    inp_filename: str
        Name of the INP file to write
    TOF_x_axis_calibration: str
        String for writing the tof_x_axis_calibration for TOPAS input
    xdd_path: str
        Full windows path to the data file for xdd
    refinement_iterations: int
        number for topas keyword 'iters'
    start_finish_x: list of int
        Time of Flight start and finish points in microseconds for refinement
    hkl_file_out: str
        Name of the output file for the hkl intensities
    simulated_xy_out: str
        Name of the output file for the calculated xy diffraction pattern

    '''
    if use_vc_macro == True:
        convergence_line = 'continue_after_convergence'
    else: convergence_line = '\'continue_after_convergence'

    refine_lines = f"""\'----------------------------------------### Refinement methods ###----------------------------------------
iters  {refinement_iterations}
chi2_convergence_criteria {conv_criteria}
\'Auto_T(0.00001)  \' simulated annealing 
\'do_errors
\'no_LIMIT_warnings
{convergence_line}
r_wp  0.0 r_exp  0.0 r_p  0.0 r_wp_dash  0.0 r_p_dash  0.0 r_exp_dash  0.0 weighted_Durbin_Watson  0.0 gof  0.0

"""

    global_lines = f"""\'----------------------------------------### Global parameters ###----------------------------------------
prm !pow  4
"""
    HRPD_lines = hrpd_inp_lines(xdd_winpath, start_finish_X, TOF_x_axis_calibration, background_line)
    POWGEN_lines = powgen_inp_lines(xdd_winpath, start_finish_X, refine_instrument_parameters=True)

    #* MULTI-PHASE WT% LINES GO HERE - UNDER XDD, BUT ABOVE STR SCOPE

    #* STR SECTION, STRAIN MODES, AND SYMMETRY MODE LINES ARE WRITTEN HERE DURING FILE WRITING
    
    if xy_out_file != None:
        output_lines_XYcalc = f""" 
        Out_X_Ycalc(\"{xy_out_file}\")

"""
    if hkl_out_file != None:
        output_lines_hkl = f""" 
        Create_d_Ip_file_dayton(\"{hkl_out_file}\")
        
"""
    if hkl_scaled_out_file != None:
        output_lines_hkl_scaled = f""" 
        Create_d_IScaled_file_dayton(\"{hkl_scaled_out_file}\")
        
"""

    endif_line = '\n#endif\n'
    usr_macros = """\'----------------------------------------### User defined macros ###----------------------------------------
"""

    with open(inp_filename, 'w') as f:
        f.write(refine_lines)
        f.write(global_lines)

        if beamline_choice == "HRPD":
            f.write(HRPD_lines)
        elif beamline_choice == "POWGEN":
            f.write(POWGEN_lines)
        else:
            raise Exception("Incorrect choice of beamline. Choose from: \nPOWGEN\nHRPD\n")
        
        for i in range(len(str_lines)):
            f.write(str_lines[i])

        if xy_out_file != None:
            f.write(output_lines_XYcalc)
        if hkl_out_file != None:
            f.write(output_lines_hkl)
        if hkl_scaled_out_file != None:
            f.write(output_lines_hkl_scaled)
            
        f.write(endif_line)
        f.write(usr_macros)
        # for i in range(len(macro_lines)):
        #     f.write(macro_lines[i])

    return None


def write_inp_for_pawley_on_subgroup_tree_BISN2O7(use_strain_modes, inp_filename, cif_name, topas_latparam_lines, macro_lines, TOF_x_axis_calibration, background_line, TOF_sample_peakshape, xdd_path, refinement_iterations, conv_criteria, start_finish_x, hkl_out_file, xy_out_file):
    '''
    Parameters
    ----------
    inp_filename: str
        Name of the INP file to write
    cif_name: str
        Name passed into topas keyword 'phase_name'
    topas_latparam_lines: list of str; tells topas which lattice parameters to refine based on spacegroup symmetry constraints
        output of write_TOPAS_funcs.generate_pawley_lattice_lines
    xdd_path: str
        Full windows path to the data file for xdd
    refinement_iterations: int
        number for topas keyword 'iters'
    conv_criteria: int
        chi2_convergence_criteria: default = 0.0001
    start_finish_x: list of float
        Time of Flight start and finish points in microseconds for refinement
    hkl_file_out: str
        Name of the output file for the hkl intensities
    simulated_xy_out: str
        Name of the output file for the calculated xy diffraction pattern

    '''


    refine_lines = f"""\'----------------------------------------### Refinement methods ###----------------------------------------
    iters  {refinement_iterations}
    chi2_convergence_criteria {conv_criteria}
    \'Auto_T(0.00001)  \' simulated annealing 
    \'do_errors
    no_LIMIT_warnings
    \'continue_after_convergence
    r_wp  0.0 r_exp  0.0 r_p  0.0 r_wp_dash  0.0 r_p_dash  0.0 r_exp_dash  0.0 weighted_Durbin_Watson  0.0 gof  0.0

"""

    global_lines = f"""\'----------------------------------------### Global parameters ###----------------------------------------

"""

    # take variables from bkg, POWGEN_instrument_2, & GSAS_btb from topas output file in 02_pawley_on_P1 and put them in data_lines_1 and data_lines_2
    # set these variables to not refine so we have consistent background and peakshape for all simulated xy files
    data_lines_1 = f"""\'----------------------------------------### Defining data sets ###----------------------------------------
    #define use_data 
    #ifdef use_data
    #define USE_BS    macro filename_bs    {{"{xdd_path}"}}
    #endif

\'----------------------------------------### Main refinement block ###----------------------------------------
    #ifdef USE_BS
    xdd filename_bs xye_format
        start_X {start_finish_x[0]}
        finish_X {start_finish_x[1]}
        local !neutron 1 local !xray 0 
        neutron_data
        r_wp  2.60439382
        {TOF_x_axis_calibration}
        local !bank_two_theta 168.567 
        local !mu  0.07076 
        local !fixb  0.0 min -10 max 10 
        {background_line}
        scale_pks = (Exp(-2*fixb/(2 D_spacing)^2)); 'fixb correction local to each bank
        weighting = If(SigmaYobs < 1e-9, 1, 1/SigmaYobs^2);
            x_calculation_step = Yobs_dx_at(Xo); convolution_step 4
            TOF_LAM(0.001)
            scale_pks = D_spacing^4;
            scale_pks = Exp(-mu*2 D_spacing Sin(0.00872664626 two_theta));   'absorption correction local to each bank

            #define peak1
            #ifdef peak1
                wifd_pkshape(!two_theta, =bank_two_theta;, !aprm, 2.50000, !tauf_2, 9.80744, !tauf_1, 0.43977, !tauf_0, 0.000, !taus_0, 25.27420, !taus_1, 2.66797, !t_eff, 144.41296, !hhh, 3.58595, !hhh_0, 11.61348)
            #endif

"""
    
    hkl_is_lines = f"""
        hkl_Is
            scale @ 1
            {TOF_sample_peakshape} 
            r_bragg  0
            
            phase_name \"{cif_name}\"	
            {topas_latparam_lines[1]}
            {topas_latparam_lines[2]}
            {topas_latparam_lines[3]}
            {topas_latparam_lines[4]}
            {topas_latparam_lines[5]}
            {topas_latparam_lines[6]}
            {topas_latparam_lines[0]}
"""

    output_lines_XYcalc = f""" 
        'Out_CIF_STR("{cif_name}_pawley.cif")
        \'Out_X_Ycalc(\"{xy_out_file}\")
"""
    output_lines_hkl = f""" 
        \'Create_d_Ip_file_dayton(\"{hkl_out_file}\")
"""
    endif_line = f"""
    #endif
"""

    usr_macros = """\'----------------------------------------### User defined macros ###----------------------------------------

"""

    with open(inp_filename, 'w') as f:
        f.write(refine_lines)
        f.write(global_lines)
        f.write(data_lines_1)

        if use_strain_modes == True:
            for i in range(len(topas_latparam_lines)):
                f.write(topas_latparam_lines[i])
        else:
            f.write(hkl_is_lines)

        f.write(output_lines_XYcalc)
        f.write(output_lines_hkl)
        f.write(endif_line)
        f.write(usr_macros)
        for i in range(len(macro_lines)):
            f.write(macro_lines[i])

    return None


def write_topas_input_pawley(topas_input_filename, 
                             instrument_lines,
                             topas_latparam_lines,
                             iterations = 0,
                             conv_criteria = 0.0001,
                             continue_after_convergence = False,
                             anneal = [False, 0.00001],
                             do_errors = False,
                             limit_warnings = False,
                             xdd_wt_percent_lines = None, 
                             xy_out_file = None, 
                             dy_out_file = None, 
                             hkl_out_file=None, 
                             hkl_scaled_out_file=None):

    '''
    Parameters
    ----------
    topas_input_filename: str
        Name of the INP file to write
    instrument_lines: str
        long f-string that contains peak fitting parameters specific to the instrument
        choose from: POWGEN, HRPD functions
    topas_latparam_lines: list of str
        list of strings defining the lattice parameters and their refinement
    iterations: int
        number of refinement iterations for TOPAS to run
    conv_criteria: float
        chi^2 convergence criteria
        default = 0.0001
    continue_after_convergence: bool
        continues the refinement after convergence
        default: False
    anneal: list[bool, float]
        performs simulated annealing
        default: [False, 0.00001]
    do_errors: bool
        calculated the error for refined parameters and generates the C-matrix (costly function to run in TOPAS)
        default: False
    limit_warnings: bool
        adds a warning to the line if refined parameter is close to the min/max limits
        default: False
    xy_out_file: str
        Path and Name of the output file for the calculated xy diffraction pattern with TOF on x axis
    dy_out_file: str
        Path and Name of the output file for the calculated xy diffraction pattern with d-spacing on x axis
    hkl_out_file: str
        Path and Name of the output file for the hkl intensities
    hkl_scaled_out_file: str
        Path and name of the output file for topas scaled hkl intensities

    Returns:
    ------------------
        Returns nothing, but writes the TOPAS input file
    '''
    if continue_after_convergence == False:
        continue_line = f'\'continue_after_convergence'
    else:
        continue_line = f'continue_after_convergence'
    
    if anneal[0] == False:
        anneal_line = f'\'Auto_T({anneal[1]})  \' simulated annealing'
    else:
        anneal_line = f'Auto_T({anneal[1]})  \' simulated annealing'

    if do_errors == False:
        error_line = f'\'do_errors'
    else:
        error_line = f'do_errors'
    
    if limit_warnings == False:
        limit_warn = f'no_LIMIT_warnings'
    else:
        limit_warn = f'\'no_LIMIT_warnings'
    
    refine_lines = f"""\'----------------------------------------### Refinement methods ###----------------------------------------
iters  {iterations}
chi2_convergence_criteria {conv_criteria}
{anneal_line}
{error_line}
{limit_warn}
{continue_line}
r_wp  0.0 r_exp  0.0 r_p  0.0 r_wp_dash  0.0 r_p_dash  0.0 r_exp_dash  0.0 weighted_Durbin_Watson  0.0 gof  0.0

"""
    global_lines = f"""\'----------------------------------------### Global parameters ###----------------------------------------
prm !pow  4
"""
    #* INSTRUMENT LINES WILL BE HERE 

    #* MULTI-PHASE WT% LINES GO HERE - UNDER XDD, BUT ABOVE STR SCOPE

    #* Pawley lattice lines will be here once written
    
    if xy_out_file != None:
        output_lines_XYcalc = f""" 
        Out_X_Ycalc(\"{xy_out_file}\")

"""
    if dy_out_file != None:
        output_lines_DYcalc = f""" 
        Out_d_from_tof_bragg_dayton("{dy_out_file}", d0_1, difc_1, difa_1)

"""
    if hkl_out_file != None:
        output_lines_hkl = f""" 
        Create_d_Ip_file_dayton(\"{hkl_out_file}\")
        
"""
    if hkl_scaled_out_file != None:
        output_lines_hkl_scaled = f""" 
        Create_d_IScaled_file_dayton(\"{hkl_scaled_out_file}\")
        
"""

    endif_line = '\n#endif\n'
    usr_macros = """\'----------------------------------------### User defined macros ###----------------------------------------

"""

    with open(topas_input_filename, 'w') as f:

        f.write(refine_lines)

        f.write(global_lines)

        f.write(instrument_lines)
        
        if xdd_wt_percent_lines != None:
            for i in range(len(xdd_wt_percent_lines)):
                f.write(xdd_wt_percent_lines[i])
        
        # topas_latparam_lines could be string or list depending on function used to make them
        if type(topas_latparam_lines) is str:
            f.write(topas_latparam_lines)
        else:
            for i in range(len(topas_latparam_lines)):
                f.write(topas_latparam_lines[i])


        if xy_out_file != None:
            f.write(output_lines_XYcalc)
        if dy_out_file != None:
            f.write(output_lines_DYcalc)
        if hkl_out_file != None:
            f.write(output_lines_hkl)
        if hkl_scaled_out_file != None:
            f.write(output_lines_hkl_scaled)
            
        f.write(endif_line)
        f.write(usr_macros)

    return None

















