'''
Author: Dr. Dayton  Kizzire
Last Updated: 08/22/2024
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

def refinement_options(iters:int = 0, converge:float = 0.0001, continue_on_convergence:bool = False, anneal:bool = False, anneal_param:float = 0.00001, do_errors:bool = False, limit_warn:bool = False):
    if continue_on_convergence == False:
        continue_line = f'\'continue_after_convergence'
    else:
        continue_line = f'continue_after_convergence'
    
    if anneal == False:
        anneal_line = f'\'Auto_T({anneal_param})  \' simulated annealing'
    else:
        anneal_line = f'Auto_T({anneal_param})  \' simulated annealing'

    if do_errors == False:
        error_line = f'\'do_errors'
    else:
        error_line = f'do_errors'
    
    if limit_warn == False:
        limit_warn_line = f'no_LIMIT_warnings'
    else:
        limit_warn_line = f'\'no_LIMIT_warnings'
    
    refine_lines = f"""\'----------------------------------------### Refinement methods ###----------------------------------------
iters  {iters}
chi2_convergence_criteria {converge}
{continue_line}
{anneal_line}
{error_line}
{limit_warn_line}

r_wp  0.0 r_exp  0.0 r_p  0.0 r_wp_dash  0.0 r_p_dash  0.0 r_exp_dash  0.0 weighted_Durbin_Watson  0.0 gof  0.0

"""
    return refine_lines


def global_parameters():
    global_lines = f"""\'----------------------------------------### Global parameters ###----------------------------------------
prm !pow  4
"""
    return global_lines


def xdd(data_winpath:str):

    xdd_line = f'''xdd "{data_winpath}" '''
    return xdd_line


def instrument_parameters(instrument_file:str):
    with open(instrument_file, 'r') as f:
        lines = f.readlines()
    instrument_lines = ''.join(lines)
    return instrument_lines


