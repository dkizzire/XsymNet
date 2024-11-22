import os
import glob
import numpy as np
import math
import write_TOPAS_funcs as funcs
import utility_functions as util


lsa, lsb, lsc, lsal, lsbe, lsga = [], [], [], [], [], []

for i in range(1):
    random_strain = np.random.uniform(low=-0.01, high=0.01, size=(6))

    # s1 = random_strain[0]
    # s2 = random_strain[1]
    # s3 = random_strain[2]
    # s4 = random_strain[3]
    # s5 = random_strain[4]
    # s6 = random_strain[5]

    s1 = 0.001
    s2 = 0.001
    s3 = 0.0
    s4 = 0.0
    s5 = 0.0
    s6 = 0.0

    # P1 
    # pe1 = 0.57735*s1 -  0.40825*s2 +  0.70711*s3
    # pe2 = 0.57735*s1 -  0.40825*s2 +  0.70711*s3
    # pe3 = 0.57735*s1 +  0.81650*s2
    # pe4 = 1.41421*s5
    # pe5 = 1.41421*s6
    # pe6 = 1.41421*s4
    # basis = [1,1,0,1,-1,0,0,0,-2]

    # BSO 0094
    pe1 = +  0.57735*s1 -  0.40825*s2
    pe2 = +  0.57735*s1 -  0.40825*s2
    pe3 = +  0.57735*s1 +  0.81650*s2
    pe4 = 0
    pe5 = 0
    pe6 = +  1.41421*s3

    basis = [0.5,0.5,0,-1,1,0,0,0,2]

    s_mat = [s1, s2, s3, s4, s5, s6]
    p_mat = [pe1, pe2, pe3, pe4, pe5, pe6]

    bv = util.straincell(10.72250,10.72250,10.72250,90.00000,90.00000,90.00000)

    sa, sb, sc, sal, sbe, sga = util.celltransform(s_mat, p_mat, bv, basis)

    print('\nMin and Max of Values')
    print(f'a = {sa:.4f}')
    print(f'b = {sb:.4f}')
    print(f'c = {sc:.4f}')
    print(f'al = {sal:.4f}')
    print(f'be = {sbe:.4f}')
    print(f'ga = {sga:.4f}')
    print()

#     lsa.append(sa)
#     lsb.append(sb)
#     lsc.append(sc)
#     lsal.append(sal)
#     lsbe.append(sbe)
#     lsga.append(sga)

# print('\nMin and Max of Values')
# print(f'a = {min(lsa):.4f},  {max(lsa):.4f}')
# print(f'b = {min(lsb):.4f},  {max(lsb):.4f}')
# print(f'c = {min(lsc):.4f},  {max(lsc):.4f}')
# print(f'al = {min(lsal):.4f},  {max(lsal):.4f}')
# print(f'be = {min(lsbe):.4f},  {max(lsbe):.4f}')
# print(f'ga = {min(lsga):.4f},  {max(lsga):.4f}')
# print()

# print(f'{7.58195*.999}, {7.58195*1.001}')
# print(f'{21.44500*.999}, {21.44500*1.001}')
# print(f'{15.16390*.999}, {15.16390*1.001}')



# strs = sorted(glob.glob('./TOPAS/topas_strs_strain/*.str'))
# cif_filenames = [os.path.basename(x) for x in strs]
# cif_noext = [os.path.splitext(x)[0] for x in cif_filenames]
# subgroup_nums = [x[5:] for x in cif_noext]

# param_counts = {}
# for i in range(len(strs)):
#     count = funcs.count_refinable_parameters_in_topas_strs(strs[i])
#     param_counts[subgroup_nums[i]] = count

# funcs.save_dict_to_json(param_counts, './subgroup_refinable_parameter_counts.json')
